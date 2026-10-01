#include "stdafx.h"

#include "Emu/System.h"
#include "RSXFIFO.h"
#include "RSXThread.h"
#include "Capture/rsx_capture.h"
#include "Core/RSXReservationLock.hpp"
#include "Emu/Memory/vm_reservation.h"
#include "Emu/Cell/lv2/sys_rsx.h"
#include "NV47/HW/context.h"

#include "util/asm.hpp"

#include <thread>

using spu_rdata_t = std::byte[128];

extern void mov_rdata(spu_rdata_t& _dst, const spu_rdata_t& _src);
extern bool cmp_rdata(const spu_rdata_t& _lhs, const spu_rdata_t& _rhs);

namespace rsx
{
	namespace FIFO
	{
		FIFO_control::FIFO_control(::rsx::thread* pctrl)
		{
			m_thread = pctrl;
			m_ctrl = pctrl->ctrl;
			m_iotable = &pctrl->iomap_table;
		}

		u32 FIFO_control::translate_address(u32 address) const
		{
			return m_iotable->get_addr(address);
		}

		void FIFO_control::sync_get() const
		{
			m_ctrl->get.release(m_internal_get);
		}

		void FIFO_control::restore_state(u32 cmd, u32 count)
		{
			m_cmd = cmd;
			m_command_inc = ((m_cmd & RSX_METHOD_NON_INCREMENT_CMD_MASK) == RSX_METHOD_NON_INCREMENT_CMD) ? 0 : 4;
			m_remaining_commands = count;
			m_internal_get = m_ctrl->get - 4;
			m_args_ptr = m_iotable->get_addr(m_internal_get);
			m_command_reg = (m_cmd & 0xffff) + m_command_inc * (((m_cmd >> 18) - count) & 0x7ff) - m_command_inc;
		}

		void FIFO_control::inc_get(bool wait)
		{
			m_internal_get += 4;

			if (wait && read_put<false>() == m_internal_get)
			{
				// NOTE: Only supposed to be invoked to wait for a single arg on command[0] (4 bytes)
				// Wait for put to allow us to procceed execution
				sync_get();
				invalidate_cache();

				while (read_put() == m_internal_get && !Emu.IsStopped())
				{
					m_thread->cpu_wait({});
				}
			}
		}

		template <bool Full>
		inline u32 FIFO_control::read_put() const
		{
			if constexpr (!Full)
			{
				return m_ctrl->put & ~3;
			}
			else
			{
				if (u32 put = m_ctrl->put; (put & 3) == 0) [[likely]]
				{
					return put;
				}

				return m_ctrl->put.and_fetch(~3);
			}
		}

		std::pair<bool, u32> FIFO_control::fetch_u32(u32 addr)
		{
			if (addr - m_cache_addr >= m_cache_size)
			{
				const u32 put = read_put();

				if (put == addr)
				{
					return {false, FIFO_EMPTY};
				}

				m_cache_addr = addr & -128;

				const u32 addr1 = m_iotable->get_addr(m_cache_addr);

				if (addr1 == umax)
				{
					m_cache_size = 0;
					return {false, FIFO_ERROR};
				}

				m_cache_size = std::min<u32>((put | 0x7f) - m_cache_addr, u32{sizeof(m_cache)} - 1) + 1;

				if (0x100000 - (m_cache_addr & 0xfffff) < m_cache_size)
				{
					// Check if memory layout changes in the next 1MB page boundary
					if ((addr1 >> 20) + 1 != (m_iotable->get_addr(m_cache_addr + 0x100000) >> 20))
					{
						// Trim cache as needed if memory layout changes
						m_cache_size = 0x100000 - (m_cache_addr & 0xfffff);
					}
				}

				// Make mask of cache lines to fetch
				u8 to_fetch = static_cast<u8>((1u << (m_cache_size / 128)) - 1);

				if (addr < put && put < m_cache_addr + m_cache_size)
				{
					// Adjust to knownly-prepared FIFO buffer bounds
					m_cache_size = put - m_cache_addr;
				}

				// Atomic FIFO debug options
				const bool force_cache_fill = g_cfg.core.rsx_fifo_accuracy == rsx_fifo_mode::atomic_ordered;
				const bool strict_fetch_ordering = g_cfg.core.rsx_fifo_accuracy >= rsx_fifo_mode::atomic_ordered;

				rsx::reservation_lock<true, 1> rsx_lock(addr1, m_cache_size, true);
				const auto src = vm::_ptr<spu_rdata_t>(addr1);

				u64 start_time = 0;
				u32 bytes_read = 0;

				// Find the next set bit after every iteration
				for (int i = 0;; i = (std::countr_zero<u32>(std::rotl<u8>(to_fetch, 0 - i - 1)) + i + 1) % 8)
				{
					// If a reservation is being updated, try to load another
					const auto& res = vm::reservation_acquire(addr1 + i * 128);
					const u64 time0 = res;

					if (!(time0 & 127))
					{
						mov_rdata(m_cache[i], src[i]);

						if (time0 == res && cmp_rdata(m_cache[i], src[i]))
						{
							// The fetch of the cache line content has been successful, unset its bit
							to_fetch &= ~(1u << i);

							if (!to_fetch)
							{
								break;
							}

							bytes_read += 128;
							continue;
						}
					}

					if (!start_time)
					{
						if (bytes_read >= 256 && !force_cache_fill)
						{
							// Cut our losses if we have something to work with.
							// This is the first time falling out of the reservation loop above, so we have clean data with no holes.
							m_cache_size = bytes_read;
							break;
						}

						start_time = get_system_time();
					}

					auto now = get_system_time();
					if (now - start_time >= 50u)
					{
						if (m_thread->is_stopped())
						{
							return {};
						}

						m_thread->cpu_wait({});

						const auto then = std::exchange(now, get_system_time());
						start_time = now;
						m_thread->performance_counters.idle_time += now - then;
					}
					else
					{
						busy_wait(200);
					}

					if (strict_fetch_ordering)
					{
						i = (i - 1) % 8;
					}
				}
			}

			const auto ret = read_from_ptr_unsafe<be_t<u32>>(+m_cache[0], addr - m_cache_addr);
			return {true, ret};
		}

		void FIFO_control::set_get(u32 get, u32 spin_cmd)
		{
			invalidate_cache();

			if (spin_cmd && m_ctrl->get == get)
			{
				m_memwatch_addr = get;
				m_memwatch_cmp = spin_cmd;
				return;
			}

			// Update ctrl registers
			m_ctrl->get.release(m_internal_get = get);
			m_remaining_commands = 0;
		}

		std::span<const u32> FIFO_control::get_current_arg_ptr(u32 length_in_words) const
		{
			if (!length_in_words)
			{
				// This means the caller is doing something stupid
				rsx_log.error("Invalid access to FIFO args data, requested length = 0");
				return {};
			}

			if (g_cfg.core.rsx_fifo_accuracy)
			{
				// Return a pointer to the cache storage with confined access
				const u32 cache_offset_in_words = (m_internal_get - m_cache_addr) / 4;
				const u32 cache_size_in_words = m_cache_size / 4;
				return {reinterpret_cast<const u32*>(&m_cache) + cache_offset_in_words, cache_size_in_words - cache_offset_in_words};
			}

			// Return a raw pointer to contiguous memory
			constexpr u32 _1M = 0x100000;
			const u32 size = length_in_words * sizeof(u32);
			const u32 from = m_iotable->get_addr(m_internal_get);

			for (u32 remaining = size, addr = m_internal_get, ptr = from; remaining > 0;)
			{
				const u32 next_block = utils::align(addr + 1, _1M);
				const u32 available = (next_block - addr);
				if (remaining <= available)
				{
					return { static_cast<const u32*>(vm::base(from)), length_in_words };
				}

				remaining -= available;
				const u32 next_ptr = m_iotable->get_addr(next_block);
				if (next_ptr != (ptr + available))
				{
					return { static_cast<const u32*>(vm::base(from)), (size - remaining) / sizeof(u32)};
				}

				ptr = next_ptr;
				addr = next_block;
			}

			fmt::throw_exception("Unreachable");
		}

		bool FIFO_control::read_unsafe(register_pair& data)
		{
			// Fast read with no processing, only safe inside a PACKET_BEGIN+count block
			if (m_remaining_commands)
			{
				bool ok{};
				u32 arg = 0;

				if (g_cfg.core.rsx_fifo_accuracy) [[ unlikely ]]
				{
					std::tie(ok, arg) = fetch_u32(m_internal_get + 4);

					if (!ok)
					{
						if (arg == FIFO_ERROR)
						{
							m_thread->recover_fifo();
						}

						return false;
					}
				}
				else
				{
					if (m_internal_get + 4 == read_put<false>())
					{
						return false;
					}

					m_args_ptr += 4;
					arg = vm::read32(m_args_ptr);
				}

				m_internal_get += 4;

				m_command_reg += m_command_inc;

				--m_remaining_commands;

				data.set(m_command_reg, arg);
				return true;
			}

			m_internal_get += 4;
			return false;
		}

		// Optimization for methods which can be batched together
		// Beware, can be easily misused
		bool FIFO_control::skip_methods(u32 count)
		{
			if (m_remaining_commands > count)
			{
				m_command_reg += m_command_inc * count;
				m_remaining_commands -= count;
				m_internal_get += 4 * count;
				m_args_ptr += 4 * count;
				return true;
			}

			m_internal_get += 4 * m_remaining_commands;
			m_remaining_commands = 0;
			return false;
		}

		void FIFO_control::abort()
		{
			m_remaining_commands = 0;
		}

		void FIFO_control::read(register_pair& data)
		{
			if (m_remaining_commands)
			{
				// Previous block aborted to wait for PUT pointer
				read_unsafe(data);
				return;
			}

			// vanillad1 selfjump (L20.2.2): how long RSX has spun on the jump-to-self at s_spin_get
			static u32 s_spin_get = umax;
			static u64 s_spin_since = 0;

			if (m_memwatch_addr)
			{
				if (m_internal_get == m_memwatch_addr)
				{
					if (const u32 addr = m_iotable->get_addr(m_memwatch_addr); addr + 1)
					{
						if (vm::read32(addr) == m_memwatch_cmp)
						{
							// vanillad1 selfjump: a self-jump the game never re-pointed while commands are queued
							// past it (put far away, a NOP after it) deadlocks Destiny (BLUS31181): its job thread
							// waits for a fence the GPU writes behind the jump. After 2 s, go on at get + 4, which is
							// what the missing edit does. The normal idle has put right behind the jump.
							const u64 now = get_system_time();

							if (s_spin_get != m_memwatch_addr)
							{
								s_spin_get = m_memwatch_addr;
								s_spin_since = now;
							}

							const u32 put = read_put();

							if (now - s_spin_since < 2'000'000 || put == m_memwatch_addr || put - m_memwatch_addr - 1 < 0x200u || vm::read32(addr + 4) != 0)
							{
								// Still spinning in place
								data.reg = FIFO_EMPTY;
								return;
							}

							rsx_log.error("vanillad1 selfjump: RSX spun %.1fs on a jump-to-self at io 0x%x (ea 0x%x, cmd 0x%x) with put 0x%x; continuing at io 0x%x",
								(now - s_spin_since) / 1'000'000., m_memwatch_addr, addr, m_memwatch_cmp, put, m_memwatch_addr + 4);

							invalidate_cache();
							m_ctrl->get.release(m_internal_get = m_memwatch_addr + 4);
							m_remaining_commands = 0;
						}
					}
				}

				s_spin_get = umax;
				m_memwatch_addr = 0;
				m_memwatch_cmp = 0;
			}

			if (!g_cfg.core.rsx_fifo_accuracy) [[ likely ]]
			{
				const u32 put = read_put();

				if (put == m_internal_get)
				{
					// Nothing to do
					data.reg = FIFO_EMPTY;
					return;
				}

				if (const u32 addr = m_iotable->get_addr(m_internal_get); addr + 1)
				{
					m_cmd = vm::read32(addr);
				}
				else
				{
					data.reg = FIFO_ERROR;
					return;
				}
			}
			else
			{
				if (auto [ok, arg] = fetch_u32(m_internal_get); ok)
				{
					m_cmd = arg;
				}
				else
				{
					data.reg = arg;
					return;
				}
			}

			if (m_cmd & RSX_METHOD_NON_METHOD_CMD_MASK) [[unlikely]]
			{
				if ((m_cmd & RSX_METHOD_OLD_JUMP_CMD_MASK) == RSX_METHOD_OLD_JUMP_CMD ||
					(m_cmd & RSX_METHOD_NEW_JUMP_CMD_MASK) == RSX_METHOD_NEW_JUMP_CMD ||
					(m_cmd & RSX_METHOD_CALL_CMD_MASK) == RSX_METHOD_CALL_CMD ||
					(m_cmd & RSX_METHOD_RETURN_MASK) == RSX_METHOD_RETURN_CMD)
				{
					// Flow control, stop reading
					data.reg = m_cmd;
					return;
				}

				// Malformed command, optional recovery
				data.reg = FIFO_ERROR;
				return;
			}

			ensure(!m_remaining_commands);
			const u32 count = (m_cmd >> 18) & 0x7ff;

			if (!count)
			{
				m_ctrl->get.release(m_internal_get += 4);
				data.reg = FIFO_NOP;
				return;
			}

			if (count > 1)
			{
				// Set up readback parameters
				m_command_reg = m_cmd & 0xfffc;
				m_command_inc = ((m_cmd & RSX_METHOD_NON_INCREMENT_CMD_MASK) == RSX_METHOD_NON_INCREMENT_CMD) ? 0 : 4;
				m_remaining_commands = count - 1;
			}

			if (g_cfg.core.rsx_fifo_accuracy)
			{
				m_internal_get += 4;

				auto [ok, arg] = fetch_u32(m_internal_get);

				if (!ok)
				{
					// Optional recovery
					if (arg == FIFO_ERROR)
					{
						data.reg = FIFO_ERROR;
					}
					else
					{
						data.reg = FIFO_EMPTY;
						m_command_reg = m_cmd & 0xfffc;
						m_remaining_commands++;
					}

					return;
				}

				data.set(m_cmd & 0xfffc, arg);
				return;
			}

			inc_get(true); // Wait for data block to become available

			// Validate the args ptr if the command attempts to read from it
			m_args_ptr = m_iotable->get_addr(m_internal_get);
			if (m_args_ptr == umax) [[unlikely]]
			{
				// Optional recovery
				data.reg = FIFO_ERROR;
				return;
			}

			data.set(m_cmd & 0xfffc, vm::read32(m_args_ptr));
		}

		// vanillad1: flatten repeat draws (P8): VANILLAD1_FLATTEN=1 keeps the flattener open across the game's constant
		// uploads between two draws of one mesh (the game instances by hand)
		bool vd1_flatten_enabled()
		{
			static const bool s_on = []()
			{
				const char* env = std::getenv("VANILLAD1_FLATTEN");
				const bool on = env && (*env == '1' || *env == '2');
				if (on) rsx_log.notice("vanillad1: FIFO flattener keeps repeat draws open (VANILLAD1_FLATTEN=%s)", env);
				return on;
			}();
			return s_on;
		}

		bool vd1_flatten_log()
		{
			static const bool s_log = []() { const char* env = std::getenv("VANILLAD1_FLATTEN"); return env && *env == '2'; }();
			return s_log;
		}

		u32 vd1_flatten_max()
		{
			static const u32 s_max = []()
			{
				const char* env = std::getenv("VANILLAD1_FLATTEN_MAX");
				const u32 n = env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
				if (n) rsx_log.notice("vanillad1: FIFO flattener runs limited to %u draws (VANILLAD1_FLATTEN_MAX)", n);
				return n;
			}();
			return s_max;
		}

		bool vd1_flatten_off(char what)
		{
			static const std::string s_off = []()
			{
				const char* env = std::getenv("VANILLAD1_FLATTEN_OFF");
				if (env && *env) rsx_log.notice("vanillad1: FIFO flattener pieces off: %s (VANILLAD1_FLATTEN_OFF)", env);
				return env ? std::string(env) : std::string();
			}();
			return s_off.find(what) != std::string::npos;
		}

		void flattening_helper::reset(bool _enabled)
		{
			enabled = _enabled;
			num_collapsed = 0;
			in_begin_end = false;
		}

		void flattening_helper::force_disable()
		{
			if (enabled)
			{
				rsx_log.warning("FIFO optimizations have been disabled as the application is not compatible with per-frame analysis");

				reset(false);
				fifo_hint = optimization_hint::application_not_compatible;
			}
		}

		void flattening_helper::evaluate_performance(u32 total_draw_count)
		{
			// vanillad1: flatten repeat draws (P8): one line a frame with VANILLAD1_FLATTEN=2
			if (enabled && vd1_flatten_log())
			{
				rsx_log.notice("vanillad1 flatten: chained=%u noconst=%u norange=%u otherrange=%u primchg=%u breaker=%u other=%u collapsed=%u",
					vd1_stat[0], vd1_stat[1], vd1_stat[2], vd1_stat[3], vd1_stat[4], vd1_stat[5], vd1_stat[6], num_collapsed);
				rsx_log.notice("vanillad1 peek: args=%u beyondput=%u translate=%u nonmethod=%u othermethod=%u mixed=%u incr=%u wordsput=%u wordtr=%u gap=%u nodraw=%u many=%u",
					vd1_pstat[1], vd1_pstat[2], vd1_pstat[3], vd1_pstat[4], vd1_pstat[5], vd1_pstat[6], vd1_pstat[7], vd1_pstat[8], vd1_pstat[9], vd1_pstat[10], vd1_pstat[11], vd1_pstat[12]);
			}
			for (u32& s : vd1_stat) s = 0;
			for (u32& s : vd1_pstat) s = 0;

			if (!enabled)
			{
				if (fifo_hint == optimization_hint::application_not_compatible)
				{
					// Not compatible, do nothing
					return;
				}

				if (total_draw_count <= 2000)
				{
					// Low draw call pressure
					fifo_hint = optimization_hint::load_low;
					return;
				}

				if (fifo_hint == optimization_hint::load_unoptimizable)
				{
					// Nope, wait for stats to change
					return;
				}
			}

			if (enabled)
			{
				// Currently activated. Check if there is any benefit
				// vanillad1: flatten repeat draws (P8): never off for lack of benefit with VANILLAD1_FLATTEN (the first frame, a loading frame, said 43 of 3889)
				if (num_collapsed < (vd1_flatten_enabled() ? 0u : 500u))
				{
					// Not worth it, disable
					if (vd1_flatten_enabled()) rsx_log.notice("vanillad1: FIFO flattener off (collapsed %u of %u draws)", num_collapsed, total_draw_count + num_collapsed);
					enabled = false;
					fifo_hint = load_unoptimizable;
				}

				u32 real_total = total_draw_count + num_collapsed;
				if (real_total <= 2000)
				{
					// Low total number of draws submitted, no need to keep trying for now
					enabled = false;
					fifo_hint = load_low;
				}

				reset(enabled);
			}
			else
			{
				// Not enabled, check if we should try enabling
				ensure(total_draw_count > 2000);
				if (fifo_hint != load_unoptimizable)
				{
					// If its set to unoptimizable, we already tried and it did not work
					// If it resets to load low (usually after some kind of loading screen) we can try again
					ensure(in_begin_end == false); // "Incorrect initial state"
					ensure(num_collapsed == 0);
					enabled = true;
					if (vd1_flatten_enabled()) rsx_log.notice("vanillad1: FIFO flattener on (%u draws a frame)", total_draw_count); // vanillad1: flatten repeat draws (P8)
				}
			}
		}

		// vanillad1: flatten repeat draws (P8): a look-ahead gave no range (VANILLAD1_FLATTEN=2: counted by reason, the first 60 logged with the words)
		void flattening_helper::vd1_peek_fail(u32 why, u32 a, u32 b, u32 c)
		{
			if (!vd1_flatten_log())
			{
				return;
			}

			vd1_pstat[why & 15]++;
			if (vd1_pfail_logged < 60)
			{
				vd1_pfail_logged++;
				rsx_log.notice("vanillad1 peek fail %u: %08x %08x %08x", why, a, b, c);
			}
		}

		flatten_op flattening_helper::test(register_pair& command, const u32* regs, u64 vd1_range)
		{
			u32 flush_cmd = ~0u;
			switch (const u32 reg = (command.reg >> 2))
			{
			case NV4097_SET_BEGIN_END:
			{
				in_begin_end = !!command.value;

				if (command.value)
				{
					// This is a BEGIN call
					if (!deferred_primitive) [[likely]]
					{
						// New primitive block
						deferred_primitive = command.value;
						vd1_last_range = (command.value == 5u) ? vd1_range : 0; // vanillad1: flatten repeat draws (P8)
					}
					// vanillad1: flatten repeat draws (P8): with the env a repeat chains only if constants came since the END, the primitive is a
					// triangle list (no restart barriers) and the next block's draw words are one range equal to the last
					// block's (the run is then one instanced draw; any other merged run would lose its constants)
					else if (deferred_primitive == command.value && (!regs || !vd1_flatten_enabled() ||
						(vd1_constants_seen && !vd1_flatten_off('c') && vd1_range && vd1_range == vd1_last_range && (!vd1_flatten_max() || draw_count < vd1_flatten_max()))))
					{
						// Same primitive can be chanined; do nothing
						vd1_constants_seen = false;
						vd1_stat[0]++;
						command.reg = FIFO_DISABLED_COMMAND;
					}
					else
					{
						// Primitive command has changed!
						// Flush
						flush_cmd = command.value;
						// vanillad1: flatten repeat draws (P8): why the run closed (VANILLAD1_FLATTEN=2)
						if (vd1_flatten_log())
						{
							vd1_stat[deferred_primitive != command.value ? 4 : !vd1_constants_seen ? 1 : !vd1_range ? 2 : vd1_range != vd1_last_range ? 3 : 6]++;
						}
						vd1_last_range = (command.value == 5u) ? vd1_range : 0; // vanillad1: flatten repeat draws (P8)
					}
				}
				else if (deferred_primitive)
				{
					command.reg = FIFO_DRAW_BARRIER;
					draw_count++;
					vd1_constants_seen = false; // vanillad1: flatten repeat draws (P8)
				}
				else
				{
					rsx_log.error("Fifo flattener misalignment, disable FIFO reordering and report to developers");
					in_begin_end = false;
					flush_cmd = 0u;
				}

				break;
			}
			case NV4097_DRAW_ARRAYS:
			case NV4097_DRAW_INDEX_ARRAY:
			{
				// TODO: Check type
				break;
			}
			default:
			{
				if (draw_count) [[unlikely]]
				{
					if (m_register_properties[reg] & register_props::always_ignore) [[unlikely]]
					{
						// Always ignore
						command.reg = FIFO_DISABLED_COMMAND;
					}
					// vanillad1: flatten repeat draws (P8): between an END and the next BEGIN the game sends constant data and
					// re-sends the constant load and the index array with their old values: neither
					// ends the run (the RSX is still inside the draw, so the data become barriers)
					else if (regs && !in_begin_end && vd1_flatten_enabled() && !vd1_flatten_off('t') &&
						(reg >= NV4097_SET_TRANSFORM_CONSTANT && reg < NV4097_SET_TRANSFORM_CONSTANT + 32))
					{
						vd1_constants_seen = true;
					}
					else if (regs && !in_begin_end && vd1_flatten_enabled() && !vd1_flatten_off('d') && regs[reg] == command.value &&
						(reg == NV4097_SET_TRANSFORM_CONSTANT_LOAD || reg == NV4097_SET_INDEX_ARRAY_ADDRESS || reg == NV4097_SET_INDEX_ARRAY_DMA))
					{
						command.reg = FIFO_DISABLED_COMMAND;
					}
					else
					{
						// Flush
						flush_cmd = (in_begin_end) ? deferred_primitive : 0u;
						if (vd1_flatten_log() && !in_begin_end) vd1_stat[5]++; // vanillad1: flatten repeat draws (P8)
					}
				}
				else
				{
					// Nothing to do
					return NOTHING;
				}

				break;
			}
			}

			if (flush_cmd != ~0u)
			{
				num_collapsed += draw_count? (draw_count - 1) : 0;
				draw_count = 0;
				deferred_primitive = flush_cmd;

				return in_begin_end ? EMIT_BARRIER : EMIT_END;
			}

			return NOTHING;
		}
	}

	void thread::run_FIFO()
	{
		FIFO::register_pair command;
		fifo_ctrl->read(command);
		const auto cmd = command.reg;

		// vanillad1: flatten repeat draws (P8): the draw words after the BEGIN just read, merged the way draw_clause::append does: one range
		// (first << 32 | count), or 0 if there is anything else, a gap, more than 256 packets, or FIFO data not yet put
		auto vd1_peek_range = [&]() -> u64
		{
			// VANILLAD1_FLATTEN=2: why a look-ahead gave nothing (counted per frame, the first 60 with the raw words)
			auto fail = [&](u32 why, u32 a = 0, u32 b = 0, u32 c = 0) -> u64 { m_flattener.vd1_peek_fail(why, a, b, c); return 0; };

			if (fifo_ctrl->get_remaining_args_count())
			{
				return fail(1, fifo_ctrl->get_remaining_args_count());
			}

			const u32 put = ctrl->put & ~3;
			u32 addr = fifo_ctrl->get_pos() + 4;
			// put behind get = the producer has wrapped the ring: everything up to the ring's end jump is written (measured:
			// put 0x0448cedc, get 0x0450xxxx in the steady plaza, so every look-ahead said 'beyond put')
			const bool bounded = put > fifo_ctrl->get_pos();
			u32 first = 0, next = 0, method = 0;
			bool have = false;

			for (u32 packets = 0; packets < 256; ++packets)
			{
				const u32 head_addr = fifo_ctrl->translate_address(addr);
				if (head_addr == umax)
				{
					return fail(3, addr);
				}

				if (bounded && addr >= put)
				{
					return fail(2, addr, put, packets);
				}

				const u32 head = vm::read32(head_addr);
				if (head & RSX_METHOD_NON_METHOD_CMD_MASK)
				{
					return fail(4, head, addr, packets); // a jump, call, return or nop: not a plain draw block
				}

				const u32 reg = (head & RSX_METHOD_METHOD_MASK) >> 2;
				const u32 count = (head & RSX_METHOD_COUNT_MASK) >> RSX_METHOD_COUNT_SHIFT;
				if (reg == NV4097_SET_BEGIN_END)
				{
					break; // the END
				}

				if (reg != NV4097_DRAW_INDEX_ARRAY && reg != NV4097_DRAW_ARRAYS)
				{
					return fail(5, head, addr, packets); // another method before the END
				}

				if (method && method != reg)
				{
					return fail(6, head);
				}

				if (count > 1 && (head & RSX_METHOD_NON_INCREMENT_CMD_MASK) != RSX_METHOD_NON_INCREMENT_CMD)
				{
					return fail(7, head);
				}

				if (bounded && addr + 4 * (count + 1) > put)
				{
					return fail(8, addr + 4 * (count + 1), put, count);
				}

				method = reg;
				for (u32 i = 1; i <= count; ++i)
				{
					const u32 word_addr = fifo_ctrl->translate_address(addr + 4 * i);
					if (word_addr == umax)
					{
						return fail(9, addr + 4 * i);
					}

					const u32 word = vm::read32(word_addr);
					const u32 start = word & 0xffffff;
					const u32 length = (word >> 24) + 1;
					if (!have)
					{
						first = start;
						next = start + length;
						have = true;
					}
					else if (start == next)
					{
						next += length;
					}
					else
					{
						return fail(10, start, next, length); // not one contiguous range
					}
				}

				addr += 4 * (count + 1);
				if (packets == 255)
				{
					return fail(12);
				}
			}

			if (!have)
			{
				return fail(11, addr);
			}

			return (static_cast<u64>(first) << 32) | (next - first);
		};

		if (cmd & (0xffff0000 | RSX_METHOD_NON_METHOD_CMD_MASK)) [[unlikely]]
		{
			// vanillad1: label batching (P9): the FIFO ran dry (FIFO_EMPTY, also the jump-to-self spin; not the game's NOP padding, FIFO_NOP): the guest may be waiting for a label
			if (m_vd1_label_open && cmd == FIFO::FIFO_EMPTY) vd1_label_flush();
			// Check for special FIFO commands
			switch (cmd)
			{
			case FIFO::FIFO_NOP:
			{
				if (performance_counters.state == FIFO::state::running)
				{
					performance_counters.FIFO_idle_timestamp = get_system_time();
					performance_counters.state = FIFO::state::nop;
				}

				return;
			}
			case FIFO::FIFO_EMPTY:
			{
				if (performance_counters.state == FIFO::state::running)
				{
					performance_counters.FIFO_idle_timestamp = get_system_time();
					performance_counters.state = FIFO::state::empty;
				}
				else
				{
					std::this_thread::yield();
				}

				return;
			}
			case FIFO::FIFO_BUSY:
			{
				// Do something else
				return;
			}
			case FIFO::FIFO_ERROR:
			{
				rsx_log.error("FIFO error: possible desync event (last cmd = 0x%x)", get_fifo_cmd());
				recover_fifo();
				return;
			}
			}

			// Check for flow control
			if (bit_set<2> jump_type; jump_type
				.set_unsafe(0, (cmd & RSX_METHOD_OLD_JUMP_CMD_MASK) == RSX_METHOD_OLD_JUMP_CMD)
				.set_unsafe(1, (cmd & RSX_METHOD_NEW_JUMP_CMD_MASK) == RSX_METHOD_NEW_JUMP_CMD)
				.any())
			{
				const u32 offs = cmd & (jump_type.test_unsafe(0) ? RSX_METHOD_OLD_JUMP_OFFSET_MASK : RSX_METHOD_NEW_JUMP_OFFSET_MASK);
				if (offs == fifo_ctrl->get_pos())
				{
					//Jump to self. Often preceded by NOP
					if (performance_counters.state == FIFO::state::running)
					{
						performance_counters.FIFO_idle_timestamp = get_system_time();
						sync_point_request.release(true);
					}

					performance_counters.state = FIFO::state::spinning;
				}
				else
				{
					last_known_code_start = offs;
				}

				//rsx_log.warning("rsx jump(0x%x) #addr=0x%x, cmd=0x%x, get=0x%x, put=0x%x", offs, m_ioAddress + get, cmd, get, put);
				fifo_ctrl->set_get(offs, cmd);
				return;
			}
			if ((cmd & RSX_METHOD_CALL_CMD_MASK) == RSX_METHOD_CALL_CMD)
			{
				if (fifo_ret_addr != RSX_CALL_STACK_EMPTY)
				{
					// Only one layer is allowed in the call stack.
					rsx_log.error("FIFO: CALL found inside a subroutine (last cmd = 0x%x)", get_fifo_cmd());
					recover_fifo();
					return;
				}

				const u32 offs = cmd & RSX_METHOD_CALL_OFFSET_MASK;
				fifo_ret_addr = fifo_ctrl->get_pos() + 4;
				fifo_ctrl->set_get(offs);
				last_known_code_start = offs;
				return;
			}
			if ((cmd & RSX_METHOD_RETURN_MASK) == RSX_METHOD_RETURN_CMD)
			{
				if (fifo_ret_addr == RSX_CALL_STACK_EMPTY)
				{
					rsx_log.error("FIFO: RET found without corresponding CALL (last cmd = 0x%x)", get_fifo_cmd());
					recover_fifo();
					return;
				}

				// Optimize returning to another CALL
				if ((ctrl->put & ~3) != fifo_ret_addr)
				{
					if (u32 addr = iomap_table.get_addr(fifo_ret_addr); addr != umax)
					{
						const u32 cmd0 = vm::read32(addr);

						// Check for missing step flags, in case the user is single-stepping in the debugger
						if ((cmd0 & RSX_METHOD_CALL_CMD_MASK) == RSX_METHOD_CALL_CMD && cpu_flag::dbg_step - state)
						{
							fifo_ctrl->set_get(cmd0 & RSX_METHOD_CALL_OFFSET_MASK);
							last_known_code_start = ctrl->get;
							fifo_ret_addr += 4;
							return;
						}
					}
				}

				fifo_ctrl->set_get(std::exchange(fifo_ret_addr, RSX_CALL_STACK_EMPTY));
				last_known_code_start = ctrl->get;
				return;
			}

			// If we reached here, this is likely an error
			fmt::throw_exception("Unexpected command 0x%x (last cmd: 0x%x)", cmd, fifo_ctrl->last_cmd());
		}

		if (const auto state = performance_counters.state;
			state != FIFO::state::running)
		{
			performance_counters.state = FIFO::state::running;

			// Hack: Delay FIFO wake-up according to setting
			// NOTE: The typical spin setup is a NOP followed by a jump-to-self
			// NOTE: There is a small delay when the jump address is dynamically edited by cell
			if (state != FIFO::state::nop)
			{
				fifo_wake_delay();
			}

			// Update performance counters with time spent in idle mode
			performance_counters.idle_time += (get_system_time() - performance_counters.FIFO_idle_timestamp);
		}

		do
		{
			if (capture_current_frame) [[unlikely]]
			{
				const u32 reg = (command.reg & 0xfffc) >> 2;
				const u32 value = command.value;

				frame_debug.command_queue.emplace_back(reg, value);

				if (!(reg == NV406E_SET_REFERENCE || reg == NV406E_SEMAPHORE_RELEASE || reg == NV406E_SEMAPHORE_ACQUIRE))
				{
					// todo: handle nv406e methods better?, do we care about call/jumps?
					rsx::frame_capture_data::replay_command replay_cmd;
					replay_cmd.rsx_command = std::make_pair((reg << 2) | (1u << 18), value);

					auto& commands = frame_capture.replay_commands;
					commands.push_back(replay_cmd);

					switch (reg)
					{
					case NV3089_IMAGE_IN:
						capture::capture_image_in(this, commands.back());
						break;
					case NV0039_BUFFER_NOTIFY:
						capture::capture_buffer_notify(this, commands.back());
						break;
					default:
					{
						static constexpr std::array<std::pair<u32, u32>, 3> ranges
						{{
							{NV308A_COLOR, 0x700},
							{NV4097_SET_TRANSFORM_PROGRAM, 32},
							{NV4097_SET_TRANSFORM_CONSTANT, 32}
						}};

						// Use legacy logic - enqueue leading command with count
						// Then enqueue each command arg alone with a no-op command
						for (const auto& range : ranges)
						{
							if (reg >= range.first && reg < range.first + range.second)
							{
								const u32 remaining = std::min<u32>(fifo_ctrl->get_remaining_args_count() + 1,
									(fifo_ctrl->last_cmd() & RSX_METHOD_NON_INCREMENT_CMD_MASK) ? -1 : (range.first + range.second) - reg);

								commands.back().rsx_command.first = (fifo_ctrl->last_cmd() & RSX_METHOD_NON_INCREMENT_CMD_MASK) | (reg << 2) | (remaining << 18);

								for (u32 i = 1; i < remaining && fifo_ctrl->get_pos() + i * 4 != (ctrl->put & ~3); i++)
								{
									replay_cmd.rsx_command = std::make_pair(0, vm::read32(iomap_table.get_addr(fifo_ctrl->get_pos()) + (i * 4)));

									commands.push_back(replay_cmd);
								}

								break;
							}
						}

						break;
					}
					}
				}
			}

			if (m_flattener.is_enabled()) [[unlikely]]
			{
				// vanillad1: flatten repeat draws (P8): at a BEGIN, look ahead at the block's draw words (one contiguous range or none)
				const bool vd1_begin = FIFO::vd1_flatten_enabled() && (command.reg >> 2) == NV4097_SET_BEGIN_END && command.value;
				switch(m_flattener.test(command, m_ctx->register_state->registers.data(), vd1_begin ? vd1_peek_range() : 0))
				{
				case FIFO::NOTHING:
				{
					break;
				}
				case FIFO::EMIT_END:
				{
					// Emit end command to close existing scope
					AUDIT(in_begin_end);
					methods[NV4097_SET_BEGIN_END](m_ctx, NV4097_SET_BEGIN_END, 0);
					break;
				}
				case FIFO::EMIT_BARRIER:
				{
					AUDIT(in_begin_end);
					methods[NV4097_SET_BEGIN_END](m_ctx, NV4097_SET_BEGIN_END, 0);
					methods[NV4097_SET_BEGIN_END](m_ctx, NV4097_SET_BEGIN_END, m_flattener.get_primitive());
					break;
				}
				default:
				{
					fmt::throw_exception("Unreachable");
				}
				}

				if (command.reg == FIFO::FIFO_DISABLED_COMMAND)
				{
					// Optimized away
					continue;
				}
			}

			const u32 reg = (command.reg & 0xffff) >> 2;
			const u32 value = command.value;
			// vanillad1: label batching (P9): burst mode (n = 1) flushes at the first command that is not a label method; any n before a wait
			if (m_vd1_label_open && (reg == NV406E_SEMAPHORE_ACQUIRE || (m_vd1_label_n == 1 &&
				reg != NV4097_SET_SEMAPHORE_OFFSET && reg != NV4097_BACK_END_WRITE_SEMAPHORE_RELEASE && reg != NV4097_TEXTURE_READ_SEMAPHORE_RELEASE))) [[unlikely]]
			{
				vd1_label_flush();
			}

			m_ctx->register_state->decode(reg, value);

			if (auto method = methods[reg])
			{
				if (m_profiler.enabled && reg != NV406E_SEMAPHORE_ACQUIRE) [[unlikely]] // vanillad1: RSX timers (P3)
				{
					const u64 vd1_t0 = get_system_time();
					method(m_ctx, reg, value);
					m_frame_stats.vd1_methods += static_cast<s64>(get_system_time() - vd1_t0);
				}
				else
				{
					method(m_ctx, reg, value);
				}

				if (state & cpu_flag::again)
				{
					m_ctx->register_state->decode(reg, m_ctx->register_state->latch);
					break;
				}
			}
			else if (m_ctx->register_state->latch != value)
			{
				// Something changed, set signal flags if any specified
				m_graphics_state |= state_signals[reg];
			}
		}
		while (fifo_ctrl->read_unsafe(command));

		fifo_ctrl->sync_get();
	}
}
