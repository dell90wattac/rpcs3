#include "stdafx.h"
#include "VKCommandStream.h"
#include "VKResourceManager.h"
#include "vkutils/descriptors.h"
#include "vkutils/sync.h"

#include "Emu/IdManager.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/system_config.h"
#include "Emu/Cell/timers.hpp" // vanillad1: X527 C1 (get_system_time)
#include "Emu/vd1_live.h"      // vanillad1: X527 C1 (CSUB)

#include <cstdlib>
#include <deque>
#include <optional>

namespace vk
{
	FORCE_INLINE static void queue_submit_impl(const queue_submit_t& submit_info);

	// vanillad1: X527 C1 stage 1 (CSUB): the submit queue (VKCommandStream.h). s_token is held while packets run, so they reach
	// the queue in FIFO order whoever runs them; s_qlock guards s_queue alone (never held while a packet runs). s_inflight
	// counts packets queued or running: help-drain is a no-op at 0 (one load on the hot paths: fence waits, event polls).
	namespace vd1_cs
	{
		static shared_mutex s_token;
		static shared_mutex s_qlock;
		static std::deque<queue_submit_t> s_queue;
		static atomic_t<u32> s_inflight{0};
		static atomic_t<u32> s_wake{0};
		static atomic_t<bool> s_running{false}; // the worker thread is up (set by the worker, cleared when it leaves)

		// Stats, logged every 10 s from the RSX thread's submits
		static atomic_t<u64> st_queued{0}, st_worker{0}, st_helped{0}, st_inline{0}, st_help_calls{0}, st_help_us{0};
		static atomic_t<u32> st_maxq{0};
		static u64 s_last_log_us = 0;

		// Runs every queued packet; the caller holds s_token. Returns how many ran.
		static u32 run_locked()
		{
			u32 n = 0;
			for (;;)
			{
				std::optional<queue_submit_t> p;
				{
					std::lock_guard q(s_qlock);
					if (s_queue.empty())
					{
						break;
					}

					p.emplace(s_queue.front());
					s_queue.pop_front();
				}

				queue_submit_impl(*p);
				s_inflight--;
				n++;
			}

			return n;
		}

		// X527 review F3: the newest stream token among the queued packets; their buffers are replayed BEFORE s_token is taken,
		// so the token covers only the vkQueueSubmit calls and fault threads' inline submits never wait behind a replay
		static u64 newest_token()
		{
			u64 newest = 0;
			std::lock_guard q(s_qlock);
			for (const auto& p : s_queue)
			{
				newest = std::max(newest, p.vd1_replay_token);
			}
			return newest;
		}

		static void maybe_log()
		{
			const u64 now = get_system_time();
			if (now - s_last_log_us < 10'000'000)
			{
				return;
			}

			s_last_log_us = now;
			rsx_log.notice("vanillad1: cs 10s: CSUB=%u queued=%u by worker=%u by helpers=%u inline=%u help calls=%u help ms=%u max queue=%u worker up=%d",
				vd1::live(vd1::lf::CSUB), static_cast<u32>(st_queued.exchange(0)), static_cast<u32>(st_worker.exchange(0)),
				static_cast<u32>(st_helped.exchange(0)), static_cast<u32>(st_inline.exchange(0)), static_cast<u32>(st_help_calls.exchange(0)),
				static_cast<u32>(st_help_us.exchange(0) / 1000), st_maxq.exchange(0), s_running.load() ? 1 : 0);
		}
	}

	bool vd1_cs_boot_on()
	{
		static const bool s_on = []()
		{
			const char* env = std::getenv("VANILLAD1_CS");
			const bool on = env && *env && *env != '0';
			if (on)
			{
				rsx_log.notice("vanillad1: command stream worker on (VANILLAD1_CS=1): deferred submits on 'VD1 Submit Worker' while the live flag CSUB is set; "
					"multithreaded RSX turns it off");
			}
			return on;
		}();
		return s_on;
	}

	void vd1_cs_help_drain()
	{
		if (!vd1_cs::s_inflight) [[likely]]
		{
			return;
		}

		const u64 t0 = get_system_time();
		u32 n = 0;
		if (const u64 newest = vd1_cs::newest_token()) // X527 CSREC (review F3): the queued buffers' replay outside the token
		{
			vd1_cs_replay_until(newest);
		}
		{
			std::lock_guard tk(vd1_cs::s_token); // waits for a packet the worker is running
			n = vd1_cs::run_locked();
		}

		vd1_cs::st_helped += n;
		vd1_cs::st_help_calls++;
		vd1_cs::st_help_us += get_system_time() - t0;
	}

	void vd1_cs_worker_t::operator()()
	{
		// vanillad1: X527: VANILLAD1_CS_CPU=<hex mask> pins the worker (e.g. 2 = the RSX thread's SMT sibling under the split, which
		// leaves the SPUs' cores alone); unset: the OS places it. Logged once.
		if (const char* env = std::getenv("VANILLAD1_CS_CPU"); env && *env)
		{
			const u64 mask = std::strtoull(env, nullptr, 16);
			if (mask)
			{
				thread_ctrl::set_thread_affinity_mask(mask);
			}
			rsx_log.notice("vanillad1: submit worker pinned to 0x%x (VANILLAD1_CS_CPU)", mask);
		}

		vd1_cs_rec_set_worker(); // vanillad1: X527 CSREC (stats: replays by the worker)
		vd1_cs::s_running = true;

		while (thread_ctrl::state() != thread_state::aborting)
		{
			const auto wake_token = vd1_cs::s_wake.observe();

			if (vd1_cs::s_inflight)
			{
				if (const u64 newest = vd1_cs::newest_token()) // X527 CSREC (review F3): replay outside the token
				{
					vd1_cs_replay_until(newest);
				}

				std::lock_guard tk(vd1_cs::s_token);
				vd1_cs::st_worker += vd1_cs::run_locked();
				continue;
			}

			if (vd1_cs_rec_backlog()) // vanillad1: X527 CSREC=3: published chunks, replayed ahead of their submit
			{
				vd1_cs_rec_replay_published();
				continue;
			}

			thread_ctrl::wait_on(vd1_cs::s_wake, wake_token);
		}

		// Leaving: nothing may stay queued (the renderer drains before it tears down; this is the last guard)
		vd1_cs::s_running = false;
		std::lock_guard tk(vd1_cs::s_token);
		vd1_cs::st_worker += vd1_cs::run_locked();
	}
	void vd1_cs_wake() // vanillad1: X527 CSREC=3: a chunk was published
	{
		vd1_cs::s_wake++;
		vd1_cs::s_wake.notify_one();
	}

	// global submit guard to prevent race condition on queue submit
	shared_mutex g_submit_mutex;

	void acquire_global_submit_lock()
	{
		g_submit_mutex.lock();
	}

	void release_global_submit_lock()
	{
		g_submit_mutex.unlock();
	}

	FORCE_INLINE
	static void queue_submit_impl(const queue_submit_t& submit_info)
	{
		ensure(submit_info.pfence);

		// vanillad1: X527 CSREC: the buffer's packets reach the real VkCommandBuffer first (no-op when already replayed: CSREC=1,
		// the replay before the token (review F3), or the worker's eager CSREC=3 batches). Never g_submit_mutex inside it.
		if (submit_info.vd1_replay_token) [[unlikely]]
		{
			vd1_cs_replay_until(submit_info.vd1_replay_token);
		}

		acquire_global_submit_lock();
		VkSubmitInfo info
		{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = submit_info.wait_semaphores_count,
			.pWaitSemaphores = submit_info.wait_semaphores.data(),
			.pWaitDstStageMask = submit_info.wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &submit_info.commands,
			.signalSemaphoreCount = submit_info.signal_semaphores_count,
			.pSignalSemaphores = submit_info.signal_semaphores.data()
		};

		vkQueueSubmit(submit_info.queue, 1, &info, submit_info.pfence->handle);
		release_global_submit_lock();

		// Signal fence
		submit_info.pfence->signal_flushed();
	}

	void queue_submit(const queue_submit_t& submit_info, VkBool32 flush)
	{
		rsx::get_current_renderer()->get_stats().submit_count++;

		// Access to this method must be externally synchronized.
		// Offloader is guaranteed to never call this for async flushes.
		{
			const bool vd1_hold = vd1_cs_rec_hold_begin(); // vanillad1: X527 CSREC=3: no worker replay while sets are written (risks R7)
			vk::descriptors::flush();
			if (vd1_hold) vd1_cs_rec_hold_end();
		}

		if (!flush && g_cfg.video.multithreaded_rsx)
		{
			auto packet = new queue_submit_t(submit_info);
			g_fxo->get<rsx::dma_manager>().backend_ctrl(rctrl_queue_submit, packet);
		}
		else if (vd1_cs::s_running) [[unlikely]] // vanillad1: X527 C1 stage 1 (CSUB): the worker is up (VANILLAD1_CS=1)
		{
			const bool on_rsx = rsx::get_current_renderer()->is_current_thread();
			if (on_rsx)
			{
				vd1_cs::maybe_log();
			}

			if (!flush && on_rsx && vd1::live(vd1::lf::CSUB))
			{
				// Queued: the worker (or the next thread to wait on GPU work) submits it, in order
				u32 depth = 0;
				{
					std::lock_guard q(vd1_cs::s_qlock);
					vd1_cs::s_queue.push_back(submit_info);
					depth = ++vd1_cs::s_inflight;
				}

				if (depth > vd1_cs::st_maxq)
				{
					vd1_cs::st_maxq = depth;
				}

				vd1_cs::st_queued++;
				vd1_cs::s_wake++;
				vd1_cs::s_wake.notify_one();
			}
			else
			{
				// Inline: whatever is queued goes first, under the token, then this one
				std::lock_guard tk(vd1_cs::s_token);
				vd1_cs::st_helped += vd1_cs::run_locked();
				queue_submit_impl(submit_info);
				vd1_cs::st_inline++;
			}
		}
		else
		{
			queue_submit_impl(submit_info);
		}
	}

	void queue_submit(const queue_submit_t* packet)
	{
		// Flush-only version used by asynchronous submit processing (MTRSX)
		queue_submit_impl(*packet);
	}

	void driver_manager_t::operator()()
	{
		while (thread_ctrl::state() != thread_state::aborting)
		{
			const auto wake_token = m_wake_event.observe();
			const auto last_eid = m_last_completed_eid.load();

			if (auto eid = m_eid_ctr.load(); eid != last_eid)
			{
				vk::get_resource_manager()->eid_completed(eid);

				m_last_completed_eid.store(eid);
				m_completed_signal++;
				m_completed_signal.notify_all();
			}

			thread_ctrl::wait_on(m_wake_event, wake_token);
		}
	}

	void driver_manager_t::notify_completed(u64 eid)
	{
		m_eid_ctr.atomic_op([eid](u64& value)
		{
			value = std::max(value, eid);
		});

		m_wake_event++;
		m_wake_event.notify_one();
	}

	void driver_manager_t::drain()
	{
		const u64 target_eid = m_eid_ctr.load(); //<- Last request at time of calling

		// Now, we spam queue wake until our EID is reached
		while (true)
		{
			const u32 completion_token = m_completed_signal.observe();
			if (m_last_completed_eid.load() >= target_eid)
			{
				// Abort, watermark reached.
				break;
			}

			if (thread_ctrl::state() == thread_state::aborting)
			{
				// Abort, emulation state changed.
				break;
			}

			// Wake the worker thread.
			m_wake_event++;
			m_wake_event.notify_one();

			// Wait for worker thread.
			thread_ctrl::wait_on(m_completed_signal, completion_token);
		}
	}
}
