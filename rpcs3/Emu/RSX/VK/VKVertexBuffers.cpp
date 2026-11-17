#include "stdafx.h"
#include "Emu/vd1_live.h"
#include "Emu/Cell/timers.hpp" // vanillad1: X301: get_system_time for the 10-s lines
#include "VKGSRender.h"
#include "../Common/BufferUtils.h"
#include "../rsx_methods.h"
#include "vkutils/buffer_object.h"

#include <span>
#include <bit> // vanillad1: X301 p24: std::rotl
#include <algorithm>
#include <cstring>

namespace vk
{
	std::pair<VkPrimitiveTopology, bool> get_appropriate_topology(rsx::primitive_type mode)
	{
		switch (mode)
		{
		case rsx::primitive_type::lines:
			return { VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false };
		case rsx::primitive_type::line_loop:
			return { VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, true };
		case rsx::primitive_type::line_strip:
			return { VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, false };
		case rsx::primitive_type::points:
			return { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false };
		case rsx::primitive_type::triangles:
			return { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false };
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip:
			return { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false };
		case rsx::primitive_type::triangle_fan:
#ifndef __APPLE__
			return { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, false };
#endif
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon:
			return { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true };
		default:
			fmt::throw_exception("Unsupported primitive topology 0x%x", static_cast<u8>(mode));
		}
	}

	bool is_primitive_native(rsx::primitive_type mode)
	{
		return !get_appropriate_topology(mode).second;
	}

	VkIndexType get_index_type(rsx::index_array_type type)
	{
		switch (type)
		{
		case rsx::index_array_type::u32:
			return VK_INDEX_TYPE_UINT32;
		case rsx::index_array_type::u16:
			return VK_INDEX_TYPE_UINT16;
		}
		fmt::throw_exception("Invalid index array type (%u)", static_cast<u8>(type));
	}
}

namespace
{
	// vanillad1: X301: EIDX (live): an emulated primitive's index list depends only on the primitive and the vertex count, so
	// this frame's earlier list is drawn again instead of generated and uploaded again. Its ring offset lasts the frame, as
	// X302's caches' do (purged with them at the flip: vd1_purge_upload_caches).
	void vd1_idx_advance(u64 offset, u64 length, u64 ring); // vanillad1: X301 p24: IXC's ring age (defined below)

	rsx::unordered_map<u64, VkDeviceSize> g_vd1_emu_index_cache;
	u32 g_vd1_emu_index_hits = 0;
	u32 g_vd1_emu_index_made = 0;

	std::tuple<u32, std::tuple<VkDeviceSize, VkIndexType>> generate_emulating_index_buffer(
		const rsx::draw_clause& clause, u32 vertex_count,
		vk::data_heap& m_index_buffer_ring_info)
	{
		u32 index_count = get_index_count(clause.primitive, vertex_count);
		u32 upload_size = index_count * sizeof(u16);

		const bool vd1_eidx = vd1::live(vd1::lf::EIDX) != 0;
		const u64 vd1_key = (u64{static_cast<u32>(clause.primitive)} << 32) | vertex_count;

		if (vd1_eidx)
		{
			if (const auto found = g_vd1_emu_index_cache.find(vd1_key); found != g_vd1_emu_index_cache.end())
			{
				g_vd1_emu_index_hits++;
				return std::make_tuple(index_count, std::make_tuple(found->second, VK_INDEX_TYPE_UINT16));
			}
		}

		VkDeviceSize offset_in_index_buffer = m_index_buffer_ring_info.alloc<256>(upload_size);
		vd1_idx_advance(offset_in_index_buffer, upload_size, m_index_buffer_ring_info.size()); // X301 p24: IXC's ring age
		void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer, upload_size);

		g_fxo->get<rsx::dma_manager>().emulate_as_indexed(buf, clause.primitive, vertex_count);

		m_index_buffer_ring_info.unmap();
		g_vd1_emu_index_made++;

		if (vd1_eidx)
		{
			g_vd1_emu_index_cache[vd1_key] = offset_in_index_buffer;
		}

		return std::make_tuple(
			index_count, std::make_tuple(offset_in_index_buffer, VK_INDEX_TYPE_UINT16));
	}

	struct vertex_input_state
	{
		VkPrimitiveTopology native_primitive_type;
		bool index_rebase;
		u32 min_index;
		u32 max_index;
		u32 vertex_draw_count;
		u32 vertex_index_offset;
		std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info;
	};

	// vanillad1: index cache (X302): this frame's uploads by source range and how they were expanded. The key is checked in
	// full and the data by its first and last 8 bytes, the same weak validation as the vertex cache, with the same
	// per-frame purge; the game's index buffers are static mesh data.
	struct vd1_index_entry
	{
		const std::byte* src;
		usz size;
		u32 upload_size;
		u32 restart_index;
		u32 key_bits;
		u64 first8, last8;
		VkDeviceSize offset;
		u32 min_index, max_index, index_count;
		u64 hash;           // X301 p24: IXC: every byte of the game's index list
		u64 total_end;      // the index ring's running byte count at this entry's end
		u32 frame;          // the frame that stored it (without IXC only this frame's entries count)
	};

	rsx::unordered_map<u64, vd1_index_entry> g_vd1_index_cache;

	// vanillad1: X301 p24: the index ring's age, as the attribute ring's for VXC (two allocation sites, both in this file)
	u64 g_vd1_idx_total = 0;
	u64 g_vd1_idx_last_end = 0;
	u64 g_vd1_idx_frame_start = 0;
	u64 g_vd1_idx_ring_size = 0;
	bool g_vd1_ixc_guard_ok = true;
	u32 g_vd1_ixc_old_hits = 0;
	u32 g_vd1_ixc_misses = 0;
	u32 g_vd1_idx_frame = 0;

	void vd1_idx_advance(u64 offset, u64 length, u64 ring)
	{
		g_vd1_idx_ring_size = ring;
		const u64 advance = offset >= g_vd1_idx_last_end ? offset + length - g_vd1_idx_last_end : ring - g_vd1_idx_last_end + offset + length;
		g_vd1_idx_total += advance;
		g_vd1_idx_last_end = offset + length;
	}

	u64 vd1_vxc_hash(u32 address, u32 bytes);
	u64 vd1_bytes_hash(const u8* p, usz bytes);

	// vanillad1: multi-block vertex cache (X302): the same for draws with 2-4 interleaved blocks (the stock cache takes one)
	struct vd1_vertex_entry
	{
		u32 count;
		u32 length;
		u32 vertex_count;
		u32 offset_in_heap;
		u32 address[4];
		u32 stride[4];
		u64 fingerprint[4];
		u64 vxc_fp[4];      // X301: VXC: the middle and last 8 bytes of each block, mixed
		u64 total_end;      // X301: VXC: the attribute ring's running byte count at this entry's end
		u32 frame;          // X301: the frame that stored it (without VXC only this frame's entries count)
	};

	rsx::unordered_map<u64, vd1_vertex_entry> g_vd1_vertex_cache;

	// vanillad1: X301: VXC (live, an experiment): X302's multi-block vertex cache across frames. An entry is used again while the
	// attribute ring has moved on less than a quarter of its size since it was written (so its bytes are still there and
	// stay there for several frames more), and its addresses, strides and three 8-byte samples a block match. The samples
	// are a weak check: this measures what a cross-frame cache could give, it is not a fix to ship as it is.
	u64 g_vd1_attr_total = 0;
	u64 g_vd1_attr_last_end = 0;
	u32 g_vd1_frame = 0;
	u32 g_vd1_vxc_old_hits = 0;
	u32 g_vd1_vxc_new_hits = 0;
	u32 g_vd1_vxc_misses = 0;
	u64 g_vd1_attr_ring_size = 0;       // X301 p24: the attribute ring's size, for the frame-end guard
	u64 g_vd1_attr_frame_start = 0;     // the running count at this frame's start
	bool g_vd1_vxc_guard_ok = true;     // the last frame allocated under 1/8 of the ring (VXC=1 reuses only then)
	u64 g_vd1_attr_frame_max = 0;       // the largest frame's allocation in the 10-s window
	u64 g_vd1_attr_frame_sum = 0;

	void vd1_attr_advance(u64 offset, u64 length, u64 ring)
	{
		const u64 advance = offset >= g_vd1_attr_last_end ? offset + length - g_vd1_attr_last_end : ring - g_vd1_attr_last_end + offset + length;
		g_vd1_attr_total += advance;
		g_vd1_attr_last_end = offset + length;
	}

	// X301 p24: VXC=1's check: every byte of the block, four independent 64-bit lanes (multiply, xor, rotate) and a final mix.
	// Not cryptographic; an accidental match of changed data is ~2^-64 a draw.
	u64 vd1_vxc_hash(u32 address, u32 bytes)
	{
		return vd1_bytes_hash(vm::get_super_ptr<u8>(address), bytes);
	}

	u64 vd1_bytes_hash(const u8* p, usz bytes)
	{
		u64 h0 = 0x9E3779B97F4A7C15ull ^ bytes, h1 = 0xC2B2AE3D27D4EB4Full, h2 = 0x165667B19E3779F9ull, h3 = 0x27D4EB2F165667C5ull;
		usz i = 0;
		for (; i + 32 <= bytes; i += 32)
		{
			u64 w[4];
			std::memcpy(w, p + i, 32);
			h0 = std::rotl((h0 ^ w[0]) * 0x9E3779B97F4A7C15ull, 31);
			h1 = std::rotl((h1 ^ w[1]) * 0xC2B2AE3D27D4EB4Full, 29);
			h2 = std::rotl((h2 ^ w[2]) * 0x165667B19E3779F9ull, 27);
			h3 = std::rotl((h3 ^ w[3]) * 0x27D4EB2F165667C5ull, 33);
		}
		for (; i + 8 <= bytes; i += 8)
		{
			u64 w;
			std::memcpy(&w, p + i, 8);
			h0 = std::rotl((h0 ^ w) * 0x9E3779B97F4A7C15ull, 31);
		}
		for (; i < bytes; i++)
		{
			h1 = (h1 ^ p[i]) * 0x100000001B3ull;
		}
		u64 h = h0 ^ std::rotl(h1, 17) ^ std::rotl(h2, 41) ^ std::rotl(h3, 7);
		h ^= h >> 33;
		h *= 0xFF51AFD7ED558CCDull;
		h ^= h >> 33;
		return h;
	}

	u64 vd1_vxc_sample(u32 address, u32 bytes)
	{
		if (bytes < 16)
		{
			return 0;
		}
		const u64 mid = *utils::bless<u64>(vm::get_super_ptr<char>(address + ((bytes / 2) & ~7u)));
		const u64 last = *utils::bless<u64>(vm::get_super_ptr<char>(address + bytes - 8));
		return mid * 0x9E3779B97F4A7C15ull ^ last;
	}

	struct draw_command_visitor
	{
		draw_command_visitor(vk::data_heap& index_buffer_ring_info, rsx::vertex_input_layout& layout)
			: m_index_buffer_ring_info(index_buffer_ring_info)
			, m_vertex_layout(layout)
		{
		}

		vertex_input_state operator()(const rsx::draw_array_command& /*command*/)
		{
			const auto [prims, primitives_emulated] = vk::get_appropriate_topology(rsx::method_registers.current_draw_clause.primitive);
			const u32 vertex_count = rsx::method_registers.current_draw_clause.get_elements_count();
			const u32 min_index = rsx::method_registers.current_draw_clause.min_index();
			const u32 max_index = (min_index + vertex_count) - 1;

			if (primitives_emulated)
			{
				u32 index_count;
				std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info;

				std::tie(index_count, index_info) =
					generate_emulating_index_buffer(rsx::method_registers.current_draw_clause,
						vertex_count, m_index_buffer_ring_info);

				return{ prims, false, min_index, max_index, index_count, 0, index_info };
			}

			return{ prims, false, min_index, max_index, vertex_count, 0, {} };
		}

		vertex_input_state operator()(const rsx::draw_indexed_array_command& command)
		{
			auto primitive = rsx::method_registers.current_draw_clause.primitive;
			const auto [prims, primitives_emulated] = vk::get_appropriate_topology(primitive);
			const bool emulate_restart = rsx::method_registers.restart_index_enabled() && vk::emulate_primitive_restart(primitive);

			rsx::index_array_type index_type = rsx::method_registers.current_draw_clause.is_immediate_draw ?
				rsx::index_array_type::u32 :
				rsx::method_registers.index_type();

			u32 type_size = get_index_type_size(index_type);

			u32 index_count = rsx::method_registers.current_draw_clause.get_elements_count();
			if (primitives_emulated)
				index_count = get_index_count(rsx::method_registers.current_draw_clause.primitive, index_count);
			u32 upload_size = index_count * type_size;

			if (emulate_restart) upload_size *= 2;

			// vanillad1: index cache (X302)
			const auto& vd1_src = command.raw_index_buffer;
			const bool vd1_ixc = vd1::live(vd1::lf::IXC) != 0; // X301 p24
			const bool vd1_cacheable = (vd1::live(vd1::lf::E9) || vd1_ixc) && vd1_src.size() >= 8 && !rsx::method_registers.current_draw_clause.is_immediate_draw;
			u64 vd1_hash = 0;
			const u32 vd1_key_bits = static_cast<u32>(index_type) | (static_cast<u32>(primitive) << 8) |
				(rsx::method_registers.restart_index_enabled() ? (1u << 16) : 0u) | (emulate_restart ? (1u << 17) : 0u);
			const u32 vd1_restart_index = rsx::method_registers.restart_index();
			const u64 vd1_key = (reinterpret_cast<u64>(vd1_src.data()) * 0x9E3779B97F4A7C15ull) ^ (u64{upload_size} << 32) ^ vd1_key_bits;
			u64 vd1_first8 = 0, vd1_last8 = 0;

			if (vd1_cacheable)
			{
				std::memcpy(&vd1_first8, vd1_src.data(), 8);
				std::memcpy(&vd1_last8, vd1_src.data() + vd1_src.size() - 8, 8);
				if (vd1_ixc)
				{
					vd1_hash = vd1_bytes_hash(reinterpret_cast<const u8*>(vd1_src.data()), vd1_src.size());
				}

				if (const auto found = g_vd1_index_cache.find(vd1_key); found != g_vd1_index_cache.end())
				{
					const auto& e = found->second;
					// X301 p24: an entry of an earlier frame only under IXC: its full hash, within 1/8 of the index ring, guard on
					const bool vd1_this_frame = e.frame == g_vd1_idx_frame;
					const bool vd1_old_ok = vd1_ixc && !vd1_this_frame && g_vd1_ixc_guard_ok && e.hash == vd1_hash &&
						g_vd1_idx_total - e.total_end < m_index_buffer_ring_info.size() / 8;
					if (e.src == vd1_src.data() && e.size == vd1_src.size() && e.upload_size == upload_size && e.key_bits == vd1_key_bits &&
						e.restart_index == vd1_restart_index && e.first8 == vd1_first8 && e.last8 == vd1_last8 && (vd1_this_frame || vd1_old_ok))
					{
						if (!vd1_this_frame) g_vd1_ixc_old_hits++;
						std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info =
							std::make_tuple(e.offset, vk::get_index_type(index_type));
						return {prims, true, e.min_index, e.max_index, e.index_count, rsx::method_registers.vertex_data_base_index(), index_info};
					}
				}
			}

			VkDeviceSize offset_in_index_buffer = m_index_buffer_ring_info.alloc<64>(upload_size);
			vd1_idx_advance(offset_in_index_buffer, upload_size, m_index_buffer_ring_info.size()); // X301 p24: IXC's ring age
			if (vd1_ixc) g_vd1_ixc_misses++;
			void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer, upload_size);

			std::span<std::byte> dst;
			stx::single_ptr<std::byte[]> tmp;
			if (emulate_restart)
			{
				tmp = stx::make_single<std::byte[], false, 64>(upload_size);
				dst = std::span<std::byte>(tmp.get(), upload_size);
			}
			else
			{
				dst = std::span<std::byte>(static_cast<std::byte*>(buf), upload_size);
			}

			/**
			* Upload index (and expands it if primitive type is not natively supported).
			*/
			u32 min_index, max_index;
			std::tie(min_index, max_index, index_count) = write_index_array_data_to_buffer(
				dst,
				command.raw_index_buffer, index_type,
				rsx::method_registers.current_draw_clause.primitive,
				rsx::method_registers.restart_index_enabled(),
				rsx::method_registers.restart_index(),
				[](auto prim) { return !vk::is_primitive_native(prim); });

			if (min_index >= max_index)
			{
				//empty set, do not draw
				m_index_buffer_ring_info.unmap();
				return{ prims, false, 0, 0, 0, 0, {} };
			}

			if (emulate_restart)
			{
				if (index_type == rsx::index_array_type::u16)
				{
					index_count = rsx::remove_restart_index(static_cast<u16*>(buf), reinterpret_cast<u16*>(tmp.get()), index_count, u16{umax});
				}
				else
				{
					index_count = rsx::remove_restart_index(static_cast<u32*>(buf), reinterpret_cast<u32*>(tmp.get()), index_count, u32{umax});
				}
			}

			m_index_buffer_ring_info.unmap();

			if (vd1_cacheable) // vanillad1: index cache (X302)
			{
				g_vd1_index_cache[vd1_key] = { vd1_src.data(), vd1_src.size(), upload_size, vd1_restart_index, vd1_key_bits, vd1_first8, vd1_last8,
					offset_in_index_buffer, min_index, max_index, index_count, vd1_hash, g_vd1_idx_total, g_vd1_idx_frame };
			}

			std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info =
				std::make_tuple(offset_in_index_buffer, vk::get_index_type(index_type));

			const auto index_offset = rsx::method_registers.vertex_data_base_index();
			return {prims, true, min_index, max_index, index_count, index_offset, index_info};
		}

		vertex_input_state operator()(const rsx::draw_inlined_array& /*command*/)
		{
			auto &draw_clause = rsx::method_registers.current_draw_clause;
			const auto [prims, primitives_emulated] = vk::get_appropriate_topology(draw_clause.primitive);

			const auto stream_length = rsx::method_registers.current_draw_clause.inline_vertex_array.size();
			const u32 vertex_count = u32(stream_length * sizeof(u32)) / m_vertex_layout.interleaved_blocks[0]->attribute_stride;

			if (!primitives_emulated)
			{
				return{ prims, false, 0, vertex_count - 1, vertex_count, 0, {} };
			}

			u32 index_count;
			std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info;
			std::tie(index_count, index_info) = generate_emulating_index_buffer(draw_clause, vertex_count, m_index_buffer_ring_info);
			return{ prims, false, 0, vertex_count - 1, index_count, 0, index_info };
		}

	private:
		vk::data_heap& m_index_buffer_ring_info;
		rsx::vertex_input_layout& m_vertex_layout;
	};
}

// vanillad1: X301: attribute views made, logged every 10 s with the other upload counters
u32 g_vd1_view_remakes = 0;

// vanillad1: index and multi-block vertex caches (X302): the ring offsets they hold last a frame, like the vertex cache's
void vd1_purge_upload_caches()
{
	// X301 p24: IXC keeps the index cache across frames (bounded); its guard: this frame's index allocation under 1/8 ring
	if (!vd1::live(vd1::lf::IXC) || g_vd1_index_cache.size() > 65536)
	{
		g_vd1_index_cache.clear();
	}
	g_vd1_idx_frame++;
	{
		const u64 frame_alloc = g_vd1_idx_total - g_vd1_idx_frame_start;
		g_vd1_idx_frame_start = g_vd1_idx_total;
		g_vd1_ixc_guard_ok = g_vd1_idx_ring_size && frame_alloc < g_vd1_idx_ring_size / 8;
	}
	g_vd1_frame++; // X301: entries from earlier frames count only under VXC

	// X301 p24: this frame's attribute allocation; VXC=1 reuses older entries next frame only if it stayed under 1/8 ring
	{
		const u64 frame_alloc = g_vd1_attr_total - g_vd1_attr_frame_start;
		g_vd1_attr_frame_start = g_vd1_attr_total;
		g_vd1_vxc_guard_ok = g_vd1_attr_ring_size && frame_alloc < g_vd1_attr_ring_size / 8;
		g_vd1_attr_frame_max = std::max(g_vd1_attr_frame_max, frame_alloc);
		g_vd1_attr_frame_sum += frame_alloc;
	}
	if (!vd1::live(vd1::lf::VXC) || g_vd1_vertex_cache.size() > 65536)
	{
		g_vd1_vertex_cache.clear();
	}
	g_vd1_emu_index_cache.clear(); // vanillad1: X301: EIDX

	// vanillad1: X301: the upload counters of the last 10 s (attribute views, EIDX, VXC)
	static u64 s_last_ms = 0;
	static u32 s_frames = 0;
	s_frames++;
	if (const u64 now = get_system_time() / 1000; now - s_last_ms >= 10'000)
	{
		if (s_last_ms)
		{
			rsx_log.notice("vanillad1: attrib views 10s: made=%u frames=%u; emulated index lists: EIDX=%u made=%u reused=%u",
				g_vd1_view_remakes, s_frames, vd1::live(vd1::lf::EIDX), g_vd1_emu_index_made, g_vd1_emu_index_hits);
			if (vd1::live(vd1::lf::VXC))
			{
				rsx_log.notice("vanillad1: vxc 10s: VXC=%u hits from earlier frames=%u, this frame=%u, misses=%u, entries=%u",
					vd1::live(vd1::lf::VXC), g_vd1_vxc_old_hits, g_vd1_vxc_new_hits, g_vd1_vxc_misses, static_cast<u32>(g_vd1_vertex_cache.size()));
			}
			if (vd1::live(vd1::lf::IXC))
			{
				rsx_log.notice("vanillad1: ixc 10s: hits from earlier frames=%u, misses=%u, entries=%u, guard=%u",
					g_vd1_ixc_old_hits, g_vd1_ixc_misses, static_cast<u32>(g_vd1_index_cache.size()), g_vd1_ixc_guard_ok ? 1u : 0u);
			}
			if (s_frames)
			{
				rsx_log.notice("vanillad1: attrib ring 10s: KB a frame avg=%u max=%u, ring KB=%u, guard=%u",
					static_cast<u32>(g_vd1_attr_frame_sum / s_frames / 1024), static_cast<u32>(g_vd1_attr_frame_max / 1024),
					static_cast<u32>(g_vd1_attr_ring_size / 1024), g_vd1_vxc_guard_ok ? 1u : 0u);
			}
		}
		s_last_ms = now;
		g_vd1_view_remakes = 0;
		g_vd1_emu_index_made = 0;
		g_vd1_vxc_old_hits = 0;
		g_vd1_vxc_new_hits = 0;
		g_vd1_vxc_misses = 0;
		g_vd1_attr_frame_max = 0;
		g_vd1_attr_frame_sum = 0;
		g_vd1_ixc_old_hits = 0;
		g_vd1_ixc_misses = 0;
		g_vd1_emu_index_hits = 0;
		s_frames = 0;
	}
}

vk::vertex_upload_info VKGSRender::upload_vertex_data()
{
	// vanillad1: X301 p24: a changed heap moves every ring offset the caches hold (X302's per-frame ones too): none survives
	if (vk::test_status_interrupt(vk::heap_changed))
	{
		g_vd1_vertex_cache.clear();
		g_vd1_index_cache.clear();
	}

	draw_command_visitor visitor(m_index_buffer_ring_info, m_vertex_layout);
	auto result = std::visit(visitor, m_draw_processor.get_draw_command(rsx::method_registers));

	const u32 vertex_count = (result.max_index - result.min_index) + 1;
	u32 vertex_base = result.min_index;
	u32 index_base = 0;

	if (result.index_rebase)
	{
		vertex_base = rsx::get_index_from_base(vertex_base, rsx::method_registers.vertex_data_base_index());
		index_base = result.min_index;
	}

	//Do actual vertex upload
	auto required = calculate_memory_requirements(m_vertex_layout, vertex_base, vertex_count);
	u32 persistent_range_base = -1, volatile_range_base = -1;
	usz persistent_offset = -1, volatile_offset = -1;

	if (required.first > 0)
	{
		//Check if cacheable
		//Only data in the 'persistent' block may be cached
		//TODO: make vertex cache keep local data beyond frame boundaries and hook notify command
		bool in_cache = false;
		bool to_store = false;
		u32  storage_address = -1;
		vd1_vertex_entry vd1_entry{}; // vanillad1: multi-block vertex cache (X302)
		u64 vd1_vertex_key = 0;
		bool vd1_store = false;

		m_frame_stats.vertex_cache_request_count++;

		// X301: VXC=1 the full-content check (p24), 2 the three-sample experiment (p23)
		const u32 vd1_vxc_mode = vd1::live(vd1::lf::VXC);
		const bool vd1_vxc = vd1_vxc_mode != 0;
		g_vd1_attr_ring_size = m_attrib_ring_info.size();

		// X301 p24: a changed heap moves every ring offset: no entry survives it (this frame's included)
		if (vk::test_status_interrupt(vk::heap_changed))
		{
			g_vd1_vertex_cache.clear();
		}

		if (!vd1_vxc && m_vertex_layout.interleaved_blocks.size() == 1 &&
			rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
		{
			const auto data_offset = (vertex_base * m_vertex_layout.interleaved_blocks[0]->attribute_stride);
			storage_address = m_vertex_layout.interleaved_blocks[0]->real_offset_address + data_offset;

			if (auto cached = m_vertex_cache->find_vertex_range(storage_address, required.first))
			{
				ensure(cached->local_address == storage_address);

				in_cache = true;
				persistent_range_base = cached->offset_in_heap;
			}
			else
			{
				to_store = true;
			}
		}
		else if (const usz vd1_blocks = m_vertex_layout.interleaved_blocks.size();
			(vd1::live(vd1::lf::E10) || vd1_vxc) && vd1_blocks > (vd1_vxc ? 0u : 1u) && vd1_blocks <= 4 && required.first >= 8 &&
			rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
		{
			vd1_entry.count = static_cast<u32>(vd1_blocks);
			vd1_entry.length = required.first;
			vd1_entry.vertex_count = vertex_count;
			vd1_vertex_key = u64{required.first} << 32 | vertex_count;

			for (usz b = 0; b < vd1_blocks; b++)
			{
				const auto& block = m_vertex_layout.interleaved_blocks[b];
				const u32 address = block->real_offset_address + (block->single_vertex ? 0 : vertex_base * block->attribute_stride);
				vd1_entry.address[b] = address;
				vd1_entry.stride[b] = block->attribute_stride;
				vd1_entry.fingerprint[b] = *utils::bless<u64>(vm::get_super_ptr<char>(address));
				if (vd1_vxc)
				{
					// the exact range the upload copies (write_vertex_data_to_memory), so a sample never reads past it
					const auto vd1_range = block->calculate_required_range(vertex_base, vertex_count);
					const u32 vd1_addr = block->real_offset_address + vd1_range.first * block->attribute_stride;
					const u32 vd1_bytes = vd1_range.second * block->attribute_stride;
					vd1_entry.vxc_fp[b] = vd1_vxc_mode == 2 ? vd1_vxc_sample(vd1_addr, vd1_bytes) : vd1_vxc_hash(vd1_addr, vd1_bytes);
				}
				vd1_vertex_key = (vd1_vertex_key ^ address ^ (u64{block->attribute_stride} << 40)) * 0x9E3779B97F4A7C15ull;
			}

			if (const auto found = g_vd1_vertex_cache.find(vd1_vertex_key); found != g_vd1_vertex_cache.end())
			{
				auto& e = found->second;
				if (e.count == vd1_entry.count && e.length == vd1_entry.length && e.vertex_count == vd1_entry.vertex_count &&
					std::memcmp(e.address, vd1_entry.address, sizeof(u32) * vd1_blocks) == 0 &&
					std::memcmp(e.stride, vd1_entry.stride, sizeof(u32) * vd1_blocks) == 0 &&
					std::memcmp(e.fingerprint, vd1_entry.fingerprint, sizeof(u64) * vd1_blocks) == 0)
				{
					// X301: an entry from an earlier frame only under VXC, young enough in the ring, with its extra samples
					const bool this_frame = e.frame == g_vd1_frame;
					// p24: VXC=1 reuses an older entry only within 1/8 of the ring of its writing, and only while the last frame
					// allocated under 1/8: the ring then needs ~7 frames to come back over it, and at most 2 are in flight
					const u64 vd1_window = vd1_vxc_mode == 2 ? m_attrib_ring_info.size() / 4 : m_attrib_ring_info.size() / 8;
					const bool old_ok = vd1_vxc && !this_frame && (vd1_vxc_mode == 2 || g_vd1_vxc_guard_ok) &&
						g_vd1_attr_total - e.total_end < vd1_window &&
						std::memcmp(e.vxc_fp, vd1_entry.vxc_fp, sizeof(u64) * vd1_blocks) == 0;

					if (this_frame || old_ok)
					{
						in_cache = true;
						persistent_range_base = e.offset_in_heap;
						if (vd1_vxc) (this_frame ? g_vd1_vxc_new_hits : g_vd1_vxc_old_hits)++;
					}
				}
			}

			vd1_store = !in_cache;
			if (vd1_vxc && !in_cache) g_vd1_vxc_misses++;
		}

		if (!in_cache)
		{
			m_frame_stats.vertex_cache_miss_count++;

			persistent_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.first));
			vd1_attr_advance(persistent_offset, required.first, m_attrib_ring_info.size()); // X301: VXC's ring age
			persistent_range_base = static_cast<u32>(persistent_offset);

			if (to_store)
			{
				//store ref in vertex cache
				m_vertex_cache->store_range(storage_address, required.first, static_cast<u32>(persistent_offset));
			}

			if (vd1_store) // vanillad1: multi-block vertex cache (X302)
			{
				vd1_entry.offset_in_heap = static_cast<u32>(persistent_offset);
				vd1_entry.total_end = g_vd1_attr_total;
				vd1_entry.frame = g_vd1_frame;
				g_vd1_vertex_cache[vd1_vertex_key] = vd1_entry;
			}
		}
	}

	if (required.second > 0)
	{
		volatile_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.second));
		vd1_attr_advance(volatile_offset, required.second, m_attrib_ring_info.size()); // X301: VXC's ring age
		volatile_range_base = static_cast<u32>(volatile_offset);
	}

	//Write all the data once if possible
	if (required.first && required.second && volatile_offset > persistent_offset)
	{
		//Do this once for both to save time on map/unmap cycles
		const usz block_end = (volatile_offset + required.second);
		const usz block_size = block_end - persistent_offset;
		const usz volatile_offset_in_block = volatile_offset - persistent_offset;

		void *block_mapping = m_attrib_ring_info.map(persistent_offset, block_size);
		m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, block_mapping, static_cast<char*>(block_mapping) + volatile_offset_in_block);
		m_attrib_ring_info.unmap();
	}
	else
	{
		if (required.first > 0 && persistent_offset != umax)
		{
			void *persistent_mapping = m_attrib_ring_info.map(persistent_offset, required.first);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, persistent_mapping, nullptr);
			m_attrib_ring_info.unmap();
		}

		if (required.second > 0)
		{
			void *volatile_mapping = m_attrib_ring_info.map(volatile_offset, required.second);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, nullptr, volatile_mapping);
			m_attrib_ring_info.unmap();
		}
	}

	if (vk::test_status_interrupt(vk::heap_changed))
	{
		// Check for validity
		if (m_persistent_attribute_storage &&
			m_persistent_attribute_storage->info.buffer != m_attrib_ring_info.heap->value)
		{
			vk::get_resource_manager()->dispose(m_persistent_attribute_storage);
		}

		if (m_volatile_attribute_storage &&
			m_volatile_attribute_storage->info.buffer != m_attrib_ring_info.heap->value)
		{
			vk::get_resource_manager()->dispose(m_volatile_attribute_storage);
		}

		m_vertex_env_buffer_info = { *m_vertex_env_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_vertex_constants_buffer_info = { *m_transform_constants_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_fragment_env_buffer_info = { *m_fragment_env_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_fragment_texture_params_buffer_info = { *m_fragment_texture_params_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_raster_env_buffer_info = { *m_raster_env_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_vertex_layout_stream_info = { *m_vertex_layout_ring_info.heap, 0, VK_WHOLE_SIZE };
		m_fragment_constants_buffer_info = { *m_fragment_constants_ring_info.heap, 0, VK_WHOLE_SIZE };

		vk::clear_status_interrupt(vk::heap_changed);
	}

	if (persistent_range_base != umax)
	{
		if (!m_persistent_attribute_storage || !m_persistent_attribute_storage->in_range(persistent_range_base, required.first, persistent_range_base))
		{
			ensure(m_texbuffer_view_size >= required.first); // "Incompatible driver (MacOS?)"
			vk::get_resource_manager()->dispose(m_persistent_attribute_storage);

			//View 64M blocks at a time (different drivers will only allow a fixed viewable heap size, 64M should be safe)
			const usz view_size = (persistent_range_base + m_texbuffer_view_size) > m_attrib_ring_info.size() ? m_attrib_ring_info.size() - persistent_range_base : m_texbuffer_view_size;
			m_persistent_attribute_storage = std::make_unique<vk::buffer_view>(*m_device, m_attrib_ring_info.heap->value, VK_FORMAT_R8_UINT, persistent_range_base, view_size);
			persistent_range_base = 0;
			g_vd1_view_remakes++; // vanillad1: X301: views made (the 10-s line)
		}
	}

	if (volatile_range_base != umax)
	{
		if (!m_volatile_attribute_storage || !m_volatile_attribute_storage->in_range(volatile_range_base, required.second, volatile_range_base))
		{
			ensure(m_texbuffer_view_size >= required.second); // "Incompatible driver (MacOS?)"
			vk::get_resource_manager()->dispose(m_volatile_attribute_storage);

			const usz view_size = (volatile_range_base + m_texbuffer_view_size) > m_attrib_ring_info.size() ? m_attrib_ring_info.size() - volatile_range_base : m_texbuffer_view_size;
			m_volatile_attribute_storage = std::make_unique<vk::buffer_view>(*m_device, m_attrib_ring_info.heap->value, VK_FORMAT_R8_UINT, volatile_range_base, view_size);
			volatile_range_base = 0;
			g_vd1_view_remakes++; // vanillad1: X301: views made (the 10-s line)
		}
	}

	return{ result.native_primitive_type,                 // Primitive
			result.vertex_draw_count,                     // Vertex count
			vertex_count,                                 // Allocated vertex count
			vertex_base,                                  // First vertex in stream
			index_base,                                   // Index of vertex at data location 0
			result.vertex_index_offset,                   // Index offset
			persistent_range_base, volatile_range_base,   // Binding range
			result.index_info };                          // Index buffer info
}
