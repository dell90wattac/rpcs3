#include "stdafx.h"
#include "Emu/vd1_live.h"
#include "VKGSRender.h"
#include "../Common/BufferUtils.h"
#include "../rsx_methods.h"
#include "vkutils/buffer_object.h"

#include <span>

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
	std::tuple<u32, std::tuple<VkDeviceSize, VkIndexType>> generate_emulating_index_buffer(
		const rsx::draw_clause& clause, u32 vertex_count,
		vk::data_heap& m_index_buffer_ring_info)
	{
		u32 index_count = get_index_count(clause.primitive, vertex_count);
		u32 upload_size = index_count * sizeof(u16);

		VkDeviceSize offset_in_index_buffer = m_index_buffer_ring_info.alloc<256>(upload_size);
		void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer, upload_size);

		g_fxo->get<rsx::dma_manager>().emulate_as_indexed(buf, clause.primitive, vertex_count);

		m_index_buffer_ring_info.unmap();
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
	};

	rsx::unordered_map<u64, vd1_index_entry> g_vd1_index_cache;

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
	};

	rsx::unordered_map<u64, vd1_vertex_entry> g_vd1_vertex_cache;

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
			const bool vd1_cacheable = vd1::live(vd1::lf::E9) && vd1_src.size() >= 8 && !rsx::method_registers.current_draw_clause.is_immediate_draw;
			const u32 vd1_key_bits = static_cast<u32>(index_type) | (static_cast<u32>(primitive) << 8) |
				(rsx::method_registers.restart_index_enabled() ? (1u << 16) : 0u) | (emulate_restart ? (1u << 17) : 0u);
			const u32 vd1_restart_index = rsx::method_registers.restart_index();
			const u64 vd1_key = (reinterpret_cast<u64>(vd1_src.data()) * 0x9E3779B97F4A7C15ull) ^ (u64{upload_size} << 32) ^ vd1_key_bits;
			u64 vd1_first8 = 0, vd1_last8 = 0;

			if (vd1_cacheable)
			{
				std::memcpy(&vd1_first8, vd1_src.data(), 8);
				std::memcpy(&vd1_last8, vd1_src.data() + vd1_src.size() - 8, 8);

				if (const auto found = g_vd1_index_cache.find(vd1_key); found != g_vd1_index_cache.end())
				{
					const auto& e = found->second;
					if (e.src == vd1_src.data() && e.size == vd1_src.size() && e.upload_size == upload_size && e.key_bits == vd1_key_bits &&
						e.restart_index == vd1_restart_index && e.first8 == vd1_first8 && e.last8 == vd1_last8)
					{
						std::optional<std::tuple<VkDeviceSize, VkIndexType>> index_info =
							std::make_tuple(e.offset, vk::get_index_type(index_type));
						return {prims, true, e.min_index, e.max_index, e.index_count, rsx::method_registers.vertex_data_base_index(), index_info};
					}
				}
			}

			VkDeviceSize offset_in_index_buffer = m_index_buffer_ring_info.alloc<64>(upload_size);
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
					offset_in_index_buffer, min_index, max_index, index_count };
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

// vanillad1: index and multi-block vertex caches (X302): the ring offsets they hold last a frame, like the vertex cache's
void vd1_purge_upload_caches()
{
	g_vd1_index_cache.clear();
	g_vd1_vertex_cache.clear();
}

vk::vertex_upload_info VKGSRender::upload_vertex_data()
{
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

		if (m_vertex_layout.interleaved_blocks.size() == 1 &&
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
			vd1::live(vd1::lf::E10) && vd1_blocks > 1 && vd1_blocks <= 4 && required.first >= 8 &&
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
					in_cache = true;
					persistent_range_base = e.offset_in_heap;
				}
			}

			vd1_store = !in_cache;
		}

		if (!in_cache)
		{
			m_frame_stats.vertex_cache_miss_count++;

			persistent_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.first));
			persistent_range_base = static_cast<u32>(persistent_offset);

			if (to_store)
			{
				//store ref in vertex cache
				m_vertex_cache->store_range(storage_address, required.first, static_cast<u32>(persistent_offset));
			}

			if (vd1_store) // vanillad1: multi-block vertex cache (X302)
			{
				vd1_entry.offset_in_heap = static_cast<u32>(persistent_offset);
				g_vd1_vertex_cache[vd1_vertex_key] = vd1_entry;
			}
		}
	}

	if (required.second > 0)
	{
		volatile_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.second));
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
