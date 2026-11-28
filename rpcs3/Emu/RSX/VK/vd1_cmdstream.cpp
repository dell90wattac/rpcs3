#include "stdafx.h"

// vanillad1: X527 C1 stage 3 (CSREC): the packet arena, the encoders and the replay (DRAFT, re/notes/X527/stage3; target
// path rpcs3/Emu/RSX/VK/vd1_cmdstream.cpp). This file calls the real vkCmd* functions, so it takes the shim's declarations
// without its macros. Defined after stdafx.h on purpose: an MSVC precompiled header would drop anything before it (the
// project builds with NotUsing today; CMake's PCH force-includes stdafx.h, which does not include VulkanAPI.h).
#define VD1_CS_NO_SHIM 1

#include "vd1_cmdstream.h"
#include "VKCommandStream.h"   // vd1_cs_wake (the stage 1 worker's wake-up)
#include "vkutils/commands.h"  // vk::command_buffer (stop)
#include "vkutils/shared.h"    // CHECK_RESULT

#include "Emu/vd1_live.h"
#include "Emu/Cell/timers.hpp" // get_system_time
#include "Utilities/mutex.h"
#include "Utilities/StrFmt.h" // fmt::throw_exception (X527 review F4)
#include "Utilities/Thread.h"
#include "util/tsc.hpp"
#include "util/sysinfo.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

// Threading model
// - The producer is the RSX thread (vd1_cs_rec_set_producer, from VKGSRender::on_init_thread). Only it encodes, and only
//   while a deferred primary buffer is open (vd1_rec::g_open). It owns s_wr_* and the st_* producer counters.
// - The replayer is whichever thread holds s_replay: the RSX thread (CSREC=1 at seal, a fallback, backpressure, quiesce),
//   the submit worker (submit packets; CSREC=3 eager batches) or a help-drainer (stage 1's vd1_cs_help_drain runs a
//   submit packet, whose queue_submit_impl replays first). It owns s_rd_*.
// - The producer publishes with a release store of s_published; a replayer reads packets [s_replayed, s_published) after
//   an acquire load. The producer never writes below s_published, the replayer never reads at or above it.
// - Command pool rule (Vulkan: host access to a pool is externally synchronised across begin / end / reset / vkCmd* of
//   all its buffers): while the stream is on, every call on the primary pool runs under s_replay, except a buffer the RSX
//   thread records directly (CSREC=0 for that buffer), which starts only after vd1_cs_rec_quiesce() emptied the arena; the
//   arena then stays empty (the RSX thread is the only producer and is recording directly) until the next deferred begin.
// - Lock order: any RPCS3 lock -> vd1_cs::s_token -> s_replay -> s_free_lock. g_submit_mutex is taken after the replay,
//   never inside it. The replay runs raw Vulkan only: no RPCS3 lock, no GC, no help-drain, no guest memory (C1 traps 1-3).

namespace vk
{
	void vd1_cs_wake(); // VKCommandStream.cpp
}

namespace vd1_rec
{
	alignas(64) std::atomic<VkCommandBuffer> g_open{VK_NULL_HANDLE}; // X527 review F1: read by every wrapper, alone on its line

	enum class op : u16
	{
		nop = 0,
		jump,                       // continue at the next chunk
		begin, end, reset,          // lifecycle (begin and end carry the buffer; reset its own buffer)
		bind_pipeline, bind_descriptor_sets, bind_index_buffer, bind_vertex_buffers,
		draw, draw_indexed, draw_multi, draw_multi_indexed, dispatch,
		set_viewport, set_scissor, set_line_width, set_depth_bias, set_blend_constants, set_depth_bounds,
		set_stencil_compare_mask, set_stencil_write_mask, set_stencil_reference,
		push_constants, update_buffer, fill_buffer,
		copy_buffer, copy_image, blit_image, copy_buffer_to_image, copy_image_to_buffer,
		clear_color_image, clear_depth_stencil_image, clear_attachments,
		pipeline_barrier, set_event, wait_events,
		begin_query, end_query, reset_query_pool, copy_query_pool_results,
		begin_render_pass, end_render_pass,
		begin_conditional_rendering, end_conditional_rendering,
		insert_debug_label,
		set_event2, wait_events2, pipeline_barrier2,
		count
	};

	struct hdr_t
	{
		op code;
		u16 reserved;
		u32 size; // the whole packet in bytes, header included, a multiple of 8
	};
	static_assert(sizeof(hdr_t) == 8);

	constexpr u64 align8(u64 v)
	{
		return (v + 7) & ~u64{7};
	}

	// Fixed-size chunks, reused through a free list: no heap allocation per command. Every chunk keeps room for one jump
	// header at its end.
	struct chunk_t
	{
		static constexpr u32 capacity = 256 * 1024 - 64;
		chunk_t* next = nullptr;
		alignas(16) u8 data[capacity];
	};

	constexpr u32 c_initial_chunks = 8;   // 2 MiB, allocated at the first enabled boot
	constexpr u32 c_max_chunks = 128;     // 32 MiB; at the cap the RSX thread replays the backlog itself (backpressure)
	constexpr u64 c_max_packet = chunk_t::capacity - sizeof(hdr_t);
	constexpr u64 c_worker_batch = 64;    // CSREC=3: packets per s_replay hold on the worker (bounds others' waits; X527 review F2: was 512)
	constexpr u64 c_sample_mask = 63;     // encode time sampled on one packet in 64

	enum class kind : u8
	{
		other = 0, // help-drainers (SPU / PPU fault paths, other threads' inline submits)
		rsx = 1,   // the producer
		worker = 2 // the stage 1 submit worker
	};

	static thread_local kind tl_kind = kind::other;

	static atomic_t<bool> s_enabled{false}; // per boot (VKGSRender ctor); the submit worker already runs then, so atomic
	static atomic_t<bool> s_stopped{false}; // VKGSRender::on_exit: everything recorded directly from then on

	// X527 review F1: the producer's and the replayer's per-packet state on separate cache lines (each is written on every
	// packet by a different thread at CSREC=2/3), and the replay lock and the two cursors on lines of their own.

	// Producer state (RSX thread only), its per-packet stats included
	struct alignas(64) producer_t
	{
		chunk_t* wr_chunk = nullptr;
		u32 wr_off = 0;
		u32 mode_open = 0; // CSREC mode of the open deferred buffer, 0 when none is open
		u64 wr_seq = 0;    // packets encoded (jumps included); the token space
		bool sample = false;
		u64 sample_t0 = 0;
		u64 st_packets = 0, st_bytes = 0, st_enc_tsc = 0, st_fallbacks = 0, st_backpressure = 0;
		u64 st_cbs[4]{};   // primary buffers begun per mode (0 = direct)
	};
	static producer_t s_p;

	// Replayer state (s_replay held)
	struct alignas(64) replayer_t
	{
		chunk_t* rd_chunk = nullptr;
		u32 rd_off = 0;
		VkCommandBuffer rd_cb = VK_NULL_HANDLE; // the buffer between the last BEGIN and its END
	};
	static replayer_t s_r;
	alignas(64) static shared_mutex s_replay;

	alignas(64) static atomic_t<u64> s_published{0};
	alignas(64) static atomic_t<u64> s_replayed{0};

	// Free chunks
	static shared_mutex s_free_lock;
	static chunk_t* s_free = nullptr;
	static u32 s_chunks_total = 0;
	static atomic_t<u32> s_chunks_live{0}; // chunks between the replay cursor and the write cursor, the write chunk included
	static atomic_t<u32> st_chunks_peak{0};

	// Stats. Producer-only counters are plain; replay counters are atomic (several replaying threads, read by the RSX thread)
	static atomic_t<u64> st_rep_packets[3]{}; // by kind
	static atomic_t<u64> st_rep_tsc[3]{};
	static atomic_t<u64> st_foreign{0};
	static atomic_t<u64> s_foreign_last_log_us{0};
	static u64 s_last_log_us = 0;

	// ---- Packets ----
	// Each starts with hdr_t; a variable tail (arrays, bytes) follows at sizeof(P), each array 8-aligned.

	struct alignas(8) p_cb { hdr_t h; VkCommandBuffer cb; u32 flags; u32 pad; };
	struct alignas(8) p_bind_pipeline { hdr_t h; VkPipeline pipeline; VkPipelineBindPoint bind_point; u32 pad; };
	struct alignas(8) p_bind_sets { hdr_t h; VkPipelineLayout layout; VkPipelineBindPoint bind_point; u32 first_set, set_count, offset_count; }; // + sets, offsets
	struct alignas(8) p_bind_index { hdr_t h; VkBuffer buffer; VkDeviceSize offset; VkIndexType type; u32 pad; };
	struct alignas(8) p_bind_vertex { hdr_t h; u32 first, count; }; // + buffers, offsets
	struct alignas(8) p_draw { hdr_t h; u32 vertex_count, instance_count, first_vertex, first_instance; };
	struct alignas(8) p_draw_indexed { hdr_t h; u32 index_count, instance_count, first_index; s32 vertex_offset; u32 first_instance, pad; };
	struct alignas(8) p_draw_multi { hdr_t h; u32 count, instance_count, first_instance, pad; }; // + VkMultiDrawInfoEXT[count], packed
	struct alignas(8) p_draw_multi_indexed { hdr_t h; u32 count, instance_count, first_instance, has_offset; s32 vertex_offset; u32 pad; }; // + VkMultiDrawIndexedInfoEXT[count]
	struct alignas(8) p_dispatch { hdr_t h; u32 x, y, z, pad; };
	struct alignas(8) p_rects { hdr_t h; u32 first, count; }; // + VkViewport[] or VkRect2D[]
	struct alignas(8) p_f32x4 { hdr_t h; f32 v[4]; };       // line width, depth bias, blend constants, depth bounds
	struct alignas(8) p_stencil { hdr_t h; VkStencilFaceFlags face; u32 value; };
	struct alignas(8) p_push_constants { hdr_t h; VkPipelineLayout layout; VkShaderStageFlags stages; u32 offset, size, pad; }; // + bytes
	struct alignas(8) p_update_buffer { hdr_t h; VkBuffer dst; VkDeviceSize offset, size; }; // + bytes
	struct alignas(8) p_fill_buffer { hdr_t h; VkBuffer dst; VkDeviceSize offset, size; u32 data, pad; };
	struct alignas(8) p_copy_buffer { hdr_t h; VkBuffer src, dst; u32 count, pad; }; // + VkBufferCopy[]
	struct alignas(8) p_copy_image { hdr_t h; VkImage src, dst; VkImageLayout src_layout, dst_layout; u32 count; VkFilter filter; }; // + VkImageCopy[] / VkImageBlit[]
	struct alignas(8) p_buffer_image { hdr_t h; VkBuffer buffer; VkImage image; VkImageLayout layout; u32 count; }; // + VkBufferImageCopy[]
	struct alignas(8) p_clear_image { hdr_t h; VkImage image; VkImageLayout layout; u32 count; union { VkClearColorValue color; VkClearDepthStencilValue depth_stencil; }; }; // + ranges
	struct alignas(8) p_clear_attachments { hdr_t h; u32 attachment_count, rect_count; }; // + VkClearAttachment[], VkClearRect[]
	struct alignas(8) p_barrier { hdr_t h; VkPipelineStageFlags src, dst; VkDependencyFlags flags; u32 mem_count, buf_count, img_count; }; // + 3 arrays
	struct alignas(8) p_wait_events { hdr_t h; VkPipelineStageFlags src, dst; u32 event_count, mem_count, buf_count, img_count; }; // + events, 3 arrays
	struct alignas(8) p_set_event { hdr_t h; VkEvent event; VkPipelineStageFlags stages; u32 pad; };
	struct alignas(8) p_query { hdr_t h; VkQueryPool pool; u32 a, b; }; // begin (query, flags), end (query), reset (first, count)
	struct alignas(8) p_copy_query { hdr_t h; VkQueryPool pool; VkBuffer dst; VkDeviceSize dst_offset, stride; u32 first, count; VkQueryResultFlags flags; u32 pad; };
	struct alignas(8) p_begin_rp { hdr_t h; VkRenderPass pass; VkFramebuffer fbo; VkRect2D area; VkSubpassContents contents; u32 clear_count; }; // + VkClearValue[]
	struct alignas(8) p_none { hdr_t h; };
	struct alignas(8) p_cond { hdr_t h; VkBuffer buffer; VkDeviceSize offset; VkConditionalRenderingFlagsEXT flags; u32 pad; };
	struct alignas(8) p_label { hdr_t h; f32 color[4]; u32 length, pad; }; // + the name, NUL-terminated
	struct alignas(8) p_dep { hdr_t h; VkEvent event; u32 event_count, pad; }; // + events (wait_events2), then event_count dep_t blocks
	struct alignas(8) dep_t { VkDependencyFlags flags; u32 mem_count, buf_count, img_count; }; // + VkMemoryBarrier2[], VkBufferMemoryBarrier2[], VkImageMemoryBarrier2[]

	template <typename T>
	static constexpr u64 bytes(u64 count)
	{
		return align8(sizeof(T) * count);
	}

	template <typename T>
	static u8* copy_tail(u8* dst, const T* src, u32 count)
	{
		if (count)
		{
			std::memcpy(dst, src, sizeof(T) * count);
		}

		return dst + bytes<T>(count);
	}

	template <typename T>
	static const T* read_tail(const u8*& src, u32 count)
	{
		const T* result = count ? reinterpret_cast<const T*>(src) : nullptr;
		src += bytes<T>(count);
		return result;
	}

	template <typename P>
	static u8* tail_of(P* p)
	{
		return reinterpret_cast<u8*>(p) + sizeof(P);
	}

	template <typename P>
	static const u8* tail_of(const P& p)
	{
		return reinterpret_cast<const u8*>(&p) + sizeof(P);
	}

	template <typename T>
	static bool chained(const T* items, u32 count)
	{
		for (u32 i = 0; i < count; i++)
		{
			if (items[i].pNext)
			{
				return true;
			}
		}

		return false;
	}

	// ---- Chunks ----

	static void release_chunk(chunk_t* c) // replayer, s_replay held
	{
		std::lock_guard lk(s_free_lock);
		c->next = s_free;
		s_free = c;
		s_chunks_live--;
	}

	static void publish() // producer
	{
		s_published.release(s_p.wr_seq);
	}

	static void replay_locked(u64 target);

	static chunk_t* take_chunk() // producer
	{
		for (u32 round = 0;; round++)
		{
			{
				std::lock_guard lk(s_free_lock);
				if (chunk_t* c = s_free)
				{
					s_free = c->next;
					c->next = nullptr;
					return c;
				}

				if (s_chunks_total < c_max_chunks)
				{
					s_chunks_total++;
					break;
				}
			}

			// At the cap: the backlog holds every chunk. Run it here; it frees every chunk but the one being written.
			ensure(round < 2, "vanillad1: CSREC: the arena stays full after a replay");
			s_p.st_backpressure++;
			publish();
			std::lock_guard lk(s_replay);
			replay_locked(s_published.load());
		}

		return new chunk_t;
	}

	static void next_chunk() // producer: the current chunk cannot hold the next packet
	{
		chunk_t* fresh = take_chunk();
		fresh->next = nullptr;

		auto* jump = reinterpret_cast<hdr_t*>(s_p.wr_chunk->data + s_p.wr_off);
		*jump = hdr_t{op::jump, 0, static_cast<u32>(sizeof(hdr_t))};
		s_p.wr_chunk->next = fresh; // before the publish below or the seal's: the replayer follows it after an acquire
		s_p.wr_chunk = fresh;
		s_p.wr_off = 0;
		s_p.wr_seq++;

		const u32 live = ++s_chunks_live;
		if (live > st_chunks_peak.observe())
		{
			st_chunks_peak.release(live);
		}

		if (s_p.mode_open >= 3)
		{
			// CSREC=3: the filled chunk goes to the worker now
			publish();
			vk::vd1_cs_wake();
		}
	}

	static void on_foreign()
	{
		// A vkCmd* into the RSX thread's open deferred buffer from another thread. Two threads recording into one command
		// buffer is a Vulkan violation without external synchronisation; here it would also interleave two writers in the arena
		// (X527 review F4: silent garbage at replay). So it stops the emulator, unless VANILLAD1_CSREC_FOREIGN_OK=1 (then it is
		// counted and logged, and encoded unsynchronised: only safe if the RSX thread is parked, risks R9).
		st_foreign++;

		static const bool s_foreign_ok = []()
		{
			const char* env = std::getenv("VANILLAD1_CSREC_FOREIGN_OK");
			return env && *env == '1';
		}();

		if (!s_foreign_ok)
		{
			fmt::throw_exception("vanillad1: CSREC: vkCmd* into the RSX thread's deferred primary buffer from thread '%s' (VANILLAD1_CSREC_FOREIGN_OK=1 lets it through)",
				thread_ctrl::get_name());
		}

		const u64 now = get_system_time();
		const u64 last = s_foreign_last_log_us.observe();
		if (now - last >= 10'000'000 && s_foreign_last_log_us.compare_and_swap_test(last, now))
		{
			rsx_log.error("vanillad1: CSREC: a vkCmd* into the RSX thread's deferred primary buffer from thread '%s' (encoded unsynchronised; total %u)",
				thread_ctrl::get_name(), static_cast<u32>(st_foreign.load()));
		}
	}

	// Reserves a packet with a tail of `tail_bytes`; nullptr when it can never fit (the caller falls back).
	template <typename P>
	static P* put(op code, u64 tail_bytes = 0)
	{
		const u64 size = align8(sizeof(P) + tail_bytes);
		if (size > c_max_packet) [[unlikely]]
		{
			return nullptr;
		}

		if (tl_kind != kind::rsx) [[unlikely]]
		{
			on_foreign();
		}

		if (s_p.wr_off + size + sizeof(hdr_t) > chunk_t::capacity) [[unlikely]]
		{
			next_chunk();
		}

		s_p.sample = (s_p.wr_seq & c_sample_mask) == 0;
		if (s_p.sample) [[unlikely]]
		{
			s_p.sample_t0 = utils::get_tsc();
		}

		auto* p = reinterpret_cast<P*>(s_p.wr_chunk->data + s_p.wr_off);
		p->h = hdr_t{code, 0, static_cast<u32>(size)};
		return p;
	}

	template <typename P>
	static void commit(P* p)
	{
		s_p.wr_off += p->h.size;
		s_p.wr_seq++;
		s_p.st_packets++;
		s_p.st_bytes += p->h.size;

		if (s_p.sample) [[unlikely]]
		{
			s_p.st_enc_tsc += utils::get_tsc() - s_p.sample_t0;
			s_p.sample = false;
		}
	}

	// The command cannot be encoded: replay the arena (the open buffer is then current in the driver) and record this one
	// call directly, under s_replay so no other replayer touches the pool meanwhile. The buffer stays deferred afterwards.
	template <typename F>
	static void fallback(F&& call)
	{
		s_p.sample = false;
		publish();
		std::lock_guard lk(s_replay);
		replay_locked(s_p.wr_seq);
		ensure(s_r.rd_cb && s_r.rd_cb == g_open.load(std::memory_order_relaxed), "vanillad1: CSREC: fallback without an open buffer");
		call(s_r.rd_cb);
		s_p.st_fallbacks++;
	}

	// ---- Dependency info (synchronization2) ----

	static bool dep_chained(const VkDependencyInfoKHR& d)
	{
		return d.pNext || chained(d.pMemoryBarriers, d.memoryBarrierCount) || chained(d.pBufferMemoryBarriers, d.bufferMemoryBarrierCount) ||
			chained(d.pImageMemoryBarriers, d.imageMemoryBarrierCount);
	}

	static u64 dep_bytes(const VkDependencyInfoKHR& d)
	{
		return sizeof(dep_t) + bytes<VkMemoryBarrier2KHR>(d.memoryBarrierCount) + bytes<VkBufferMemoryBarrier2KHR>(d.bufferMemoryBarrierCount) +
			bytes<VkImageMemoryBarrier2KHR>(d.imageMemoryBarrierCount);
	}

	static u8* dep_write(u8* t, const VkDependencyInfoKHR& d)
	{
		auto* x = reinterpret_cast<dep_t*>(t);
		*x = dep_t{d.dependencyFlags, d.memoryBarrierCount, d.bufferMemoryBarrierCount, d.imageMemoryBarrierCount};
		t += sizeof(dep_t);
		t = copy_tail(t, d.pMemoryBarriers, d.memoryBarrierCount);
		t = copy_tail(t, d.pBufferMemoryBarriers, d.bufferMemoryBarrierCount);
		t = copy_tail(t, d.pImageMemoryBarriers, d.imageMemoryBarrierCount);
		return t;
	}

	static VkDependencyInfoKHR dep_read(const u8*& t)
	{
		const auto* x = reinterpret_cast<const dep_t*>(t);
		t += sizeof(dep_t);

		VkDependencyInfoKHR d{};
		d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
		d.pNext = nullptr;
		d.dependencyFlags = x->flags;
		d.memoryBarrierCount = x->mem_count;
		d.pMemoryBarriers = read_tail<VkMemoryBarrier2KHR>(t, x->mem_count);
		d.bufferMemoryBarrierCount = x->buf_count;
		d.pBufferMemoryBarriers = read_tail<VkBufferMemoryBarrier2KHR>(t, x->buf_count);
		d.imageMemoryBarrierCount = x->img_count;
		d.pImageMemoryBarriers = read_tail<VkImageMemoryBarrier2KHR>(t, x->img_count);
		return d;
	}

	constexpr u32 c_max_wait_events2 = 16;

	// ---- Encoders ----

	void enc::bind_pipeline(VkPipelineBindPoint bind_point, VkPipeline pipeline)
	{
		auto* p = put<p_bind_pipeline>(op::bind_pipeline);
		p->pipeline = pipeline;
		p->bind_point = bind_point;
		commit(p);
	}

	void enc::bind_descriptor_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout, u32 first_set, u32 set_count, const VkDescriptorSet* sets, u32 dynamic_offset_count, const u32* dynamic_offsets)
	{
		auto* p = put<p_bind_sets>(op::bind_descriptor_sets, bytes<VkDescriptorSet>(set_count) + bytes<u32>(dynamic_offset_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdBindDescriptorSets(cb, bind_point, layout, first_set, set_count, sets, dynamic_offset_count, dynamic_offsets); });
		}

		p->layout = layout;
		p->bind_point = bind_point;
		p->first_set = first_set;
		p->set_count = set_count;
		p->offset_count = dynamic_offset_count;
		u8* t = copy_tail(tail_of(p), sets, set_count);
		copy_tail(t, dynamic_offsets, dynamic_offset_count);
		commit(p);
	}

	void enc::bind_index_buffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType index_type)
	{
		auto* p = put<p_bind_index>(op::bind_index_buffer);
		p->buffer = buffer;
		p->offset = offset;
		p->type = index_type;
		commit(p);
	}

	void enc::bind_vertex_buffers(u32 first_binding, u32 binding_count, const VkBuffer* buffers, const VkDeviceSize* offsets)
	{
		auto* p = put<p_bind_vertex>(op::bind_vertex_buffers, bytes<VkBuffer>(binding_count) + bytes<VkDeviceSize>(binding_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdBindVertexBuffers(cb, first_binding, binding_count, buffers, offsets); });
		}

		p->first = first_binding;
		p->count = binding_count;
		u8* t = copy_tail(tail_of(p), buffers, binding_count);
		copy_tail(t, offsets, binding_count);
		commit(p);
	}

	void enc::draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance)
	{
		auto* p = put<p_draw>(op::draw);
		p->vertex_count = vertex_count;
		p->instance_count = instance_count;
		p->first_vertex = first_vertex;
		p->first_instance = first_instance;
		commit(p);
	}

	void enc::draw_indexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset, u32 first_instance)
	{
		auto* p = put<p_draw_indexed>(op::draw_indexed);
		p->index_count = index_count;
		p->instance_count = instance_count;
		p->first_index = first_index;
		p->vertex_offset = vertex_offset;
		p->first_instance = first_instance;
		commit(p);
	}

	void enc::draw_multi(u32 draw_count, const VkMultiDrawInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride)
	{
		// VKDraw.cpp:1441-1443 passes the draw clause's subranges with stride sizeof(rsx::draw_range_t): copied packed
		auto* p = put<p_draw_multi>(op::draw_multi, bytes<VkMultiDrawInfoEXT>(draw_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdDrawMultiEXT(cb, draw_count, info, instance_count, first_instance, stride); });
		}

		p->count = draw_count;
		p->instance_count = instance_count;
		p->first_instance = first_instance;

		auto* dst = reinterpret_cast<VkMultiDrawInfoEXT*>(tail_of(p));
		const auto* src = reinterpret_cast<const u8*>(info);
		for (u32 i = 0; i < draw_count; i++)
		{
			std::memcpy(dst + i, src + u64{i} * stride, sizeof(VkMultiDrawInfoEXT));
		}

		commit(p);
	}

	void enc::draw_multi_indexed(u32 draw_count, const VkMultiDrawIndexedInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride, const s32* vertex_offset)
	{
		auto* p = put<p_draw_multi_indexed>(op::draw_multi_indexed, bytes<VkMultiDrawIndexedInfoEXT>(draw_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdDrawMultiIndexedEXT(cb, draw_count, info, instance_count, first_instance, stride, vertex_offset); });
		}

		p->count = draw_count;
		p->instance_count = instance_count;
		p->first_instance = first_instance;
		p->has_offset = vertex_offset ? 1 : 0;
		p->vertex_offset = vertex_offset ? *vertex_offset : 0;

		auto* dst = reinterpret_cast<VkMultiDrawIndexedInfoEXT*>(tail_of(p));
		const auto* src = reinterpret_cast<const u8*>(info);
		for (u32 i = 0; i < draw_count; i++)
		{
			std::memcpy(dst + i, src + u64{i} * stride, sizeof(VkMultiDrawIndexedInfoEXT));
		}

		commit(p);
	}

	void enc::dispatch(u32 x, u32 y, u32 z)
	{
		auto* p = put<p_dispatch>(op::dispatch);
		p->x = x;
		p->y = y;
		p->z = z;
		commit(p);
	}

	void enc::set_viewport(u32 first, u32 count, const VkViewport* viewports)
	{
		auto* p = put<p_rects>(op::set_viewport, bytes<VkViewport>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdSetViewport(cb, first, count, viewports); });
		}

		p->first = first;
		p->count = count;
		copy_tail(tail_of(p), viewports, count);
		commit(p);
	}

	void enc::set_scissor(u32 first, u32 count, const VkRect2D* scissors)
	{
		auto* p = put<p_rects>(op::set_scissor, bytes<VkRect2D>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdSetScissor(cb, first, count, scissors); });
		}

		p->first = first;
		p->count = count;
		copy_tail(tail_of(p), scissors, count);
		commit(p);
	}

	void enc::set_line_width(float width)
	{
		auto* p = put<p_f32x4>(op::set_line_width);
		p->v[0] = width;
		p->v[1] = p->v[2] = p->v[3] = 0.f;
		commit(p);
	}

	void enc::set_depth_bias(float constant_factor, float clamp, float slope_factor)
	{
		auto* p = put<p_f32x4>(op::set_depth_bias);
		p->v[0] = constant_factor;
		p->v[1] = clamp;
		p->v[2] = slope_factor;
		p->v[3] = 0.f;
		commit(p);
	}

	void enc::set_blend_constants(const float constants[4])
	{
		auto* p = put<p_f32x4>(op::set_blend_constants);
		std::memcpy(p->v, constants, sizeof(p->v));
		commit(p);
	}

	void enc::set_depth_bounds(float min_bounds, float max_bounds)
	{
		auto* p = put<p_f32x4>(op::set_depth_bounds);
		p->v[0] = min_bounds;
		p->v[1] = max_bounds;
		p->v[2] = p->v[3] = 0.f;
		commit(p);
	}

	static void put_stencil(op code, VkStencilFaceFlags face, u32 value)
	{
		auto* p = put<p_stencil>(code);
		p->face = face;
		p->value = value;
		commit(p);
	}

	void enc::set_stencil_compare_mask(VkStencilFaceFlags face, u32 mask) { put_stencil(op::set_stencil_compare_mask, face, mask); }
	void enc::set_stencil_write_mask(VkStencilFaceFlags face, u32 mask) { put_stencil(op::set_stencil_write_mask, face, mask); }
	void enc::set_stencil_reference(VkStencilFaceFlags face, u32 reference) { put_stencil(op::set_stencil_reference, face, reference); }

	void enc::push_constants(VkPipelineLayout layout, VkShaderStageFlags stages, u32 offset, u32 size, const void* values)
	{
		auto* p = put<p_push_constants>(op::push_constants, align8(size));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdPushConstants(cb, layout, stages, offset, size, values); });
		}

		p->layout = layout;
		p->stages = stages;
		p->offset = offset;
		p->size = size;
		std::memcpy(tail_of(p), values, size);
		commit(p);
	}

	void enc::update_buffer(VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, const void* data)
	{
		// The data is copied now, as the driver copies it at record time today. VKGSRender.cpp:3171 passes a pointer into
		// the live host context (last_label_acquire_event): its value at this call is what reaches the GPU, as before.
		auto* p = size <= 65536 ? put<p_update_buffer>(op::update_buffer, align8(size)) : nullptr;
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdUpdateBuffer(cb, dst, offset, size, data); });
		}

		p->dst = dst;
		p->offset = offset;
		p->size = size;
		std::memcpy(tail_of(p), data, static_cast<usz>(size));
		commit(p);
	}

	void enc::fill_buffer(VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, u32 data)
	{
		auto* p = put<p_fill_buffer>(op::fill_buffer);
		p->dst = dst;
		p->offset = offset;
		p->size = size;
		p->data = data;
		commit(p);
	}

	void enc::copy_buffer(VkBuffer src, VkBuffer dst, u32 count, const VkBufferCopy* regions)
	{
		auto* p = put<p_copy_buffer>(op::copy_buffer, bytes<VkBufferCopy>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdCopyBuffer(cb, src, dst, count, regions); });
		}

		p->src = src;
		p->dst = dst;
		p->count = count;
		copy_tail(tail_of(p), regions, count);
		commit(p);
	}

	void enc::copy_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageCopy* regions)
	{
		auto* p = put<p_copy_image>(op::copy_image, bytes<VkImageCopy>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdCopyImage(cb, src, src_layout, dst, dst_layout, count, regions); });
		}

		p->src = src;
		p->dst = dst;
		p->src_layout = src_layout;
		p->dst_layout = dst_layout;
		p->count = count;
		p->filter = VK_FILTER_NEAREST;
		copy_tail(tail_of(p), regions, count);
		commit(p);
	}

	void enc::blit_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageBlit* regions, VkFilter filter)
	{
		auto* p = put<p_copy_image>(op::blit_image, bytes<VkImageBlit>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdBlitImage(cb, src, src_layout, dst, dst_layout, count, regions, filter); });
		}

		p->src = src;
		p->dst = dst;
		p->src_layout = src_layout;
		p->dst_layout = dst_layout;
		p->count = count;
		p->filter = filter;
		copy_tail(tail_of(p), regions, count);
		commit(p);
	}

	void enc::copy_buffer_to_image(VkBuffer src, VkImage dst, VkImageLayout dst_layout, u32 count, const VkBufferImageCopy* regions)
	{
		auto* p = put<p_buffer_image>(op::copy_buffer_to_image, bytes<VkBufferImageCopy>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdCopyBufferToImage(cb, src, dst, dst_layout, count, regions); });
		}

		p->buffer = src;
		p->image = dst;
		p->layout = dst_layout;
		p->count = count;
		copy_tail(tail_of(p), regions, count);
		commit(p);
	}

	void enc::copy_image_to_buffer(VkImage src, VkImageLayout src_layout, VkBuffer dst, u32 count, const VkBufferImageCopy* regions)
	{
		auto* p = put<p_buffer_image>(op::copy_image_to_buffer, bytes<VkBufferImageCopy>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdCopyImageToBuffer(cb, src, src_layout, dst, count, regions); });
		}

		p->buffer = dst;
		p->image = src;
		p->layout = src_layout;
		p->count = count;
		copy_tail(tail_of(p), regions, count);
		commit(p);
	}

	void enc::clear_color_image(VkImage image, VkImageLayout layout, const VkClearColorValue* color, u32 count, const VkImageSubresourceRange* ranges)
	{
		auto* p = put<p_clear_image>(op::clear_color_image, bytes<VkImageSubresourceRange>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdClearColorImage(cb, image, layout, color, count, ranges); });
		}

		p->image = image;
		p->layout = layout;
		p->count = count;
		p->color = *color;
		copy_tail(tail_of(p), ranges, count);
		commit(p);
	}

	void enc::clear_depth_stencil_image(VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, u32 count, const VkImageSubresourceRange* ranges)
	{
		auto* p = put<p_clear_image>(op::clear_depth_stencil_image, bytes<VkImageSubresourceRange>(count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdClearDepthStencilImage(cb, image, layout, value, count, ranges); });
		}

		p->image = image;
		p->layout = layout;
		p->count = count;
		p->color = {};
		p->depth_stencil = *value;
		copy_tail(tail_of(p), ranges, count);
		commit(p);
	}

	void enc::clear_attachments(u32 attachment_count, const VkClearAttachment* attachments, u32 rect_count, const VkClearRect* rects)
	{
		auto* p = put<p_clear_attachments>(op::clear_attachments, bytes<VkClearAttachment>(attachment_count) + bytes<VkClearRect>(rect_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdClearAttachments(cb, attachment_count, attachments, rect_count, rects); });
		}

		p->attachment_count = attachment_count;
		p->rect_count = rect_count;
		u8* t = copy_tail(tail_of(p), attachments, attachment_count);
		copy_tail(t, rects, rect_count);
		commit(p);
	}

	void enc::pipeline_barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
		u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img)
	{
		const bool has_chain = chained(mem, mem_count) || chained(buf, buf_count) || chained(img, img_count);
		auto* p = has_chain ? nullptr : put<p_barrier>(op::pipeline_barrier,
			bytes<VkMemoryBarrier>(mem_count) + bytes<VkBufferMemoryBarrier>(buf_count) + bytes<VkImageMemoryBarrier>(img_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdPipelineBarrier(cb, src, dst, flags, mem_count, mem, buf_count, buf, img_count, img); });
		}

		p->src = src;
		p->dst = dst;
		p->flags = flags;
		p->mem_count = mem_count;
		p->buf_count = buf_count;
		p->img_count = img_count;
		u8* t = copy_tail(tail_of(p), mem, mem_count);
		t = copy_tail(t, buf, buf_count);
		copy_tail(t, img, img_count);
		commit(p);
	}

	void enc::set_event(VkEvent event, VkPipelineStageFlags stages)
	{
		auto* p = put<p_set_event>(op::set_event);
		p->event = event;
		p->stages = stages;
		commit(p);
	}

	void enc::wait_events(u32 event_count, const VkEvent* events, VkPipelineStageFlags src, VkPipelineStageFlags dst,
		u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img)
	{
		const bool has_chain = chained(mem, mem_count) || chained(buf, buf_count) || chained(img, img_count);
		auto* p = has_chain ? nullptr : put<p_wait_events>(op::wait_events, bytes<VkEvent>(event_count) +
			bytes<VkMemoryBarrier>(mem_count) + bytes<VkBufferMemoryBarrier>(buf_count) + bytes<VkImageMemoryBarrier>(img_count));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdWaitEvents(cb, event_count, events, src, dst, mem_count, mem, buf_count, buf, img_count, img); });
		}

		p->src = src;
		p->dst = dst;
		p->event_count = event_count;
		p->mem_count = mem_count;
		p->buf_count = buf_count;
		p->img_count = img_count;
		u8* t = copy_tail(tail_of(p), events, event_count);
		t = copy_tail(t, mem, mem_count);
		t = copy_tail(t, buf, buf_count);
		copy_tail(t, img, img_count);
		commit(p);
	}

	void enc::begin_query(VkQueryPool pool, u32 query, VkQueryControlFlags flags)
	{
		auto* p = put<p_query>(op::begin_query);
		p->pool = pool;
		p->a = query;
		p->b = flags;
		commit(p);
	}

	void enc::end_query(VkQueryPool pool, u32 query)
	{
		auto* p = put<p_query>(op::end_query);
		p->pool = pool;
		p->a = query;
		p->b = 0;
		commit(p);
	}

	void enc::reset_query_pool(VkQueryPool pool, u32 first, u32 count)
	{
		auto* p = put<p_query>(op::reset_query_pool);
		p->pool = pool;
		p->a = first;
		p->b = count;
		commit(p);
	}

	void enc::copy_query_pool_results(VkQueryPool pool, u32 first, u32 count, VkBuffer dst, VkDeviceSize dst_offset, VkDeviceSize stride, VkQueryResultFlags flags)
	{
		auto* p = put<p_copy_query>(op::copy_query_pool_results);
		p->pool = pool;
		p->dst = dst;
		p->dst_offset = dst_offset;
		p->stride = stride;
		p->first = first;
		p->count = count;
		p->flags = flags;
		commit(p);
	}

	void enc::begin_render_pass(const VkRenderPassBeginInfo* info, VkSubpassContents contents)
	{
		auto* p = info->pNext ? nullptr : put<p_begin_rp>(op::begin_render_pass, bytes<VkClearValue>(info->clearValueCount));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::vkCmdBeginRenderPass(cb, info, contents); });
		}

		p->pass = info->renderPass;
		p->fbo = info->framebuffer;
		p->area = info->renderArea;
		p->contents = contents;
		p->clear_count = info->clearValueCount;
		copy_tail(tail_of(p), info->pClearValues, info->clearValueCount);
		commit(p);
	}

	void enc::end_render_pass()
	{
		commit(put<p_none>(op::end_render_pass));
	}

	void enc::begin_conditional_rendering(const VkConditionalRenderingBeginInfoEXT* info)
	{
		auto* p = info->pNext ? nullptr : put<p_cond>(op::begin_conditional_rendering);
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdBeginConditionalRenderingEXT(cb, info); });
		}

		p->buffer = info->buffer;
		p->offset = info->offset;
		p->flags = info->flags;
		commit(p);
	}

	void enc::end_conditional_rendering()
	{
		commit(put<p_none>(op::end_conditional_rendering));
	}

	void enc::insert_debug_label(const VkDebugUtilsLabelEXT* label)
	{
		const usz length = label->pLabelName ? std::strlen(label->pLabelName) : 0;
		auto* p = (label->pNext || length > 4096) ? nullptr : put<p_label>(op::insert_debug_label, length + 1);
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdInsertDebugUtilsLabelEXT(cb, label); });
		}

		std::memcpy(p->color, label->color, sizeof(p->color));
		p->length = static_cast<u32>(length);
		char* name = reinterpret_cast<char*>(tail_of(p));
		if (length)
		{
			std::memcpy(name, label->pLabelName, length);
		}

		name[length] = '\0';
		commit(p);
	}

	void enc::set_event2(VkEvent event, const VkDependencyInfoKHR* dependency)
	{
		auto* p = dep_chained(*dependency) ? nullptr : put<p_dep>(op::set_event2, dep_bytes(*dependency));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdSetEvent2KHR(cb, event, dependency); });
		}

		p->event = event;
		p->event_count = 1;
		dep_write(tail_of(p), *dependency);
		commit(p);
	}

	void enc::wait_events2(u32 event_count, const VkEvent* events, const VkDependencyInfoKHR* dependencies)
	{
		u64 tail = bytes<VkEvent>(event_count);
		bool encodable = event_count <= c_max_wait_events2;
		for (u32 i = 0; encodable && i < event_count; i++)
		{
			encodable = !dep_chained(dependencies[i]);
			tail += dep_bytes(dependencies[i]);
		}

		auto* p = encodable ? put<p_dep>(op::wait_events2, tail) : nullptr;
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdWaitEvents2KHR(cb, event_count, events, dependencies); });
		}

		p->event = VK_NULL_HANDLE;
		p->event_count = event_count;
		u8* t = copy_tail(tail_of(p), events, event_count);
		for (u32 i = 0; i < event_count; i++)
		{
			t = dep_write(t, dependencies[i]);
		}

		commit(p);
	}

	void enc::pipeline_barrier2(const VkDependencyInfoKHR* dependency)
	{
		auto* p = dep_chained(*dependency) ? nullptr : put<p_dep>(op::pipeline_barrier2, dep_bytes(*dependency));
		if (!p) [[unlikely]]
		{
			return fallback([&](VkCommandBuffer cb) { ::_vkCmdPipelineBarrier2KHR(cb, dependency); });
		}

		p->event = VK_NULL_HANDLE;
		p->event_count = 1;
		dep_write(tail_of(p), *dependency);
		commit(p);
	}

	// ---- Replay ----

	template <typename P>
	static const P& as(const hdr_t& h)
	{
		return reinterpret_cast<const P&>(h);
	}

	static void execute(const hdr_t& h) // s_replay held
	{
		const VkCommandBuffer cb = s_r.rd_cb;

		if (h.code > op::reset && !cb) [[unlikely]]
		{
			fmt::throw_exception("vanillad1: CSREC: packet %u outside a begin / end pair", static_cast<u32>(h.code));
		}

		switch (h.code)
		{
		case op::nop:
			break;
		case op::begin:
		{
			const auto& p = as<p_cb>(h);
			ensure(!s_r.rd_cb, "vanillad1: CSREC: begin while a buffer is open");
			VkCommandBufferInheritanceInfo inheritance{};
			inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
			VkCommandBufferBeginInfo info{};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags = p.flags;
			info.pInheritanceInfo = &inheritance;
			CHECK_RESULT(::vkBeginCommandBuffer(p.cb, &info));
			s_r.rd_cb = p.cb;
			break;
		}
		case op::end:
		{
			const auto& p = as<p_cb>(h);
			ensure(p.cb == cb, "vanillad1: CSREC: end of a buffer that is not open");
			CHECK_RESULT(::vkEndCommandBuffer(p.cb));
			s_r.rd_cb = VK_NULL_HANDLE;
			break;
		}
		case op::reset:
		{
			const auto& p = as<p_cb>(h);
			ensure(p.cb != cb, "vanillad1: CSREC: reset of the open buffer");
			CHECK_RESULT(::vkResetCommandBuffer(p.cb, p.flags));
			break;
		}
		case op::bind_pipeline:
		{
			const auto& p = as<p_bind_pipeline>(h);
			::vkCmdBindPipeline(cb, p.bind_point, p.pipeline);
			break;
		}
		case op::bind_descriptor_sets:
		{
			const auto& p = as<p_bind_sets>(h);
			const u8* t = tail_of(p);
			const auto* sets = read_tail<VkDescriptorSet>(t, p.set_count);
			const auto* offsets = read_tail<u32>(t, p.offset_count);
			::vkCmdBindDescriptorSets(cb, p.bind_point, p.layout, p.first_set, p.set_count, sets, p.offset_count, offsets);
			break;
		}
		case op::bind_index_buffer:
		{
			const auto& p = as<p_bind_index>(h);
			::vkCmdBindIndexBuffer(cb, p.buffer, p.offset, p.type);
			break;
		}
		case op::bind_vertex_buffers:
		{
			const auto& p = as<p_bind_vertex>(h);
			const u8* t = tail_of(p);
			const auto* buffers = read_tail<VkBuffer>(t, p.count);
			const auto* offsets = read_tail<VkDeviceSize>(t, p.count);
			::vkCmdBindVertexBuffers(cb, p.first, p.count, buffers, offsets);
			break;
		}
		case op::draw:
		{
			const auto& p = as<p_draw>(h);
			::vkCmdDraw(cb, p.vertex_count, p.instance_count, p.first_vertex, p.first_instance);
			break;
		}
		case op::draw_indexed:
		{
			const auto& p = as<p_draw_indexed>(h);
			::vkCmdDrawIndexed(cb, p.index_count, p.instance_count, p.first_index, p.vertex_offset, p.first_instance);
			break;
		}
		case op::draw_multi:
		{
			const auto& p = as<p_draw_multi>(h);
			const u8* t = tail_of(p);
			const auto* info = read_tail<VkMultiDrawInfoEXT>(t, p.count);
			::_vkCmdDrawMultiEXT(cb, p.count, info, p.instance_count, p.first_instance, sizeof(VkMultiDrawInfoEXT));
			break;
		}
		case op::draw_multi_indexed:
		{
			const auto& p = as<p_draw_multi_indexed>(h);
			const u8* t = tail_of(p);
			const auto* info = read_tail<VkMultiDrawIndexedInfoEXT>(t, p.count);
			::_vkCmdDrawMultiIndexedEXT(cb, p.count, info, p.instance_count, p.first_instance, sizeof(VkMultiDrawIndexedInfoEXT), p.has_offset ? &p.vertex_offset : nullptr);
			break;
		}
		case op::dispatch:
		{
			const auto& p = as<p_dispatch>(h);
			::vkCmdDispatch(cb, p.x, p.y, p.z);
			break;
		}
		case op::set_viewport:
		{
			const auto& p = as<p_rects>(h);
			const u8* t = tail_of(p);
			::vkCmdSetViewport(cb, p.first, p.count, read_tail<VkViewport>(t, p.count));
			break;
		}
		case op::set_scissor:
		{
			const auto& p = as<p_rects>(h);
			const u8* t = tail_of(p);
			::vkCmdSetScissor(cb, p.first, p.count, read_tail<VkRect2D>(t, p.count));
			break;
		}
		case op::set_line_width:
			::vkCmdSetLineWidth(cb, as<p_f32x4>(h).v[0]);
			break;
		case op::set_depth_bias:
		{
			const auto& p = as<p_f32x4>(h);
			::vkCmdSetDepthBias(cb, p.v[0], p.v[1], p.v[2]);
			break;
		}
		case op::set_blend_constants:
			::vkCmdSetBlendConstants(cb, as<p_f32x4>(h).v);
			break;
		case op::set_depth_bounds:
		{
			const auto& p = as<p_f32x4>(h);
			::vkCmdSetDepthBounds(cb, p.v[0], p.v[1]);
			break;
		}
		case op::set_stencil_compare_mask:
		{
			const auto& p = as<p_stencil>(h);
			::vkCmdSetStencilCompareMask(cb, p.face, p.value);
			break;
		}
		case op::set_stencil_write_mask:
		{
			const auto& p = as<p_stencil>(h);
			::vkCmdSetStencilWriteMask(cb, p.face, p.value);
			break;
		}
		case op::set_stencil_reference:
		{
			const auto& p = as<p_stencil>(h);
			::vkCmdSetStencilReference(cb, p.face, p.value);
			break;
		}
		case op::push_constants:
		{
			const auto& p = as<p_push_constants>(h);
			::vkCmdPushConstants(cb, p.layout, p.stages, p.offset, p.size, tail_of(p));
			break;
		}
		case op::update_buffer:
		{
			const auto& p = as<p_update_buffer>(h);
			::vkCmdUpdateBuffer(cb, p.dst, p.offset, p.size, tail_of(p));
			break;
		}
		case op::fill_buffer:
		{
			const auto& p = as<p_fill_buffer>(h);
			::vkCmdFillBuffer(cb, p.dst, p.offset, p.size, p.data);
			break;
		}
		case op::copy_buffer:
		{
			const auto& p = as<p_copy_buffer>(h);
			const u8* t = tail_of(p);
			::vkCmdCopyBuffer(cb, p.src, p.dst, p.count, read_tail<VkBufferCopy>(t, p.count));
			break;
		}
		case op::copy_image:
		{
			const auto& p = as<p_copy_image>(h);
			const u8* t = tail_of(p);
			::vkCmdCopyImage(cb, p.src, p.src_layout, p.dst, p.dst_layout, p.count, read_tail<VkImageCopy>(t, p.count));
			break;
		}
		case op::blit_image:
		{
			const auto& p = as<p_copy_image>(h);
			const u8* t = tail_of(p);
			::vkCmdBlitImage(cb, p.src, p.src_layout, p.dst, p.dst_layout, p.count, read_tail<VkImageBlit>(t, p.count), p.filter);
			break;
		}
		case op::copy_buffer_to_image:
		{
			const auto& p = as<p_buffer_image>(h);
			const u8* t = tail_of(p);
			::vkCmdCopyBufferToImage(cb, p.buffer, p.image, p.layout, p.count, read_tail<VkBufferImageCopy>(t, p.count));
			break;
		}
		case op::copy_image_to_buffer:
		{
			const auto& p = as<p_buffer_image>(h);
			const u8* t = tail_of(p);
			::vkCmdCopyImageToBuffer(cb, p.image, p.layout, p.buffer, p.count, read_tail<VkBufferImageCopy>(t, p.count));
			break;
		}
		case op::clear_color_image:
		{
			const auto& p = as<p_clear_image>(h);
			const u8* t = tail_of(p);
			::vkCmdClearColorImage(cb, p.image, p.layout, &p.color, p.count, read_tail<VkImageSubresourceRange>(t, p.count));
			break;
		}
		case op::clear_depth_stencil_image:
		{
			const auto& p = as<p_clear_image>(h);
			const u8* t = tail_of(p);
			::vkCmdClearDepthStencilImage(cb, p.image, p.layout, &p.depth_stencil, p.count, read_tail<VkImageSubresourceRange>(t, p.count));
			break;
		}
		case op::clear_attachments:
		{
			const auto& p = as<p_clear_attachments>(h);
			const u8* t = tail_of(p);
			const auto* attachments = read_tail<VkClearAttachment>(t, p.attachment_count);
			const auto* rects = read_tail<VkClearRect>(t, p.rect_count);
			::vkCmdClearAttachments(cb, p.attachment_count, attachments, p.rect_count, rects);
			break;
		}
		case op::pipeline_barrier:
		{
			const auto& p = as<p_barrier>(h);
			const u8* t = tail_of(p);
			const auto* mem = read_tail<VkMemoryBarrier>(t, p.mem_count);
			const auto* buf = read_tail<VkBufferMemoryBarrier>(t, p.buf_count);
			const auto* img = read_tail<VkImageMemoryBarrier>(t, p.img_count);
			::vkCmdPipelineBarrier(cb, p.src, p.dst, p.flags, p.mem_count, mem, p.buf_count, buf, p.img_count, img);
			break;
		}
		case op::set_event:
		{
			const auto& p = as<p_set_event>(h);
			::vkCmdSetEvent(cb, p.event, p.stages);
			break;
		}
		case op::wait_events:
		{
			const auto& p = as<p_wait_events>(h);
			const u8* t = tail_of(p);
			const auto* events = read_tail<VkEvent>(t, p.event_count);
			const auto* mem = read_tail<VkMemoryBarrier>(t, p.mem_count);
			const auto* buf = read_tail<VkBufferMemoryBarrier>(t, p.buf_count);
			const auto* img = read_tail<VkImageMemoryBarrier>(t, p.img_count);
			::vkCmdWaitEvents(cb, p.event_count, events, p.src, p.dst, p.mem_count, mem, p.buf_count, buf, p.img_count, img);
			break;
		}
		case op::begin_query:
		{
			const auto& p = as<p_query>(h);
			::vkCmdBeginQuery(cb, p.pool, p.a, p.b);
			break;
		}
		case op::end_query:
		{
			const auto& p = as<p_query>(h);
			::vkCmdEndQuery(cb, p.pool, p.a);
			break;
		}
		case op::reset_query_pool:
		{
			const auto& p = as<p_query>(h);
			::vkCmdResetQueryPool(cb, p.pool, p.a, p.b);
			break;
		}
		case op::copy_query_pool_results:
		{
			const auto& p = as<p_copy_query>(h);
			::vkCmdCopyQueryPoolResults(cb, p.pool, p.first, p.count, p.dst, p.dst_offset, p.stride, p.flags);
			break;
		}
		case op::begin_render_pass:
		{
			const auto& p = as<p_begin_rp>(h);
			const u8* t = tail_of(p);
			VkRenderPassBeginInfo info{};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = p.pass;
			info.framebuffer = p.fbo;
			info.renderArea = p.area;
			info.clearValueCount = p.clear_count;
			info.pClearValues = read_tail<VkClearValue>(t, p.clear_count);
			::vkCmdBeginRenderPass(cb, &info, p.contents);
			break;
		}
		case op::end_render_pass:
			::vkCmdEndRenderPass(cb);
			break;
		case op::begin_conditional_rendering:
		{
			const auto& p = as<p_cond>(h);
			VkConditionalRenderingBeginInfoEXT info{};
			info.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT;
			info.buffer = p.buffer;
			info.offset = p.offset;
			info.flags = p.flags;
			::_vkCmdBeginConditionalRenderingEXT(cb, &info);
			break;
		}
		case op::end_conditional_rendering:
			::_vkCmdEndConditionalRenderingEXT(cb);
			break;
		case op::insert_debug_label:
		{
			const auto& p = as<p_label>(h);
			VkDebugUtilsLabelEXT label{};
			label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
			label.pLabelName = reinterpret_cast<const char*>(tail_of(p));
			std::memcpy(label.color, p.color, sizeof(label.color));
			::_vkCmdInsertDebugUtilsLabelEXT(cb, &label);
			break;
		}
		case op::set_event2:
		{
			const auto& p = as<p_dep>(h);
			const u8* t = tail_of(p);
			const VkDependencyInfoKHR dependency = dep_read(t);
			::_vkCmdSetEvent2KHR(cb, p.event, &dependency);
			break;
		}
		case op::wait_events2:
		{
			const auto& p = as<p_dep>(h);
			const u8* t = tail_of(p);
			const auto* events = read_tail<VkEvent>(t, p.event_count);
			VkDependencyInfoKHR dependencies[c_max_wait_events2];
			for (u32 i = 0; i < p.event_count; i++)
			{
				dependencies[i] = dep_read(t);
			}

			::_vkCmdWaitEvents2KHR(cb, p.event_count, events, dependencies);
			break;
		}
		case op::pipeline_barrier2:
		{
			const auto& p = as<p_dep>(h);
			const u8* t = tail_of(p);
			const VkDependencyInfoKHR dependency = dep_read(t);
			::_vkCmdPipelineBarrier2KHR(cb, &dependency);
			break;
		}
		case op::jump:
		case op::count:
		default:
			fmt::throw_exception("vanillad1: CSREC: bad packet %u", static_cast<u32>(h.code));
		}
	}

	static void replay_locked(u64 target) // s_replay held; target <= s_published
	{
		u64 done = s_replayed.observe(); // written only under s_replay
		if (done >= target)
		{
			return;
		}

		const u64 first = done;
		const u64 t0 = utils::get_tsc();

		while (done < target)
		{
			const auto* h = reinterpret_cast<const hdr_t*>(s_r.rd_chunk->data + s_r.rd_off);
			if (h->code == op::jump)
			{
				chunk_t* old = s_r.rd_chunk;
				s_r.rd_chunk = old->next;
				s_r.rd_off = 0;
				ensure(s_r.rd_chunk, "vanillad1: CSREC: jump to no chunk");
				release_chunk(old);
			}
			else
			{
				execute(*h);
				s_r.rd_off += h->size;
			}

			done++;
		}

		s_replayed.release(done);

		const u32 k = static_cast<u32>(tl_kind);
		st_rep_packets[k] += done - first;
		st_rep_tsc[k] += utils::get_tsc() - t0;
	}

	static u64 tsc_to_us(u64 tsc)
	{
		const u64 freq = utils::get_tsc_freq();
		return freq ? tsc * 1'000'000 / freq : tsc / 3'000;
	}

	static void maybe_log() // producer, at seal
	{
		const u64 now = get_system_time();
		if (now - s_last_log_us < 10'000'000)
		{
			return;
		}

		s_last_log_us = now;

		u64 rep[3], rep_us[3];
		for (u32 i = 0; i < 3; i++)
		{
			rep[i] = st_rep_packets[i].exchange(0);
			rep_us[i] = tsc_to_us(st_rep_tsc[i].exchange(0));
		}

		u32 total = 0;
		{
			std::lock_guard lk(s_free_lock);
			total = s_chunks_total;
		}

		rsx_log.notice("vanillad1: csrec 10s: CSREC=%u buffers direct=%u m1=%u m2=%u m3=%u packets=%u kB=%u encode ms~%u fallbacks=%u foreign=%u "
			"replayed rsx=%u worker=%u helpers=%u replay ms rsx=%u worker=%u helpers=%u chunks=%u peak live=%u backpressure=%u",
			vd1::live(vd1::lf::CSREC), static_cast<u32>(s_p.st_cbs[0]), static_cast<u32>(s_p.st_cbs[1]), static_cast<u32>(s_p.st_cbs[2]), static_cast<u32>(s_p.st_cbs[3]),
			static_cast<u32>(s_p.st_packets), static_cast<u32>(s_p.st_bytes / 1024), static_cast<u32>(tsc_to_us(s_p.st_enc_tsc * (c_sample_mask + 1)) / 1000),
			static_cast<u32>(s_p.st_fallbacks), static_cast<u32>(st_foreign.exchange(0)),
			static_cast<u32>(rep[1]), static_cast<u32>(rep[2]), static_cast<u32>(rep[0]),
			static_cast<u32>(rep_us[1] / 1000), static_cast<u32>(rep_us[2] / 1000), static_cast<u32>(rep_us[0] / 1000),
			total, st_chunks_peak.exchange(s_chunks_live.load()), static_cast<u32>(s_p.st_backpressure));

		s_p.st_packets = s_p.st_bytes = s_p.st_enc_tsc = s_p.st_fallbacks = s_p.st_backpressure = 0;
		s_p.st_cbs[0] = s_p.st_cbs[1] = s_p.st_cbs[2] = s_p.st_cbs[3] = 0;
	}
}

// The API's bodies live in vd1_rec::api, so unqualified names resolve to this file's state and never to a vk:: name; the
// vk:: entry points at the end forward to them.
namespace vd1_rec::api
{
	void vd1_cs_rec_init(bool enabled)
	{
		std::lock_guard lk(s_replay);

		s_enabled = enabled;
		s_stopped = false;
		g_open.store(VK_NULL_HANDLE, std::memory_order_relaxed);
		s_p.mode_open = 0;
		s_r.rd_cb = VK_NULL_HANDLE;

		if (!enabled)
		{
			return;
		}

		if (!s_p.wr_chunk)
		{
			// First enabled boot: the arena
			std::lock_guard fl(s_free_lock);
			for (u32 i = 0; i < c_initial_chunks; i++)
			{
				auto* c = new chunk_t;
				c->next = s_free;
				s_free = c;
			}

			s_chunks_total = c_initial_chunks;
			s_p.wr_chunk = s_free;
			s_free = s_free->next;
			s_p.wr_chunk->next = nullptr;
			s_r.rd_chunk = s_p.wr_chunk;
			s_p.wr_off = s_r.rd_off = 0;
			s_chunks_live = 1;
		}
		else if (s_replayed.load() != s_p.wr_seq)
		{
			// A previous boot ended without on_exit's stop (failed init): its packets belong to a dead device. Drop them.
			rsx_log.error("vanillad1: CSREC: %u packets left from the last boot dropped", static_cast<u32>(s_p.wr_seq - s_replayed.load()));
			while (s_r.rd_chunk != s_p.wr_chunk)
			{
				chunk_t* old = s_r.rd_chunk;
				s_r.rd_chunk = old->next;
				release_chunk(old);
			}

			s_r.rd_off = s_p.wr_off;
			s_published.release(s_p.wr_seq);
			s_replayed.release(s_p.wr_seq);
		}

		rsx_log.notice("vanillad1: CSREC stream ready (VANILLAD1_CS=1): the live flag CSREC=1 records the RSX thread's primary command buffers through "
			"the packet stream and replays them at close, 2 replays them at submit (submit worker or a waiter), 3 also on the worker as each %u KiB chunk fills",
			chunk_t::capacity / 1024);
	}

	void vd1_cs_rec_set_producer()
	{
		tl_kind = kind::rsx;
	}

	void vd1_cs_rec_set_worker()
	{
		tl_kind = kind::worker;
	}

	bool vd1_cs_rec_active()
	{
		return s_enabled.observe() && tl_kind == kind::rsx && !s_stopped.observe() && vd1::live(vd1::lf::CSREC) != 0;
	}

	u32 vd1_cs_rec_begin_mode()
	{
		if (!s_enabled.observe())
		{
			return 0;
		}

		const u32 mode = vd1_cs_rec_active() ? std::min<u32>(vd1::live(vd1::lf::CSREC), 3) : 0;
		if (tl_kind == kind::rsx)
		{
			s_p.st_cbs[mode]++;
		}

		return mode;
	}

	void vd1_cs_rec_begin(VkCommandBuffer cb, VkCommandBufferUsageFlags flags, u32 mode)
	{
		ensure(mode >= 1 && mode <= 3);
		ensure(g_open.load(std::memory_order_relaxed) == VK_NULL_HANDLE, "vanillad1: CSREC: a deferred primary buffer begun while another is open");

		auto* p = put<p_cb>(op::begin);
		p->cb = cb;
		p->flags = flags;
		p->pad = 0;
		commit(p);

		s_p.mode_open = mode;
		g_open.store(cb, std::memory_order_relaxed);
	}

	void vd1_cs_rec_reset(VkCommandBuffer cb, VkCommandBufferResetFlags flags)
	{
		ensure(g_open.load(std::memory_order_relaxed) != cb, "vanillad1: CSREC: reset of the open buffer");

		auto* p = put<p_cb>(op::reset);
		p->cb = cb;
		p->flags = flags;
		p->pad = 0;
		commit(p);
	}

	u64 vd1_cs_seal(VkCommandBuffer cb)
	{
		ensure(g_open.load(std::memory_order_relaxed) == cb, "vanillad1: CSREC: seal of a buffer that is not open");

		auto* p = put<p_cb>(op::end);
		p->cb = cb;
		p->flags = 0;
		p->pad = 0;
		commit(p);

		g_open.store(VK_NULL_HANDLE, std::memory_order_relaxed);
		const u32 mode = std::exchange(s_p.mode_open, 0u);
		const u64 token = s_p.wr_seq;
		publish();

		if (mode == 1)
		{
			// CSREC=1: the encoders' proof on one thread: replay here, before the submit (queued or not)
			std::lock_guard lk(s_replay);
			replay_locked(token);
		}

		maybe_log();
		return token;
	}

	void vd1_cs_replay_until(u64 token)
	{
		if (!token || s_replayed.load() >= token) [[likely]]
		{
			return;
		}

		std::lock_guard lk(s_replay);
		const u64 published = s_published.load();
		ensure(token <= published, "vanillad1: CSREC: a submit token beyond what is published");
		replay_locked(token);
	}

	// X527 CSTRY: as vd1_cs_replay_until, but never waits for the replay lock: false when another thread (the worker) holds it,
	// so a poller (event::status) does not stall behind a whole buffer's replay
	bool vd1_cs_replay_until_try(u64 token)
	{
		if (!token || s_replayed.load() >= token) [[likely]]
		{
			return true;
		}

		if (!s_replay.try_lock())
		{
			return false;
		}

		const u64 published = s_published.load();
		ensure(token <= published, "vanillad1: CSREC: a submit token beyond what is published");
		replay_locked(token);
		s_replay.unlock();
		return true;
	}

	void vd1_cs_rec_quiesce()
	{
		if (!s_enabled.observe())
		{
			return;
		}

		if (tl_kind == kind::rsx)
		{
			if (s_replayed.load() >= s_p.wr_seq)
			{
				return;
			}

			publish();
			std::lock_guard lk(s_replay);
			replay_locked(s_p.wr_seq);
			return;
		}

		// Another thread (the constructor before the RSX thread starts, the destructor after stop): only what is published
		// can be pending, and after stop nothing is
		const u64 published = s_published.load();
		if (s_replayed.load() >= published)
		{
			return;
		}

		std::lock_guard lk(s_replay);
		replay_locked(published);
	}

	bool vd1_cs_rec_backlog()
	{
		return s_enabled.observe() && vd1::live(vd1::lf::CSREC) >= 3 && s_published.load() > s_replayed.load();
	}

	void vd1_cs_rec_replay_published()
	{
		std::lock_guard lk(s_replay);
		const u64 published = s_published.load();
		replay_locked(std::min(published, s_replayed.observe() + c_worker_batch));
	}

	void vd1_cs_rec_publish()
	{
		if (s_enabled.observe() && tl_kind == kind::rsx && s_p.mode_open >= 3 && s_published.observe() < s_p.wr_seq)
		{
			publish();
			vk::vd1_cs_wake();
		}
	}

	bool vd1_cs_rec_hold_begin()
	{
		if (!s_enabled.observe() || vd1::live(vd1::lf::CSREC) < 3)
		{
			return false;
		}

		s_replay.lock();
		return true;
	}

	void vd1_cs_rec_hold_end()
	{
		s_replay.unlock();
	}

	void vd1_cs_rec_stop(vk::command_buffer* open)
	{
		if (!s_enabled.observe() || s_stopped.observe())
		{
			return;
		}

		ensure(tl_kind == kind::rsx, "vanillad1: CSREC: stop from a thread other than the RSX thread");

		// Materialise: the open buffer (if deferred) is now really open in the driver with every command so far, so the rest
		// of its life (and every later buffer) is recorded directly
		publish();
		{
			std::lock_guard lk(s_replay);
			replay_locked(s_p.wr_seq);
			s_r.rd_cb = VK_NULL_HANDLE;
		}

		s_stopped = true;
		g_open.store(VK_NULL_HANDLE, std::memory_order_relaxed);
		s_p.mode_open = 0;

		if (open)
		{
			open->vd1_rec_mode = 0;
		}

		rsx_log.notice("vanillad1: CSREC stream stopped (renderer exit): %u packets replayed in all", static_cast<u32>(s_replayed.load()));
	}
}

namespace vk
{
	void vd1_cs_rec_init(bool enabled) { vd1_rec::api::vd1_cs_rec_init(enabled); }
	void vd1_cs_rec_set_producer() { vd1_rec::api::vd1_cs_rec_set_producer(); }
	void vd1_cs_rec_set_worker() { vd1_rec::api::vd1_cs_rec_set_worker(); }
	void vd1_cs_rec_stop(vk::command_buffer* open) { vd1_rec::api::vd1_cs_rec_stop(open); }
	u32 vd1_cs_rec_begin_mode() { return vd1_rec::api::vd1_cs_rec_begin_mode(); }
	bool vd1_cs_rec_active() { return vd1_rec::api::vd1_cs_rec_active(); }
	void vd1_cs_rec_begin(VkCommandBuffer cb, VkCommandBufferUsageFlags flags, u32 mode) { vd1_rec::api::vd1_cs_rec_begin(cb, flags, mode); }
	void vd1_cs_rec_reset(VkCommandBuffer cb, VkCommandBufferResetFlags flags) { vd1_rec::api::vd1_cs_rec_reset(cb, flags); }
	u64 vd1_cs_seal(VkCommandBuffer cb) { return vd1_rec::api::vd1_cs_seal(cb); }
	void vd1_cs_replay_until(u64 token) { vd1_rec::api::vd1_cs_replay_until(token); }
	bool vd1_cs_replay_until_try(u64 token) { return vd1_rec::api::vd1_cs_replay_until_try(token); }
	void vd1_cs_rec_quiesce() { vd1_rec::api::vd1_cs_rec_quiesce(); }
	bool vd1_cs_rec_backlog() { return vd1_rec::api::vd1_cs_rec_backlog(); }
	void vd1_cs_rec_replay_published() { vd1_rec::api::vd1_cs_rec_replay_published(); }
	void vd1_cs_rec_publish() { vd1_rec::api::vd1_cs_rec_publish(); }
	bool vd1_cs_rec_hold_begin() { return vd1_rec::api::vd1_cs_rec_hold_begin(); }
	void vd1_cs_rec_hold_end() { vd1_rec::api::vd1_cs_rec_hold_end(); }
}
