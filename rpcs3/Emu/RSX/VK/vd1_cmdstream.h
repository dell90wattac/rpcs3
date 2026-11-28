#pragma once

// vanillad1: X527 C1 stage 3 (CSREC): the RSX thread's vkCmd* stream (DRAFT, re/notes/X527/stage3; target path
// rpcs3/Emu/RSX/VK/vd1_cmdstream.h). VulkanAPI.h includes this file at its end, after vulkan_core.h and VKProcTable.h.
//
// While a primary-chain command buffer is open in a deferred mode (live flag CSREC 1/2/3, boot VANILLAD1_CS=1), every
// vkCmd* call the RSX thread makes on it is encoded into a packet arena (plain values and Vulkan handles, every pointer
// argument deep-copied) instead of reaching the driver; begin / end / reset of that buffer become packets too. The packets
// are replayed into the real VkCommandBuffer, in order, before the buffer's vkQueueSubmit:
//   CSREC=1  the RSX thread replays at end() (proves the encoders, no threading)
//   CSREC=2  whoever runs the buffer's submit packet replays first (the submit worker, or a help-drainer)
//   CSREC=3  as 2, and the worker also replays each arena chunk as soon as it fills (lags one chunk)
//   CSREC=0  (or VANILLAD1_CS unset, or Multithreaded RSX on) the stock path: the wrappers call straight through.
//
// The shim: each name below is a FUNCTION-LIKE macro, `#define vkCmdDraw(...) ::vd1_rec::vkCmdDraw(__VA_ARGS__)`.
//  - It is defined after the Vulkan headers were parsed, so the prototypes in vulkan_core.h are untouched (and
//    re-including them is a no-op behind their include guard).
//  - A function-like macro expands only where the name is followed by '(': `PFN_vkCmdDraw`, the member
//    `PFN_vkCmdCopyBuffer vkCmdCopyBuffer;` in VMA's VmaVulkanFunctions, `(PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;` in VMA's
//    importer, `_##func = ...` / `extern PFN_##func _##func` in VKProcTable.h / VulkanAPI.cpp, and `if (_vkCmdDrawMultiEXT)`
//    stay as they are. (An object-like macro would rename VMA's struct member and break VulkanAPI.cpp's proc table.)
//  - The target is a qualified name, so argument-dependent lookup never runs and ::vkCmdDraw is never a competing overload
//    (inline wrappers with the same names in a namespace, without the macro, would be ambiguous through ADL on
//    VkCommandBuffer_T and invisible from VKGSRender's global-namespace members).
//  - The wrappers below are defined BEFORE the macros and call the real function as `(::vkCmdDraw)(...)`: a parenthesised
//    name is never macro-expanded, so there is no recursion even if this file were reached twice.
//  - Unused inline functions draw no warning on MSVC, GCC or clang (they are `inline`, not `static`).
// Define VD1_CS_NO_SHIM before the first include of VulkanAPI.h to get the declarations without the macros
// (vd1_cmdstream.cpp, which calls the real functions).

#include "VulkanAPI.h" // no-op when reached from VulkanAPI.h itself (pragma once on both)

#include <atomic>

namespace vk
{
	class command_buffer;

	// Lifecycle (vkutils/commands.cpp, VKGSRender.cpp, VKCommandStream.cpp). All but the replay entry points are called by
	// the RSX thread ("the producer"); see vd1_cmdstream.cpp for the threading rules.
	void vd1_cs_rec_init(bool enabled);              // VKGSRender ctor: VANILLAD1_CS=1 and not Multithreaded RSX
	void vd1_cs_rec_set_producer();                  // VKGSRender::on_init_thread: this thread encodes
	void vd1_cs_rec_set_worker();                    // the submit worker's thread entry (stats only)
	void vd1_cs_rec_stop(vk::command_buffer* open);  // VKGSRender::on_exit (RSX thread): replay all, record directly from now
	u32  vd1_cs_rec_begin_mode();                    // 0 = record directly; 1..3 = CSREC mode for the buffer being begun
	bool vd1_cs_rec_active();                        // the calling thread encodes primary-chain lifecycle calls
	void vd1_cs_rec_begin(VkCommandBuffer cb, VkCommandBufferUsageFlags flags, u32 mode); // BEGIN packet, opens the stream
	void vd1_cs_rec_reset(VkCommandBuffer cb, VkCommandBufferResetFlags flags);            // RESET packet
	u64  vd1_cs_seal(VkCommandBuffer cb);            // END packet; returns the token the buffer's submit carries (CSREC=1: replayed here)
	void vd1_cs_replay_until(u64 token);             // any thread: runs every packet before `token` (no-op if done)
	bool vd1_cs_replay_until_try(u64 token);         // X527 CSTRY: as above, false at once when the replay lock is held
	void vd1_cs_rec_quiesce();                       // before a direct call on a primary-chain buffer: replay everything
	bool vd1_cs_rec_backlog();                       // CSREC=3: published packets the worker may replay ahead of the submit
	void vd1_cs_rec_replay_published();              // worker: replay a bounded batch of what is published
	void vd1_cs_rec_publish();                       // RSX thread: let the worker see what is encoded (CSREC=3 idle hint)
	bool vd1_cs_rec_hold_begin();                    // CSREC=3: hold off the worker's replay around descriptors::flush()
	void vd1_cs_rec_hold_end();                      //   (call _end only when _begin returned true)
}

namespace vd1_rec
{
	// The deferred primary command buffer now open on the RSX thread, or null. Written only by the RSX thread (begin, seal,
	// stop); read by every wrapper on every thread. Relaxed: a thread other than the RSX thread only ever compares it with
	// its own (different) buffer, and the RSX thread reads its own writes.
	alignas(64) extern std::atomic<VkCommandBuffer> g_open;

	inline bool deferred(VkCommandBuffer cb)
	{
		const VkCommandBuffer open = g_open.load(std::memory_order_relaxed);
		return cb == open && open != VK_NULL_HANDLE;
	}

	// Encoders (vd1_cmdstream.cpp). Each copies its arguments into the arena; one that cannot (a pNext chain, a packet larger
	// than a chunk) replays the arena and calls the real function on the RSX thread instead ("fallback", counted).
	namespace enc
	{
		void bind_pipeline(VkPipelineBindPoint bind_point, VkPipeline pipeline);
		void bind_descriptor_sets(VkPipelineBindPoint bind_point, VkPipelineLayout layout, u32 first_set, u32 set_count, const VkDescriptorSet* sets, u32 dynamic_offset_count, const u32* dynamic_offsets);
		void bind_index_buffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType index_type);
		void bind_vertex_buffers(u32 first_binding, u32 binding_count, const VkBuffer* buffers, const VkDeviceSize* offsets);
		void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance);
		void draw_indexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset, u32 first_instance);
		void draw_multi(u32 draw_count, const VkMultiDrawInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride);
		void draw_multi_indexed(u32 draw_count, const VkMultiDrawIndexedInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride, const s32* vertex_offset);
		void dispatch(u32 x, u32 y, u32 z);
		void set_viewport(u32 first, u32 count, const VkViewport* viewports);
		void set_scissor(u32 first, u32 count, const VkRect2D* scissors);
		void set_line_width(float width);
		void set_depth_bias(float constant_factor, float clamp, float slope_factor);
		void set_blend_constants(const float constants[4]);
		void set_depth_bounds(float min_bounds, float max_bounds);
		void set_stencil_compare_mask(VkStencilFaceFlags face, u32 mask);
		void set_stencil_write_mask(VkStencilFaceFlags face, u32 mask);
		void set_stencil_reference(VkStencilFaceFlags face, u32 reference);
		void push_constants(VkPipelineLayout layout, VkShaderStageFlags stages, u32 offset, u32 size, const void* values);
		void update_buffer(VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, const void* data);
		void fill_buffer(VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, u32 data);
		void copy_buffer(VkBuffer src, VkBuffer dst, u32 count, const VkBufferCopy* regions);
		void copy_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageCopy* regions);
		void blit_image(VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageBlit* regions, VkFilter filter);
		void copy_buffer_to_image(VkBuffer src, VkImage dst, VkImageLayout dst_layout, u32 count, const VkBufferImageCopy* regions);
		void copy_image_to_buffer(VkImage src, VkImageLayout src_layout, VkBuffer dst, u32 count, const VkBufferImageCopy* regions);
		void clear_color_image(VkImage image, VkImageLayout layout, const VkClearColorValue* color, u32 count, const VkImageSubresourceRange* ranges);
		void clear_depth_stencil_image(VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, u32 count, const VkImageSubresourceRange* ranges);
		void clear_attachments(u32 attachment_count, const VkClearAttachment* attachments, u32 rect_count, const VkClearRect* rects);
		void pipeline_barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags, u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img);
		void set_event(VkEvent event, VkPipelineStageFlags stages);
		void wait_events(u32 event_count, const VkEvent* events, VkPipelineStageFlags src, VkPipelineStageFlags dst, u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img);
		void begin_query(VkQueryPool pool, u32 query, VkQueryControlFlags flags);
		void end_query(VkQueryPool pool, u32 query);
		void reset_query_pool(VkQueryPool pool, u32 first, u32 count);
		void copy_query_pool_results(VkQueryPool pool, u32 first, u32 count, VkBuffer dst, VkDeviceSize dst_offset, VkDeviceSize stride, VkQueryResultFlags flags);
		void begin_render_pass(const VkRenderPassBeginInfo* info, VkSubpassContents contents);
		void end_render_pass();
		void begin_conditional_rendering(const VkConditionalRenderingBeginInfoEXT* info);
		void end_conditional_rendering();
		void insert_debug_label(const VkDebugUtilsLabelEXT* label);
		void set_event2(VkEvent event, const VkDependencyInfoKHR* dependency);
		void wait_events2(u32 event_count, const VkEvent* events, const VkDependencyInfoKHR* dependencies);
		void pipeline_barrier2(const VkDependencyInfoKHR* dependency);
	}

	// ---- Core 1.0 prototypes (called directly) ----

	inline void vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipeline pipeline)
	{
		if (deferred(cb)) return enc::bind_pipeline(bind_point, pipeline);
		(::vkCmdBindPipeline)(cb, bind_point, pipeline);
	}

	inline void vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipelineLayout layout, u32 first_set, u32 set_count, const VkDescriptorSet* sets, u32 dynamic_offset_count, const u32* dynamic_offsets)
	{
		if (deferred(cb)) return enc::bind_descriptor_sets(bind_point, layout, first_set, set_count, sets, dynamic_offset_count, dynamic_offsets);
		(::vkCmdBindDescriptorSets)(cb, bind_point, layout, first_set, set_count, sets, dynamic_offset_count, dynamic_offsets);
	}

	inline void vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType index_type)
	{
		if (deferred(cb)) return enc::bind_index_buffer(buffer, offset, index_type);
		(::vkCmdBindIndexBuffer)(cb, buffer, offset, index_type);
	}

	inline void vkCmdBindVertexBuffers(VkCommandBuffer cb, u32 first_binding, u32 binding_count, const VkBuffer* buffers, const VkDeviceSize* offsets)
	{
		if (deferred(cb)) return enc::bind_vertex_buffers(first_binding, binding_count, buffers, offsets);
		(::vkCmdBindVertexBuffers)(cb, first_binding, binding_count, buffers, offsets);
	}

	inline void vkCmdDraw(VkCommandBuffer cb, u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance)
	{
		if (deferred(cb)) return enc::draw(vertex_count, instance_count, first_vertex, first_instance);
		(::vkCmdDraw)(cb, vertex_count, instance_count, first_vertex, first_instance);
	}

	inline void vkCmdDrawIndexed(VkCommandBuffer cb, u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset, u32 first_instance)
	{
		if (deferred(cb)) return enc::draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
		(::vkCmdDrawIndexed)(cb, index_count, instance_count, first_index, vertex_offset, first_instance);
	}

	inline void vkCmdDispatch(VkCommandBuffer cb, u32 x, u32 y, u32 z)
	{
		if (deferred(cb)) return enc::dispatch(x, y, z);
		(::vkCmdDispatch)(cb, x, y, z);
	}

	inline void vkCmdSetViewport(VkCommandBuffer cb, u32 first, u32 count, const VkViewport* viewports)
	{
		if (deferred(cb)) return enc::set_viewport(first, count, viewports);
		(::vkCmdSetViewport)(cb, first, count, viewports);
	}

	inline void vkCmdSetScissor(VkCommandBuffer cb, u32 first, u32 count, const VkRect2D* scissors)
	{
		if (deferred(cb)) return enc::set_scissor(first, count, scissors);
		(::vkCmdSetScissor)(cb, first, count, scissors);
	}

	inline void vkCmdSetLineWidth(VkCommandBuffer cb, float width)
	{
		if (deferred(cb)) return enc::set_line_width(width);
		(::vkCmdSetLineWidth)(cb, width);
	}

	inline void vkCmdSetDepthBias(VkCommandBuffer cb, float constant_factor, float clamp, float slope_factor)
	{
		if (deferred(cb)) return enc::set_depth_bias(constant_factor, clamp, slope_factor);
		(::vkCmdSetDepthBias)(cb, constant_factor, clamp, slope_factor);
	}

	inline void vkCmdSetBlendConstants(VkCommandBuffer cb, const float constants[4])
	{
		if (deferred(cb)) return enc::set_blend_constants(constants);
		(::vkCmdSetBlendConstants)(cb, constants);
	}

	inline void vkCmdSetDepthBounds(VkCommandBuffer cb, float min_bounds, float max_bounds)
	{
		if (deferred(cb)) return enc::set_depth_bounds(min_bounds, max_bounds);
		(::vkCmdSetDepthBounds)(cb, min_bounds, max_bounds);
	}

	inline void vkCmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags face, u32 mask)
	{
		if (deferred(cb)) return enc::set_stencil_compare_mask(face, mask);
		(::vkCmdSetStencilCompareMask)(cb, face, mask);
	}

	inline void vkCmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags face, u32 mask)
	{
		if (deferred(cb)) return enc::set_stencil_write_mask(face, mask);
		(::vkCmdSetStencilWriteMask)(cb, face, mask);
	}

	inline void vkCmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags face, u32 reference)
	{
		if (deferred(cb)) return enc::set_stencil_reference(face, reference);
		(::vkCmdSetStencilReference)(cb, face, reference);
	}

	inline void vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, u32 offset, u32 size, const void* values)
	{
		if (deferred(cb)) return enc::push_constants(layout, stages, offset, size, values);
		(::vkCmdPushConstants)(cb, layout, stages, offset, size, values);
	}

	inline void vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, const void* data)
	{
		if (deferred(cb)) return enc::update_buffer(dst, offset, size, data);
		(::vkCmdUpdateBuffer)(cb, dst, offset, size, data);
	}

	inline void vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, u32 data)
	{
		if (deferred(cb)) return enc::fill_buffer(dst, offset, size, data);
		(::vkCmdFillBuffer)(cb, dst, offset, size, data);
	}

	inline void vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, u32 count, const VkBufferCopy* regions)
	{
		if (deferred(cb)) return enc::copy_buffer(src, dst, count, regions);
		(::vkCmdCopyBuffer)(cb, src, dst, count, regions);
	}

	inline void vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageCopy* regions)
	{
		if (deferred(cb)) return enc::copy_image(src, src_layout, dst, dst_layout, count, regions);
		(::vkCmdCopyImage)(cb, src, src_layout, dst, dst_layout, count, regions);
	}

	inline void vkCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout, VkImage dst, VkImageLayout dst_layout, u32 count, const VkImageBlit* regions, VkFilter filter)
	{
		if (deferred(cb)) return enc::blit_image(src, src_layout, dst, dst_layout, count, regions, filter);
		(::vkCmdBlitImage)(cb, src, src_layout, dst, dst_layout, count, regions, filter);
	}

	inline void vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout dst_layout, u32 count, const VkBufferImageCopy* regions)
	{
		if (deferred(cb)) return enc::copy_buffer_to_image(src, dst, dst_layout, count, regions);
		(::vkCmdCopyBufferToImage)(cb, src, dst, dst_layout, count, regions);
	}

	inline void vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout, VkBuffer dst, u32 count, const VkBufferImageCopy* regions)
	{
		if (deferred(cb)) return enc::copy_image_to_buffer(src, src_layout, dst, count, regions);
		(::vkCmdCopyImageToBuffer)(cb, src, src_layout, dst, count, regions);
	}

	inline void vkCmdClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearColorValue* color, u32 count, const VkImageSubresourceRange* ranges)
	{
		if (deferred(cb)) return enc::clear_color_image(image, layout, color, count, ranges);
		(::vkCmdClearColorImage)(cb, image, layout, color, count, ranges);
	}

	inline void vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, u32 count, const VkImageSubresourceRange* ranges)
	{
		if (deferred(cb)) return enc::clear_depth_stencil_image(image, layout, value, count, ranges);
		(::vkCmdClearDepthStencilImage)(cb, image, layout, value, count, ranges);
	}

	inline void vkCmdClearAttachments(VkCommandBuffer cb, u32 attachment_count, const VkClearAttachment* attachments, u32 rect_count, const VkClearRect* rects)
	{
		if (deferred(cb)) return enc::clear_attachments(attachment_count, attachments, rect_count, rects);
		(::vkCmdClearAttachments)(cb, attachment_count, attachments, rect_count, rects);
	}

	inline void vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
		u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img)
	{
		if (deferred(cb)) return enc::pipeline_barrier(src, dst, flags, mem_count, mem, buf_count, buf, img_count, img);
		(::vkCmdPipelineBarrier)(cb, src, dst, flags, mem_count, mem, buf_count, buf, img_count, img);
	}

	inline void vkCmdSetEvent(VkCommandBuffer cb, VkEvent event, VkPipelineStageFlags stages)
	{
		if (deferred(cb)) return enc::set_event(event, stages);
		(::vkCmdSetEvent)(cb, event, stages);
	}

	inline void vkCmdWaitEvents(VkCommandBuffer cb, u32 event_count, const VkEvent* events, VkPipelineStageFlags src, VkPipelineStageFlags dst,
		u32 mem_count, const VkMemoryBarrier* mem, u32 buf_count, const VkBufferMemoryBarrier* buf, u32 img_count, const VkImageMemoryBarrier* img)
	{
		if (deferred(cb)) return enc::wait_events(event_count, events, src, dst, mem_count, mem, buf_count, buf, img_count, img);
		(::vkCmdWaitEvents)(cb, event_count, events, src, dst, mem_count, mem, buf_count, buf, img_count, img);
	}

	inline void vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, u32 query, VkQueryControlFlags flags)
	{
		if (deferred(cb)) return enc::begin_query(pool, query, flags);
		(::vkCmdBeginQuery)(cb, pool, query, flags);
	}

	inline void vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, u32 query)
	{
		if (deferred(cb)) return enc::end_query(pool, query);
		(::vkCmdEndQuery)(cb, pool, query);
	}

	inline void vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, u32 first, u32 count)
	{
		if (deferred(cb)) return enc::reset_query_pool(pool, first, count);
		(::vkCmdResetQueryPool)(cb, pool, first, count);
	}

	inline void vkCmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool, u32 first, u32 count, VkBuffer dst, VkDeviceSize dst_offset, VkDeviceSize stride, VkQueryResultFlags flags)
	{
		if (deferred(cb)) return enc::copy_query_pool_results(pool, first, count, dst, dst_offset, stride, flags);
		(::vkCmdCopyQueryPoolResults)(cb, pool, first, count, dst, dst_offset, stride, flags);
	}

	inline void vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info, VkSubpassContents contents)
	{
		if (deferred(cb)) return enc::begin_render_pass(info, contents);
		(::vkCmdBeginRenderPass)(cb, info, contents);
	}

	inline void vkCmdEndRenderPass(VkCommandBuffer cb)
	{
		if (deferred(cb)) return enc::end_render_pass();
		(::vkCmdEndRenderPass)(cb);
	}

	// ---- Extension functions (VKProcTable.h pointers, filled by vk::init) ----

	inline void pfn_vkCmdBeginConditionalRenderingEXT(VkCommandBuffer cb, const VkConditionalRenderingBeginInfoEXT* info)
	{
		if (deferred(cb)) return enc::begin_conditional_rendering(info);
		(::_vkCmdBeginConditionalRenderingEXT)(cb, info);
	}

	inline void pfn_vkCmdEndConditionalRenderingEXT(VkCommandBuffer cb)
	{
		if (deferred(cb)) return enc::end_conditional_rendering();
		(::_vkCmdEndConditionalRenderingEXT)(cb);
	}

	inline void pfn_vkCmdInsertDebugUtilsLabelEXT(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* label)
	{
		if (deferred(cb)) return enc::insert_debug_label(label);
		(::_vkCmdInsertDebugUtilsLabelEXT)(cb, label);
	}

	inline void pfn_vkCmdSetEvent2KHR(VkCommandBuffer cb, VkEvent event, const VkDependencyInfoKHR* dependency)
	{
		if (deferred(cb)) return enc::set_event2(event, dependency);
		(::_vkCmdSetEvent2KHR)(cb, event, dependency);
	}

	inline void pfn_vkCmdWaitEvents2KHR(VkCommandBuffer cb, u32 event_count, const VkEvent* events, const VkDependencyInfoKHR* dependencies)
	{
		if (deferred(cb)) return enc::wait_events2(event_count, events, dependencies);
		(::_vkCmdWaitEvents2KHR)(cb, event_count, events, dependencies);
	}

	inline void pfn_vkCmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfoKHR* dependency)
	{
		if (deferred(cb)) return enc::pipeline_barrier2(dependency);
		(::_vkCmdPipelineBarrier2KHR)(cb, dependency);
	}

	inline void pfn_vkCmdDrawMultiEXT(VkCommandBuffer cb, u32 draw_count, const VkMultiDrawInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride)
	{
		if (deferred(cb)) return enc::draw_multi(draw_count, info, instance_count, first_instance, stride);
		(::_vkCmdDrawMultiEXT)(cb, draw_count, info, instance_count, first_instance, stride);
	}

	inline void pfn_vkCmdDrawMultiIndexedEXT(VkCommandBuffer cb, u32 draw_count, const VkMultiDrawIndexedInfoEXT* info, u32 instance_count, u32 first_instance, u32 stride, const s32* vertex_offset)
	{
		if (deferred(cb)) return enc::draw_multi_indexed(draw_count, info, instance_count, first_instance, stride, vertex_offset);
		(::_vkCmdDrawMultiIndexedEXT)(cb, draw_count, info, instance_count, first_instance, stride, vertex_offset);
	}
}

#if !defined(VD1_CS_NO_SHIM)

// ---- The 36 core names RPCS3 records with ----
#define vkCmdBindPipeline(...)                  ::vd1_rec::vkCmdBindPipeline(__VA_ARGS__)
#define vkCmdBindDescriptorSets(...)            ::vd1_rec::vkCmdBindDescriptorSets(__VA_ARGS__)
#define vkCmdBindIndexBuffer(...)               ::vd1_rec::vkCmdBindIndexBuffer(__VA_ARGS__)
#define vkCmdBindVertexBuffers(...)             ::vd1_rec::vkCmdBindVertexBuffers(__VA_ARGS__)
#define vkCmdDraw(...)                          ::vd1_rec::vkCmdDraw(__VA_ARGS__)
#define vkCmdDrawIndexed(...)                   ::vd1_rec::vkCmdDrawIndexed(__VA_ARGS__)
#define vkCmdDispatch(...)                      ::vd1_rec::vkCmdDispatch(__VA_ARGS__)
#define vkCmdSetViewport(...)                   ::vd1_rec::vkCmdSetViewport(__VA_ARGS__)
#define vkCmdSetScissor(...)                    ::vd1_rec::vkCmdSetScissor(__VA_ARGS__)
#define vkCmdSetLineWidth(...)                  ::vd1_rec::vkCmdSetLineWidth(__VA_ARGS__)
#define vkCmdSetDepthBias(...)                  ::vd1_rec::vkCmdSetDepthBias(__VA_ARGS__)
#define vkCmdSetBlendConstants(...)             ::vd1_rec::vkCmdSetBlendConstants(__VA_ARGS__)
#define vkCmdSetDepthBounds(...)                ::vd1_rec::vkCmdSetDepthBounds(__VA_ARGS__)
#define vkCmdSetStencilCompareMask(...)         ::vd1_rec::vkCmdSetStencilCompareMask(__VA_ARGS__)
#define vkCmdSetStencilWriteMask(...)           ::vd1_rec::vkCmdSetStencilWriteMask(__VA_ARGS__)
#define vkCmdSetStencilReference(...)           ::vd1_rec::vkCmdSetStencilReference(__VA_ARGS__)
#define vkCmdPushConstants(...)                 ::vd1_rec::vkCmdPushConstants(__VA_ARGS__)
#define vkCmdUpdateBuffer(...)                  ::vd1_rec::vkCmdUpdateBuffer(__VA_ARGS__)
#define vkCmdFillBuffer(...)                    ::vd1_rec::vkCmdFillBuffer(__VA_ARGS__)
#define vkCmdCopyBuffer(...)                    ::vd1_rec::vkCmdCopyBuffer(__VA_ARGS__)
#define vkCmdCopyImage(...)                     ::vd1_rec::vkCmdCopyImage(__VA_ARGS__)
#define vkCmdBlitImage(...)                     ::vd1_rec::vkCmdBlitImage(__VA_ARGS__)
#define vkCmdCopyBufferToImage(...)             ::vd1_rec::vkCmdCopyBufferToImage(__VA_ARGS__)
#define vkCmdCopyImageToBuffer(...)             ::vd1_rec::vkCmdCopyImageToBuffer(__VA_ARGS__)
#define vkCmdClearColorImage(...)               ::vd1_rec::vkCmdClearColorImage(__VA_ARGS__)
#define vkCmdClearDepthStencilImage(...)        ::vd1_rec::vkCmdClearDepthStencilImage(__VA_ARGS__)
#define vkCmdClearAttachments(...)              ::vd1_rec::vkCmdClearAttachments(__VA_ARGS__)
#define vkCmdPipelineBarrier(...)               ::vd1_rec::vkCmdPipelineBarrier(__VA_ARGS__)
#define vkCmdSetEvent(...)                      ::vd1_rec::vkCmdSetEvent(__VA_ARGS__)
#define vkCmdWaitEvents(...)                    ::vd1_rec::vkCmdWaitEvents(__VA_ARGS__)
#define vkCmdBeginQuery(...)                    ::vd1_rec::vkCmdBeginQuery(__VA_ARGS__)
#define vkCmdEndQuery(...)                      ::vd1_rec::vkCmdEndQuery(__VA_ARGS__)
#define vkCmdResetQueryPool(...)                ::vd1_rec::vkCmdResetQueryPool(__VA_ARGS__)
#define vkCmdCopyQueryPoolResults(...)          ::vd1_rec::vkCmdCopyQueryPoolResults(__VA_ARGS__)
#define vkCmdBeginRenderPass(...)               ::vd1_rec::vkCmdBeginRenderPass(__VA_ARGS__)
#define vkCmdEndRenderPass(...)                 ::vd1_rec::vkCmdEndRenderPass(__VA_ARGS__)

// ---- The 8 extension pointers RPCS3 records with (VKProcTable.h) ----
#define _vkCmdBeginConditionalRenderingEXT(...) ::vd1_rec::pfn_vkCmdBeginConditionalRenderingEXT(__VA_ARGS__)
#define _vkCmdEndConditionalRenderingEXT(...)   ::vd1_rec::pfn_vkCmdEndConditionalRenderingEXT(__VA_ARGS__)
#define _vkCmdInsertDebugUtilsLabelEXT(...)     ::vd1_rec::pfn_vkCmdInsertDebugUtilsLabelEXT(__VA_ARGS__)
#define _vkCmdSetEvent2KHR(...)                 ::vd1_rec::pfn_vkCmdSetEvent2KHR(__VA_ARGS__)
#define _vkCmdWaitEvents2KHR(...)               ::vd1_rec::pfn_vkCmdWaitEvents2KHR(__VA_ARGS__)
#define _vkCmdPipelineBarrier2KHR(...)          ::vd1_rec::pfn_vkCmdPipelineBarrier2KHR(__VA_ARGS__)
#define _vkCmdDrawMultiEXT(...)                 ::vd1_rec::pfn_vkCmdDrawMultiEXT(__VA_ARGS__)
#define _vkCmdDrawMultiIndexedEXT(...)          ::vd1_rec::pfn_vkCmdDrawMultiIndexedEXT(__VA_ARGS__)

// ---- Poison: recording functions the stream does not wrap. A new call fails to compile ("identifier not found:
// VD1_CSREC_UNWRAPPED_..."): wrap it in vd1_cmdstream.h/.cpp first, or it would reach a deferred buffer out of order
// (C1 trap 7). Undeclared names in older SDK headers are harmless (a macro that is never invoked).
#define vkCmdDrawIndirect(...)                  VD1_CSREC_UNWRAPPED_vkCmdDrawIndirect(__VA_ARGS__)
#define vkCmdDrawIndexedIndirect(...)           VD1_CSREC_UNWRAPPED_vkCmdDrawIndexedIndirect(__VA_ARGS__)
#define vkCmdDrawIndirectCount(...)             VD1_CSREC_UNWRAPPED_vkCmdDrawIndirectCount(__VA_ARGS__)
#define vkCmdDrawIndexedIndirectCount(...)      VD1_CSREC_UNWRAPPED_vkCmdDrawIndexedIndirectCount(__VA_ARGS__)
#define vkCmdDispatchIndirect(...)              VD1_CSREC_UNWRAPPED_vkCmdDispatchIndirect(__VA_ARGS__)
#define vkCmdDispatchBase(...)                  VD1_CSREC_UNWRAPPED_vkCmdDispatchBase(__VA_ARGS__)
#define vkCmdResolveImage(...)                  VD1_CSREC_UNWRAPPED_vkCmdResolveImage(__VA_ARGS__)
#define vkCmdResetEvent(...)                    VD1_CSREC_UNWRAPPED_vkCmdResetEvent(__VA_ARGS__)
#define vkCmdWriteTimestamp(...)                VD1_CSREC_UNWRAPPED_vkCmdWriteTimestamp(__VA_ARGS__)
#define vkCmdNextSubpass(...)                   VD1_CSREC_UNWRAPPED_vkCmdNextSubpass(__VA_ARGS__)
#define vkCmdExecuteCommands(...)               VD1_CSREC_UNWRAPPED_vkCmdExecuteCommands(__VA_ARGS__)
#define vkCmdSetDeviceMask(...)                 VD1_CSREC_UNWRAPPED_vkCmdSetDeviceMask(__VA_ARGS__)
#define vkCmdBeginRenderPass2(...)              VD1_CSREC_UNWRAPPED_vkCmdBeginRenderPass2(__VA_ARGS__)
#define vkCmdNextSubpass2(...)                  VD1_CSREC_UNWRAPPED_vkCmdNextSubpass2(__VA_ARGS__)
#define vkCmdEndRenderPass2(...)                VD1_CSREC_UNWRAPPED_vkCmdEndRenderPass2(__VA_ARGS__)
#define vkCmdBeginRendering(...)                VD1_CSREC_UNWRAPPED_vkCmdBeginRendering(__VA_ARGS__)
#define vkCmdEndRendering(...)                  VD1_CSREC_UNWRAPPED_vkCmdEndRendering(__VA_ARGS__)
#define vkCmdSetEvent2(...)                     VD1_CSREC_UNWRAPPED_vkCmdSetEvent2(__VA_ARGS__)
#define vkCmdResetEvent2(...)                   VD1_CSREC_UNWRAPPED_vkCmdResetEvent2(__VA_ARGS__)
#define vkCmdWaitEvents2(...)                   VD1_CSREC_UNWRAPPED_vkCmdWaitEvents2(__VA_ARGS__)
#define vkCmdPipelineBarrier2(...)              VD1_CSREC_UNWRAPPED_vkCmdPipelineBarrier2(__VA_ARGS__)
#define vkCmdWriteTimestamp2(...)               VD1_CSREC_UNWRAPPED_vkCmdWriteTimestamp2(__VA_ARGS__)
#define vkCmdCopyBuffer2(...)                   VD1_CSREC_UNWRAPPED_vkCmdCopyBuffer2(__VA_ARGS__)
#define vkCmdCopyImage2(...)                    VD1_CSREC_UNWRAPPED_vkCmdCopyImage2(__VA_ARGS__)
#define vkCmdCopyBufferToImage2(...)            VD1_CSREC_UNWRAPPED_vkCmdCopyBufferToImage2(__VA_ARGS__)
#define vkCmdCopyImageToBuffer2(...)            VD1_CSREC_UNWRAPPED_vkCmdCopyImageToBuffer2(__VA_ARGS__)
#define vkCmdBlitImage2(...)                    VD1_CSREC_UNWRAPPED_vkCmdBlitImage2(__VA_ARGS__)
#define vkCmdResolveImage2(...)                 VD1_CSREC_UNWRAPPED_vkCmdResolveImage2(__VA_ARGS__)
#define vkCmdSetCullMode(...)                   VD1_CSREC_UNWRAPPED_vkCmdSetCullMode(__VA_ARGS__)
#define vkCmdSetFrontFace(...)                  VD1_CSREC_UNWRAPPED_vkCmdSetFrontFace(__VA_ARGS__)
#define vkCmdSetPrimitiveTopology(...)          VD1_CSREC_UNWRAPPED_vkCmdSetPrimitiveTopology(__VA_ARGS__)
#define vkCmdSetDepthTestEnable(...)            VD1_CSREC_UNWRAPPED_vkCmdSetDepthTestEnable(__VA_ARGS__)
#define vkCmdSetDepthWriteEnable(...)           VD1_CSREC_UNWRAPPED_vkCmdSetDepthWriteEnable(__VA_ARGS__)
#define vkCmdSetDepthCompareOp(...)             VD1_CSREC_UNWRAPPED_vkCmdSetDepthCompareOp(__VA_ARGS__)
#define vkCmdSetStencilTestEnable(...)          VD1_CSREC_UNWRAPPED_vkCmdSetStencilTestEnable(__VA_ARGS__)
#define vkCmdSetStencilOp(...)                  VD1_CSREC_UNWRAPPED_vkCmdSetStencilOp(__VA_ARGS__)
#define vkCmdBindVertexBuffers2(...)            VD1_CSREC_UNWRAPPED_vkCmdBindVertexBuffers2(__VA_ARGS__)
#define vkCmdPushDescriptorSetKHR(...)          VD1_CSREC_UNWRAPPED_vkCmdPushDescriptorSetKHR(__VA_ARGS__)
#define vkCmdBeginDebugUtilsLabelEXT(...)       VD1_CSREC_UNWRAPPED_vkCmdBeginDebugUtilsLabelEXT(__VA_ARGS__)
#define vkCmdEndDebugUtilsLabelEXT(...)         VD1_CSREC_UNWRAPPED_vkCmdEndDebugUtilsLabelEXT(__VA_ARGS__)
#define vkCmdInsertDebugUtilsLabelEXT(...)      VD1_CSREC_UNWRAPPED_vkCmdInsertDebugUtilsLabelEXT(__VA_ARGS__)
#define vkCmdBeginConditionalRenderingEXT(...)  VD1_CSREC_UNWRAPPED_vkCmdBeginConditionalRenderingEXT(__VA_ARGS__)
#define vkCmdEndConditionalRenderingEXT(...)    VD1_CSREC_UNWRAPPED_vkCmdEndConditionalRenderingEXT(__VA_ARGS__)
#define vkCmdSetEvent2KHR(...)                  VD1_CSREC_UNWRAPPED_vkCmdSetEvent2KHR(__VA_ARGS__)
#define vkCmdWaitEvents2KHR(...)                VD1_CSREC_UNWRAPPED_vkCmdWaitEvents2KHR(__VA_ARGS__)
#define vkCmdPipelineBarrier2KHR(...)           VD1_CSREC_UNWRAPPED_vkCmdPipelineBarrier2KHR(__VA_ARGS__)
#define vkCmdDrawMultiEXT(...)                  VD1_CSREC_UNWRAPPED_vkCmdDrawMultiEXT(__VA_ARGS__)
#define vkCmdDrawMultiIndexedEXT(...)           VD1_CSREC_UNWRAPPED_vkCmdDrawMultiIndexedEXT(__VA_ARGS__)

#endif // !VD1_CS_NO_SHIM
