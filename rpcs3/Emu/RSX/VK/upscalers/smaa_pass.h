#pragma once

// vanillad1: X467.4 p28: SMAA 1x on the output frame, before the output scaling (VANILLAD1_SMAA).
// SMAA by Jorge Jimenez et al. (MIT, see smaa/smaa_core.inc).

#include "../vkutils/sampler.h"
#include "../VKCompute.h"

#include "upscaling.h"

namespace vk
{
	// VANILLAD1_SMAA: unset / 0 = off (as before), 1 = SMAA 1x high preset, 2 = ultra preset. Read once, logged once.
	u32 vd1_smaa_level();

	namespace SMAA
	{
		// One of SMAA's three passes as a compute shader: n sampled inputs at bindings 0..n-1, the target at binding n
		class smaa_pass : public compute_task
		{
		protected:
			std::unique_ptr<vk::sampler> m_linear_sampler;
			std::unique_ptr<vk::sampler> m_point_sampler;
			std::array<const vk::image_view*, 3> m_inputs {};
			const vk::image_view* m_output = nullptr;
			u32 m_input_count = 0;
			u32 m_point_mask = 0; // Bit n: input n is sampled with nearest filtering
			std::array<f32, 4> m_rt_metrics {};

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;

		public:
			smaa_pass(u32 pass_id, u32 input_count, u32 point_mask);
			void run(const vk::command_buffer& cmd, std::initializer_list<vk::viewable_image*> inputs, vk::viewable_image* dst);
		};

		// The frame -> edges
		struct edge_detection_pass : public smaa_pass
		{
			edge_detection_pass();
		};

		// Edges, AreaTex, SearchTex -> blending weights
		struct blending_weight_pass : public smaa_pass
		{
			blending_weight_pass();
		};

		// The frame, blending weights -> the anti-aliased frame
		struct neighborhood_blending_pass : public smaa_pass
		{
			neighborhood_blending_pass();
		};
	}

	// Owns SMAA's targets and lookup tables next to the chosen upscaler, so they are freed with it.
	// antialias() runs SMAA on the frame (its whole image, at its size); scale_output() is the chosen upscaler's.
	class smaa_upscale_pass : public upscaler
	{
		std::unique_ptr<upscaler> m_inner;

		std::unique_ptr<vk::viewable_image> m_edges;
		std::unique_ptr<vk::viewable_image> m_weights;
		std::unique_ptr<vk::viewable_image> m_output;
		std::unique_ptr<vk::viewable_image> m_area_tex;
		std::unique_ptr<vk::viewable_image> m_search_tex;

		VkFormat m_source_format = VK_FORMAT_UNDEFINED;
		VkFormat m_skipped_format = VK_FORMAT_UNDEFINED;
		bool m_failed = false;

		void dispose_images();
		bool initialize_images(u32 width, u32 height, VkFormat source_format);
		bool upload_lookup_textures(const vk::command_buffer& cmd);

	public:
		explicit smaa_upscale_pass(std::unique_ptr<upscaler> inner);
		~smaa_upscale_pass() override;

		// The anti-aliased copy of src (same size; B8G8R8A8 / R8G8B8A8 sources only), or src itself when SMAA can't run
		vk::viewable_image* antialias(const vk::command_buffer& cmd, vk::viewable_image* src);

		vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,          // CB
			vk::viewable_image* src,                // Source input
			VkImage present_surface,                // Present target. May be VK_NULL_HANDLE for some passes
			VkImageLayout present_surface_layout,   // Present surface layout, or VK_IMAGE_LAYOUT_UNDEFINED if no present target is provided
			const VkImageBlit& request,             // Scaling request information
			rsx::flags32_t mode                     // Mode
		) override;
	};
}
