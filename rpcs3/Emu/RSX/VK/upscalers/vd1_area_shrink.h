#pragma once

// vanillad1: X515.3: a true shrink at present: VANILLAD1_AREA_SHRINK=1 (read once, logged; unset / 0 = as before).
// When the frame is bigger than the output area (a resolution scale above the window's size), the output stage would
// let vkCmdBlitImage (or the calibration pass's sampler) blend two pixels a side, which keeps most of the crawl a
// supersampled frame should lose. Here the frame is first shrunk to exactly the output area by an area average: each
// output pixel is the coverage-weighted mean of every frame pixel under it, in linear light. Everything after it (the
// F12 / recording copy, the calibration pass, the chosen upscaler) takes the shrunk frame and copies it 1:1.
// Header only, included by VKPresent.cpp alone. Not for 3D (two eyes); a frame that is not 8-bit RGBA goes out as before.

#include "../vkutils/barriers.h"
#include "../vkutils/sampler.h"
#include "../VKCompute.h"
#include "../VKHelpers.h"
#include "../VKResourceManager.h"

#include "upscaling.h"

#include <array>
#include <cstdlib>
#include <cstring>

namespace vk
{
	inline bool vd1_area_shrink_on()
	{
		static const bool s_on = []()
		{
			const char* env = std::getenv("VANILLAD1_AREA_SHRINK");
			const bool on = env && *env && std::strcmp(env, "0") != 0;
			if (on) rsx_log.notice("vanillad1: a frame bigger than the window is shrunk by an area average (VANILLAD1_AREA_SHRINK=%s)", env);
			return on;
		}();
		return s_on;
	}

	namespace vd1_area
	{
		// One invocation an output pixel: the frame's pixels under its footprint, weighted by the area they cover
		struct shrink_task : public compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_input = nullptr;
			const vk::image_view* m_output = nullptr;
			std::array<f32, 4> m_sizes {}; // source w, h (the frame's region), output w, h

			shrink_task()
			{
				m_src = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform area_params
{
	vec4 sizes; // source w, h; output w, h
};

layout(set=0, binding=0) uniform sampler2D src_tex;
layout(set=0, binding=1, rgba8) uniform writeonly image2D dst_img;

vec3 to_linear(vec3 c)
{
	return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

vec3 to_srgb(vec3 c)
{
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}

void main()
{
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (pos.x >= int(sizes.z) || pos.y >= int(sizes.w))
	{
		return;
	}

	vec2 ratio = sizes.xy / sizes.zw;
	vec2 lo = vec2(pos) * ratio;
	vec2 hi = lo + ratio;
	ivec2 first = ivec2(floor(lo));
	ivec2 last = min(ivec2(ceil(hi)) - 1, ivec2(sizes.xy) - 1);

	vec4 sum = vec4(0.0);
	float total = 0.0;
	for (int y = first.y; y <= min(last.y, first.y + 15); ++y)
	{
		float wy = max(min(hi.y, float(y + 1)) - max(lo.y, float(y)), 0.0);
		for (int x = first.x; x <= min(last.x, first.x + 15); ++x)
		{
			float wx = max(min(hi.x, float(x + 1)) - max(lo.x, float(x)), 0.0);
			vec4 c = texelFetch(src_tex, ivec2(x, y), 0);
			sum += vec4(to_linear(clamp(c.rgb, 0.0, 1.0)), c.a) * (wx * wy);
			total += wx * wy;
		}
	}

	vec4 avg = sum / max(total, 1e-6);
	imageStore(dst_img, pos, vec4(to_srgb(clamp(avg.rgb, 0.0, 1.0)), avg.a));
}
)";

				ssbo_count = 0;
				use_push_constants = true;
				push_constants_size = 16;
			}

			std::vector<glsl::program_input> get_inputs() override
			{
				auto result = compute_task::get_inputs();

				result.push_back(glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"InputTexture",
					vk::glsl::input_type_texture,
					0,
					0));

				result.push_back(glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"OutputTexture",
					vk::glsl::input_type_storage_texture,
					0,
					1));

				return result;
			}

			void bind_resources(const vk::command_buffer& /*cmd*/) override
			{
				if (!m_sampler)
				{
					const auto pdev = vk::get_current_renderer();
					m_sampler = std::make_unique<vk::sampler>(*pdev,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
				}

				const vk::sampler& sampler = *m_sampler;
				m_program->bind_uniform({ *m_input, sampler }, 0, 0);
				m_program->bind_uniform({ *m_output }, 0, 1);
			}

			void run(const vk::command_buffer& cmd, vk::viewable_image* src, u32 src_w, u32 src_h, vk::viewable_image* dst)
			{
				m_input = src->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
				m_output = dst->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
				m_sizes = { static_cast<f32>(src_w), static_cast<f32>(src_h), static_cast<f32>(dst->width()), static_cast<f32>(dst->height()) };

				if (!m_program)
				{
					load_program(cmd);
				}

				vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, m_sizes.data());
				compute_task::run(cmd, utils::aligned_div(dst->width(), 8u), utils::aligned_div(dst->height(), 8u), 1);
			}
		};
	}

	// Owns the shrunk frame next to the chosen upscaler (so it is freed with it, as SMAA's targets are). shrink() makes the
	// output-sized copy; scale_output() is the chosen upscaler's.
	class vd1_area_shrink_pass : public upscaler
	{
		std::unique_ptr<upscaler> m_inner;
		std::unique_ptr<vk::viewable_image> m_output;
		VkFormat m_source_format = VK_FORMAT_UNDEFINED;
		VkFormat m_skipped_format = VK_FORMAT_UNDEFINED;
		bool m_failed = false;

		void dispose()
		{
			// Through the resource manager: a frame in flight may still read the image
			if (m_output && m_output->value)
			{
				vk::get_resource_manager()->dispose(m_output);
			}
			m_output.reset();
		}

		bool initialize(u32 width, u32 height, VkFormat source_format)
		{
			dispose();
			m_source_format = source_format;

			const auto pdev = vk::get_current_renderer();
			const VkFlags features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
				VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

			VkFormat format = VK_FORMAT_UNDEFINED;
			const VkFormat other_format = (source_format == VK_FORMAT_B8G8R8A8_UNORM) ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
			for (const VkFormat f : { source_format, other_format })
			{
				if ((pdev->get_format_properties(f).optimalTilingFeatures & features) == features)
				{
					format = f;
					break;
				}
			}

			if (format == VK_FORMAT_UNDEFINED)
			{
				rsx_log.error("vanillad1: area shrink: no 8-bit RGBA storage image on this device: off");
				m_failed = true;
				return false;
			}

			m_output = std::make_unique<vk::viewable_image>(
				*pdev,
				pdev->get_memory_mapping().device_local,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D,
				format,
				width, height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3,
				VMM_ALLOCATION_POOL_SWAPCHAIN,
				RSX_FORMAT_CLASS_COLOR);

			if (!m_output->value)
			{
				m_output.reset();
				rsx_log.warning("vanillad1: area shrink target %ux%u: out of memory, off until the output stage is reset", width, height);
				m_failed = true;
				return false;
			}

			rsx_log.notice("vanillad1: area shrink target %ux%u", width, height);
			return true;
		}

	public:
		explicit vd1_area_shrink_pass(std::unique_ptr<upscaler> inner)
			: m_inner(std::move(inner))
		{
			ensure(m_inner);
		}

		~vd1_area_shrink_pass() override
		{
			dispose();
		}

		// The frame's region (0, 0, src_w, src_h) shrunk to out_w x out_h, or src itself when it can't run
		vk::viewable_image* shrink(const vk::command_buffer& cmd, vk::viewable_image* src, u32 src_w, u32 src_h, u32 out_w, u32 out_h)
		{
			if (m_failed || !src || !out_w || !out_h || src_w > src->width() || src_h > src->height())
			{
				return src;
			}

			const VkFormat source_format = src->format();
			if ((source_format != VK_FORMAT_B8G8R8A8_UNORM && source_format != VK_FORMAT_R8G8B8A8_UNORM) || src->samples() != 1)
			{
				if (m_skipped_format != source_format)
				{
					m_skipped_format = source_format;
					rsx_log.warning("vanillad1: area shrink skips a frame of VkFormat %d, %u samples (8-bit RGBA, single sample only)",
						static_cast<int>(source_format), src->samples());
				}
				return src;
			}

			if (!m_output || m_output->width() != out_w || m_output->height() != out_h || m_source_format != source_format)
			{
				if (!initialize(out_w, out_h, source_format))
				{
					return src;
				}
			}

			const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			// The frame's writers (draws, SMAA, a blit) before the compute read
			src->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			vk::insert_global_memory_barrier(cmd,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

			// Last frame's readers of the target (the screenshot copy, the output pass) before the write
			m_output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			vk::insert_image_memory_barrier(cmd, m_output->value,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
				range);

			vk::get_compute_task<vd1_area::shrink_task>()->run(cmd, src, src_w, src_h, m_output.get());

			// The write is visible to the readers after shrink() (transfer, fragment or compute)
			vk::insert_image_memory_barrier(cmd, m_output->value,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
				range);

			// The read ends before the frame goes back to its layout (that transition waits for fragment / vertex stages only)
			vk::insert_global_memory_barrier(cmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
				0, 0);
			src->pop_layout(cmd);

			return m_output.get();
		}

		vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,          // CB
			vk::viewable_image* src,                // Source input
			VkImage present_surface,                // Present target. May be VK_NULL_HANDLE for some passes
			VkImageLayout present_surface_layout,   // Present surface layout, or VK_IMAGE_LAYOUT_UNDEFINED if no present target is provided
			const VkImageBlit& request,             // Scaling request information
			rsx::flags32_t mode                     // Mode
		) override
		{
			return m_inner->scale_output(cmd, src, present_surface, present_surface_layout, request, mode);
		}
	};
}
