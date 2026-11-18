// vanillad1: X467.4 p28: SMAA 1x on the output frame, before the output scaling (VANILLAD1_SMAA).
// The shader is SMAA.fxh (SMAA 2.8, Copyright (C) 2013 Jorge Jimenez, Jose I. Echevarria, Belen Masia, Fernando Navarro,
// Diego Gutierrez; MIT licence, its notice in smaa_core.inc and smaa_lut.h) through its SMAA_GLSL_4 port, as three
// compute passes (smaa_cs.glsl). Check offline: re/notes/gfx_quality/smaa_bench.py (SPIR-V and real frames).

#include "../../vkutils/barriers.h"
#include "../../VKHelpers.h"
#include "../../VKResourceManager.h"

#include "../smaa_pass.h"
#include "smaa_lut.h"

#include "Emu/system_config.h"
#include "util/asm.hpp"

#include <cstdlib>

namespace vk
{
	u32 vd1_smaa_level()
	{
		static const u32 s_level = []()
		{
			const char* env = std::getenv("VANILLAD1_SMAA");
			u32 level = env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;

			if (level > 2)
			{
				rsx_log.warning("vanillad1: VANILLAD1_SMAA=%s is not a preset (1 = high, 2 = ultra): using 2", env);
				level = 2;
			}

			if (level)
			{
				rsx_log.notice("vanillad1: SMAA 1x on the output frame before the output scaling (VANILLAD1_SMAA=%u: %s preset, colour edges)",
					level, level == 2 ? "ultra" : "high");
			}

			return level;
		}();

		return s_level;
	}

	namespace SMAA
	{
		smaa_pass::smaa_pass(u32 pass_id, u32 input_count, u32 point_mask)
			: m_input_count(input_count)
			, m_point_mask(point_mask)
		{
			ensure(input_count <= m_inputs.size());

			const char* shader_core =
				#include "Emu/RSX/VK/upscalers/smaa/smaa_cs.glsl"
			;

			const char* smaa_core =
				#include "Emu/RSX/VK/upscalers/smaa/smaa_core.inc"
			;

			// SMAA's edge pixel shaders discard where there is no edge (into a cleared target, with the stencil marking the
			// rest); a compute pass has no discard, so it writes "no edge" there: the same edges texture, every pixel written.
			// Every "discard;" in SMAA.fxh is in one of its edge detection functions, which return float2.
			const std::string core = fmt::replace_all(smaa_core, "discard;", "return float2(0.0, 0.0);");

			const std::string definitions = fmt::format(
				"#define %s 1\n"
				"#define SMAA_VD1_PASS %u",
				vd1_smaa_level() >= 2 ? "SMAA_PRESET_ULTRA" : "SMAA_PRESET_HIGH",
				pass_id);

			const std::pair<std::string_view, std::string> replacement_table[] =
			{
				{ "%SMAA_DEFINITIONS%", definitions },
				{ "%SMAA_CORE%", core },
				{ "%push_block%", "push_constant" },
				{ "%set_binding%", "set=0, binding=" }
			};

			m_src = fmt::replace_all(shader_core, replacement_table);

			// No ssbo usage
			ssbo_count = 0;

			// SMAA_RT_METRICS as push constants
			use_push_constants = true;
			push_constants_size = 16;

			create();
		}

		std::vector<glsl::program_input> smaa_pass::get_inputs()
		{
			auto result = compute_task::get_inputs();

			for (u32 i = 0; i < m_input_count; ++i)
			{
				result.push_back(glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"InputTexture" + std::to_string(i),
					vk::glsl::input_type_texture,
					0,
					i));
			}

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"OutputTexture",
				vk::glsl::input_type_storage_texture,
				0,
				m_input_count));

			return result;
		}

		void smaa_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			if (!m_linear_sampler)
			{
				const auto pdev = vk::get_current_renderer();
				m_linear_sampler = std::make_unique<vk::sampler>(*pdev,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
				m_point_sampler = std::make_unique<vk::sampler>(*pdev,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE, 0.f, 1.f, 0.f, 0.f, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
			}

			for (u32 i = 0; i < m_input_count; ++i)
			{
				const auto& sampler = (m_point_mask & (1u << i)) ? *m_point_sampler : *m_linear_sampler;
				m_program->bind_uniform({ *m_inputs[i], sampler }, 0, i);
			}

			m_program->bind_uniform({ *m_output }, 0, m_input_count);
		}

		void smaa_pass::run(const vk::command_buffer& cmd, std::initializer_list<vk::viewable_image*> inputs, vk::viewable_image* dst)
		{
			ensure(inputs.size() == m_input_count);

			u32 i = 0;
			for (auto input : inputs)
			{
				m_inputs[i++] = input->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			}

			m_output = dst->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));

			// All three targets are the frame's size: SMAA_RT_METRICS
			const f32 width = static_cast<f32>(dst->width());
			const f32 height = static_cast<f32>(dst->height());
			m_rt_metrics = { 1.f / width, 1.f / height, width, height };

			if (!m_program)
			{
				load_program(cmd);
			}

			vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, m_rt_metrics.data());

			// 8x8 invocations a group, one a pixel (smaa_cs.glsl)
			compute_task::run(cmd, utils::aligned_div(dst->width(), 8u), utils::aligned_div(dst->height(), 8u), 1);
		}

		edge_detection_pass::edge_detection_pass()
			: smaa_pass(0, 1, 0) // colorTex
		{}

		blending_weight_pass::blending_weight_pass()
			: smaa_pass(1, 3, 1u << 2) // edgesTex, areaTex, searchTex (point)
		{}

		neighborhood_blending_pass::neighborhood_blending_pass()
			: smaa_pass(2, 2, 0) // colorTex, blendTex
		{}
	}

	// Through the resource manager, as fsr_upscale_pass: a frame in flight may still read the image
	static void smaa_dispose(std::unique_ptr<vk::viewable_image>& data)
	{
		if (data && data->value)
		{
			vk::get_resource_manager()->dispose(data);
		}
		else if (data)
		{
			data.reset();
		}
	}

	smaa_upscale_pass::smaa_upscale_pass(std::unique_ptr<upscaler> inner)
		: m_inner(std::move(inner))
	{
		ensure(m_inner);
	}

	smaa_upscale_pass::~smaa_upscale_pass()
	{
		dispose_images();
		smaa_dispose(m_area_tex);
		smaa_dispose(m_search_tex);
	}

	void smaa_upscale_pass::dispose_images()
	{
		smaa_dispose(m_edges);
		smaa_dispose(m_weights);
		smaa_dispose(m_output);
	}

	bool smaa_upscale_pass::initialize_images(u32 width, u32 height, VkFormat source_format)
	{
		dispose_images();
		m_source_format = source_format;

		const auto pdev = vk::get_current_renderer();
		auto initialize_image_impl = [pdev, width, height](VkImageUsageFlags usage, VkFormat format)
		{
			return std::make_unique<vk::viewable_image>(
				*pdev,                                               // Owner
				pdev->get_memory_mapping().device_local,             // Must be in device optimal memory
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D,
				format,
				width, height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,       // Dimensions (w, h, d, mips, layers, samples)
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_TILING_OPTIMAL,
				usage,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3,                    // Allow creation to fail if there is no memory
				VMM_ALLOCATION_POOL_SWAPCHAIN,
				RSX_FORMAT_CLASS_COLOR);
		};

		// The output stands in for the frame: the screenshot copy and the upscalers read it (transfer or sampled).
		// Its format is the frame's when the device can store to it, else the other 8-bit RGBA order (as FSR's fallback).
		const VkFlags output_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
		const VkFlags work_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		const auto supports = [pdev](VkFormat format, VkFlags features)
		{
			return (pdev->get_format_properties(format).optimalTilingFeatures & features) == features;
		};

		VkFormat output_format = VK_FORMAT_UNDEFINED;
		const VkFormat other_format = (source_format == VK_FORMAT_B8G8R8A8_UNORM) ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
		for (const VkFormat format : { source_format, other_format })
		{
			if (supports(format, output_features | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
			{
				output_format = format;
				break;
			}
		}

		// Edges and weights: RGBA8 (a storage format every device has; edges use RG of it)
		if (output_format == VK_FORMAT_UNDEFINED || !supports(VK_FORMAT_R8G8B8A8_UNORM, work_features))
		{
			rsx_log.error("vanillad1: SMAA is not supported by this driver and hardware combination (no 8-bit RGBA storage image): off");
			m_failed = true;
			return false;
		}

		const VkFlags usage_output = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		const VkFlags usage_work = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		m_edges = initialize_image_impl(usage_work, VK_FORMAT_R8G8B8A8_UNORM);
		m_weights = initialize_image_impl(usage_work, VK_FORMAT_R8G8B8A8_UNORM);
		m_output = initialize_image_impl(usage_output, output_format);

		if (!m_edges->value || !m_weights->value || !m_output->value)
		{
			dispose_images();
			rsx_log.warning("vanillad1: SMAA targets %ux%u: out of memory, SMAA off until the output stage is reset", width, height);
			m_failed = true;
			return false;
		}

		rsx_log.notice("vanillad1: SMAA targets %ux%u (output format %s)", width, height,
			output_format == VK_FORMAT_B8G8R8A8_UNORM ? "B8G8R8A8" : "R8G8B8A8");
		return true;
	}

	bool smaa_upscale_pass::upload_lookup_textures(const vk::command_buffer& cmd)
	{
		const auto pdev = vk::get_current_renderer();
		auto initialize_image_impl = [pdev](VkFormat format, u32 width, u32 height)
		{
			return std::make_unique<vk::viewable_image>(
				*pdev,
				pdev->get_memory_mapping().device_local,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D,
				format,
				width, height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3,
				VMM_ALLOCATION_POOL_SWAPCHAIN,
				RSX_FORMAT_CLASS_COLOR);
		};

		m_area_tex = initialize_image_impl(VK_FORMAT_R8G8_UNORM, smaa_lut::area_width, smaa_lut::area_height);
		m_search_tex = initialize_image_impl(VK_FORMAT_R8_UNORM, smaa_lut::search_width, smaa_lut::search_height);

		if (!m_area_tex->value || !m_search_tex->value)
		{
			m_area_tex.reset();
			m_search_tex.reset();
			rsx_log.warning("vanillad1: SMAA lookup textures: out of memory, SMAA off until the output stage is reset");
			m_failed = true;
			return false;
		}

		constexpr u32 area_size = sizeof(smaa_lut::area_tex);
		constexpr u32 search_size = sizeof(smaa_lut::search_tex);
		static_assert(area_size == smaa_lut::area_width * smaa_lut::area_height * 2);
		static_assert(search_size == smaa_lut::search_width * smaa_lut::search_height);

		// Once: a staging buffer, freed through the resource manager when this frame is done
		auto staging = std::make_unique<vk::buffer>(*pdev, utils::align(area_size + search_size, 0x10000), pdev->get_memory_mapping().host_visible_coherent,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0, VMM_ALLOCATION_POOL_UNDEFINED);

		auto dst = static_cast<u8*>(staging->map(0, area_size + search_size));
		std::memcpy(dst, smaa_lut::area_tex, area_size);
		std::memcpy(dst + area_size, smaa_lut::search_tex, search_size);
		staging->unmap();

		m_area_tex->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		m_search_tex->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkBufferImageCopy region
		{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
			.imageOffset = {},
			.imageExtent = { smaa_lut::area_width, smaa_lut::area_height, 1u }
		};
		vkCmdCopyBufferToImage(cmd, staging->value, m_area_tex->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		region.bufferOffset = area_size;
		region.imageExtent = { smaa_lut::search_width, smaa_lut::search_height, 1u };
		vkCmdCopyBufferToImage(cmd, staging->value, m_search_tex->value, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// The compute reads wait on these through antialias()'s barrier ahead of the passes
		m_area_tex->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_search_tex->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vk::get_resource_manager()->dispose(staging);
		return true;
	}

	vk::viewable_image* smaa_upscale_pass::antialias(const vk::command_buffer& cmd, vk::viewable_image* src)
	{
		if (m_failed || !src)
		{
			return src;
		}

		const VkFormat source_format = src->format();
		if ((source_format != VK_FORMAT_B8G8R8A8_UNORM && source_format != VK_FORMAT_R8G8B8A8_UNORM) || src->samples() != 1)
		{
			if (m_skipped_format != source_format)
			{
				m_skipped_format = source_format;
				rsx_log.warning("vanillad1: SMAA skips a %ux%u frame of VkFormat %d, %u samples (8-bit RGBA, single sample only)",
					src->width(), src->height(), static_cast<int>(source_format), src->samples());
			}

			return src;
		}

		if (!m_output || m_output->width() != src->width() || m_output->height() != src->height() || m_source_format != source_format)
		{
			if (!initialize_images(src->width(), src->height(), source_format))
			{
				return src;
			}
		}

		if (!m_area_tex && !upload_lookup_textures(cmd))
		{
			return src;
		}

		const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		// The frame: whatever wrote it (draws, a blit, an upload) and the lookup tables' upload, before the compute reads.
		// (change_image_layout's SHADER_READ_ONLY transition waits for fragment / vertex stages, not compute.)
		src->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		vk::insert_global_memory_barrier(cmd,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

		// A target in GENERAL: last frame's readers (these passes, the screenshot copy, the upscaler) are done before the write
		const auto prepare_write = [&](vk::viewable_image* image)
		{
			image->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			vk::insert_image_memory_barrier(cmd, image->value,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
				range);
		};

		// The write is visible to the next pass and to the readers after antialias() (transfer, fragment or compute)
		const auto finish_write = [&](vk::viewable_image* image)
		{
			vk::insert_image_memory_barrier(cmd, image->value,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
				range);
		};

		auto edge_task = vk::get_compute_task<vk::SMAA::edge_detection_pass>();
		auto weight_task = vk::get_compute_task<vk::SMAA::blending_weight_pass>();
		auto blend_task = vk::get_compute_task<vk::SMAA::neighborhood_blending_pass>();

		prepare_write(m_edges.get());
		edge_task->run(cmd, { src }, m_edges.get());
		finish_write(m_edges.get());

		prepare_write(m_weights.get());
		weight_task->run(cmd, { m_edges.get(), m_area_tex.get(), m_search_tex.get() }, m_weights.get());
		finish_write(m_weights.get());

		prepare_write(m_output.get());
		blend_task->run(cmd, { src, m_weights.get() }, m_output.get());
		finish_write(m_output.get());

		// The reads end before the frame goes back to its layout (that transition waits for fragment / vertex stages only)
		vk::insert_global_memory_barrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			0, 0);
		src->pop_layout(cmd);

		return m_output.get();
	}

	vk::viewable_image* smaa_upscale_pass::scale_output(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		VkImage present_surface,
		VkImageLayout present_surface_layout,
		const VkImageBlit& request,
		rsx::flags32_t mode)
	{
		// SMAA ran in antialias(); the scaling is the chosen upscaler's
		return m_inner->scale_output(cmd, src, present_surface, present_surface_layout, request, mode);
	}
}
