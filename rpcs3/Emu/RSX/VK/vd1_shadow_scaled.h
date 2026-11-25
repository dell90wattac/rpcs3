#pragma once

// vanillad1: X515.2 shadows at a scale: VANILLAD1_SHADOW_SCALED=1 (read once, logged; unset / 0 = as before).
// Destiny's sun-shadow resolve (the disc's ps3_pcf) samples the cascade depth with unnormalised coordinates and a 4-tap
// kernel whose only size is an inline constant (w, h, 1/w, 1/h) in guest texels (704 or 960). At a resolution scale the
// depth surface is s times bigger on the host and the hardware compare's bilinear footprint is one HOST texel, so the
// kernel no longer joins up: the penumbra is cut into steps one guest texel wide (re/notes/gfx_upgrade/shadows.md
// section 2). Here, for a fragment unit that samples a depth format with UN coordinates from a resolution-scaled surface
// (whole, no clip), the descriptor's coordinate scale becomes 1 / host size and every fragment constant equal to the
// guest (w, h, 1/w, 1/h) becomes the host's: the PS3's own kernel on the finer map. Constants are uniforms: no shader
// change. Vulkan only, the compiled path (not the shader interpreter).

#include "util/types.hpp"
#include "util/logs.hpp"
#include "Emu/system_config.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/RSXTexture.h"
#include "Emu/RSX/Common/TextureUtils.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vk::vd1_shadow
{
	inline bool on()
	{
		static const bool s_on = []()
		{
			const char* env = std::getenv("VANILLAD1_SHADOW_SCALED");
			const bool on = env && *env && std::strcmp(env, "0") != 0;
			if (on) rsx_log.notice("vanillad1: shadow depth sampled at its host size at a resolution scale (VANILLAD1_SHADOW_SCALED=%s)", env);
			return on;
		}();
		return s_on;
	}


	struct unit_t
	{
		u32 guest_w = 0, guest_h = 0, host_w = 0, host_h = 0;
		bool found = false; // the last constants fill held this unit's (w, h, 1/w, 1/h) and moved it to the host's
		bool operator==(const unit_t&) const = default;
	};

	inline std::array<unit_t, 16>& units()
	{
		static std::array<unit_t, 16> s_units{};
		return s_units;
	}

	inline void clear(u32 unit)
	{
		if (unit < 16) units()[unit] = {};
	}

	// load_texture_env, a fragment unit's freshly resolved descriptor (left as it is). Returns true when the unit's state
	// changed: the texture parameters are read again and the constants filled again.
	template <typename Descriptor>
	bool note(u32 unit, const Descriptor& desc, const rsx::fragment_texture& tex)
	{
		if (unit >= 16) return false;
		auto& u = units()[unit];
		const unit_t before = u;
		u = {};

		const u32 raw = tex.format();
		const u32 fmt = raw & ~(CELL_GCM_TEXTURE_LN | CELL_GCM_TEXTURE_UN);
		const u32 pct = static_cast<u32>(g_cfg.video.resolution_scale_percent.get());
		if ((raw & CELL_GCM_TEXTURE_UN) && fmt >= CELL_GCM_TEXTURE_DEPTH24_D8 && fmt <= CELL_GCM_TEXTURE_DEPTH16_FLOAT &&
			pct != 100 && desc.upload_context == rsx::texture_upload_context::framebuffer_storage && desc.image_handle &&
			!desc.texcoord_xform.clamp)
		{
			const u32 gw = tex.width(), gh = tex.height();
			const u32 hw = desc.image_handle->image()->width(), hh = desc.image_handle->image()->height();
			const auto vd1_close = [pct](u32 host, u32 guest)
			{
				const s64 want = static_cast<s64>(guest) * pct / 100;
				return std::abs(static_cast<s64>(host) - want) <= 2;
			};

			// The whole surface at the configured scale; anything else (a bigger reused image, a copy) stays as before
			if (gw && gh && hw != gw && vd1_close(hw, gw) && vd1_close(hh, gh))
			{
				u.guest_w = gw;
				u.guest_h = gh;
				u.host_w = hw;
				u.host_h = hh;
				u.found = before.found && before.guest_w == gw && before.guest_h == gh && before.host_w == hw && before.host_h == hh;
			}
		}

		return !(u == before);
	}

	inline bool close(f32 a, f32 b)
	{
		return std::fabs(a - b) <= std::fabs(b) * 1e-5f;
	}

	// load_program_env, right after the fragment constants are written: every (w, h, 1/w, 1/h) of a noted unit -> the host's
	inline void patch_constants(f32* buf, usz floats, u32 referenced_mask)
	{
		static u32 s_logged = 0;
		for (u32 i = 0; i < 16; i++)
		{
			auto& u = units()[i];
			u.found = false;
			if (!(referenced_mask & (1u << i)) || !u.host_w) continue;

			const f32 gw = static_cast<f32>(u.guest_w), gh = static_cast<f32>(u.guest_h);
			for (usz n = 0; n + 4 <= floats; n += 4)
			{
				f32* v = buf + n;
				if (close(v[0], gw) && close(v[1], gh) && close(v[2], 1.f / gw) && close(v[3], 1.f / gh))
				{
					v[0] = static_cast<f32>(u.host_w);
					v[1] = static_cast<f32>(u.host_h);
					v[2] = 1.f / static_cast<f32>(u.host_w);
					v[3] = 1.f / static_cast<f32>(u.host_h);
					u.found = true;
				}
			}

			if (u.found && s_logged < 16)
			{
				s_logged++;
				rsx_log.notice("vanillad1: shadow scaled: unit %u %ux%u: texel constant and coordinates at %ux%u", i, u.guest_w, u.guest_h, u.host_w, u.host_h);
			}
		}
	}

	// load_program_env, before the texture parameters are written: the UN coordinates of a unit whose constant moved are
	// divided by the host size instead of the guest's. Returns true when a parameter changed (they must be written).
	template <typename TextureConfig>
	bool apply_scale(TextureConfig& params, u32 referenced_mask)
	{
		bool changed = false;
		for (u32 i = 0; i < 16; i++)
		{
			const auto& u = units()[i];
			if (!(referenced_mask & (1u << i)) || !u.found) continue;

			const f32 sx = 1.f / static_cast<f32>(u.host_w), sy = 1.f / static_cast<f32>(u.host_h);
			auto& slot = params[i];
			if (slot.scale[0] != sx || slot.scale[1] != sy)
			{
				slot.scale[0] = sx;
				slot.scale[1] = sy;
				changed = true;
			}
		}
		return changed;
	}
}