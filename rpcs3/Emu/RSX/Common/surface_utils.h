#pragma once

#include "util/types.hpp"
#include "Utilities/geometry.h"
#include "TextureUtils.h"
#include "../Utils/rsx_utils.h"
#include "Emu/Memory/vm.h"
#include "Emu/system_config.h"

#include <atomic> // vanillad1: RCB keep (P13.1)
#include <chrono>
#include <cstdlib> // vanillad1: RCB range (P13)
#include <string_view>
#include <utility>
#include <vector>

#define ENABLE_SURFACE_CACHE_DEBUG 0

namespace rsx
{
	// vanillad1: RCB range (P13): VANILLAD1_RCB_RANGE=<lo>-<hi>[,<lo>-<hi>...] (hex, hi exclusive, read once). Colour
	// surfaces overlapping a range read colour buffers back whatever Read Color Buffers says: a read
	// reloads them from guest memory once per frame (VKRenderTargets.cpp) and views converted from
	// them are never cached (texture_cache.h). Unset / empty / unparsable = false (stock).
	inline bool vd1_rcb_covers(const rsx::address_range32& range)
	{
		static const std::vector<std::pair<u32, u32>> s_ranges = []()
		{
			std::vector<std::pair<u32, u32>> out;
			const char* env = std::getenv("VANILLAD1_RCB_RANGE");
			if (!env || !*env) return out;
			for (const char* p = env;;)
			{
				char* rest = nullptr;
				const u32 lo = static_cast<u32>(std::strtoul(p, &rest, 16));
				u32 hi = 0;
				if (rest && *rest == '-') hi = static_cast<u32>(std::strtoul(rest + 1, &rest, 16));
				if (hi <= lo)
				{
					rsx_log.error("vanillad1: VANILLAD1_RCB_RANGE=%s not understood; colour buffers read back as configured", env);
					out.clear();
					return out;
				}
				out.emplace_back(lo, hi);
				if (!rest || *rest != ',') break;
				p = rest + 1;
			}
			for (const auto& [lo, hi] : out)
				rsx_log.notice("vanillad1: colour buffers read back every frame for 0x%x-0x%x (VANILLAD1_RCB_RANGE)", lo, hi);
			return out;
		}();

		for (const auto& [lo, hi] : s_ranges)
		{
			if (range.start < hi && range.end >= lo)
			{
				return true;
			}
		}

		return false;
	}

	// vanillad1: RCB stock range (P13.2): VANILLAD1_RCB_STOCK=<lo>-<hi>[:<w>x<h>][:init][:own][:keep][,...] (hex range, hi
	// exclusive, read once). Colour surfaces overlapping a range follow stock Read Color Buffers (initialised from guest
	// memory, reloaded when the 3-word tag changed) whatever the config says. Unset / empty / unparsable = false (stock).
	// vanillad1: RCB stock size (P13.2): <w>x<h> = only surfaces of that size (w = 0: the caller has no size, e.g.
	// validate_fbo_integrity: no match); init = the creation load only (barrier = true asks for the reload rule: no match).
	// vanillad1: RCB stock own (P13.3): own = no stock reload from a texture lookup of another address; keep = such a lookup leaves the
	// surface out instead of invalidating it (vd1_rcb_stock_flags).
	struct vd1_rcb_stock_entry
	{
		u32 lo, hi, w, h;
		bool init_only;
		bool own, keep; // vanillad1: RCB stock own (P13.3)
		bool noinherit; // vanillad1: RCB stock noinherit (P13.3)
	};

	enum : u32 { vd1_stock_own = 1, vd1_stock_keep = 2, vd1_stock_noinherit = 4 }; // vanillad1: RCB stock own (P13.3), noinherit: vanillad1: RCB stock noinherit (P13.3)

	inline const std::vector<vd1_rcb_stock_entry>& vd1_rcb_stock_entries()
	{
		static const std::vector<vd1_rcb_stock_entry> s_ranges = []()
		{
			std::vector<vd1_rcb_stock_entry> out;
			const char* env = std::getenv("VANILLAD1_RCB_STOCK");
			if (!env || !*env) return out;
			for (const char* p = env;;)
			{
				char* rest = nullptr;
				vd1_rcb_stock_entry e{};
				e.lo = static_cast<u32>(std::strtoul(p, &rest, 16));
				if (rest && *rest == '-') e.hi = static_cast<u32>(std::strtoul(rest + 1, &rest, 16));
				if (rest && *rest == ':' && rest[1] >= '0' && rest[1] <= '9')
				{
					e.w = static_cast<u32>(std::strtoul(rest + 1, &rest, 10));
					if (rest && (*rest == 'x' || *rest == 'X')) e.h = static_cast<u32>(std::strtoul(rest + 1, &rest, 10));
				}
				bool bad = false;
				while (rest && *rest == ':') // vanillad1: RCB stock own (P13.3): the mode flags, any order
				{
					const std::string_view r(rest);
					if (r.substr(0, 5) == ":init") { e.init_only = true; rest += 5; }
					else if (r.substr(0, 4) == ":own") { e.own = true; rest += 4; }
					else if (r.substr(0, 5) == ":keep") { e.keep = true; rest += 5; }
					else if (r.substr(0, 10) == ":noinherit") { e.noinherit = true; rest += 10; } // vanillad1: RCB stock noinherit (P13.3)
					else { bad = true; break; }
				}
				if (bad || e.hi <= e.lo || (e.w && !e.h))
				{
					rsx_log.error("vanillad1: VANILLAD1_RCB_STOCK=%s not understood; colour buffers read back as configured", env);
					out.clear();
					return out;
				}
				out.push_back(e);
				if (!rest || *rest != ',') break;
				p = rest + 1;
			}
			for (const auto& e : out)
				rsx_log.notice("vanillad1: colour buffers read back as stock Read Color Buffers for 0x%x-0x%x, size %ux%u (0 = any)%s%s%s (VANILLAD1_RCB_STOCK)",
					e.lo, e.hi, e.w, e.h, e.init_only ? ", the creation load only" : "",
					e.own ? ", no reload from another address's lookup" : "", e.keep ? ", kept (not dropped) by another address's lookup" : "");
			for (const auto& e : out) // vanillad1: RCB stock noinherit (P13.3)
				if (e.noinherit) rsx_log.notice("vanillad1: 0x%x-0x%x %ux%u: surfaces at other addresses do not inherit its contents (VANILLAD1_RCB_STOCK :noinherit)", e.lo, e.hi, e.w, e.h);
			return out;
		}();
		return s_ranges;
	}

	inline bool vd1_rcb_stock_covers(const rsx::address_range32& range, u32 w = 0, u32 h = 0, bool barrier = false)
	{
		for (const auto& e : vd1_rcb_stock_entries())
		{
			if (!(range.start < e.hi && range.end >= e.lo)) continue;
			if (e.w && (w != e.w || h != e.h)) continue;
			if (barrier && e.init_only) continue;
			return true;
		}

		return false;
	}

	// vanillad1: RCB stock own (P13.3): the own / keep flags of the entries covering a surface (range + size)
	inline u32 vd1_rcb_stock_flags(const rsx::address_range32& range, u32 w, u32 h)
	{
		u32 out = 0;
		for (const auto& e : vd1_rcb_stock_entries())
		{
			if (!(range.start < e.hi && range.end >= e.lo)) continue;
			if (e.w && (w != e.w || h != e.h)) continue;
			if (e.own) out |= vd1_stock_own;
			if (e.keep) out |= vd1_stock_keep;
			if (e.noinherit) out |= vd1_stock_noinherit; // vanillad1: RCB stock noinherit (P13.3)
		}
		return out;
	}

	// vanillad1: RCB stock own (P13.3): the address a texture lookup (surface_store.h get_merged_texture_memory_region) is resolving, 0 outside one
	inline thread_local u32 vd1_lookup_addr = 0;

	// vanillad1: RCB range (P13): set while the texture cache copies a surface out to guest memory (VKTextureCache.h
	// copy_texture, the Write Color Buffers readback): that read must not reload a surface in
	// VANILLAD1_RCB_RANGE (the SPUs have not lit the fresh blit yet).
	inline thread_local bool vd1_rcb_in_readback = false;

	// vanillad1: RCB changed (P13.2): VANILLAD1_RCB_SIZE=<w>x<h> (decimal, read once; unset = any size): under VANILLAD1_RCB_RANGE only
	// colour surfaces of that size read colour buffers back (the hand-off buffers are 1024x624; in orbit
	// other surfaces live in the same range).
	inline bool vd1_rcb_size_ok(u32 w, u32 h)
	{
		static const std::pair<u32, u32> s_size = []()
		{
			std::pair<u32, u32> out{0u, 0u};
			const char* env = std::getenv("VANILLAD1_RCB_SIZE");
			if (!env || !*env) return out;
			char* rest = nullptr;
			out.first = static_cast<u32>(std::strtoul(env, &rest, 10));
			out.second = (rest && (*rest == 'x' || *rest == 'X')) ? static_cast<u32>(std::strtoul(rest + 1, nullptr, 10)) : 0u;
			if (!out.first || !out.second)
			{
				rsx_log.error("vanillad1: VANILLAD1_RCB_SIZE=%s not understood; any size", env);
				out = {0u, 0u};
				return out;
			}
			rsx_log.notice("vanillad1: colour buffers read back only for %ux%u surfaces (VANILLAD1_RCB_SIZE)", out.first, out.second);
			return out;
		}();
		return !s_size.first || (w == s_size.first && h == s_size.second);
	}

	// vanillad1: RCB range (P13): VANILLAD1_RCB_MODE=any (read once): any read reloads, as build 8631bdf did (A/B only).
	inline bool vd1_rcb_any_read()
	{
		static const bool s_any = []()
		{
			const char* env = std::getenv("VANILLAD1_RCB_MODE");
			const bool any = env && std::string_view(env) == "any";
			if (any)
			{
				rsx_log.warning("vanillad1: rcb reloads on any read, readbacks included (VANILLAD1_RCB_MODE=any)");
			}
			return any;
		}();
		return s_any;
	}

	// vanillad1: RCB keep (P13.1): counters for the stats line every 10 s (VKRenderTargets.cpp, texture_cache.h, surface_store.h)
	struct vd1_rcb_counters
	{
		std::atomic<u32> reads{0}, writes{0}, reloads{0}, rb_kept{0}, hk{0}, drops{0}, nul_kept{0}, forced{0}, tag0{0};
		// vanillad1: RCB changed (P13.2): reloads by the half that changed, accesses that found nothing new, reloads again in a frame,
		// reloads at a write, first accesses, unsampled surfaces, the band / settle waits
		std::atomic<u32> up{0}, lo{0}, both{0}, same{0}, again{0}, wr_rl{0}, first{0}, nosamp{0};
		std::atomic<u32> wait_ok{0}, wait_to{0}, settled{0}, unsettled{0}, wait_us{0};
		std::atomic<u32> stock_rd{0}, stock_rl{0}; // vanillad1: RCB stock range (P13.2): barriers on VANILLAD1_RCB_STOCK surfaces, reloads
		std::atomic<u32> stock_fg{0}, stock_kept{0}; // vanillad1: RCB stock own (P13.3): foreign barriers that skipped the stock rule (own), surfaces left out (keep)
		std::atomic<u32> stock_inh{0}, stock_noinh{0}; // vanillad1: RCB stock noinherit (P13.3): inheritances from a flagged STOCK surface, those skipped
	};
	inline vd1_rcb_counters g_vd1_rcb;

	// vanillad1: RCB changed (P13.2): the flip index the VANILLAD1_RCB_TRACE window ends at (VKRenderTargets.cpp sets it, VKTextureCache.h reads it)
	inline std::atomic<u64> g_vd1_rcb_trace_until{0};

	inline void vd1_rcb_tick()
	{
		static std::atomic<s64> s_last{0};
		const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
		s64 last = s_last.load();
		if (last == 0)
		{
			s_last.compare_exchange_strong(last, now);
			return;
		}
		if (now - last < 10000 || !s_last.compare_exchange_strong(last, now))
		{
			return;
		}
		auto& c = g_vd1_rcb;
		rsx_log.notice("vanillad1: rcb 10s: reads=%u writes=%u reloads=%u rb_kept=%u hk=%u drops=%u nul_kept=%u forced=%u tag0=%u",
			c.reads.exchange(0), c.writes.exchange(0), c.reloads.exchange(0), c.rb_kept.exchange(0), c.hk.exchange(0),
			c.drops.exchange(0), c.nul_kept.exchange(0), c.forced.exchange(0), c.tag0.exchange(0));
		rsx_log.notice("vanillad1: rcb 10s changed: up=%u lo=%u both=%u same=%u again=%u wr_rl=%u first=%u nosamp=%u", // vanillad1: RCB changed (P13.2)
			c.up.exchange(0), c.lo.exchange(0), c.both.exchange(0), c.same.exchange(0), c.again.exchange(0),
			c.wr_rl.exchange(0), c.first.exchange(0), c.nosamp.exchange(0));
		rsx_log.notice("vanillad1: rcb 10s waits: wait_ok=%u wait_to=%u settled=%u unsettled=%u wait_ms=%u stock_rd=%u stock_rl=%u stock_fg=%u stock_kept=%u",
			c.wait_ok.exchange(0), c.wait_to.exchange(0), c.settled.exchange(0), c.unsettled.exchange(0), c.wait_us.exchange(0) / 1000,
			c.stock_rd.exchange(0), c.stock_rl.exchange(0), // stock_*: vanillad1: RCB stock range (P13.2)
			c.stock_fg.exchange(0), c.stock_kept.exchange(0)); // vanillad1: RCB stock own (P13.3)
		rsx_log.notice("vanillad1: rcb 10s stock: inh=%u noinh=%u", c.stock_inh.exchange(0), c.stock_noinh.exchange(0)); // vanillad1: RCB stock noinherit (P13.3)
	}

	enum surface_state_flags : u32
	{
		ready               = 0x00,
		erase_bkgnd         = 0x01,
		require_resolve     = 0x02,
		require_unresolve   = 0x04,
		force_data_load     = 0x08
	};

	enum class surface_sample_layout : u32
	{
		null = 0,
		ps3 = 1
	};

	enum class surface_inheritance_result : u32
	{
		none = 0,
		partial,
		full
	};

	template <typename surface_type>
	struct surface_overlap_info_t
	{
		surface_type surface = nullptr;
		u32 base_address = 0;
		bool is_depth = false;
		bool is_clipped = false;
		bool is_reloaded = false;

		coordu src_area;  //<- Always computed in source image coordinates
		coordu dst_area;  //<- Always computed in destination (requester) image coordinates
	};

	template <typename surface_type>
	struct deferred_clipped_region
	{
		u16 src_x, src_y, dst_x, dst_y, width, height;
		f32 transfer_scale_x, transfer_scale_y;
		surface_type target;
		surface_type source;

		template <typename T>
		deferred_clipped_region<T> cast() const
		{
			deferred_clipped_region<T> ret;
			ret.src_x = src_x;
			ret.src_y = src_y;
			ret.dst_x = dst_x;
			ret.dst_y = dst_y;
			ret.width = width;
			ret.height = height;
			ret.transfer_scale_x = transfer_scale_x;
			ret.transfer_scale_y = transfer_scale_y;
			ret.target = static_cast<T>(target);
			ret.source = static_cast<T>(source);

			return ret;
		}

		operator bool() const
		{
			return (source != nullptr);
		}

		template <typename T>
		void init_transfer(T target_surface)
		{
			if (!width)
			{
				// Perform intersection here
				const auto region = rsx::get_transferable_region(target_surface);

				auto src_w = std::get<0>(region);
				auto src_h = std::get<1>(region);
				auto dst_w = std::get<2>(region);
				auto dst_h = std::get<3>(region);

				// Apply resolution scale if needed
				auto src = static_cast<T>(source);
				std::tie(src_w, src_h) = rsx::apply_resolution_scale<true>(
					src->resolution_scaling_config,
					src_w, src_h,
					src->template get_surface_width<rsx::surface_metrics::pixels>(),
					src->template get_surface_height<rsx::surface_metrics::pixels>());

				std::tie(dst_w, dst_h) = rsx::apply_resolution_scale<true>(
					target_surface->resolution_scaling_config,
					dst_w, dst_h,
					target_surface->template get_surface_width<rsx::surface_metrics::pixels>(),
					target_surface->template get_surface_height<rsx::surface_metrics::pixels>());

				width = src_w;
				height = src_h;
				transfer_scale_x = f32(dst_w) / src_w;
				transfer_scale_y = f32(dst_h) / src_h;

				target = target_surface;
			}
		}

		areai src_rect() const
		{
			ensure(width);
			return { src_x, src_y, src_x + width, src_y + height };
		}

		areai dst_rect() const
		{
			ensure(width);
			return { dst_x, dst_y, dst_x + u16(width * transfer_scale_x + 0.5f), dst_y + u16(height * transfer_scale_y + 0.5f) };
		}
	};

	template <typename image_storage_type>
	struct render_target_descriptor : public rsx::ref_counted
	{
		u64 last_use_tag = 0;         // tag indicating when this block was last confirmed to have been written to
		u32 base_addr = 0;

#if (ENABLE_SURFACE_CACHE_DEBUG)
		u64 memory_hash = 0;
#else
		std::array<std::pair<u32, u64>, 3> memory_tag_samples;
#endif

		std::vector<deferred_clipped_region<image_storage_type>> old_contents;

		// Surface properties
		u32 rsx_pitch = 0;
		u32 native_pitch = 0;
		u16 surface_width = 0;
		u16 surface_height = 0;
		u8  spp = 1;
		u8  samples_x = 1;
		u8  samples_y = 1;

		// AA mode
		rsx::surface_antialiasing aa_mode = rsx::surface_antialiasing::center_1_sample;

		// Scaling configuration
		surface_scaling_config_t resolution_scaling_config;

		rsx::address_range32 memory_range;

		std::unique_ptr<typename std::remove_pointer_t<image_storage_type>> resolve_surface;
		surface_sample_layout sample_layout = surface_sample_layout::null;
		surface_raster_type raster_type = surface_raster_type::linear;

		flags32_t memory_usage_flags = surface_usage_flags::unknown;
		flags32_t state_flags = surface_state_flags::ready;
		flags32_t msaa_flags = surface_state_flags::ready;
		flags32_t stencil_init_flags = 0;

		union
		{
			rsx::surface_color_format gcm_color_format;
			rsx::surface_depth_format2 gcm_depth_format;
		}
		format_info;

		struct
		{
			u64 timestamp = 0;
			bool locked = false;
		}
		texture_cache_metadata;

		render_target_descriptor() {}

		virtual ~render_target_descriptor()
		{
			if (!old_contents.empty())
			{
				// Cascade resource derefs
				rsx_log.error("Resource was destroyed whilst holding a resource reference!");
			}
		}

		virtual image_storage_type get_surface(rsx::surface_access access_type) = 0;
		virtual bool is_depth_surface() const = 0;

		void reset()
		{
			texture_cache_metadata = {};
		}

		template<rsx::surface_metrics Metrics = rsx::surface_metrics::pixels, typename T = u32>
		T get_surface_width() const
		{
			if constexpr (Metrics == rsx::surface_metrics::samples)
			{
				return static_cast<T>(surface_width * samples_x);
			}
			else if constexpr (Metrics == rsx::surface_metrics::pixels)
			{
				return static_cast<T>(surface_width);
			}
			else if constexpr (Metrics == rsx::surface_metrics::bytes)
			{
				return static_cast<T>(native_pitch);
			}
			else
			{
				fmt::throw_exception("Unreachable");
			}
		}

		template<rsx::surface_metrics Metrics = rsx::surface_metrics::pixels, typename T = u32>
		T get_surface_height() const
		{
			if constexpr (Metrics == rsx::surface_metrics::samples)
			{
				return static_cast<T>(surface_height * samples_y);
			}
			else if constexpr (Metrics == rsx::surface_metrics::pixels)
			{
				return static_cast<T>(surface_height);
			}
			else if constexpr (Metrics == rsx::surface_metrics::bytes)
			{
				return static_cast<T>(surface_height * samples_y);
			}
			else
			{
				fmt::throw_exception("Unreachable");
			}
		}

		inline u32 get_rsx_pitch() const
		{
			return rsx_pitch;
		}

		inline u32 get_native_pitch() const
		{
			return native_pitch;
		}

		inline u8 get_bpp() const
		{
			return u8(get_native_pitch() / get_surface_width<rsx::surface_metrics::samples>());
		}

		inline u8 get_spp() const
		{
			return spp;
		}

		void set_aa_mode(rsx::surface_antialiasing aa)
		{
			aa_mode = aa;

			switch (aa)
			{
				case rsx::surface_antialiasing::center_1_sample:
					samples_x = samples_y = spp = 1;
					break;
				case rsx::surface_antialiasing::diagonal_centered_2_samples:
					samples_x = spp = 2;
					samples_y = 1;
					break;
				case rsx::surface_antialiasing::square_centered_4_samples:
				case rsx::surface_antialiasing::square_rotated_4_samples:
					samples_x = samples_y = 2;
					spp = 4;
					break;
				default:
					fmt::throw_exception("Unknown AA mode 0x%x", static_cast<u8>(aa));
			}
		}

		rsx::surface_antialiasing get_aa_mode() const
		{
			return aa_mode;
		}

		void set_spp(u8 count)
		{
			switch (count)
			{
			case 1:
				samples_x = samples_y = spp = 1;
				break;
			case 2:
				samples_x = spp = 2;
				samples_y = 1;
				break;
			case 4:
				samples_x = samples_y = 2;
				spp = 4;
				break;
			default:
				fmt::throw_exception("Unexpected sample count 0x%x", count);
			}
		}

		void set_format(rsx::surface_color_format format)
		{
			format_info.gcm_color_format = format;
		}

		void set_format(rsx::surface_depth_format2 format)
		{
			format_info.gcm_depth_format = format;
		}

		void set_resolution_scaling_config(const surface_scaling_config_t& config)
		{
			resolution_scaling_config = config;
		}

		inline rsx::surface_color_format get_surface_color_format() const
		{
			return format_info.gcm_color_format;
		}

		inline rsx::surface_depth_format2 get_surface_depth_format() const
		{
			return format_info.gcm_depth_format;
		}

		inline u32 get_gcm_format() const
		{
			return
			(
				is_depth_surface() ?
					get_compatible_gcm_format(format_info.gcm_depth_format).first :
					get_compatible_gcm_format(format_info.gcm_color_format).first
			);
		}

		inline const rsx::surface_scaling_config_t& get_resolution_scaling_config() const
		{
			return resolution_scaling_config;
		}

		inline bool dirty() const
		{
			return (state_flags != rsx::surface_state_flags::ready) || !old_contents.empty();
		}

		inline bool write_through() const
		{
			return (state_flags & rsx::surface_state_flags::erase_bkgnd) && old_contents.empty();
		}

		inline bool needs_cpu_upload() const
		{
			if ((state_flags & rsx::surface_state_flags::erase_bkgnd) == 0) [[ likely ]]
			{
				return false;
			}

			if (state_flags & rsx::surface_state_flags::force_data_load)
			{
				return true;
			}

			return is_depth_surface()
				? !!g_cfg.video.read_depth_buffer
				: (!!g_cfg.video.read_color_buffers || (vd1_rcb_covers(get_memory_range()) && vd1_rcb_size_ok(surface_width, surface_height)) || // vanillad1: RCB range (P13), size (vanillad1: RCB changed (P13.2))
					vd1_rcb_stock_covers(get_memory_range(), surface_width, surface_height)); // vanillad1: RCB stock range (P13.2), size: vanillad1: RCB stock size (P13.2)
		}

#if (ENABLE_SURFACE_CACHE_DEBUG)
		u64 hash_block() const
		{
			const auto padding = (rsx_pitch - native_pitch) / 8;
			const auto row_length = (native_pitch) / 8;
			auto num_rows = (surface_height * samples_y);
			auto ptr = reinterpret_cast<u64*>(vm::g_sudo_addr + base_addr);

			auto col = row_length;
			u64 result = 0;

			while (num_rows--)
			{
				while (col--)
				{
					result ^= *ptr++;
				}

				ptr += padding;
				col = row_length;
			}

			return result;
		}

		void queue_tag(u32 address)
		{
			ensure(native_pitch);
			ensure(rsx_pitch);

			base_addr = address;

			const u32 internal_height = get_surface_height<rsx::surface_metrics::samples>();
			const u32 excess = (rsx_pitch - native_pitch);
			memory_range = rsx::address_range32::start_length(base_addr, internal_height * rsx_pitch - excess);
		}

		void sync_tag()
		{
			memory_hash = hash_block();
		}

		void shuffle_tag()
		{
			memory_hash = ~memory_hash;
		}

		bool test() const
		{
			return hash_block() == memory_hash;
		}

#else
		void queue_tag(u32 address)
		{
			ensure(native_pitch);
			ensure(rsx_pitch);

			// Clear metadata
			reset();

			base_addr = address;

			const u32 size_x = (native_pitch > 8)? (native_pitch - 8) : 0u;
			const u32 size_y = u32(surface_height * samples_y) - 1u;
			const position2u samples[] =
			{
				// NOTE: Sorted by probability to catch dirty flag
				{0, 0},
				{size_x, size_y},
				{size_x / 2, size_y / 2},

				// Auxilliary, highly unlikely to ever catch anything
				// NOTE: Currently unused as length of samples is truncated to 3
				{size_x, 0},
				{0, size_y},
			};

			for (uint n = 0; n < memory_tag_samples.size(); ++n)
			{
				const auto sample_offset = (samples[n].y * rsx_pitch) + samples[n].x;
				memory_tag_samples[n].first = (sample_offset + base_addr);
			}

			const u32 internal_height = get_surface_height<rsx::surface_metrics::samples>();
			const u32 excess = (rsx_pitch - native_pitch);
			memory_range = rsx::address_range32::start_length(base_addr, internal_height * rsx_pitch - excess);
		}

		void sync_tag()
		{
			for (auto &e : memory_tag_samples)
			{
				e.second = *reinterpret_cast<nse_t<u64, 1>*>(vm::g_sudo_addr + e.first);
			}
		}

		void shuffle_tag()
		{
			memory_tag_samples[0].second = ~memory_tag_samples[0].second;
		}

		bool test() const
		{
			for (const auto& e : memory_tag_samples)
			{
				if (e.second != *reinterpret_cast<nse_t<u64, 1>*>(vm::g_sudo_addr + e.first))
					return false;
			}

			return true;
		}
#endif

		void invalidate_GPU_memory()
		{
			// Here be dragons. Use with caution.
			shuffle_tag();
			state_flags |= rsx::surface_state_flags::erase_bkgnd;
		}

		void clear_rw_barrier()
		{
			for (auto &e : old_contents)
			{
				ensure(dynamic_cast<rsx::ref_counted*>(e.source))->release();
			}

			old_contents.clear();
		}

		template <typename T>
		u32 prepare_rw_barrier_for_transfer(T *target)
		{
			if (old_contents.size() <= 1)
				return 0;

			// Sort here before doing transfers since surfaces may have been updated in the meantime
			std::sort(old_contents.begin(), old_contents.end(), [](const auto& a, const auto &b)
			{
				const auto _a = static_cast<const T*>(a.source);
				const auto _b = static_cast<const T*>(b.source);
				return (_a->last_use_tag < _b->last_use_tag);
			});

			// Try and optimize by omitting possible overlapped transfers
			for (usz i = old_contents.size() - 1; i > 0 /* Intentional */; i--)
			{
				old_contents[i].init_transfer(target);

				const auto dst_area = old_contents[i].dst_rect();
				if (unsigned(dst_area.x2) == target->width() && unsigned(dst_area.y2) == target->height() &&
					!dst_area.x1 && !dst_area.y1)
				{
					// This transfer will overwrite everything older
					return u32(i);
				}
			}

			return 0;
		}

		template<typename T>
		void set_old_contents(T* other)
		{
			ensure(old_contents.empty());

			if (!other || other->get_rsx_pitch() != this->get_rsx_pitch())
			{
				return;
			}

			old_contents.emplace_back();
			old_contents.back().source = other;
			other->add_ref();
		}

		template<typename T>
		void set_old_contents_region(const T& region, bool normalized)
		{
			// NOTE: This method will not perform pitch verification!
			ensure(region.source);
			ensure(region.source != static_cast<decltype(region.source)>(this));

			old_contents.push_back(region.template cast<image_storage_type>());
			auto &slice = old_contents.back();
			region.source->add_ref();

			// Reverse normalization process if needed
			if (normalized)
			{
				const u16 bytes_to_texels_x = region.source->get_bpp() * region.source->samples_x;
				const u16 rows_to_texels_y = region.source->samples_y;
				slice.src_x /= bytes_to_texels_x;
				slice.src_y /= rows_to_texels_y;
				slice.width /= bytes_to_texels_x;
				slice.height /= rows_to_texels_y;

				const u16 bytes_to_texels_x2 = (get_bpp() * samples_x);
				const u16 rows_to_texels_y2 = samples_y;
				slice.dst_x /= bytes_to_texels_x2;
				slice.dst_y /= rows_to_texels_y2;

				slice.transfer_scale_x = f32(bytes_to_texels_x) / bytes_to_texels_x2;
				slice.transfer_scale_y = f32(rows_to_texels_y) / rows_to_texels_y2;
			}

			// Apply resolution scale if needed
			if (resolution_scaling_config.scale_percent != 100 ||
				region.source->resolution_scaling_config.scale_percent != 100)
			{
				const auto& src_res_scale = region.source->resolution_scaling_config;
				const auto& dst_res_scale = resolution_scaling_config;
				const auto src_surface = ensure(dynamic_cast<const render_target_descriptor*>(slice.source));
				const auto dst_surface = ensure(dynamic_cast<const render_target_descriptor*>(slice.target));

				auto [src_width, src_height] = rsx::apply_resolution_scale<true>(src_res_scale, slice.width, slice.height, src_surface->get_surface_width(), src_surface->get_surface_height());
				auto [dst_width, dst_height] = rsx::apply_resolution_scale<true>(dst_res_scale, slice.width, slice.height, dst_surface->get_surface_width(), dst_surface->get_surface_height());

				slice.transfer_scale_x *= f32(dst_width) / src_width;
				slice.transfer_scale_y *= f32(dst_height) / src_height;

				slice.width = src_width;
				slice.height = src_height;

				std::tie(slice.src_x, slice.src_y) = rsx::apply_resolution_scale<false>(src_res_scale, slice.src_x, slice.src_y, src_surface->get_surface_width(), src_surface->get_surface_height());
				std::tie(slice.dst_x, slice.dst_y) = rsx::apply_resolution_scale<false>(dst_res_scale, slice.dst_x, slice.dst_y, dst_surface->get_surface_width(), dst_surface->get_surface_height());
			}
		}

		template <typename T>
		surface_inheritance_result inherit_surface_contents(T* surface)
		{
			const auto child_w = get_surface_width<rsx::surface_metrics::bytes>();
			const auto child_h = get_surface_height<rsx::surface_metrics::bytes>();

			const auto parent_w = surface->template get_surface_width<rsx::surface_metrics::bytes>();
			const auto parent_h = surface->template get_surface_height<rsx::surface_metrics::bytes>();

			const auto [src_offset, dst_offset, size] = rsx::intersect_region(surface->base_addr, parent_w, parent_h, base_addr, child_w, child_h, get_rsx_pitch());

			if (!size.width || !size.height)
			{
				return surface_inheritance_result::none;
			}

			ensure(src_offset.x < parent_w && src_offset.y < parent_h);
			ensure(dst_offset.x < child_w && dst_offset.y < child_h);

			// TODO: Eventually need to stack all the overlapping regions, but for now just do the latest rect in the space
			deferred_clipped_region<T*> region{};
			region.src_x = src_offset.x;
			region.src_y = src_offset.y;
			region.dst_x = dst_offset.x;
			region.dst_y = dst_offset.y;
			region.width = size.width;
			region.height = size.height;
			region.source = surface;
			region.target = static_cast<T*>(this);

			set_old_contents_region(region, true);
			return (region.width == parent_w && region.height == parent_h) ?
				surface_inheritance_result::full :
				surface_inheritance_result::partial;
		}

		void on_write(u64 write_tag = 0,
			rsx::surface_state_flags resolve_flags = surface_state_flags::require_resolve,
			surface_raster_type type = rsx::surface_raster_type::undefined)
		{
			if (write_tag)
			{
				// Update use tag if requested
				last_use_tag = write_tag;
			}

			// Tag unconditionally without introducing new data
			sync_tag();

			// HACK!! This should be cleared through memory barriers only
			state_flags = rsx::surface_state_flags::ready;

			if (spp > 1 && sample_layout != surface_sample_layout::null)
			{
				msaa_flags = resolve_flags;
			}

			if (!old_contents.empty())
			{
				clear_rw_barrier();
			}

			if (type != rsx::surface_raster_type::undefined)
			{
				raster_type = type;
			}
		}

		void on_write_copy(u64 write_tag = 0,
			bool keep_optimizations = false,
			surface_raster_type type = rsx::surface_raster_type::undefined)
		{
			on_write(write_tag, rsx::surface_state_flags::require_unresolve, type);

			if (!keep_optimizations && is_depth_surface())
			{
				// A successful write-copy occured, cannot guarantee flat contents in stencil area
				stencil_init_flags |= (1 << 9);
			}
		}

		inline void on_write_fast(u64 write_tag)
		{
			ensure(write_tag);
			last_use_tag = write_tag;

			if (spp > 1 && sample_layout != surface_sample_layout::null)
			{
				msaa_flags |= rsx::surface_state_flags::require_resolve;
			}
		}

		// Returns the rect area occupied by this surface expressed as an 8bpp image with no AA
		inline areau get_normalized_memory_area() const
		{
			const u16 internal_width = get_surface_width<rsx::surface_metrics::bytes>();
			const u16 internal_height = get_surface_height<rsx::surface_metrics::bytes>();

			return { 0, 0, internal_width, internal_height };
		}

		inline rsx::address_range32 get_memory_range() const
		{
			return memory_range;
		}

		template <typename T>
		void transform_samples_to_pixels(area_base<T>& area)
		{
			if (spp == 1) [[likely]] return;

			area.x1 /= samples_x;
			area.x2 /= samples_x;
			area.y1 /= samples_y;
			area.y2 /= samples_y;
		}

		template <typename T>
		void transform_pixels_to_samples(area_base<T>& area)
		{
			if (spp == 1) [[likely]] return;

			area.x1 *= samples_x;
			area.x2 *= samples_x;
			area.y1 *= samples_y;
			area.y2 *= samples_y;
		}

		template <typename T>
		void transform_samples_to_pixels(T& x1, T& x2, T& y1, T& y2)
		{
			if (spp == 1) [[likely]] return;

			x1 /= samples_x;
			x2 /= samples_x;
			y1 /= samples_y;
			y2 /= samples_y;
		}

		template <typename T>
		void transform_pixels_to_samples(T& x1, T& x2, T& y1, T& y2)
		{
			if (spp == 1) [[likely]] return;

			x1 *= samples_x;
			x2 *= samples_x;
			y1 *= samples_y;
			y2 *= samples_y;
		}

		template<typename T>
		void transform_blit_coordinates(rsx::surface_access access_type, area_base<T>& region)
		{
			if (spp == 1 || sample_layout == rsx::surface_sample_layout::ps3)
				return;

			ensure(access_type.is_read() || access_type.is_transfer());
			transform_samples_to_pixels(region);
		}

		void on_lock()
		{
			add_ref();
			texture_cache_metadata.locked = true;
			texture_cache_metadata.timestamp = rsx::get_shared_tag();
		}

		void on_unlock()
		{
			texture_cache_metadata.locked = false;
			texture_cache_metadata.timestamp = rsx::get_shared_tag();
			release();
		}

		void on_swap_out()
		{
			if (is_locked())
			{
				on_unlock();
			}
			else
			{
				release();
			}
		}

		void on_swap_in(bool lock)
		{
			if (!is_locked() && lock)
			{
				on_lock();
			}
			else
			{
				add_ref();
			}
		}

		void on_clone_from(const render_target_descriptor* ref)
		{
			if (ref->is_locked() && !is_locked())
			{
				// Propagate locked state only.
				texture_cache_metadata = ref->texture_cache_metadata;
			}

			rsx_pitch = ref->get_rsx_pitch();
			last_use_tag = ref->last_use_tag;
			raster_type = ref->raster_type;     // Can't actually cut up swizzled data
		}

		bool is_locked() const
		{
			return texture_cache_metadata.locked;
		}

		bool has_flushable_data() const
		{
			ensure(is_locked());
			ensure(texture_cache_metadata.timestamp);
			return (texture_cache_metadata.timestamp < last_use_tag);
		}
	};
}