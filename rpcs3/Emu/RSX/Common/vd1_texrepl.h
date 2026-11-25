#pragma once

// vanillad1: X515.1 texture replacement (patches A-D of the project's texrepl design). Header only, RSX thread only (the
// live-bytes counter is atomic: a section can be destroyed under a fault on another thread).
//
// Switches, each read once and logged; unset / 0 = as before:
// - VANILLAD1_TEXREPL_LOG=1 (A): per CPU upload of a fragment 2D DXT texture with >= 2 mips and a shorter side >= 32: its two
//   keys, shape, pitch and hash time, one line per new (key, shape); a "texrepl 10s" summary line.
// - VANILLAD1_TEXREPL=<pack file> (B, C): a bind whose key is in the pack is uploaded from the pack's bytes instead of guest
//   memory. Same shape as the bind (B), or 2^s times it on both sides (C: the section keeps the guest's shape, only its
//   host image is bigger; VKTextureCache.cpp create_new_texture, VKDraw.cpp the sampler's LOD rule).
// - VANILLAD1_TEXREPL_BUDGET_MB=<n> (D): bigger images (C) are declined while their live bytes would pass n MiB.
//
// The key: (gcm format without LN / UN, width, height, hash of mip 0's block rows) of the texture's TAIL. A bind of the full
// texture (the hi-mip block: streamed top mip + a copy of the tail) is looked up by its mip 1, a tail bind by its mip 0.
// The hash: FNV-1a 64 over the little-endian u64 words of the block rows (width_in_block x block bytes per row, pitch
// padding skipped), offset basis 0xcbf29ce484222325, prime 0x100000001b3, one xor + multiply per word.
//
// The pack (VD1TPAK1, little-endian; written by the project's tools/texpack.py):
//   header 32 bytes: "VD1TPAK1", u32 version = 1, u32 count, u64 index offset, u64 reserved
//   index: count entries of 40 bytes, sorted by (hash, format, key_w, key_h):
//     u64 hash, u32 format (0x86 DXT1, 0x87 DXT23, 0x88 DXT45), u16 key_w, u16 key_h,
//     u16 w, u16 h, u16 mips (the replacement image), u16 flags (0), u64 offset, u64 length
//   data: per entry its mips, largest first, each ceil(w/4) x ceil(h/4) blocks row by row with no padding, in the guest's
//   own DXT byte order (as on the disc); offsets 16-byte aligned; length = the sum of the mips.
//
// Guards kept here: 2D fragment binds only, DXT only (compressed formats are never blit sources:
// is_blit_source_format_compatible), no UN coordinates, >= 2 guest mips, shorter side >= 32, a replacement only inside RSX
// local heap 0 (0xC0000000-0xCD960000), host sides <= 4096, the pack bytes uploaded as a user pointer (source_is_userptr).

#include "util/types.hpp"
#include "util/logs.hpp"
#include "Utilities/File.h"
#include "Emu/Memory/vm.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/Common/TextureUtils.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace rsx::vd1_texrepl
{
	inline bool log_on()
	{
		static const bool s_on = []()
		{
			const char* env = std::getenv("VANILLAD1_TEXREPL_LOG");
			const bool on = env && *env && std::strcmp(env, "0") != 0;
			if (on) rsx_log.notice("vanillad1: texrepl log on (VANILLAD1_TEXREPL_LOG=%s)", env);
			return on;
		}();
		return s_on;
	}

	inline u64 budget_bytes()
	{
		static const u64 s_bytes = []() -> u64
		{
			const char* env = std::getenv("VANILLAD1_TEXREPL_BUDGET_MB");
			const u64 mb = (env && *env) ? std::strtoull(env, nullptr, 10) : 0;
			if (mb) rsx_log.notice("vanillad1: texrepl budget %llu MiB for bigger images (VANILLAD1_TEXREPL_BUDGET_MB)", mb);
			return mb << 20;
		}();
		return s_bytes;
	}

	// The live bytes of bigger (C) replacement images: added when a section takes one, taken off when it is destroyed
	inline std::atomic<u64> g_live_bytes{0};

	struct pack_entry
	{
		u64 hash;
		u32 format;
		u16 key_w, key_h;
		u16 w, h, mips, flags;
		u64 offset;
		u64 length;
	};
	static_assert(sizeof(pack_entry) == 40);

	inline u32 block_bytes(u32 gcm_format)
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1: return 8;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45: return 16;
		default: return 0;
		}
	}

	inline u64 chain_bytes(u32 gcm_format, u32 w, u32 h, u32 mips)
	{
		u64 total = 0;
		for (u32 i = 0; i < mips; i++)
		{
			const u64 wb = (std::max<u32>(w >> i, 1) + 3) / 4;
			const u64 hb = (std::max<u32>(h >> i, 1) + 3) / 4;
			total += wb * hb * block_bytes(gcm_format);
		}
		return total;
	}

	struct pack
	{
		fs::file file;
		std::vector<pack_entry> index;
	};

	// The pack named by VANILLAD1_TEXREPL, loaded once; nullptr when unset, 0 or unusable (logged)
	inline pack* get_pack()
	{
		static pack* s_pack = []() -> pack*
		{
			const char* env = std::getenv("VANILLAD1_TEXREPL");
			if (!env || !*env || std::strcmp(env, "0") == 0) return nullptr;

			auto p = std::make_unique<pack>();
			p->file.open(std::string(env));
			if (!p->file)
			{
				rsx_log.error("vanillad1: texrepl pack %s could not be opened; replacement off", env);
				return nullptr;
			}

			const u64 size = p->file.size();
			char header[32]{};
			u32 version = 0, count = 0;
			u64 index_offset = 0;
			if (size < 32 || p->file.read_at(0, header, 32) != 32 || std::memcmp(header, "VD1TPAK1", 8) != 0)
			{
				rsx_log.error("vanillad1: texrepl pack %s is not a VD1TPAK1 file; replacement off", env);
				return nullptr;
			}
			std::memcpy(&version, header + 8, 4);
			std::memcpy(&count, header + 12, 4);
			std::memcpy(&index_offset, header + 16, 8);
			if (version != 1 || count > (1u << 22) || index_offset < 32 || index_offset + u64{count} * sizeof(pack_entry) > size)
			{
				rsx_log.error("vanillad1: texrepl pack %s: version %u, %u entries, index at 0x%llx not understood; replacement off", env, version, count, index_offset);
				return nullptr;
			}

			std::vector<pack_entry> raw(count);
			if (count && p->file.read_at(index_offset, raw.data(), u64{count} * sizeof(pack_entry)) != u64{count} * sizeof(pack_entry))
			{
				rsx_log.error("vanillad1: texrepl pack %s: index read failed; replacement off", env);
				return nullptr;
			}

			u32 dropped = 0;
			u64 bytes = 0;
			p->index.reserve(count);
			for (const auto& e : raw)
			{
				const u32 max_mips = 1 + std::countr_zero(std::bit_floor<u32>(std::max<u32>(std::max<u32>(e.w, e.h), 1)));
				const bool ok = block_bytes(e.format) && e.w && e.h && e.w <= 4096 && e.h <= 4096 && e.key_w && e.key_h &&
					e.mips >= 1 && e.mips <= max_mips && (e.offset % 16) == 0 &&
					e.length == chain_bytes(e.format, e.w, e.h, e.mips) && e.offset + e.length <= size;
				if (!ok)
				{
					dropped++;
					continue;
				}
				p->index.push_back(e);
				bytes += e.length;
			}

			std::sort(p->index.begin(), p->index.end(), [](const pack_entry& a, const pack_entry& b)
			{
				if (a.hash != b.hash) return a.hash < b.hash;
				if (a.format != b.format) return a.format < b.format;
				if (a.key_w != b.key_w) return a.key_w < b.key_w;
				return a.key_h < b.key_h;
			});

			rsx_log.notice("vanillad1: texrepl pack %s: %u entries (%u dropped), %llu MiB (VANILLAD1_TEXREPL)", env, ::size32(p->index), dropped, bytes >> 20);
			if (p->index.empty()) return nullptr;
			return p.release();
		}();
		return s_pack;
	}

	inline bool repl_on()
	{
		return get_pack() != nullptr;
	}

	inline bool any_on()
	{
		return log_on() || repl_on();
	}

	inline const pack_entry* find(u64 hash, u32 format, u16 key_w, u16 key_h)
	{
		const pack* p = get_pack();
		if (!p) return nullptr;

		auto it = std::lower_bound(p->index.begin(), p->index.end(), hash, [](const pack_entry& e, u64 h) { return e.hash < h; });
		for (; it != p->index.end() && it->hash == hash; ++it)
		{
			if (it->format == format && it->key_w == key_w && it->key_h == key_h)
			{
				return &*it;
			}
		}
		return nullptr;
	}

	// FNV-1a 64 over the block rows of one subresource, read through the always-mapped view (no fault on a locked page)
	inline bool hash_rows(const rsx::subresource_layout& l, u32 bb, u64& out_hash, u64& out_bytes)
	{
		const u8* p = l.data.data<u8>();
		const u64 row = u64{l.width_in_block} * bb;
		const u64 stride = u64{l.pitch_in_block} * bb;
		const u32 rows = l.height_in_block * std::max<u32>(l.depth, 1);
		if (!p || !row || (row % 8) || stride < row || !rows || stride * (rows - 1) + row > l.data.size())
		{
			return false;
		}

		const uptr a = reinterpret_cast<uptr>(p);
		const uptr base = reinterpret_cast<uptr>(vm::g_base_addr);
		if (a >= base && a < base + 0x1'0000'0000ull)
		{
			p = reinterpret_cast<const u8*>(vm::g_sudo_addr) + (a - base);
		}

		u64 h = 0xcbf29ce484222325ull;
		for (u32 r = 0; r < rows; r++)
		{
			const u8* q = p + stride * r;
			for (u64 i = 0; i < row; i += 8)
			{
				u64 w;
				std::memcpy(&w, q + i, 8);
				h ^= w;
				h *= 0x100000001b3ull;
			}
		}

		out_hash = h;
		out_bytes = row * rows;
		return true;
	}

	struct stats_t
	{
		u32 uploads = 0, eligible = 0, new_keys = 0, hit_same = 0, hit_bigger = 0, declined = 0, shape_miss = 0, read_fail = 0, stale = 0;
		u64 hashed_bytes = 0, hash_ns = 0;
		std::chrono::steady_clock::time_point last{};
	};

	inline stats_t& stats()
	{
		static stats_t s;
		return s;
	}

	inline void maybe_summary()
	{
		auto& st = stats();
		const auto now = std::chrono::steady_clock::now();
		if (st.last == std::chrono::steady_clock::time_point{})
		{
			st.last = now;
			return;
		}
		if (now - st.last < std::chrono::seconds(10)) return;

		rsx_log.notice("vanillad1: texrepl 10s: %u uploads, %u eligible, %u new keys, hashed %.1f MiB in %.2f ms; hits %u same-size, %u bigger; declined by budget %u, "
			"shape mismatch %u, pack read failed %u, changed before the protect %u; bigger images live %.1f MiB (budget %llu MiB)",
			st.uploads, st.eligible, st.new_keys, st.hashed_bytes / 1048576.0, st.hash_ns / 1e6, st.hit_same, st.hit_bigger, st.declined,
			st.shape_miss, st.read_fail, st.stale, g_live_bytes.load() / 1048576.0, budget_bytes() >> 20);

		const auto last = now;
		st = {};
		st.last = last;
	}

	// One prepared replacement, handed from the common miss path (texture_cache.h upload_texture) to the backend's
	// upload_image_from_cpu, which takes it only for the same address, format and guest shape
	struct plan_t
	{
		bool armed = false;
		bool taken = false;
		u32 address = 0, gcm_format = 0;
		u16 guest_w = 0, guest_h = 0, guest_mips = 0;
		u16 w = 0, h = 0, mips = 0;
		u8 shift = 0;
		u64 image_bytes = 0;
		u32 check_index = 0;
		u64 check_hash = 0;
		std::vector<u128> storage;
		std::vector<rsx::subresource_layout> layout;
		// set by the backend when the guest bytes changed between the hash and the protect; the draw drops the section
		u32 stale_start = 0, stale_length = 0;
	};

	inline plan_t& plan()
	{
		static plan_t s;
		return s;
	}

	inline void disarm()
	{
		auto& pl = plan();
		pl.armed = false;
		pl.taken = false;
		pl.layout.clear();
	}

	// The miss path: log (A) and, with a pack, look the bind up and read its replacement (B, C, D). Returns true when a
	// replacement is armed for the upload that follows.
	inline bool prepare(bool fragment, u32 address, u16 w, u16 h, u16 mips, u32 pitch, u32 gcm_format, bool swizzled, bool unnormalized,
		rsx::texture_dimension_extended dim, u32 min_filter, f32 min_lod, f32 max_lod, const std::vector<rsx::subresource_layout>& layout)
	{
		disarm();
		auto& st = stats();
		maybe_summary();
		st.uploads++;

		const u32 bb = block_bytes(gcm_format);
		if (!fragment || unnormalized || dim != rsx::texture_dimension_extended::texture_dimension_2d || !bb ||
			mips < 2 || std::min(w, h) < 32 || layout.size() < 2)
		{
			return false;
		}
		st.eligible++;

		const bool in_heap = address >= 0xC0000000u && address < 0xCD960000u;
		pack* const pk = in_heap ? get_pack() : nullptr;
		if (!pk && !log_on()) return false;

		const auto t0 = std::chrono::steady_clock::now();
		u64 h_tail = 0, h_own = 0, n_tail = 0, n_own = 0;
		const bool ok_tail = hash_rows(layout[1], bb, h_tail, n_tail);
		const pack_entry* e = nullptr;
		u32 key_index = 0;
		if (pk && ok_tail)
		{
			if ((e = find(h_tail, gcm_format, layout[1].width_in_texel, layout[1].height_in_texel))) key_index = 1;
		}
		bool ok_own = false;
		if (!e || log_on())
		{
			ok_own = hash_rows(layout[0], bb, h_own, n_own);
			if (!e && pk && ok_own)
			{
				e = find(h_own, gcm_format, w, h);
			}
		}
		st.hashed_bytes += n_tail + n_own;
		st.hash_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());

		// The entry's shape against the bind's: the same, or 2^s on both sides
		const char* verdict = pk ? "miss" : (in_heap ? "log only" : "outside heap 0");
		u32 s = 0;
		bool use = false;
		if (e)
		{
			const bool even = (e->w % w) == 0 && (e->h % h) == 0 && (e->w / w) == (e->h / h) && std::has_single_bit<u32>(e->w / w);
			s = even ? std::countr_zero<u32>(e->w / w) : 0;
			if (!even || s > 3 || (s == 0 && e->mips < mips))
			{
				st.shape_miss++;
				verdict = "shape mismatch";
			}
			else if (s && budget_bytes() && g_live_bytes.load() + e->length > budget_bytes())
			{
				st.declined++;
				verdict = "declined by budget";
			}
			else
			{
				use = true;
				verdict = s ? "bigger" : "same size";
			}
		}

		if (log_on())
		{
			static std::unordered_set<u64> s_seen;
			const u64 seen_key = h_own ^ (u64{w} << 48) ^ (u64{h} << 32) ^ (u64{gcm_format} << 16) ^ mips;
			if (s_seen.size() > 200000) s_seen.clear();
			if (s_seen.insert(seen_key).second)
			{
				st.new_keys++;
				rsx_log.notice("vanillad1: texrepl 0x%08x %ux%u mips %u fmt 0x%x pitch %u%s: tail key %ux%u %016x%s, own key %016x%s; min filter %u lod %.2f-%.2f -> %s s=%u",
					address, w, h, mips, gcm_format, pitch, swizzled ? " swizzled" : "",
					layout[1].width_in_texel, layout[1].height_in_texel, h_tail, ok_tail ? "" : " (not hashed)", h_own, ok_own ? "" : " (not hashed)",
					min_filter, static_cast<f64>(min_lod), static_cast<f64>(max_lod), verdict, s);
			}
		}

		if (!use) return false;

		// Read the entry; the image's mips: the bind's own count for a same-size swap, the entry's for a bigger one
		auto& pl = plan();
		pl.storage.resize((e->length + 15) / 16);
		if (pk->file.read_at(e->offset, pl.storage.data(), e->length) != e->length)
		{
			st.read_fail++;
			return false;
		}

		pl.w = e->w;
		pl.h = e->h;
		pl.mips = s ? e->mips : mips;
		const u8* base = reinterpret_cast<const u8*>(pl.storage.data());
		u64 offset = 0;
		for (u16 i = 0; i < pl.mips; i++)
		{
			const u16 tw = static_cast<u16>(std::max<u32>(e->w >> i, 1));
			const u16 th = static_cast<u16>(std::max<u32>(e->h >> i, 1));
			const u16 wb = static_cast<u16>((tw + 3) / 4);
			const u16 hb = static_cast<u16>((th + 3) / 4);
			const u32 size = u32{wb} * hb * bb;

			rsx::subresource_layout sub{};
			sub.data = rsx::io_buffer(static_cast<const void*>(base + offset), size);
			sub.width_in_texel = tw;
			sub.height_in_texel = th;
			sub.width_in_block = wb;
			sub.height_in_block = hb;
			sub.depth = 1;
			sub.level = i;
			sub.layer = 0;
			sub.border = 0;
			sub.pitch_in_block = wb;
			pl.layout.push_back(std::move(sub));
			offset += size;
		}

		pl.armed = true;
		pl.address = address;
		pl.gcm_format = gcm_format;
		pl.guest_w = w;
		pl.guest_h = h;
		pl.guest_mips = mips;
		pl.shift = static_cast<u8>(s);
		pl.image_bytes = s ? e->length : 0;
		pl.check_index = key_index;
		pl.check_hash = key_index ? h_tail : h_own;
		(s ? st.hit_bigger : st.hit_same)++;
		return true;
	}

	// The backend: the armed plan when it is for this upload
	inline plan_t* take(u32 address, u32 gcm_format, u16 w, u16 h, u16 mips)
	{
		auto& pl = plan();
		if (!pl.armed || pl.taken || pl.address != address || pl.gcm_format != gcm_format || pl.guest_w != w || pl.guest_h != h || pl.guest_mips != mips)
		{
			return nullptr;
		}
		pl.taken = true;
		return &pl;
	}

	// The backend, after the section is protected: true when the guest bytes still have the key the replacement was chosen by
	inline bool recheck(const plan_t& pl, const std::vector<rsx::subresource_layout>& guest_layout)
	{
		u64 hsh = 0, n = 0;
		if (pl.check_index >= guest_layout.size() || !hash_rows(guest_layout[pl.check_index], block_bytes(pl.gcm_format), hsh, n) || hsh != pl.check_hash)
		{
			stats().stale++;
			return false;
		}
		return true;
	}

	// The draw: a section to drop because its replacement no longer matches the guest bytes
	inline bool take_stale(u32& start, u32& length)
	{
		auto& pl = plan();
		if (!pl.stale_length) return false;
		start = pl.stale_start;
		length = pl.stale_length;
		pl.stale_start = pl.stale_length = 0;
		return true;
	}
}
