#include "stdafx.h"
#include "nv0039.h"

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Core/RSXReservationLock.hpp"
#include "Emu/RSX/Host/MM.h"

#include "Utilities/deferred_op.hpp"

// vanillad1: NV0039 stride 4 -> 1 in SIMD (P10)
#ifdef ARCH_X64
#include <emmintrin.h>
#endif
#include <cstdlib>
#include <cstring> // vanillad1: G-buffer byte on the GPU (X300.7)

#include "context_accessors.define.h"

namespace rsx
{
	namespace nv0039
	{
		// Transfer with stride
		inline void block2d_copy_with_stride(u8* dst, const u8* src, u32 column_count, u32 row_count, s32 src_pitch, s32 dst_pitch, u8 src_stride, u8 dst_stride)
		{
			for (u32 row = 0; row < row_count; ++row)
			{
				auto dst_ptr = dst;
				auto src_ptr = src;
				for (u32 column = 0; column < column_count; ++column)
				{
					*dst_ptr = *src_ptr;

					src_ptr += src_stride;
					dst_ptr += dst_stride;
				}

				dst += dst_pitch;
				src += src_pitch;
			}
		}

		// vanillad1: NV0039 stride 4 -> 1 in SIMD (P10): VANILLAD1_NV0039_SCALAR=1 keeps the byte loop (A/B)
		inline bool vd1_nv0039_scalar()
		{
			static const bool s_scalar = []()
			{
				const char* env = std::getenv("VANILLAD1_NV0039_SCALAR");
				const bool on = env && *env == '1';
				if (on) rsx_log.notice("vanillad1: NV0039 byte copy: scalar loop (VANILLAD1_NV0039_SCALAR=1)");
				else rsx_log.notice("vanillad1: NV0039 byte copy: 16 pixels a step for stride 4 -> 1");
				return on;
			}();
			return s_scalar;
		}

		// One byte (the first) of every 4-byte pixel, 16 pixels a step; the row's tail with the old loop.
		// Reads at most column_count * 4 source bytes of a row, writes column_count destination bytes.
		inline void block2d_copy_stride4_to_1(u8* dst, const u8* src, u32 column_count, u32 row_count, s32 src_pitch, s32 dst_pitch)
		{
			for (u32 row = 0; row < row_count; ++row)
			{
				u32 column = 0;
#ifdef ARCH_X64
				const __m128i low_byte = _mm_set1_epi32(0xff);
				for (; column + 16 <= column_count; column += 16)
				{
					const __m128i* in = reinterpret_cast<const __m128i*>(src + column * 4);
					const __m128i a = _mm_and_si128(_mm_loadu_si128(in + 0), low_byte);
					const __m128i b = _mm_and_si128(_mm_loadu_si128(in + 1), low_byte);
					const __m128i c = _mm_and_si128(_mm_loadu_si128(in + 2), low_byte);
					const __m128i d = _mm_and_si128(_mm_loadu_si128(in + 3), low_byte);
					// dwords 0..255 -> words -> bytes, in order
					const __m128i ab = _mm_packs_epi32(a, b);
					const __m128i cd = _mm_packs_epi32(c, d);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + column), _mm_packus_epi16(ab, cd));
				}
#endif
				for (; column < column_count; ++column)
				{
					dst[column] = src[column * 4];
				}

				dst += dst_pitch;
				src += src_pitch;
			}
		}

		inline void block2d_copy(u8* dst, const u8* src, u32 width, u32 height, s32 src_pitch, s32 dst_pitch)
		{
			for (u32 i = 0; i < height; ++i)
			{
				std::memcpy(dst, src, width);
				dst += dst_pitch;
				src += src_pitch;
			}
		}

		inline bool validate_buffer_notify(s32 col_count, s32 row_count, s32 in_stride, s32 out_stride)
		{
			if (!col_count || !row_count)
			{
				rsx_log.warning("NV0039_BUFFER_NOTIFY NOPed out: 2D area is zero.");
				return false;
			}

			if (in_stride <= 0 || in_stride > 4)
			{
				rsx_log.error("NV0039_BUFFER_NOTIFY NOPed out: Invalid input stride (=%d)", in_stride);
				return false;
			}

			if (out_stride <= 0 || out_stride > 4)
			{
				rsx_log.error("NV0039_BUFFER_NOTIFY NOPed out: Invalid output stride (=%d)", out_stride);
				return false;
			}

			return true;
		}

		// vanillad1: G-buffer byte on the GPU (X300.7): VANILLAD1_GBUF_GPU=1 holds back Destiny's NV0039 4 -> 1 byte copy out of
		// a local surface (G-buffer A 0xCE8E0000 -> the 8-bit buffer 0x444EFA80 the lighting job reads) while the GPU lighting
		// pass is on: the pass takes the byte from the surface on the GPU (VKGSRender::vd1_gpu_light_ready) and the copy, with
		// its readback of the surface (the RSX thread's own fault, ~1.5 ms a frame), never runs. A held copy the pass did not
		// take runs at the ready word (before it lands: the SPUs wait for it), at the next NV0039 or at the flip.
		// check = the copy runs as before and the pass compares its own bytes with it (every channel) every 10th frame.
		int vd1_gbuf_gpu_mode()
		{
			static const int s_mode = []()
			{
				const char* env = std::getenv("VANILLAD1_GBUF_GPU");
				const char* gl = std::getenv("VANILLAD1_GPU_LIGHT");
				int m = !env || !*env || *env == '0' ? 0 : !std::strcmp(env, "check") ? 2 : 1;
				if (m && (!gl || !*gl || !std::strcmp(gl, "0") || !std::strcmp(gl, "probe")))
				{
					rsx_log.warning("vanillad1: VANILLAD1_GBUF_GPU=%s ignored: the GPU lighting pass is off (VANILLAD1_GPU_LIGHT)", env);
					m = 0;
				}
				if (m) rsx_log.success("vanillad1: VANILLAD1_GBUF_GPU=%s: the G-buffer byte %s", env,
					m == 2 ? "copied as before and compared with the GPU's" : "on the GPU (the NV0039 copy held back)");
				return m;
			}();
			return s_mode;
		}

		static void do_buffer_notify(thread* rthr, const vd1_nv0039_args& a)
		{
			const s32 in_pitch = a.in_pitch;
			const s32 out_pitch = a.out_pitch;
			const u32 line_length = a.line_length;
			const u32 line_count = a.line_count;
			const u8 out_format = a.out_format;
			const u8 in_format = a.in_format;
			const u32 src_offset = a.src_offset;
			const u32 src_dma = a.src_dma;
			const u32 dst_offset = a.dst_offset;
			const u32 dst_dma = a.dst_dma;

			const auto in_width_in_bytes = line_length * in_format;
			const auto out_width_in_bytes = line_length * out_format;
			const bool is_block_transfer =
				in_format == 1 && out_format == 1 &&
				in_pitch + 0u == in_width_in_bytes &&
				out_pitch + 0u == out_width_in_bytes;
			const auto read_address = get_address(src_offset, src_dma);
			const auto write_address = get_address(dst_offset, dst_dma);
			const auto read_length = in_pitch * (line_count - 1) + in_width_in_bytes;
			const auto write_length = out_pitch * (line_count - 1) + out_width_in_bytes;

			rthr->invalidate_fragment_program(dst_dma, dst_offset, write_length);

			if (const auto result = rthr->read_barrier(read_address, read_length, !is_block_transfer);
				result == rsx::result_zcull_intr)
			{
				// This transfer overlaps will zcull data pool
				if (rthr->copy_zcull_stats(read_address, read_length, write_address) >= write_length)
				{
					// All writes deferred
					return;
				}
			}

			// Deferred write_barrier on RSX side
			utils::deferred_op deferred([&]()
			{
				rthr->write_barrier(write_address, write_length);
				// res->release(0);
			});

			auto res = ::rsx::reservation_lock<true>(write_address, write_length, read_address, read_length);

			u8* dst = vm::_ptr<u8>(write_address);
			const u8* src = vm::_ptr<u8>(read_address);

			const rsx::simple_array<utils::address_range64> flush_mm_ranges =
			{
				utils::address_range64::start_length(reinterpret_cast<u64>(dst), write_length),
				utils::address_range64::start_length(reinterpret_cast<u64>(src), read_length)
			};
			rsx::mm_flush(flush_mm_ranges);

			const bool is_overlapping = dst_dma == src_dma && [&]() -> bool
			{
				const u32 src_max = src_offset + read_length;
				const u32 dst_max = dst_offset + write_length;
				return (src_offset >= dst_offset && src_offset < dst_max) ||
				 (dst_offset >= src_offset && dst_offset < src_max);
			}();

			if (in_format > 1 || out_format > 1) [[ unlikely ]]
			{
				// The formats are just input channel strides. You can use this to do cool tricks like gathering channels
				// Very rare, only seen in use by Destiny
				// TODO: Hw accel
				// vanillad1: NV0039 stride 4 -> 1 in SIMD (P10): the G-buffer copy for the SPUs; not for overlapping ranges (the loop's order matters there)
				if (in_format == 4 && out_format == 1 && !is_overlapping && !vd1_nv0039_scalar())
				{
					block2d_copy_stride4_to_1(dst, src, line_length, line_count, in_pitch, out_pitch);
					return;
				}
				block2d_copy_with_stride(dst, src, line_length, line_count, in_pitch, out_pitch, in_format, out_format);
				return;
			}

			if (!is_overlapping)
			{
				if (is_block_transfer)
				{
					std::memcpy(dst, src, read_length);
					return;
				}

				block2d_copy(dst, src, line_length, line_count, in_pitch, out_pitch);
				return;
			}

			if (is_block_transfer)
			{
				std::memmove(dst, src, read_length);
				return;
			}

			// Handle overlapping 2D range using double-copy to temp.
			std::vector<u8> temp(line_length * line_count);
			u8* buf = temp.data();

			block2d_copy(buf, src, line_length, line_count, in_pitch, line_length);
			block2d_copy(dst, buf, line_length, line_count, line_length, out_pitch);
		}

		// vanillad1: G-buffer byte on the GPU (X300.7): a held copy runs now (the pass did not take it)
		void vd1_release_held(thread* rthr)
		{
			const vd1_nv0039_args h = rthr->m_vd1_nv0039_held;
			rthr->m_vd1_nv0039_held.state = vd1_nv0039_args::none;
			if (h.state == vd1_nv0039_args::held)
			{
				rthr->m_vd1_nv0039_released++;
				do_buffer_notify(rthr, h);
			}
		}

		void buffer_notify(context* ctx, u32, u32 arg)
		{
			vd1_nv0039_args a{};
			a.in_pitch = REGS(ctx)->nv0039_input_pitch();
			a.out_pitch = REGS(ctx)->nv0039_output_pitch();
			a.line_length = REGS(ctx)->nv0039_line_length();  // Number of columns per row
			a.line_count = REGS(ctx)->nv0039_line_count();    // Number of rows to copy
			a.out_format = REGS(ctx)->nv0039_output_format();  // Column stride in bytes. Only the first byte is actually written to.
			a.in_format = REGS(ctx)->nv0039_input_format();    // Column stride in bytes. Only the first byte is actually read from.
			const u32 notify = arg;

			if (!validate_buffer_notify(a.line_length, a.line_count, a.in_format, a.out_format))
			{
				return;
			}

			rsx_log.trace("NV0039_BUFFER_NOTIFY: pitch(in=0x%x, out=0x%x), line(len=0x%x, cnt=0x%x), fmt(in=0x%x, out=0x%x), notify=0x%x",
				a.in_pitch, a.out_pitch, a.line_length, a.line_count, a.in_format, a.out_format, notify);

			a.src_offset = REGS(ctx)->nv0039_input_offset();
			a.src_dma = REGS(ctx)->nv0039_input_location();

			a.dst_offset = REGS(ctx)->nv0039_output_offset();
			a.dst_dma = REGS(ctx)->nv0039_output_location();

			if (const int vd1_mode = vd1_gbuf_gpu_mode()) [[unlikely]] // vanillad1: G-buffer byte on the GPU (X300.7)
			{
				vd1_release_held(RSX(ctx));

				// Only the 4 -> 1 byte gather out of local memory (a render target) into main memory, one byte a 4-byte pixel
				if (a.in_format == 4 && a.out_format == 1 && a.src_dma != a.dst_dma && a.in_pitch > 0 && a.out_pitch > 0)
				{
					a.src = get_address(a.src_offset, a.src_dma);
					a.dst = get_address(a.dst_offset, a.dst_dma);
					if (a.src >= rsx::constants::local_mem_base && a.dst < rsx::constants::local_mem_base)
					{
						a.state = vd1_mode == 1 ? vd1_nv0039_args::held : vd1_nv0039_args::check;
						RSX(ctx)->m_vd1_nv0039_held = a;
						static u32 s_logs = 0;
						if (s_logs++ < 4)
							rsx_log.notice("vanillad1: gbuf gpu: NV0039 0x%x (pitch %d) -> 0x%x (pitch %d), %u x %u, %s",
								a.src, a.in_pitch, a.dst, a.out_pitch, a.line_length, a.line_count, vd1_mode == 1 ? "held" : "copied (check)");
						if (vd1_mode == 1)
						{
							return;
						}
					}
				}
			}

			do_buffer_notify(RSX(ctx), a);
		}
	}
}
