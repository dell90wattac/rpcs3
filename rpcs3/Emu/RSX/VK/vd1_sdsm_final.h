#pragma once
// vanillad1: sdsm on the GPU (X301.2): the job's finaliser (LS 0x060cc-0x066a8), on the CPU in f32 as RPCS3 runs the SPU
// code: fused multiply-add where the SPU has one, exact reciprocals (RPCS3's frest + fi), the SPU's own log2 / exp2 /
// floor polynomials for the power-of-two quantisation. A port of finalise in re/notes/perf_2026-11-01/deep_re/sdsm_oracle.py
// (docs/findings/spu_sdsm.md). Header-only, no RPCS3 dependencies (re/notes/X301.2/final_test.cpp compiles it alone).
#include <cmath>
#include <cstdint>
#include <cstring>

namespace vd1_sdsm
{
	inline float bf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
	inline uint32_t fb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
	inline uint32_t be32(const uint8_t* p) { return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | uint32_t{p[3]}; }
	inline void put_be32(uint8_t* p, uint32_t v) { p[0] = static_cast<uint8_t>(v >> 24); p[1] = static_cast<uint8_t>(v >> 16); p[2] = static_cast<uint8_t>(v >> 8); p[3] = static_cast<uint8_t>(v); }
	inline float fma32(float a, float b, float c) { return std::fma(a, b, c); }
	inline float rcp(float x) { return 1.0f / x; }
	inline float min_first(float a, float b) { return (b < a) ? b : a; }   // Python's min(a, b): a unless b < a

	// LS 0x050a8
	inline float spu_log2(float x)
	{
		if (x == 0.0f)
		{
			return 0.0f;
		}
		const uint32_t bits = fb(x);
		float m = bf((bits & 0x007fffffu) | 0x3f000000u);
		const int32_t e = static_cast<int32_t>((bits & 0x7f800000u) >> 23);
		float ef;
		if (bf(0x3f3504f3u) > m)
		{
			m = m + m;
			ef = static_cast<float>(e - 127);
		}
		else
		{
			ef = static_cast<float>(e - 126);
		}
		const float f = m - 1.0f;
		const float g = m + 1.0f;
		const float rg = rcp(g);
		float t = f * rg;
		t = fma32(fma32(-g, rg, bf(0x3f800001u)), t, t);
		const float t2 = t * t;
		const float p = fma32(fma32(t2, bf(0x3e990214u), bf(0x3eccae10u)), t2, bf(0x3f2aaabfu));
		const float r2 = fma32(p, t2, -f);
		const float r8 = t * r2;
		const float hi = fma32(f, bf(0x3fb8aa3au), ef);
		const float lo = fma32(f, bf(0x32a5705fu), r8 * bf(0x3fb8aa3bu));
		return hi + lo;
	}

	// LS 0x052a8
	inline float spu_floor(float x)
	{
		if (!(std::fabs(x) < 8388608.0f))
		{
			return x;
		}
		const int32_t i = static_cast<int32_t>(x);   // toward zero
		const float fi = static_cast<float>(i);
		const float fm1 = static_cast<float>(i - 1);
		return (fi > x) ? fm1 : fi;
	}

	// LS 0x051b8
	inline float spu_exp2(float x)
	{
		if (x > bf(0x4300ffffu))
		{
			return bf(0x7fffffffu);
		}
		if (bf(0xc2fc0000u) > x)
		{
			return 0.0f;
		}
		const float bias = (x >= 0.0f) ? bf(0x3f7fffffu) : 0.0f;
		const int32_t n = static_cast<int32_t>(x + bias);   // cflts: toward zero
		const float fnf = static_cast<float>(n);
		float s = bf(static_cast<uint32_t>(n + 126) << 23);
		s = s + s;
		const float r = (fnf - x) * bf(0x3f317217u);
		const float p21 = fma32(r, bf(0xb9142e40u), bf(0x3aae4f6fu));
		const float p19 = fma32(r, bf(0xbe2aaa4fu), bf(0x3efffffdu));
		const float r2 = r * r;
		const float p16 = fma32(r, p21, bf(0xbc08026du));
		const float p15 = fma32(r, p19, bf(0xbf800000u));
		const float r4 = r2 * r2;
		const float p6 = fma32(r, p16, bf(0x3d2aa0e5u));
		const float p11 = fma32(r, p15, bf(0x3f800000u));
		const float p7 = fma32(r4, p6, p11);
		return p7 * s;
	}

	struct final_t
	{
		float result[3][4];    // per cascade {off_u, off_v, scale_u, scale_v}
		uint32_t cmd[3][19];   // per cascade the RSX subroutine: 0x00441efc, 4, M' (16 words), 0x00020000 (RET)
	};

	// P: the 0x2b0-B parameter block as in guest memory (big-endian); bounds: per cascade (min u, min v, max u, max v) after
	// the merge with (2, 2, -2, -2)
	inline final_t finalise(const uint8_t* P, const float bounds[3][4])
	{
		auto pw = [&](uint32_t off) { return be32(P + off); };
		auto pf = [&](uint32_t off) { return bf(be32(P + off)); };
		const uint32_t uvmask = pw(0x240), qmask = pw(0x244);
		final_t o{};

		for (uint32_t i = 0; i < 3; i++)
		{
			float M[16];
			for (uint32_t k = 0; k < 16; k++)
			{
				M[k] = pf(0x120 + 64 * i + 4 * k);
			}

			if ((uvmask >> i) & 1)
			{
				const float* bb = bounds[i];
				float u0 = fma32(bb[0], 0.5f, 0.5f);
				float v1 = fma32(bb[1], -0.5f, 0.5f);
				float v0 = fma32(bb[3], -0.5f, 0.5f);
				float u1 = fma32(bb[2], 0.5f, 0.5f);
				const float pad0 = pf(0x1f8 + 4 * i), pad1 = pf(0x204 + 4 * i);
				u0 = u0 - pad0;
				v1 = v1 + pad1;
				v0 = v0 - pad0;
				u1 = u1 + pad1;
				const float w = u1 - u0, h = v1 - v0;
				float iw = rcp(w);
				iw = fma32(fma32(-iw, w, 1.0f), iw, iw);
				float ih = rcp(h);
				ih = fma32(fma32(-ih, h, 1.0f), ih, ih);
				const float nu0 = (-u0) * iw, nv0 = (-v0) * ih;
				float su = min_first(iw, pf(0x1e0 + 4 * i));
				float sv = min_first(ih, pf(0x1ec + 4 * i));
				const bool quant = (qmask >> i) & 1;
				if (quant)
				{
					const float qu = spu_exp2(spu_floor(spu_log2(su)));
					su = (1.0f > qu) ? 1.0f : qu;
					const float qv = spu_exp2(spu_floor(spu_log2(sv)));
					sv = (1.0f > qv) ? 1.0f : qv;
				}
				const float ww = rcp(iw);
				float t = su * ww;
				const float suw = fma32(fma32(-t, iw, su), ww, t);
				const float hh2 = rcp(ih);
				t = sv * hh2;
				const float svh = fma32(fma32(-t, ih, sv), hh2, t);
				float off_u = fma32(suw, nu0 - 0.5f, 0.5f);
				float off_v = fma32(svh, nv0 - 0.5f, 0.5f);
				if (quant)
				{
					const float nu = static_cast<float>(pw(0x228 + 4 * i)), nv = static_cast<float>(pw(0x234 + 4 * i));
					const float fu = spu_floor(off_u * nu);
					const float inu = rcp(nu);
					t = fu * inu;
					off_u = fma32(fma32(-t, nu, fu), inu, t);
					const float fv = spu_floor(off_v * nv);
					const float inv = rcp(nv);
					t = fv * inv;
					off_v = fma32(fma32(-t, nv, fv), inv, t);
				}
				o.result[i][0] = off_u;
				o.result[i][1] = off_v;
				o.result[i][2] = su;
				o.result[i][3] = sv;

				const float m03 = M[3], m13 = M[7];
				for (uint32_t k = 0; k < 4; k++)
				{
					M[k] = M[k] * su;
					M[4 + k] = M[4 + k] * sv;
				}
				M[3] = fma32(su, m03, ((off_u + off_u) + su) - 1.0f);
				M[7] = ((off_v * -2.0f) - sv) + fma32(sv, m13, 1.0f);
			}
			else
			{
				o.result[i][0] = 0.0f;
				o.result[i][1] = 0.0f;
				o.result[i][2] = 1.0f;
				o.result[i][3] = 1.0f;
			}

			o.cmd[i][0] = 0x00441efcu;
			o.cmd[i][1] = 4u;
			for (uint32_t k = 0; k < 16; k++)
			{
				o.cmd[i][2 + k] = fb(M[k]);
			}
			o.cmd[i][18] = 0x00020000u;
		}
		return o;
	}
}
