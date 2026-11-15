R"(
#version 450
// vanillad1 X301.2: Destiny's sdsm SPU job (spu_sdsm_job, Sample Distribution Shadow Maps) as one compute pass.
// A port of re/notes/perf_2026-11-01/deep_re/sdsm_oracle.py (docs/findings/spu_sdsm.md). One workgroup = one 32x24 tile,
// one invocation = one 2x2 quad (= one half-res pixel). One storage buffer holds everything (32-bit words); the push
// block gives the offsets. Bit-exact intent: f32 with fused multiply-add where the SPU has one, exact reciprocals
// (RPCS3's frest/fi) by one Newton step on the GPU's own, integer ops for the f16 codes.
#define VD1_VK %vk
layout(local_size_x = 16, local_size_y = 12, local_size_z = 1) in;
#if VD1_VK
layout(set = 0, binding = 0, std430) buffer ssbo0 { uint data[]; };
layout(push_constant) uniform push_block {
#else
layout(std430, binding = 0) buffer ssbo0 { uint data[]; };
layout(std140, binding = 0) uniform push_block {
#endif
	uint o_in;     // the input surface, w x h words, host values: (f32bits(z) >> 7) << 8 | stencil (RPCS3's z24s8_float)
	uint o_lin;    // out: the in-place linear depth, w x h words, host values (the stencil byte kept)
	uint o_ha;     // out: half_a, (w/2) x (h/2), guest byte order (the 2x2 minimum raw word)
	uint o_hb;     // out: half_b, (w/2) x (h/2), guest byte order (f16(lin(max)) << 16 | f16(lin(min)))
	uint o_q;      // out: quarter, (w/4) x (h/4), guest byte order (the 2x2 minimum of half_a)
	uint o_b;      // in / out: 12 ordered words, cascade i at 4i: min u, min v, max u, max v (the host presets (2, 2, -2, -2))
	uint o_k;      // constants (f32 bits): a, b, 1/(w/2), 1/(h/2), 0.5/(w/2), 0.5/(h/2), w/2, h/2
	uint o_p;      // the slot's 0x2b0-B parameter block, host words
	uint w;
	uint h;
	uint flags;    // unused
	uint pad0;
};

shared uint s_ha[12][16];
shared uint s_b[12];

float K(uint i) { return uintBitsToFloat(data[o_k + i]); }
float P(uint off) { return uintBitsToFloat(data[o_p + (off >> 2)]); }

// 1/x correctly rounded (RPCS3's frest + fi give the exact reciprocal): the GPU's estimate, one Newton step with fma
float rcp_exact(float x)
{
	precise float r = 1.0 / x;
	precise float e = fma(-x, r, 1.0);
	precise float o = fma(e, r, r);
	return o;
}

// LS 0x06810: 1 / (as_f32((raw >> 1) & ~0x7f) * a + b)
float lin_depth(uint raw, float a, float b)
{
	const float v = uintBitsToFloat((raw >> 1) & 0xffffff80u);
	precise float den = fma(v, a, b);
	return rcp_exact(den);
}

// LS 0x069b4-0x06a80: IEEE half, truncated, by integer ops; 0 below 2^-15, 0x7bff from 65536 up
uint code16(float x)
{
	const uint bits = floatBitsToUint(x);
	const uint hix = bits >> 16;
	const uint hi = ((bits - 0x38000000u) >> 16) & 0xffffu;
	const uint rot = ((hi << 3) | (hi >> 13)) & 0xffffu;
	const uint c = hix < 0x4780u ? rot : 0x7bffu;
	return hix < 0x3800u ? 0u : c;
}

uint bswap(uint v)
{
	return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

// order-preserving float <-> uint for atomicMin / atomicMax
uint f2o(float f)
{
	const uint u = floatBitsToUint(f);
	return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

void main()
{
	const uint lx = gl_LocalInvocationID.x, ly = gl_LocalInvocationID.y;
	const uint gx = gl_WorkGroupID.x, gy = gl_WorkGroupID.y;
	const uint li = ly * 16u + lx;
	const uint hx = gx * 16u + lx, hy = gy * 12u + ly;   // the half-res pixel
	const uint hw = w >> 1;

	if (li < 12u)
	{
		s_b[li] = ((li & 3u) < 2u) ? 0xffffffffu : 0u;    // min lanes start high, max lanes low (identities)
	}

	const float a = K(0), b = K(1);

	// The quad's four raw words, the in-place write (stencil byte kept)
	const uint x0 = hx * 2u, y0 = hy * 2u;
	const uint i00 = y0 * w + x0, i10 = i00 + 1u, i01 = i00 + w, i11 = i01 + 1u;
	const uint r00 = data[o_in + i00], r10 = data[o_in + i10], r01 = data[o_in + i01], r11 = data[o_in + i11];
	data[o_lin + i00] = (floatBitsToUint(lin_depth(r00, a, b)) & 0xffffff00u) | (r00 & 0xffu);
	data[o_lin + i10] = (floatBitsToUint(lin_depth(r10, a, b)) & 0xffffff00u) | (r10 & 0xffu);
	data[o_lin + i01] = (floatBitsToUint(lin_depth(r01, a, b)) & 0xffffff00u) | (r01 & 0xffu);
	data[o_lin + i11] = (floatBitsToUint(lin_depth(r11, a, b)) & 0xffffff00u) | (r11 & 0xffu);

	// LS 0x068d8: the 2x2 minimum and maximum of the raw words (unsigned)
	const uint mn = min(min(r00, r10), min(r01, r11));
	const uint mx = max(max(r00, r10), max(r01, r11));
	const float d = lin_depth(mn, a, b);
	const float dmx = lin_depth(mx, a, b);
	const uint hidx = hy * hw + hx;
	data[o_ha + hidx] = bswap(mn);
	data[o_hb + hidx] = bswap((code16(dmx) << 16) | code16(d));
	s_ha[ly][lx] = mn;

	// LS 0x06b88: the bounds over the half-res lin(min) depth d
	const float inv_hw = K(2), inv_hh = K(3), inv_w = K(4), inv_h = K(5), fhw = K(6), fhh = K(7);
	const float col = float(gx * 16u), row = float(gy * 12u);
	precise float tx0 = col * inv_hw;
	precise float tx = fma(fma(-tx0, fhw, col), inv_hw, tx0);
	precise float ty0 = row * inv_hh;
	precise float ty = fma(fma(-ty0, fhh, row), inv_hh, ty0);
	const uint q = lx >> 2, k = lx & 3u;
	precise float txw = tx + inv_w;
	precise float kw = float(k) * inv_hw;
	precise float xbase = txw + kw;
	precise float xu = fma(float(q * 4u), inv_hw, xbase);
	precise float tyh = ty + inv_h;
	precise float yu = fma(float(ly), inv_hh, tyh);
	precise float xn = fma(2.0, xu, -1.0);
	precise float y2 = 2.0 * yu;
	precise float yn = 1.0 - y2;
	precise float rx = fma(xn, P(0x30), P(0x60));          // M_A[0][0], M_A[3][0]
	precise float ry = fma(yn, P(0x44), P(0x64));          // M_A[1][1], M_A[3][1]
	precise float vx = rx * d;
	precise float vy = ry * d;
	const float vz = -d;

	barrier();   // s_b initialised, s_ha written

	for (uint c = 0u; c < 3u; c++)
	{
		precise float t = fma(d, P(0x210 + 4u * c), P(0x21c + 4u * c));
		if (abs(t) > 1.0)
		{
			continue;
		}
		const uint pu = 0xc0u + 16u * c, pv = 0xf0u + 16u * c;
		precise float u = fma(vx, P(pu), fma(vy, P(pu + 4u), fma(vz, P(pu + 8u), P(pu + 12u))));
		precise float v = fma(vx, P(pv), fma(vy, P(pv + 4u), fma(vz, P(pv + 8u), P(pv + 12u))));
		atomicMin(s_b[4u * c + 0u], f2o(u));
		atomicMin(s_b[4u * c + 1u], f2o(v));
		atomicMax(s_b[4u * c + 2u], f2o(u));
		atomicMax(s_b[4u * c + 3u], f2o(v));
	}

	// LS 0x06aa0: quarter = the 2x2 minimum of half_a
	if (lx < 8u && ly < 6u)
	{
		const uint m = min(min(s_ha[2u * ly][2u * lx], s_ha[2u * ly][2u * lx + 1u]), min(s_ha[2u * ly + 1u][2u * lx], s_ha[2u * ly + 1u][2u * lx + 1u]));
		data[o_q + (gy * 6u + ly) * (w >> 2) + gx * 8u + lx] = bswap(m);
	}

	barrier();   // the tile's bounds complete

	if (li < 12u)
	{
		const uint s = s_b[li];
		if ((li & 3u) < 2u)
		{
			if (s != 0xffffffffu) atomicMin(data[o_b + li], s);
		}
		else
		{
			if (s != 0u) atomicMax(data[o_b + li], s);
		}
	}
}
)"
