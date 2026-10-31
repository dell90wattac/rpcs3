R"(
#version 450
// vanillad1 X300.2: Destiny's deferred lighting SPU job (spu_deferred_lighting_job) as one compute pass.
// One workgroup = one 32x24 tile, one invocation = one pixel. Port of re/notes/X300.2/oracle.py.
// One storage buffer holds everything (32-bit words, host order); the push block gives the offsets.
#define VD1_VK %vk
layout(local_size_x = 32, local_size_y = 24, local_size_z = 1) in;
#if VD1_VK
layout(set = 0, binding = 0, std430) buffer ssbo0 { uint data[]; };
layout(push_constant) uniform push_block {
#else
layout(std430, binding = 0) buffer ssbo0 { uint data[]; };
layout(std140, binding = 0) uniform push_block {
#endif
	uint o_z;      // linear depth, f32 bits, w x h
	uint o_g;      // G-buffer word A<<24|R<<16|G<<8|B (in), the 2 x FP16 write-back (out, lit tiles)
	uint o_m;      // the 8-bit buffer, 4 bytes a word, little-endian, pitch w
	uint o_l;      // the light target, 2 words a texel (packHalf2x16(R,G), packHalf2x16(B,A)), in place
	uint o_p;      // the slot's 0x280-B parameter block
	uint o_lt;     // the light records, 60 words each
	uint o_t;      // the static 0x9350-B table (TAB, KPAIRS at word 8960, CURVE at word 8980)
	uint n_lights;
	uint flags;    // bit 0: write the light target, bit 1: write the G-buffer pair
	uint o_lut;    // the 8-bit buffer's lookup: 256 x (c = s^P+0x1cc, d = s^P+0x16c), s = max((i/255)^2, 1e-4)
	uint pad1, pad2;
};

shared uint s_zmin, s_zmax;
shared uint s_vis[8];
shared uint s_qa[192];
shared uint s_q[3][192];

float P(uint off) { return uintBitsToFloat(data[o_p + (off >> 2)]); }
vec3 P3(uint off) { return vec3(P(off), P(off + 4u), P(off + 8u)); }
float LF(uint l, uint k) { return uintBitsToFloat(data[o_lt + l * 60u + k]); }
vec3 L3(uint l, uint k) { return vec3(LF(l, k), LF(l, k + 1u), LF(l, k + 2u)); }
float TB(uint i) { return uintBitsToFloat(data[o_t + i]); }
float sat(float x) { return clamp(x, 0.0, 1.0); }
float dot3(vec3 a, vec3 b) { return a.z * b.z + (a.y * b.y + a.x * b.x); }

const vec4 LUT8[8] = vec4[8](
	vec4(-0.0, 0.037037, -0.0, 0.0),
	vec4(0.13270199, 0.006273, 0.002531, -7.1e-05),
	vec4(0.28589299, -0.002873, 0.001772, -2.0e-05),
	vec4(0.42568401, -0.000989, 0.000493, 2.6e-05),
	vec4(0.56427503, 0.000201, -2.0e-06, 3.3e-05),
	vec4(0.70318103, 0.000598, -0.000143, 2.7e-05),
	vec4(0.84124100, 0.000556, -0.00012, 1.9e-05),
	vec4(0.98002899, 0.000319, -7.8e-05, 1.2e-05));
vec4 lut8(float k)
{
	float x = 7.0 - k * 7.0;
	int i = int(clamp(floor(x), 0.0, 7.0));
	float fr = x - float(i);
	return LUT8[min(i + 1, 7)] * fr + (1.0 - fr) * LUT8[i];
}

// 0x0a780 / 0x09588's shared terms: H = normalize(L + V); Cv over sqrt(1 - N.H); Tb from the table; inv = 1/(|L+V|^2 + 0.3)
void brdf(vec3 N, vec3 V, vec3 L, uint mat, float a, float b, out float Cv, out float Tb, out float inv)
{
	vec3 h = L + V;
	float l2 = dot3(h, h);
	float ndh = dot3(N, h * inversesqrt(l2));
	float u = sat(sqrt(1.0 - sat(ndh)));
	float u31 = u * 31.0; uint i = uint(u31); float fr = u31 - float(i);
	uint ci = mat * 32u + i;
	Cv = TB(8980u + min(ci + 1u, 319u)) * fr + TB(8980u + ci) * (1.0 - fr);
	uint ti = mat * 896u + i * 28u + uint(15.0 - a * 15.0);
	Tb = (TB(min(ti + 28u, 8959u)) * fr + TB(ti) * (1.0 - fr)) * b;
	inv = 1.0 / (l2 + 0.3);
}

float plane(float nx, float ny, float tt, float nn, vec3 p, float r)
{
	float inv = 1.0 / sqrt(nn + tt * tt);
	return (nx * inv) * p.x + (ny * inv) * p.y + (tt * inv) * p.z + r;
}
bool out_of(float nx, float ny, float tt, float nn, vec3 p0, float r0, vec3 p1, float r1, bool two)
{
	return plane(nx, ny, tt, nn, p0, r0) < 0.0 && (!two || plane(nx, ny, tt, nn, p1, r1) < 0.0);
}
// 0x0a150: type 0 a sphere, 2-5 a capsule, 8-9 a cone (apex radius 0), against the tile's 8 planes and depth range
bool visible(uint l, uint t, float X0, float X1, float Y0, float Y1, float zmin, float zmax)
{
	vec3 p0 = L3(l, 0u); float r = LF(l, 3u); vec3 p1 = L3(l, 5u);
	bool two = t >= 2u, cone = t >= 8u;
	float r0 = cone ? 0.0 : r;
	if (out_of(1.0, 0.0, X0, 1.0, p0, r0, p1, r, two) || out_of(-1.0, 0.0, -X1, 1.0, p0, r0, p1, r, two) ||
		out_of(0.0, 1.0, Y0, 1.0, p0, r0, p1, r, two) || out_of(0.0, -1.0, -Y1, 1.0, p0, r0, p1, r, two) ||
		out_of(1.0, 1.0, X0 + Y0, 2.0, p0, r0, p1, r, two) || out_of(-1.0, -1.0, -X1 - Y1, 2.0, p0, r0, p1, r, two) ||
		out_of(1.0, -1.0, X0 - Y1, 2.0, p0, r0, p1, r, two) || out_of(-1.0, 1.0, -X1 + Y0, 2.0, p0, r0, p1, r, two))
		return false;
	float zfar = two ? max(p1.z + r, cone ? p0.z : p0.z + r) : p0.z + r;
	float znear = two ? min(p1.z - r, cone ? p0.z : p0.z - r) : p0.z - r;
	return !(-zmax > zfar) && !(znear > -zmin);
}

// 0x09ad8: the ambient's front lobe around the light's axis A.xyz (wrap A.w), colour B.xyz, specular weight B.w, times att
void fill_lobe(vec3 N, vec3 V, uint mat, float a, float b, float att, uint l, uint o, float kx, float ky, inout vec4 acc)
{
	const vec3 A3 = L3(l, o), B3 = L3(l, o + 4u);
	const float w = LF(l, o + 3u);
	const vec4 co = lut8(w);
	const float ndl = dot3(A3, N);
	const float front = sat(ndl * w + (0.5 - w * 0.5));
	acc.rgb += ((B3 * (front * front)) * att) * kx;
	const vec3 h = A3 + V;
	const float u = sat(sqrt(sat(1.0 - dot3(N, h * inversesqrt(dot3(h, h))))));
	const float u31 = u * 31.0; const uint i = uint(u31); const float fr = u31 - float(i);
	const float rr = 15.0 - a * 15.0;
	const float poly = (rr * rr * rr) * co.w + ((rr * rr) * co.z + (rr * co.y + co.x));
	const uint idx = mat * 896u + i * 28u + uint(sat(poly) * 27.0);
	const float T = TB(min(idx + 28u, 8959u)) * fr + TB(idx) * (1.0 - fr);
	acc.a += ((LF(l, o + 7u) * ((T * b) * sat(ndl))) * att) * ky;
}

bool handled(uint t) { return t <= 5u || t == 8u || t == 9u; }

void main()
{
	const uint w = floatBitsToUint(P(0x64u)), hh = floatBitsToUint(P(0x68u));
	const uvec2 g = gl_GlobalInvocationID.xy;
	const uint li = gl_LocalInvocationIndex, lane = g.x & 3u;
	const uint qi = gl_LocalInvocationID.y * 8u + (gl_LocalInvocationID.x >> 2);
	const uint pix = g.y * w + g.x;
	if (li == 0u) { s_zmin = 0xffffffffu; s_zmax = 0u; }
	if (li < 8u) s_vis[li] = 0u;
	if (li < 192u) { s_qa[li] = 0u; s_q[0][li] = 0u; s_q[1][li] = 0u; s_q[2][li] = 0u; }
	barrier();
	const uint zw = data[o_z + pix];
	atomicMin(s_zmin, zw & 0xffffff00u);
	atomicMax(s_zmax, zw & 0xffffff00u);
	barrier();
	const float zmin = uintBitsToFloat(s_zmin), zmax = uintBitsToFloat(s_zmax);
	if (!(55188.0 > zmin)) return;   // 0x0571c: the job leaves a far tile alone (the whole group returns)

	// ---- the tile's light list (0x0a150), one light per invocation
	if (li < n_lights)
	{
		const uint t = data[o_lt + li * 60u + 51u] >> 24;
		const float sx = P(0x80u), sy = P(0x94u);
		const float c0 = float(gl_WorkGroupID.x * 32u), r0 = float(gl_WorkGroupID.y * 24u);
		const float X0 = ((c0 / float(w)) * 2.0 - 1.0) / sx, X1 = (((c0 + 32.0) / float(w)) * 2.0 - 1.0) / sx;
		const float Y0 = -(((r0 + 24.0) / float(hh)) * 2.0 - 1.0) / sy, Y1 = -((r0 / float(hh)) * 2.0 - 1.0) / sy;
		if (handled(t) && visible(li, t, X0, X1, Y0, Y1, zmin, zmax)) atomicOr(s_vis[li >> 5], 1u << (li & 31u));
	}

	// ---- decode (0x07390)
	const float depth = uintBitsToFloat(zw);
	const float u = float(g.x) / float(w), v = float(g.y) / float(hh);
	const vec3 cam = P3(0x110u);
	const vec3 ray = (P3(0xc0u) * (1.0 - u) + P3(0xd0u) * u) * (1.0 - v) + (P3(0xe0u) * (1.0 - u) + P3(0xf0u) * u) * v;
	const vec3 pos = cam + depth * ray;
	const uint gw = data[o_g + pix];
	const vec3 rawn = vec3((gw >> 16) & 255u, (gw >> 8) & 255u, gw & 255u) * (2.0 / 255.0) - 1.0;
	const float ln = length(rawn);
	const vec3 N = rawn / ln;
	const float a = sat(4.0 * ln - 3.0);
	const vec3 dv = cam - pos;
	const float l2xy = dv.y * dv.y + dv.x * dv.x;
	const vec3 V = dv * inversesqrt(dv.z * dv.z + l2xy);
	const float f0 = sat(sqrt(l2xy) * P(0x19cu) + P(0x1acu));
	const uint mat = min(((gw >> 24) * 0xaabu) >> 16, 9u);
	const float sb = (15.0 - a * 15.0) * TB(8961u + mat * 2u) + TB(8960u + mat * 2u);
	const float b = sb * sb;
	const uint mk = (data[o_m + (pix >> 2)] >> (8u * (pix & 3u))) & 255u;
	const float c = uintBitsToFloat(data[o_lut + mk * 2u]), d = uintBitsToFloat(data[o_lut + mk * 2u + 1u]);

	// ---- hemisphere ambient (0x08410) and the sun pair's quad gate
	const vec3 D = P3(0x1c0u);
	const float kF = P(0x17cu), kB = P(0x18cu);
	const float ndl0 = dot3(D, N);
	const bool neg = (floatBitsToUint(ndl0) >> 31) != 0u;
	const vec3 S = P3(0x160u);
	const float nds = dot3(N, S);
	atomicOr(s_qa[qi], (neg ? (1u << lane) : 0u) | (nds > 0.0 ? 16u : 0u));
	barrier();
	const uint qa = s_qa[qi];
	const float front = sat(ndl0 * kF + (0.5 - kF * 0.5)), back = sat((0.5 - kB * 0.5) - ndl0 * kB);
	const vec3 CF = P3(0x190u) * f0 + P3(0x170u), CB = P3(0x1a0u) * f0 + P3(0x180u);
	vec4 acc;
	acc.rgb = CB * ((back * back) * c) + CF * ((front * front) * c);
	{
		const vec3 Hh = normalize(D * (neg ? -1.0 : 1.0) + V);
		const float uu = sat(sqrt(sat(1.0 - dot3(N, Hh))));
		const float u31 = uu * 31.0; const uint i = uint(u31); const float fr = u31 - float(i);
		const vec4 cF = lut8(kF), cB = lut8(kB);
		const vec4 co = vec4((qa & 1u) != 0u ? cB.x : cF.x, (qa & 2u) != 0u ? cB.y : cF.y,
			(qa & 4u) != 0u ? cB.z : cF.z, (qa & 8u) != 0u ? cB.w : cF.w);   // lane k of the quad supplies coefficient k
		const float rr = 15.0 - a * 15.0;
		const float poly = (rr * rr * rr) * co.w + ((rr * rr) * co.z + (rr * co.y + co.x));
		const uint idx = mat * 896u + i * 28u + uint(sat(poly) * 27.0);
		const float T = TB(min(idx + 28u, 8959u)) * fr + TB(idx) * (1.0 - fr);
		const float spec = neg ? P(0x1bcu) * f0 + P(0x1b4u) : P(0x1b8u) * f0 + P(0x1b0u);
		acc.a = spec * (((T * b) * abs(ndl0)) * c);
	}
	float hi = 0.0, lo = 0.0;
	if ((qa & 16u) != 0u)
	{
		float Cv, Tb, inv;
		brdf(N, V, S, mat, a, b, Cv, Tb, inv);
		const float sn = sat(nds);
		hi = (sn * Cv) * d; lo = ((inv * sn) * Tb) * d;
	}

	// ---- per light (type functions via LS 0xdd40, the N.L gate 0x0a908, accumulate 0x09588)
	const bool bit5 = ((zw >> 5) & 1u) != 0u;
	const uint dbg = (flags >> 8) & 15u;   // test bench only: 1 ambient alone, 2 (N, a) * 4, 3 pos / 64
	if (dbg == 2u) acc = vec4(N, a) * 4.0;
	if (dbg == 4u) acc = vec4(hi, lo, d, c / 3.0) * 4.0;
	if (dbg == 3u) acc = vec4(pos / 64.0, depth / 64.0) * vec4(4.0, 4.0, 4.0, 4.0 / 3.0);
	uint k = 0u;
	for (uint l = 0u; l < (dbg == 0u ? n_lights : 0u); ++l)
	{
		if ((s_vis[l >> 5] & (1u << (l & 31u))) == 0u) continue;
		const uint tw = data[o_lt + l * 60u + 51u];
		const uint t = tw >> 24, ce = (tw >> 8) & 255u, cf = tw & 255u;
		const vec3 LP = L3(l, 8u);
		vec3 dl;
		if (t >= 2u && t <= 5u)
		{
			const vec3 Dn = L3(l, 52u); const float hw = abs(LF(l, 55u));
			const float s0 = ((0.0 - LP.x * Dn.x) - LP.y * Dn.y) - LP.z * Dn.z;
			const float tp = Dn.z * pos.z + (Dn.y * pos.y + (Dn.x * pos.x + s0));
			dl = Dn * clamp(tp, -hw, hw) + LP - pos;
		}
		else dl = LP - pos;
		const float d2 = dot3(dl, dl), rs = inversesqrt(d2);
		const vec3 Lv = dl * rs;
		const float fall = sat((rs * d2) * LF(l, 11u) + LF(l, 15u));
		float att = fall * fall;
		if (t == 4u || t == 5u || t == 8u || t == 9u)
		{
			const vec3 E = t <= 5u ? L3(l, 56u) : L3(l, 52u);
			att *= sat((((0.0 - E.x * Lv.x) - E.y * Lv.y) - E.z * Lv.z) * LF(l, 36u) + LF(l, 37u));
		}
		if (t == 1u || t == 3u || t == 5u || t == 9u)   // the odd types: the even type's shape and two clip planes
		{
			const float p1 = pos.z * LF(l, 42u) + (pos.y * LF(l, 41u) + (pos.x * LF(l, 40u) + LF(l, 43u)));
			const float p2 = pos.z * LF(l, 46u) + (pos.y * LF(l, 45u) + (pos.x * LF(l, 44u) + LF(l, 47u)));
			if (!(p1 > 0.0 && p2 > 0.0)) att = 0.0;
		}
		const bool lo_m = (((ce & 15u) >> (3u - lane)) & 1u) != 0u, hi_m = (((ce >> 4) >> (3u - lane)) & 1u) != 0u;
		if (!((bit5 != lo_m) || hi_m)) att = 0.0;
		const float ndl = dot3(N, Lv);
		const uint bits = (att > 0.0 ? 1u : 0u) | (ndl > 0.0 ? 2u : 0u);
		const uint q = k % 3u; k++;
		if (bits != 0u) atomicOr(s_q[q][qi], bits);
		barrier();
		const uint qb = s_q[q][qi];
		if (li < 192u) s_q[(q + 2u) % 3u][li] = 0u;   // read last round, written again two rounds on
		if ((qb & 1u) == 0u) continue;
		const float kx = LF(l, 48u), ky = LF(l, 49u);
		// +0xcf bit 0 (0x05bd4): no N.L gate, back-facing quads still get the +0x30 term
		if ((qb & 2u) != 0u || (cf & 1u) != 0u)
		{
			float Cv, Tb, inv;
			brdf(N, V, Lv, mat, a, b, Cv, Tb, inv);
			const float sn = sat(ndl);
			acc.rgb += ((sn * Cv * L3(l, 16u) + L3(l, 12u)) * att) * kx;
			acc.a += ((((inv * sn) * Tb) * LF(l, 19u)) * att) * ky;
		}
		// +0xcd bit 0 (0x05c94) / bit 1 (0x05ce0): a fill lobe (0x09ad8) around +0x50 / +0x70, colour +0x60 / +0x80
		const uint cd = (tw >> 16) & 255u;
		if ((cd & 1u) != 0u) fill_lobe(N, V, mat, a, b, att, l, 20u, kx, ky, acc);
		if ((cd & 2u) != 0u) fill_lobe(N, V, mat, a, b, att, l, 28u, kx, ky, acc);
	}

	if ((flags & 1u) != 0u)
	{
		data[o_l + pix * 2u] = packHalf2x16(vec2(acc.r * 0.25, acc.g * 0.25));
		data[o_l + pix * 2u + 1u] = packHalf2x16(vec2(acc.b * 0.25, acc.a * 0.75));
	}
	if ((flags & 2u) != 0u) data[o_g + pix] = packHalf2x16(vec2(lo, hi));
}
)"
