#pragma once

#include "util/types.hpp"

// vanillad1: live switches (X302): one shared-memory block of u32 flags, flipped by tools/live.py while the game runs, so a
// 1 % gain can be measured on one boot (boot-to-boot spread is ~7 %). VANILLAD1_LIVE=<name> maps Local\vanillad1_live_<name>;
// without it the flags live in static storage. VANILLAD1_LIVE_SET=E24=1,E25=1 sets boot values (launch.ps1 keeps the
// defaults there). A read is one load and no branch, so the sites can sit on hot paths. Unset = 0 = stock behaviour.
namespace vd1
{
#define VD1_LIVE_FLAGS(X) \
	X(E1) X(E24) X(E25) X(E27) X(E13) X(E3A) X(E3B) X(E14) X(E18) X(E22) \
	X(E7) X(E6) X(E9) X(E10) X(E20) X(E21) X(E16) X(E23) X(E26) X(E12) \
	X(B1) X(A1) X(B3) X(LABEL) X(B7) X(UPLOADS)

	enum class lf : u32
	{
#define X(name) name,
		VD1_LIVE_FLAGS(X)
#undef X
		count
	};

	inline constexpr u32 live_max_flags = 64;
	static_assert(static_cast<u32>(lf::count) <= live_max_flags);

	// Never null: points at static storage until the block is mapped (before main)
	extern volatile u32* g_live;

	inline u32 live(lf flag)
	{
		return g_live[static_cast<u32>(flag)];
	}
}
