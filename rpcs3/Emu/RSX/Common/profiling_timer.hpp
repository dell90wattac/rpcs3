#pragma once

#include <util/types.hpp>
#include "Emu/Cell/timers.hpp"

namespace rsx
{
	struct profiling_timer
	{
		bool enabled = false;
		u64 last;

		profiling_timer() = default;

		void start()
		{
			if (enabled) [[unlikely]]
			{
				last = get_system_time();
			}
		}

		s64 duration()
		{
			if (!enabled) [[likely]]
			{
				return 0ll;
			}

			auto old = last;
			last = get_system_time();
			return static_cast<s64>(last - old);
		}
	};

	// vanillad1: RSX timers (P3): adds the scope's host time (us) to a frame stat when on
	struct scoped_stat_timer
	{
		s64* dst;
		u64 t0;

		scoped_stat_timer(bool on, s64& stat)
			: dst(on ? &stat : nullptr), t0(on ? get_system_time() : 0)
		{
		}

		~scoped_stat_timer()
		{
			if (dst) [[unlikely]]
			{
				*dst += static_cast<s64>(get_system_time() - t0);
			}
		}
	};
}
