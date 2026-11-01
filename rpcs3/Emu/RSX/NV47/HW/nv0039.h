// NV47 Buffer Management
#pragma once

#include "context.h"

namespace rsx
{
	namespace nv0039
	{
		void buffer_notify(context* ctx, u32 reg, u32 arg);

		// vanillad1: G-buffer byte on the GPU (X300.7): VANILLAD1_GBUF_GPU (0 off, 1 hold the copy, 2 check); a held copy runs
		int vd1_gbuf_gpu_mode();
		void vd1_release_held(thread* rthr);
	}
}
