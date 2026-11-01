#include "stdafx.h"
#include "vd1_live.h"

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#endif

// vanillad1: live switches (X302): the block tools/live.py reads. Layout: +0 magic, +4 version, +8 count, +12 pid,
// +64 u32 flags[64], +1024 char names[64][16] (written here, so the tool can't mismatch a build).
namespace vd1
{
	static constexpr u32 s_magic = 0x4c314456; // "VD1L"
	static constexpr u32 s_flags_offset = 64;
	static constexpr u32 s_names_offset = 1024;
	static constexpr u32 s_name_size = 16;
	static constexpr u32 s_block_size = s_names_offset + live_max_flags * s_name_size;

	static volatile u32 s_storage[live_max_flags]{};

	constinit volatile u32* g_live = s_storage;

	static const char* const s_names[]
	{
#define X(name) #name,
		VD1_LIVE_FLAGS(X)
#undef X
	};

	static void apply_boot_values(volatile u32* flags)
	{
		// The range-gated switches stay on when their env is set, as before they became switchable
		if (const char* env = std::getenv("VANILLAD1_FAULT_FAST"); env && *env) flags[static_cast<u32>(lf::A1)] = 1;
		if (const char* env = std::getenv("VANILLAD1_WCB_ONLY"); env && *env) flags[static_cast<u32>(lf::B3)] = 1;

		const char* env = std::getenv("VANILLAD1_LIVE_SET");
		if (!env) return;

		const std::string list = env;
		for (usz pos = 0; pos < list.size();)
		{
			usz end = list.find(',', pos);
			if (end == std::string::npos) end = list.size();
			const std::string item = list.substr(pos, end - pos);
			pos = end + 1;

			const usz eq = item.find('=');
			if (eq == std::string::npos) continue;
			const std::string key = item.substr(0, eq);
			for (u32 i = 0; i < static_cast<u32>(lf::count); i++)
			{
				if (key == s_names[i]) flags[i] = static_cast<u32>(std::strtoul(item.c_str() + eq + 1, nullptr, 10));
			}
		}
	}

	// Runs before main; g_live is constant-initialised, so anything reading earlier sees static zeroes
	static const bool s_live_ready = []()
	{
#ifdef _WIN32
		if (const char* env = std::getenv("VANILLAD1_LIVE"); env && *env)
		{
			const std::string name = std::string("Local\\vanillad1_live_") + env;
			if (const HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, s_block_size, name.c_str()))
			{
				if (auto* base = static_cast<char*>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, s_block_size)))
				{
					std::memset(base, 0, s_block_size);
					auto* head = reinterpret_cast<u32*>(base);
					head[1] = 1;
					head[2] = static_cast<u32>(lf::count);
					head[3] = GetCurrentProcessId();
					for (u32 i = 0; i < static_cast<u32>(lf::count); i++)
					{
						std::strncpy(base + s_names_offset + i * s_name_size, s_names[i], s_name_size - 1);
					}
					auto* flags = reinterpret_cast<volatile u32*>(base + s_flags_offset);
					apply_boot_values(flags);
					g_live = flags;
					head[0] = s_magic;
					return true;
				}
			}
		}
#endif
		apply_boot_values(s_storage);
		return false;
	}();
}
