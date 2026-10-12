// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Debug helper: sample hot HLE functions and log a guest backtrace.
//
// Guest code runs natively on host threads (the guest stack *is* the host stack), so a host
// backtrace captured inside an HLE function contains the guest call chain. Addresses can be
// resolved to modules with the "Loading module <name> to <addr>" lines in the emulator log.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"

#if defined(_WIN32)
#include <windows.h>
extern "C" NTSYSAPI USHORT NTAPI RtlCaptureStackBackTrace(ULONG FramesToSkip, ULONG FramesToCapture,
                                                          PVOID* BackTrace, PULONG BackTraceHash);
#endif

namespace Common::Profiler {

constexpr int MaxFrames = 20;
constexpr int MaxSites = 16;

struct Site {
    std::atomic<const char*> name{nullptr};
    std::atomic<u32> calls{0};
    std::atomic<u32> logged{0};
};

inline Site* GetSite(const char* name) {
    static Site sites[MaxSites];
    for (int i = 0; i < MaxSites; i++) {
        const char* cur = sites[i].name.load();
        if (cur == name) {
            return &sites[i];
        }
        if (cur == nullptr) {
            const char* expected = nullptr;
            if (sites[i].name.compare_exchange_strong(expected, name)) {
                return &sites[i];
            }
            if (sites[i].name.load() == name) {
                return &sites[i];
            }
        }
    }
    return nullptr;
}

// Log a guest backtrace for every `every`-th call, at most `max_logs` times per site.
inline void Sample(const char* name, u32 every = 1, u32 max_logs = 12) {
    Site* site = GetSite(name);
    if (site == nullptr) {
        return;
    }
    const u32 n = ++site->calls;
    if (every > 1 && (n % every) != 0) {
        return;
    }
    if (n % 500 == 0) {
        LOG_INFO(Core, "PROFILE-COUNT {} #{}", name, n);
    }
    if (site->logged.fetch_add(1) >= max_logs) {
        return;
    }
#if defined(_WIN32)
    // Windows unwinding cannot walk guest frames (they have no unwind info), so scan the stack
    // conservatively instead: guest return addresses are plain values on the current stack.
    // Guest modules are mapped below 4 GB, while the emulator and host DLLs are not.
    constexpr u64 GuestLow = 0x10000;
    constexpr u64 GuestHigh = 0x100000000ull;
    const u64* sp = reinterpret_cast<const u64*>(__builtin_frame_address(0));
    std::string line;
    int found = 0;
    for (int i = 0; i < 4096 && found < 16; i++) {
        const u64 v = sp[i];
        if (v >= GuestLow && v < GuestHigh) {
            line += fmt::format("{:#x} ", v);
            found++;
        }
    }
    LOG_INFO(Core, "PROFILE {} #{} stack({}): {}", name, n, found, line);
#else
    LOG_INFO(Core, "PROFILE {} #{} (no backtrace)", name, n);
#endif
}

// Total number of calls seen for a site (for periodic summaries).
inline u32 Count(const char* name) {
    Site* site = GetSite(name);
    return site ? site->calls.load() : 0;
}

} // namespace Common::Profiler
