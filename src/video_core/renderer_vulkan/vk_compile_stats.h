// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Diagnostic: phase timing for pipeline/shader bring-up, to locate stalls.
#pragma once

#include <atomic>
#include <chrono>
#include "common/logging/log.h"
#include "common/types.h"

namespace Vulkan::CompileStats {

inline std::atomic<u64> translate_ns{0};
inline std::atomic<u64> translate_count{0};
inline std::atomic<u64> emit_ns{0};
inline std::atomic<u64> shader_module_ns{0};
inline std::atomic<u64> shader_module_count{0};
inline std::atomic<u64> graphics_pipeline_ns{0};
inline std::atomic<u64> graphics_pipeline_count{0};
inline std::atomic<u64> compute_pipeline_ns{0};
inline std::atomic<u64> compute_pipeline_count{0};
inline std::atomic<u64> preload_ns{0};
inline std::atomic<u64> events{0};

using Clock = std::chrono::steady_clock;

inline u64 Since(const Clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

inline void MaybeLog() {
    // Log every 64 recorded events; totals are cumulative since start.
    if ((events.fetch_add(1, std::memory_order_relaxed) + 1) % 64 != 0) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "COMPILESTATS: translate={}ms/{} emit={}ms module={}ms/{} gfxPipeline={}ms/{} "
             "csPipeline={}ms/{} preload={}ms",
             translate_ns.load() / 1000000, translate_count.load(), emit_ns.load() / 1000000,
             shader_module_ns.load() / 1000000, shader_module_count.load(),
             graphics_pipeline_ns.load() / 1000000, graphics_pipeline_count.load(),
             compute_pipeline_ns.load() / 1000000, compute_pipeline_count.load(),
             preload_ns.load() / 1000000);
}

struct ScopedTimer {
    std::atomic<u64>& acc;
    std::atomic<u64>* count;
    Clock::time_point start;

    explicit ScopedTimer(std::atomic<u64>& accumulator, std::atomic<u64>* count_ptr = nullptr)
        : acc{accumulator}, count{count_ptr}, start{Clock::now()} {}
    ~ScopedTimer() {
        acc.fetch_add(Since(start), std::memory_order_relaxed);
        if (count) {
            count->fetch_add(1, std::memory_order_relaxed);
        }
    }
};

} // namespace Vulkan::CompileStats