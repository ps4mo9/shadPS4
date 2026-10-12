// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Debug helper: per-frame accounting of the GPU work that dominates frame time.
#pragma once

#include <atomic>
#include <chrono>
#include "common/logging/log.h"
#include "common/types.h"

namespace Common::FrameStats {

inline std::atomic<u64> draws{0};
inline std::atomic<u64> dispatches{0};
inline std::atomic<u64> bind_resources{0};
inline std::atomic<u64> bind_textures{0};
inline std::atomic<u64> bind_buffers{0};
inline std::atomic<u64> flips{0};

inline std::atomic<u64> draw_ns{0};
inline std::atomic<u64> bind_resources_ns{0};
inline std::atomic<u64> bind_textures_ns{0};
inline std::atomic<u64> bind_buffers_ns{0};
inline std::atomic<u64> prepare_state_ns{0};

inline std::atomic<s64> last_flip_us{0};

// Total device memory handed out for GPU-resident copies of guest memory, plus bookkeeping so
// growth (and the lack of any shrink) is visible in the log.
inline std::atomic<u64> resident_bytes{0};
inline std::atomic<u64> resident_allocs{0};
inline std::atomic<u64> resident_largest{0};

struct Timer {
    std::atomic<u64>& acc;
    std::chrono::steady_clock::time_point start;

    explicit Timer(std::atomic<u64>& accumulator)
        : acc{accumulator}, start{std::chrono::steady_clock::now()} {}
    ~Timer() {
        acc.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count());
    }
};

inline void OnFlip() {
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    const auto prev = last_flip_us.exchange(now);
    const u64 n = ++flips;
    LOG_INFO(Core,
             "FRAMESTATS flip #{} dt={}ms | draws={} ({}ms) dispatches={} | bindRes={} ({}ms) "
             "bindTex={} ({}ms) bindBuf={} ({}ms) prepState={}ms | gpuResident={}MB (allocs={}, "
             "largest={}MB)",
             n, prev != 0 ? (now - prev) / 1000 : 0, draws.load(), draw_ns.load() / 1000000,
             dispatches.load(), bind_resources.load(), bind_resources_ns.load() / 1000000,
             bind_textures.load(), bind_textures_ns.load() / 1000000, bind_buffers.load(),
             bind_buffers_ns.load() / 1000000, prepare_state_ns.load() / 1000000,
             resident_bytes.load() >> 20, resident_allocs.load(), resident_largest.load() >> 20);
}

} // namespace Common::FrameStats
