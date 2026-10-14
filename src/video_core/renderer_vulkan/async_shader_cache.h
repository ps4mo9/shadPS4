// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <thread>
#include <unordered_map>

#include "common/logging/log.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/params.h"
#include "shader_recompiler/recompiler.h"

namespace Vulkan {

// Background translator for shader programs.
//
// The heavy part of bringing up a new pipeline state is not the Vulkan driver compile but the
// GCN -> IR -> SPIR-V translation performed on the GPU command processor thread; a title that
// streams hundreds of materials makes that thread stall for the duration of every translation.
// This cache moves the translation off that thread: the first request for a (program, permutation)
// pair enqueues the translation onto a small worker pool (each worker owns its shader pools, so
// no sharing), records the SPIR-V words and satisfies subsequent requests from the record.
// Vulkan-side compilation (shader module creation) still happens on the requesting thread.
class AsyncShaderCache {
public:
    explicit AsyncShaderCache(const Shader::Profile& profile_) : profile{profile_} {
        // Leave the command processor thread and a game thread their cores.
        const auto hw = std::max(1u, std::thread::hardware_concurrency());
        num_workers = std::clamp(hw > 2 ? hw - 2 : 1u, 1u, 4u);
        for (u32 i = 0; i < num_workers; ++i) {
            workers.emplace_back([this] { WorkerLoop(); });
        }
    }

    ~AsyncShaderCache() {
        {
            std::scoped_lock lk{queue_mutex};
            stopping = true;
        }
        queue_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    AsyncShaderCache(const AsyncShaderCache&) = delete;
    AsyncShaderCache& operator=(const AsyncShaderCache&) = delete;

    // Keyed by program hash and permutation index.
    static u64 MakeKey(u64 program_hash, size_t perm_idx) {
        return program_hash ^ (0x9e3779b97f4a7c15ull * (perm_idx + 1));
    }

    // Returns the translated SPIR-V words if a background translation with the SAME descriptor
    // binding layout has finished, else nullopt and the caller uses its synchronous path.
    // Binding numbers are baked into SPIR-V, so a record made from a different layout must not
    // be reused.
    std::optional<std::vector<u32>> TryGetSpv(u64 program_hash, size_t perm_idx,
                                             const Shader::Backend::Bindings& binding) {
        std::scoped_lock lk{result_mutex};
        const auto key = MakeKey(program_hash, perm_idx);
        const auto it = results.find(key);
        if (it == results.end() || !(it->second.first == binding)) {
            return std::nullopt;
        }
        return it->second.second;
    }

    // Enqueues a translation if one is not in flight. Inputs are copied; the guest code span is
    // stable for the lifetime of the request (the title does not rewrite its own shader code).
    void Request(u64 program_hash, Shader::Stage stage, Shader::LogicalStage l_stage,
                 Shader::ShaderParams params, Shader::RuntimeInfo runtime_info,
                 Shader::Backend::Bindings binding) {
        {
            std::scoped_lock q{queue_mutex};
            if (stopping || queued.contains(program_hash)) {
                // A translation for this program is already in flight; the caller still has its
                // synchronous path for this and later permutations.
                return;
            }
            queued.insert(program_hash);
            queue.push_back(Job{program_hash, stage, l_stage, params, runtime_info, binding});
        }
        queue_cv.notify_one();
    }

    // Number of background translations completed since start (diagnostics).
    u64 Completed() const {
        return completed.load();
    }

private:
    struct Job {
        u64 program_hash;
        Shader::Stage stage;
        Shader::LogicalStage l_stage;
        Shader::ShaderParams params;
        Shader::RuntimeInfo runtime_info;
        Shader::Backend::Bindings binding;
    };

    void WorkerLoop() {
        while (true) {
            std::optional<Job> job;
            {
                std::unique_lock lk{queue_mutex};
                queue_cv.wait(lk, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            Translate(*job);
        }
    }

    void Translate(const Job& job) {
        try {
            // Worker-local pools; nothing shared with other translations.
            Shader::Pools pools;
            auto runtime_info = job.runtime_info;
            auto binding = job.binding;
            auto info = Shader::Info(job.stage, job.l_stage, job.params);
            const auto ir_program =
                Shader::TranslateProgram(job.params.code, pools, info, runtime_info, profile);
            auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program,
                                                        binding);
            {
                std::scoped_lock r{result_mutex};
                results[MakeKey(job.program_hash, 0)] = {job.binding, std::move(spv)};
            }
            ++completed;
        } catch (const std::exception& e) {
            LOG_ERROR(Render_Vulkan, "Async shader translation failed: {}", e.what());
        }
    }

    const Shader::Profile& profile;
    u32 num_workers{};
    std::vector<std::thread> workers;
    std::deque<Job> queue;
    std::unordered_set<u64> queued;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::unordered_map<u64, std::pair<Shader::Backend::Bindings, std::vector<u32>>> results;
    std::mutex result_mutex;
    std::atomic<bool> stopping{false};
    std::atomic<u64> completed{0};
};

} // namespace Vulkan