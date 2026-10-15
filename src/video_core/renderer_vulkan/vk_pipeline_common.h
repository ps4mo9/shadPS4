// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::LogicalStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::LogicalStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    /// Completes the driver pipeline creation when construction was deferred. Runs on the
    /// pipeline-compile worker thread for deferred pipelines and inline at the end of the
    /// constructor otherwise. Must be called exactly once.
    virtual void CompleteCreate(vk::PipelineCache pipeline_cache) = 0;

    /// True once the driver pipeline exists and may be used.
    bool IsReady() const noexcept {
        return create_state.load(std::memory_order_acquire) == CreateState::Ready;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    using BufferBarriers = boost::container::small_vector<vk::BufferMemoryBarrier2, 16>;

    void BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                       const Shader::PushData& push_data) const;

protected:
    [[nodiscard]] std::string GetDebugString() const;

    enum class CreateState : u32 {
        Pending, ///< Deferred driver creation has not finished yet; the pipeline is unusable.
        Ready,   ///< The pipeline handle is valid.
        Failed,  ///< Driver creation failed; work using this state is skipped.
    };

    void SetCreateState(CreateState state) noexcept {
        create_state.store(state, std::memory_order_release);
    }

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    std::atomic<CreateState> create_state{CreateState::Pending};
    bool uses_push_descriptors{};
    bool is_compute;
};

} // namespace Vulkan
