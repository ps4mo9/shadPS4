// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_compile_stats.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

// Owns the structures that vk::ComputePipelineCreateInfo references, so the driver call can run
// after the constructor returned (deferred creation) without dangling pointers.
struct ComputePipeline::CreateData {
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci{};
    vk::PipelineShaderStageCreateInfo shader_ci{};
    vk::ComputePipelineCreateInfo pipeline_info{};
    std::string debug_str{};
};

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading /*=false*/,
                                 bool defer_creation /*=false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_}, create_data{std::make_unique<CreateData>()} {
    auto& cd = *create_data;
    auto& info = stages[int(Shader::LogicalStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();

    cd.subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    cd.shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &cd.subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    u32 binding{};
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        // During deserialization, we don't have access to the UD to fetch sharp data. To address
        // this properly we need to track shaprs or portion of them in `sdata`, but since we're
        // interested only in "is storage" flag (which is not even effective atm), we can take a
        // shortcut there.
        const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
        bindings.push_back({
            .binding = binding++,
            .descriptorType = buffer.IsStorage(sharp) ? vk::DescriptorType::eStorageBuffer
                                                      : vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    for (const auto& image : info->images) {
        const u32 num_bindings = image.NumBindings(*info);
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    const auto device = instance.GetDevice();
    auto [descriptor_set_result, descriptor_set] =
        device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(descriptor_set_result == vk::Result::eSuccess,
               "Failed to create compute descriptor set layout: {}",
               vk::to_string(descriptor_set_result));
    desc_layout = std::move(descriptor_set);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create compute pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);

    cd.pipeline_info = {
        .stage = cd.shader_ci,
        .layout = *pipeline_layout,
    };
    cd.debug_str = debug_str;

    if (defer_creation) {
        // The driver call is left to the pipeline-compile worker (see PipelineCache); dispatches
        // using this pipeline are skipped until IsReady() becomes true.
        return;
    }
    CompleteCreate(pipeline_cache);
}

void ComputePipeline::CompleteCreate(vk::PipelineCache pipeline_cache) {
    ASSERT(create_data != nullptr);

    const vk::Device device = instance.GetDevice();
    Vulkan::CompileStats::ScopedTimer cs_timer{Vulkan::CompileStats::compute_pipeline_ns,
                                               &Vulkan::CompileStats::compute_pipeline_count};
    auto [pipeline_result, pipe] =
        device.createComputePipelineUnique(pipeline_cache, create_data->pipeline_info);
    Vulkan::CompileStats::MaybeLog();

    if (pipeline_result == vk::Result::eSuccess) {
        pipeline = std::move(pipe);
        SetObjectName(device, *pipeline, "Compute Pipeline {}", create_data->debug_str);
        SetCreateState(CreateState::Ready);
    } else {
        LOG_CRITICAL(Render_Vulkan, "Failed to create compute pipeline: {}",
                     vk::to_string(pipeline_result));
        SetCreateState(CreateState::Failed);
    }
    create_data.reset();
}

ComputePipeline::~ComputePipeline() = default;

} // namespace Vulkan
