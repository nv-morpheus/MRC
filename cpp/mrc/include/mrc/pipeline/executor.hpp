/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mrc/utils/macros.hpp"

#include <map>
#include <memory>

namespace mrc {
class Options;
}  // namespace mrc
namespace mrc::pipeline {
class IPipeline;
}  // namespace mrc::pipeline

namespace mrc::pipeline {
class ISystem;

class SegmentMapping
{
  public:
    SegmentMapping(std::string segment_name);

    bool is_enabled() const;

    void set_enabled(bool is_enabled);

  private:
    std::string m_segment_name;

    bool m_is_enabled{true};
};

class PipelineMapping
{
  public:
    PipelineMapping(std::shared_ptr<IPipeline> pipeline);

    // Disable copy construction to avoid accidental copies. There should only be one instance per pipeline. Equal copy
    // is ok
    PipelineMapping(const PipelineMapping&) = delete;
    PipelineMapping(PipelineMapping&&)      = default;

    PipelineMapping& operator=(const PipelineMapping&) = default;
    PipelineMapping& operator=(PipelineMapping&&)      = default;

    SegmentMapping& get_segment(const std::string& segment_name);
    const SegmentMapping& get_segment(const std::string& segment_name) const;

  private:
    std::map<std::string, SegmentMapping> m_segment_mappings;
};

class IExecutor
{
  public:
    virtual ~IExecutor() = default;

    DELETE_COPYABILITY(IExecutor);

    virtual PipelineMapping& register_pipeline(std::shared_ptr<IPipeline> pipeline) = 0;
    virtual void start()                                                            = 0;
    virtual void stop()                                                             = 0;
    virtual void join()                                                             = 0;

  protected:
    IExecutor() = default;
};

}  // namespace mrc::pipeline

namespace mrc {

// For backwards compatibility, make utility implementation which holds onto a unique_ptr
class Executor
{
  public:
    Executor();
    Executor(std::shared_ptr<Options> options);
    ~Executor();

    pipeline::PipelineMapping& register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline);
    void start();
    void stop();
    void join();

  private:
    std::unique_ptr<pipeline::IExecutor> m_impl;
};

std::unique_ptr<pipeline::IExecutor> make_executor_impl(std::shared_ptr<Options> options);

std::unique_ptr<pipeline::IExecutor> make_executor_impl(std::unique_ptr<pipeline::ISystem> system);

}  // namespace mrc
