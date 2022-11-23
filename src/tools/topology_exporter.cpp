/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/gpu_info.hpp"
#include "internal/system/topology.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/options.hpp"
#include "mrc/protos/architect.pb.h"

#include <glog/logging.h>

#include <cstdio>
#include <memory>

using namespace mrc;
using namespace internal;

int main(int argc, char* argv[])
{
    auto options = std::make_shared<Options>();
    {
        // std::ofstream output("topology.bin", std::ios::binary);
        auto* file       = fopen("topology.bin", "wb");
        int posix_handle = fileno(file);
        auto topology    = system::Topology::Create(options->topology());
        auto system      = topology->serialize().SerializePartialToFileDescriptor(posix_handle);
        fclose(file);
    }

    {
        protos::Topology msg;
        auto* file       = fopen("topology.bin", "rb");
        int posix_handle = fileno(file);
        CHECK(msg.ParseFromFileDescriptor(posix_handle));
        fclose(file);
        auto topology = system::Topology::Create(options->topology(), msg);
    }

    return 0;
}
