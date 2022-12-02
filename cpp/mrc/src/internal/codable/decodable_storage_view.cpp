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

#include "internal/codable/decodable_storage_view.hpp"

#include "internal/data_plane/client.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/memory/device_resources.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/remote_registration_cache.hpp"

#include "mrc/memory/memory_kind.hpp"
#include "mrc/protos/codable.pb.h"

#include <glog/logging.h>
#include <ucp/api/ucp_def.h>

#include <cstring>
#include <optional>
#include <ostream>
#include <string>

namespace mrc::internal::codable {

std::size_t DecodableStorageView::buffer_size(const idx_t& idx) const
{
    DCHECK_LT(idx, descriptor_count());
    const auto& desc = proto().descriptors().at(idx);

    CHECK(desc.has_eager_desc() || desc.has_remote_desc());

    if (desc.has_eager_desc())
    {
        return desc.eager_desc().data().size();
    }

    // if (desc.has_remote_desc())
    // {
    return desc.remote_desc().bytes();
    // }
}

void DecodableStorageView::copy_from_buffer(const idx_t& idx, mrc::memory::buffer_view dst_view) const
{
    CHECK_LT(idx, descriptor_count());
    const auto& desc = proto().descriptors().at(idx);

    if (desc.has_eager_desc())
    {
        return copy_from_eager_buffer(idx, dst_view);
    }

    if (desc.has_remote_desc())
    {
        return copy_from_registered_buffer(idx, dst_view);
    }

    LOG(FATAL) << "descriptor " << idx << " not backed by a buffered resource";
}

void DecodableStorageView::copy_from_registered_buffer(const idx_t& idx, mrc::memory::buffer_view& dst_view) const
{
    const auto& remote = proto().descriptors().at(idx).remote_desc();
    CHECK_LE(dst_view.bytes(), remote.bytes());

    // todo(ryan) - check locality, if we are on the same machine but a different instance, use direct method
    if (resources().network()->instance_id() == remote.instance_id())
    {
        LOG(FATAL) << "implement local copy";
    }
    else
    {
        LOG(INFO) << "performing rmda get ";
        bool cached_registration{false};
        ucp_rkey_h rkey;
        data_plane::Request request;
        data_plane::Client& client = resources().network()->data_plane().client();

        // get endpoint to remote instance_id
        auto ep = client.endpoint_shared(remote.instance_id());

        const void* remote_address = reinterpret_cast<const void*>(remote.address());

        // determine if remote memory region is in the remote memory cache on this endpoint
        auto block = ep->registration_cache().lookup(remote_address);

        // rkey from cache
        if (block)
        {
            LOG(INFO) << "remote memory region in cache";
            rkey = block->remote_key_handle();
        }
        else
        {
            cached_registration = true;
            auto block =
                ep->registration_cache().add_block(reinterpret_cast<const void*>(remote.memory_block_address()),
                                                   remote.memory_block_size(),
                                                   remote.remote_key());
            rkey = block.remote_key_handle();
        }

        // issue rdma get
        client.async_get(dst_view.data(), dst_view.bytes(), *ep, remote.address(), rkey, request);

        // await and yield on get
        request.await_complete();

        if (cached_registration && !remote.should_cache())
        {
            ep->registration_cache().drop_block(reinterpret_cast<const void*>(remote.memory_block_address()));
        }
    }
}

void DecodableStorageView::copy_from_eager_buffer(const idx_t& idx, mrc::memory::buffer_view& dst_view) const
{
    const auto& eager_buffer = proto().descriptors().at(idx).eager_desc();
    CHECK_LE(dst_view.bytes(), eager_buffer.data().size());

    if (dst_view.kind() == mrc::memory::memory_kind::device)
    {
        LOG(FATAL) << "implement async device copies";
    }

    if (dst_view.kind() == mrc::memory::memory_kind::none)
    {
        LOG(WARNING) << "got a memory::kind::none";
    }
    std::memcpy(dst_view.data(), eager_buffer.data().data(), dst_view.bytes());
}

std::shared_ptr<mrc::memory::memory_resource> DecodableStorageView::host_memory_resource() const
{
    return resources().host().arena_memory_resource();
}

std::shared_ptr<mrc::memory::memory_resource> DecodableStorageView::device_memory_resource() const
{
    if (resources().device())
    {
        return resources().device()->arena_memory_resource();
    }
    return nullptr;
}

}  // namespace mrc::internal::codable
