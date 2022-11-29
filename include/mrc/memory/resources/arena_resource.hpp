/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "mrc/memory/adaptors.hpp"
#include "mrc/memory/resources/detail/arena.hpp"
#include "mrc/utils/bytes_to_string.hpp"

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>
#include <spdlog/common.h>
#include <spdlog/fmt/bundled/ostream.h>

#include <cstddef>
#include <map>
#include <shared_mutex>

namespace mrc::memory {

// Ignore naming conventions here to match RMM
// NOLINTBEGIN(readability-identifier-naming)

/**
 * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * GPU memory is divided into a global arena, per-thread arenas for default streams, and per-stream
 * arenas for non-default streams. Each arena allocates memory from the global arena in chunks
 * called superblocks.
 *
 * Blocks in each arena are allocated using address-ordered first fit. When a block is freed, it is
 * coalesced with neighbouring free blocks if the addresses are contiguous. Free superblocks are
 * returned to the global arena.
 *
 * In real-world applications, allocation sizes tend to follow a power law distribution in which
 * large allocations are rare, but small ones quite common. By handling small allocations in the
 * per-thread arena, adequate performance can be achieved without introducing excessive memory
 * fragmentation under high concurrency.
 *
 * This design is inspired by several existing CPU memory allocators targeting multi-threaded
 * applications (glibc malloc, Hoard, jemalloc, TCMalloc), albeit in a simpler form. Possible future
 * improvements include using size classes, allocation caches, and more fine-grained locking or
 * lock-free approaches.
 *
 * \see Wilson, P. R., Johnstone, M. S., Neely, M., & Boles, D. (1995, September). Dynamic storage
 * allocation: A survey and critical review. In International Workshop on Memory Management (pp.
 * 1-116). Springer, Berlin, Heidelberg.
 * \see Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. (2000). Hoard: A scalable
 * memory allocator for multithreaded applications. ACM Sigplan Notices, 35(11), 117-128.
 * \see Evans, J. (2006, April). A scalable concurrent malloc (3) implementation for FreeBSD. In
 * Proc. of the bsdcan conference, ottawa, canada.
 * \see https://sourceware.org/glibc/wiki/MallocInternals
 * \see http://hoard.org/
 * \see http://jemalloc.net/
 * \see https://github.com/google/tcmalloc
 *
 * @tparam Upstream Memory resource to use for allocating memory for the global arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena_resource final : public adaptor<Upstream>
{
    using pointer_type = typename adaptor<Upstream>::pointer_type;

    using global_arena = detail::arena::global_arena<pointer_type>;
    using arena        = detail::arena::arena<pointer_type>;
    using read_lock    = std::shared_lock<std::shared_timed_mutex>;
    using write_lock   = std::lock_guard<std::shared_timed_mutex>;

  public:
    /**
     * @brief Construct an `arena_resource`.
     *
     * @throws rmm::logic_error if `upstream_mr == nullptr`.
     * @throws rmm::logic_error if `initial_size` is neither the default nor aligned to a multiple of
     * 256 bytes.
     * @throws rmm::logic_error if `maximum_size` is neither the default nor aligned to a multiple of
     * 256 bytes.
     *
     * @param upstream_mr The memory resource from which to allocate blocks for the pool
     * @param initial_size Minimum size, in bytes, of the initial global arena. Defaults to half of
     * the available memory on the current device.
     * @param maximum_size Maximum size, in bytes, that the global arena can grow to. Defaults to all
     * of the available memory on the current device.
     */
    explicit arena_resource(Upstream upstream_mr,
                            std::size_t initial_size = global_arena::default_initial_size,
                            std::size_t maximum_size = global_arena::default_maximum_size,
                            bool dump_log_on_failure = false) :
      adaptor<Upstream>(std::move(upstream_mr)),
      global_arena_(std::make_shared<global_arena>(&this->resource(), initial_size, maximum_size)),
      dump_log_on_failure_(dump_log_on_failure)
    {
        if (dump_log_on_failure_)
        {
            logger_ = spdlog::basic_logger_mt("arena_memory_dump", "rmm_arena_memory_dump.log");
        }
    }

    ~arena_resource() override = default;

    // Disable copy (and move) semantics.
    arena_resource(arena_resource const&) = delete;
    arena_resource& operator=(arena_resource const&) = delete;
    arena_resource(arena_resource&&) noexcept        = delete;
    arena_resource& operator=(arena_resource&&) noexcept = delete;

  private:
    /**
     * @brief Allocates memory of size at least `bytes`.
     *
     * The returned pointer has at least 256-byte alignment.
     *
     * @throws `std::bad_alloc` if the requested allocation could not be fulfilled.
     *
     * @param bytes The size in bytes of the allocation.
     * @param stream The stream to associate this allocation with.
     * @return void* Pointer to the newly allocated memory.
     */
    void* do_allocate(std::size_t bytes) final
    {
        if (bytes <= 0)
        {
            return nullptr;
        }

        bytes         = detail::arena::align_up(bytes);
        auto& arena   = get_thread_arena();
        void* pointer = arena.allocate(bytes);

        if (pointer == nullptr)
        {
            write_lock lock(mtx_);
            defragment();
            pointer = arena.allocate(bytes);
            if (pointer == nullptr)
            {
                if (dump_log_on_failure_)
                {
                    dump_memory_log(bytes);
                }
                LOG(FATAL) << "Maximum pool size exceeded";
                // RMM_FAIL("Maximum pool size exceeded", rmm::out_of_memory);
            }
        }

        return pointer;
    }

    /**
     * @brief Deallocate memory pointed to by `ptr`.
     *
     * @param ptr Pointer to be deallocated.
     * @param bytes The size in bytes of the allocation. This must be equal to the
     * value of `bytes` that was passed to the `allocate` call that returned `p`.
     * @param stream Stream on which to perform deallocation.
     */
    void do_deallocate(void* ptr, std::size_t bytes) final
    {
        if (ptr == nullptr || bytes <= 0)
        {
            return;
        }

        bytes = detail::arena::align_up(bytes);
        get_thread_arena().deallocate(ptr, bytes);
    }

    /**
     * @brief Defragment memory by returning all free blocks to the global arena.
     */
    void defragment()
    {
        // RMM_CUDA_TRY(cudaDeviceSynchronize());
        for (auto& thread_arena : thread_arenas_)
        {
            thread_arena.second->clean();
        }
        /*
        for (auto& stream_arena : stream_arenas_)
        {
            stream_arena.second.clean();
        }
        */
    }

    /**
     * @brief Get the arena associated with the current thread or the given stream.
     *
     * @param stream The stream associated with the arena.
     * @return arena& The arena associated with the current thread or the given stream.
    arena& get_arena(cuda_stream_view stream)
    {
        if (use_per_thread_arena(stream))
        {
            return get_thread_arena();
        }
        return get_stream_arena(stream);
    }
     */

    /**
     * @brief Get the arena associated with the current thread.
     *
     * @return arena& The arena associated with the current thread.
     */
    arena& get_thread_arena()
    {
        auto const thread_id = std::this_thread::get_id();
        {
            read_lock lock(mtx_);
            auto const iter = thread_arenas_.find(thread_id);
            if (iter != thread_arenas_.end())
            {
                return *iter->second;
            }
        }
        {
            write_lock lock(mtx_);
            auto thread_arena = std::make_shared<arena>(global_arena_);
            thread_arenas_.emplace(thread_id, thread_arena);
            thread_local detail::arena::arena_cleaner<pointer_type> cleaner{thread_arena};
            return *thread_arena;
        }
    }

    /**
     * @brief Get the arena associated with the given stream.
     *
     * @return arena& The arena associated with the given stream.
    arena& get_stream_arena(cuda_stream_view stream)
    {
        RMM_LOGGING_ASSERT(!use_per_thread_arena(stream));
        {
            read_lock lock(mtx_);
            auto const iter = stream_arenas_.find(stream.value());
            if (iter != stream_arenas_.end())
            {
                return iter->second;
            }
        }
        {
            write_lock lock(mtx_);
            stream_arenas_.emplace(stream.value(), global_arena_);
            return stream_arenas_.at(stream.value());
        }
    }
     */

    /**
     * Dump memory to log.
     *
     * @param bytes the number of bytes requested for allocation
     */
    void dump_memory_log(size_t bytes)
    {
        /*
        logger_->info("**************************************************");
        logger_->info("Ran out of memory trying to allocate {}.", bytes;
        logger_->info("**************************************************");
        logger_->info("Global arena:");
        global_arena_.dump_memory_log(logger_);
        logger_->info("Per-thread arenas:");
        for (auto const& thread_arena : thread_arenas_)
        {
            logger_->info("  Thread {}:", thread_arena.first);
            thread_arena.second->dump_memory_log(logger_);
        }
        if (!stream_arenas_.empty())
        {
            logger_->info("Per-stream arenas:");
            for (auto const& stream_arena : stream_arenas_)
            {
                logger_->info("  Stream {}:", static_cast<void*>(stream_arena.first));
                stream_arena.second.dump_memory_log(logger_);
            }
        }
        */
    }

    /**
     * @brief Should a per-thread arena be used given the CUDA stream.
     *
     * @param stream to check.
     * @return true if per-thread arena should be used, false otherwise.
    static bool use_per_thread_arena(cuda_stream_view stream)
    {
        return stream.is_per_thread_default();
    }
     */

    /// The global arena to allocate superblocks from.
    std::shared_ptr<global_arena> global_arena_;  // NOLINT
    /// Arenas for default streams, one per thread.
    /// Implementation note: for small sizes, map is more efficient than unordered_map.
    std::map<std::thread::id, std::shared_ptr<arena>> thread_arenas_;  // NOLINT
    /// Arenas for non-default streams, one per stream.
    /// Implementation note: for small sizes, map is more efficient than unordered_map.
    // std::map<cudaStream_t, arena> stream_arenas_;
    /// If true, dump memory information to log on allocation failure.
    bool dump_log_on_failure_;  // NOLINT
    /// The logger for memory dump.
    std::shared_ptr<spdlog::logger> logger_{};  // NOLINT
    /// Mutex for read and write locks.
    mutable std::shared_timed_mutex mtx_;  // NOLINT
};

// NOLINTEND(readability-identifier-naming)

}  // namespace mrc::memory
