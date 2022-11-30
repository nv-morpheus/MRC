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

#include "../test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/core/logging.hpp"

#include <glog/logging.h>

using namespace mrc;

TEST_CLASS(Logging);

TEST_F(TestLogging, Logging)
{
    EXPECT_FALSE(is_initialized());

    // should be safe to call prior to init
    EXPECT_LT(static_cast<int>(get_log_level()), google::NUM_SEVERITIES);

    EXPECT_TRUE(init_logging("test_logging", LogLevels::ERROR));
    EXPECT_TRUE(is_initialized());
    EXPECT_EQ(get_log_level(), LogLevels::ERROR);
    EXPECT_EQ(FLAGS_minloglevel, google::ERROR);

    EXPECT_FALSE(init_logging("should be a noop", LogLevels::INFO));
    EXPECT_EQ(FLAGS_minloglevel, google::ERROR);  // verify the above didn't alter the log level

    set_log_level(LogLevels::INFO);
    EXPECT_EQ(get_log_level(), LogLevels::INFO);
    EXPECT_EQ(FLAGS_minloglevel, google::INFO);
}
