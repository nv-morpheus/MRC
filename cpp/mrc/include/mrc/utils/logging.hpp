/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// #include <glog/logging.h>
// #include <boost/fiber/operations.hpp>
// #include <iomanip>
// #include <ostream>

namespace mrc::utils {
// void glog_custom_prefix(std::ostream &s, const LogMessageInfo &l, void *)
// {
//     s << l.severity[0]                         // Severity
//       << std::setw(2) << l.time.year() - 100   // Year
//       << std::setw(2) << 1 + l.time.month()    // Month
//       << std::setw(2) << l.time.day()          // Day
//       << ' '                                   // Break
//       << std::setw(2) << l.time.hour() << ':'  // Hour
//       << std::setw(2) << l.time.min() << ':'   // Min
//       << std::setw(2) << l.time.sec() << "."   // Sec
//       << std::setw(6) << l.time.usec()         // microsec
//       << ' ' << std::setfill(' ')              // Break
//       << std::setw(5) << l.thread_id           // ThreadID
//       << std::setfill('0') << '/'              // /
//       << boost::this_fiber::get_id()           // FiberID
//       << std::setfill('0') << ' '              // Break
//       << l.filename << ':' << l.line_number << "]";
// }
}  // namespace mrc::utils
