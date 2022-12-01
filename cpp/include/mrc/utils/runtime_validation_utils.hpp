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

/*
template<typename T>
struct is_container : std::integral_constant<bool, has_const_iterator<T>::value
      && has_begin_end<T>::beg_value && has_begin_end<T>::end_value>
{ };

template<typename CType, typename std::enable_if_t<is_container<CType>, bool> = true>
bool has_duplicates() {
    for (auto x : CType.begin())
};*/
