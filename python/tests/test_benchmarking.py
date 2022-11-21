# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================

import pathlib
import random

import srf.benchmarking

whereami = pathlib.Path(__file__).parent.resolve()

# Add a little jitter to testing
TEST_ITERATIONS = random.randint(10, 250)


def tracer_test_f(x):
    return x


# @pytest.mark.xfail  # issue#161
def init_tracer_segment(seg: srf.Builder, watcher: srf.benchmarking.LatencyWatcher):
    # CXX double source with heterogesrfus segment node composition
    # print("Made it into init_tracer_segment\n", flush=True)
    python_tracer_source = watcher.make_tracer_source(seg, "tracer_source", False)
    python_node_1 = watcher.make_traced_node(seg, "python_traced_1", tracer_test_f)
    seg.make_edge(python_tracer_source, python_node_1)

    python_tracer_sink = watcher.make_tracer_sink(seg, "tracer_sink", lambda x: x)
    seg.make_edge(python_node_1, python_tracer_sink)


# @pytest.mark.xfail  # issue#161
# def test_tracer_creation():
#    options = srf.Options()
#    executor = srf.Executor(options)
#
#    latency_watcher = srf.benchmarking.LatencyWatcher(executor)
#    latency_watcher.make_segment("tracer_segment", init_tracer_segment)
#
#    latency_watcher.tracer_count(TEST_ITERATIONS)
#    latency_watcher.trace_until_notified()
#    latency_watcher.shutdown()
#
#    tracer_metrics = latency_watcher.aggregate_tracers()
#    # print(json.dumps(tracer_metrics, indent=2))
#    metadata = tracer_metrics["metadata"]
#    assert (metadata["tracer_count"] == TEST_ITERATIONS)

# @pytest.mark.xfail  # issue#161
# def test_latency_tracer_counts_match_framework_stats():
#     srf.benchmarking.reset_tracing_stats()
#     assert (srf.benchmarking.trace_operators() == (False, False))
#     assert (srf.benchmarking.trace_channels() == (False, False))

#     srf.benchmarking.trace_operators(True)
#     assert (srf.benchmarking.trace_operators() == (True, True))
#     srf.benchmarking.trace_channels(True)
#     assert (srf.benchmarking.trace_channels() == (True, True))
#     required_components = [("tracer_source", "src"), ("python_traced_1", "internal"), ("tracer_sink", "sink")]

#     options = srf.Options()

#     options.placement.cpu_strategy = PlacementStrategy.PerMachine
#     executor = srf.Executor(options)

#     latency_watcher = srf.benchmarking.LatencyWatcher(executor)
#     latency_watcher.make_segment("tracer_segment", init_tracer_segment)

#     latency_watcher.tracer_count(TEST_ITERATIONS)
#     latency_watcher.trace_until_notified()
#     latency_watcher.shutdown()

#     framework_stats_info = srf.benchmarking.get_tracing_stats()
#     component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]
#     # Verify tracer benchmark data looks correct
#     tracer_metrics = latency_watcher.aggregate_tracers()
#     counters = tracer_metrics["aggregations"]["metrics"]["counter"]

#     metadata = tracer_metrics["metadata"]
#     assert (metadata["tracer_count"] == TEST_ITERATIONS)
#     assert (metadata["node_count"] == 3)

#     latency_tracers = counters["component_latency_seconds_mean"]
#     latency_tracers_op = {
#         val["labels"]["source_name"]: val["value"]
#         for val in latency_tracers if val["labels"]["type"] == "operator"
#     }

#     # Collect all framework counter ids.
#     for key, _type in required_components:
#         assert (key in component_metrics)
#         component = component_metrics[key]
#         assert (len(component.keys()) > 0)
#         if (_type in ("src", "internal")):
#             assert (component["component_channel_write_total"] == TEST_ITERATIONS)
#             assert (component["component_emissions_total"] == TEST_ITERATIONS)
#             assert (component["component_operator_proc_latency_seconds"] >= latency_tracers_op[key])
#         if (_type in ("sink", "internal")):
#             assert (component["component_channel_read_total"] == TEST_ITERATIONS)
#             assert (component["component_receive_total"] == TEST_ITERATIONS)

#     srf.benchmarking.reset_tracing_stats()

# @pytest.mark.xfail  # issue#161
# def test_throughput_tracer_counts_match_framework_stats():
#     srf.benchmarking.reset_tracing_stats()
#     srf.benchmarking.trace_operators(True)
#     srf.benchmarking.trace_channels(True)
#     required_components = [("tracer_source", "src"), ("python_traced_1", "internal"), ("tracer_sink", "sink")]

#     options = srf.Options()

#     options.placement.cpu_strategy = PlacementStrategy.PerMachine
#     executor = srf.Executor(options)

#     throughput_watcher = srf.benchmarking.ThroughputWatcher(executor)
#     throughput_watcher.make_segment("tracer_segment", init_tracer_segment)

#     throughput_watcher.tracer_count(TEST_ITERATIONS)
#     throughput_watcher.trace_until_notified()
#     throughput_watcher.shutdown()

#     framework_stats_info = srf.benchmarking.get_tracing_stats()
#     component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]

#     tracer_metrics = throughput_watcher.aggregate_tracers()
#     counters = tracer_metrics["aggregations"]["metrics"]["counter"]

#     metadata = tracer_metrics["metadata"]
#     assert (metadata["tracer_count"] == TEST_ITERATIONS)
#     assert (metadata["node_count"] == 3)

#     throughput_tracers = counters["component_mean_throughput"]
#     throughput_tracers_op = {
#         val["labels"]["source_name"]: val["value"]
#         for val in throughput_tracers if val["labels"]["type"] == "operator"
#     }

#     # Collect all framework counter ids.
#     for key, _type in required_components:
#         assert (key in component_metrics)

#         component = component_metrics[key]
#         assert (len(component.keys()) > 0)
#         if (_type in ("src", "internal")):
#             assert (component["component_channel_write_total"] == TEST_ITERATIONS)
#             assert (component["component_emissions_total"] == TEST_ITERATIONS)
#             # The tracer approach should always produce throughput numbers greater than or equal to the framework
#             # stats
#             # This is because the framework stat collection will include the overhead of tracer updates, while the
#             # tracer itself will not.
#             assert (component["component_emission_rate_seconds"] <= throughput_tracers_op[key])
#             assert (component["component_received_rate_seconds"] <= throughput_tracers_op[key])
#         if (_type in ("sink", "internal")):
#             assert (component["component_channel_read_total"] == TEST_ITERATIONS)
#             assert (component["component_receive_total"] == TEST_ITERATIONS)

#     srf.benchmarking.reset_tracing_stats()

if (__name__ in ("__main__", )):
    # init_tracer_segment()
    # test_tracer_creation()
    # test_latency_tracer_counts_match_framework_stats()
    # test_throughput_tracer_counts_match_framework_stats()
    pass
