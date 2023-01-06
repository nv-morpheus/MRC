//
// Created by drobison on 1/6/23.
//

#pragma once

#include "mrc/benchmarking/tracer.hpp"

#include <pybind11/pytypes.h>  // for object, dict

namespace mrc::pymrc {
    using latency_tracer_t = mrc::benchmarking::TracerEnsemble<pybind11::object, mrc::benchmarking::LatencyTracer>;
    using throughput_tracer_t = mrc::benchmarking::TracerEnsemble<pybind11::object, mrc::benchmarking::ThroughputTracer>;
}
