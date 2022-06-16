#include <pysrf/node.hpp>
#include <pysrf/types.hpp>

#include <srf/segment/egress_port.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace srf::segment {
namespace py = pybind11;

template <>
EgressPort<pysrf::PyHolder>::EgressPort(SegmentAddress address, PortName name) :
  m_segment_address(address),
  m_port_name(std::move(name)),
  m_sink(std::make_unique<pysrf::PythonNode<pysrf::PyHolder, pysrf::PyHolder>>())
{
    this->set_name(m_port_name);
}

template <>
node::SinkProperties<pysrf::PyHolder>* EgressPort<pysrf::PyHolder>::get_object() const
{
    CHECK(m_sink) << "failed to acquire backing runnable for egress port " << m_port_name;
    return m_sink.get();
}

template <>
std::unique_ptr<runnable::Launcher> EgressPort<pysrf::PyHolder>::prepare_launcher(runnable::LaunchControl& launch_control)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_sink);
    CHECK(m_manifold_connected) << "manifold not set for egress port";
    return launch_control.prepare_launcher(std::move(m_sink));
}

template <>
void EgressPort<pysrf::PyHolder>::connect_to_manifold(std::shared_ptr<manifold::Interface> manifold)
{
    // egress ports connect to manifold inputs
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    DCHECK_EQ(manifold->port_name(), m_port_name);
    CHECK(m_sink);
    CHECK(!m_manifold_connected);
    manifold->add_input(m_segment_address, m_sink.get());
    m_manifold_connected = true;
}
}  // namespace srf::segment