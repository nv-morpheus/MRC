#include <pysrf/node.hpp>
#include <pysrf/types.hpp>

#include <srf/segment/ingress_port.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace srf::segment {
namespace py = pybind11;

template <>
IngressPort<pysrf::PyHolder>::IngressPort(SegmentAddress address, PortName name) :
  m_segment_address(address),
  m_port_name(std::move(name)),
  m_source(std::make_unique<pysrf::PythonNode<pysrf::PyHolder, pysrf::PyHolder>>())

{
    this->set_name(m_port_name);
}

template <>
node::SourceProperties<pysrf::PyHolder>* IngressPort<pysrf::PyHolder>::get_object() const
{
    CHECK(m_source);
    return m_source.get();
}

template <>
std::unique_ptr<runnable::Launcher> IngressPort<pysrf::PyHolder>::prepare_launcher(
    runnable::LaunchControl& launch_control)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_source);
    return launch_control.prepare_launcher(std::move(m_source));
}

template <>
std::shared_ptr<manifold::Interface> IngressPort<pysrf::PyHolder>::make_manifold(pipeline::Resources& resources)
{
    return manifold::Factory<pysrf::PyHolder>::make_manifold(m_port_name, resources);
}

template <>
void IngressPort<pysrf::PyHolder>::connect_to_manifold(std::shared_ptr<manifold::Interface> manifold)
{
    // ingress ports connect to manifold outputs
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_source);
    manifold->add_output(m_segment_address, m_source.get());
}
}  // namespace srf::segment
