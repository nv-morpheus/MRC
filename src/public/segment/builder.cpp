#include <srf/segment/builder.hpp>

// Non-main includes
#include <srf/engine/segment/ibuilder.hpp>
#include <srf/runnable/launchable.hpp>
#include <srf/segment/egress_port.hpp>
#include <srf/segment/object.hpp>

namespace srf::segment {

Builder::Builder(internal::segment::IBuilder& backend) : m_backend(backend) {}

const std::string& Builder::name() const
{
    return m_backend.name();
}

bool Builder::has_object(const std::string& name) const
{
    return m_backend.has_object(name);
}

ObjectProperties& Builder::find_object(const std::string& name)
{
    return m_backend.find_object(name);
}

void Builder::add_object(const std::string& name, std::shared_ptr<ObjectProperties> object)
{
    return m_backend.add_object(name, std::move(object));
}

void Builder::add_runnable(const std::string& name, std::shared_ptr<runnable::Launchable> runnable)
{
    return m_backend.add_runnable(name, std::move(runnable));
}

std::shared_ptr<IngressPortBase> Builder::get_ingress_base(const std::string& name)
{
    return m_backend.get_ingress_base(name);
}
std::shared_ptr<EgressPortBase> Builder::get_egress_base(const std::string& name)
{
    return m_backend.get_egress_base(name);
}

std::function<void(std::int64_t)> Builder::make_throughput_counter(const std::string& name)
{
    return m_backend.make_throughput_counter(name);
}
}  // namespace srf::segment
