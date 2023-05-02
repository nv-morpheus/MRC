#include "internal/control_plane/state/root_state.hpp"

#include "mrc/protos/architect_state.pb.h"

#include <google/protobuf/util/message_differencer.h>

namespace mrc::internal::control_plane::state {

ControlPlaneStateBase::ControlPlaneStateBase(const google::protobuf::Message& message) : m_internal_message(message) {}

bool ControlPlaneStateBase::operator==(const ControlPlaneStateBase& other) const
{
    return google::protobuf::util::MessageDifferencer::Equals(m_internal_message, other.m_internal_message);
}

ControlPlaneState::ControlPlaneState(protos::ControlPlaneState& message) :
  ControlPlaneStateBase(message),
  m_message(message)
{
    // For each message type, create a wrapper
    for (const auto& id : m_message.connections().ids())
    {
        m_connections.emplace(id, Connection(*this, message.connections().entities().at(id)));
    }

    for (const auto& id : m_message.workers().ids())
    {
        m_workers.emplace(id, Worker(*this, message.workers().entities().at(id)));
    }

    for (const auto& id : m_message.pipeline_definitions().ids())
    {
        m_pipeline_definitions.emplace(id, PipelineDefinition(*this, message.pipeline_definitions().entities().at(id)));
    }

    for (const auto& id : m_message.pipeline_instances().ids())
    {
        m_pipeline_instances.emplace(id, PipelineInstance(*this, message.pipeline_instances().entities().at(id)));
    }

    for (const auto& id : m_message.segment_definitions().ids())
    {
        m_segment_definitions.emplace(id, SegmentDefinition(*this, message.segment_definitions().entities().at(id)));
    }

    for (const auto& id : m_message.segment_instances().ids())
    {
        m_segment_instances.emplace(id, SegmentInstance(*this, message.segment_instances().entities().at(id)));
    }
}

const std::map<uint64_t, Connection>& ControlPlaneState::connections() const
{
    return m_connections;
}

const std::map<uint64_t, Worker>& ControlPlaneState::workers() const
{
    return m_workers;
}

const std::map<uint64_t, PipelineDefinition>& ControlPlaneState::pipeline_definitions() const
{
    return m_pipeline_definitions;
}

const std::map<uint64_t, PipelineInstance>& ControlPlaneState::pipeline_instances() const
{
    return m_pipeline_instances;
}

const std::map<uint64_t, SegmentDefinition>& ControlPlaneState::segment_definitions() const
{
    return m_segment_definitions;
}

const std::map<uint64_t, SegmentInstance>& ControlPlaneState::segment_instances() const
{
    return m_segment_instances;
}

Connection::Connection(ControlPlaneState& root, const protos::Connection& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t Connection::id() const
{
    return m_message.id();
}

std::string Connection::peer_info() const
{
    return m_message.peer_info();
}

std::map<uint64_t, const Worker&> Connection::workers() const
{
    std::map<uint64_t, const Worker&> child_objs;

    for (const auto& id : m_message.worker_ids())
    {
        child_objs.emplace(id, m_root.workers().at(id));
    }

    return child_objs;
}

std::map<uint64_t, const PipelineInstance&> Connection::assigned_pipelines() const
{
    std::map<uint64_t, const PipelineInstance&> child_objs;

    for (const auto& id : m_message.assigned_pipeline_ids())
    {
        child_objs.emplace(id, m_root.pipeline_instances().at(id));
    }

    return child_objs;
}

Worker::Worker(ControlPlaneState& root, const protos::Worker& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t Worker::id() const
{
    return m_message.id();
}

std::string Worker::worker_address() const
{
    return m_message.worker_address();
}

uint64_t Worker::machine_id() const
{
    return m_message.machine_id();
}

WorkerStates Worker::state() const
{
    return static_cast<WorkerStates>(m_message.state());
}

std::map<uint64_t, const SegmentInstance&> Worker::assigned_segments() const
{
    std::map<uint64_t, const SegmentInstance&> child_objs;

    for (const auto& id : m_message.assigned_segment_ids())
    {
        child_objs.emplace(id, m_root.segment_instances().at(id));
    }

    return child_objs;
}

PipelineDefinition::PipelineDefinition(ControlPlaneState& root, const protos::PipelineDefinition& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t PipelineDefinition::id() const
{
    return m_message.id();
}

std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> PipelineDefinition::segments() const
{
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

    for (const auto& id : m_message.segment_ids())
    {
        child_objs.emplace(id, m_root.segment_instances().at(id));
    }

    return child_objs;
}

std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> PipelineDefinition::instances() const
{
    std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> child_objs;

    for (const auto& id : m_message.instance_ids())
    {
        child_objs.emplace(id, m_root.pipeline_instances().at(id));
    }

    return child_objs;
}

PipelineInstance::PipelineInstance(ControlPlaneState& root, const protos::PipelineInstance& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t PipelineInstance::id() const
{
    return m_message.id();
}

const PipelineDefinition& PipelineInstance::definition() const
{
    return m_root.pipeline_definitions().at(m_message.definition_id());
}

uint64_t PipelineInstance::machine_id() const
{
    return m_message.machine_id();
}

std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> PipelineInstance::segments() const
{
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

    for (const auto& id : m_message.segment_ids())
    {
        child_objs.emplace(id, m_root.segment_instances().at(id));
    }

    return child_objs;
}

SegmentDefinition::SegmentDefinition(ControlPlaneState& root, const protos::SegmentDefinition& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t SegmentDefinition::id() const
{
    return m_message.id();
}

std::string SegmentDefinition::name() const
{
    return m_message.name();
}

const PipelineDefinition& SegmentDefinition::pipeline() const
{
    return m_root.pipeline_definitions().at(m_message.pipeline_id());
}

std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> SegmentDefinition::instances() const
{
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

    for (const auto& id : m_message.instance_ids())
    {
        child_objs.emplace(id, m_root.segment_instances().at(id));
    }

    return child_objs;
}

SegmentInstance::SegmentInstance(ControlPlaneState& root, const protos::SegmentInstance& message) :
  ControlPlaneStateBase(message),
  m_root(root),
  m_message(message)
{}

uint64_t SegmentInstance::id() const
{
    return m_message.id();
}

uint32_t SegmentInstance::address() const
{
    return m_message.address();
}

const SegmentDefinition& SegmentInstance::definition() const
{
    return m_root.segment_definitions().at(m_message.definition_id());
}

const Worker& SegmentInstance::worker() const
{
    return m_root.workers().at(m_message.worker_id());
}

const PipelineInstance& SegmentInstance::pipeline() const
{
    return m_root.pipeline_instances().at(m_message.pipeline_id());
}

}  // namespace mrc::internal::control_plane::state
