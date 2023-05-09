#include "internal/control_plane/state/root_state.hpp"

#include "mrc/protos/architect_state.pb.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <memory>
#include <utility>

namespace mrc::internal::control_plane::state {

ControlPlaneStateBase::ControlPlaneStateBase(const google::protobuf::Message& message) : m_internal_message(message) {}

bool ControlPlaneStateBase::operator==(const ControlPlaneStateBase& other) const
{
    // // Make sure neither are null
    // if (m_internal_message == nullptr || other.m_internal_message == nullptr)
    // {
    //     return m_internal_message == other.m_internal_message;
    // }

    std::string left;
    std::string right;

    google::protobuf::util::MessageToJsonString(m_internal_message, &left);
    google::protobuf::util::MessageToJsonString(other.m_internal_message, &right);

    return google::protobuf::util::MessageDifferencer::Equals(m_internal_message, other.m_internal_message);
}

ControlPlaneNormalizedState::ControlPlaneNormalizedState(std::unique_ptr<protos::ControlPlaneState> message) :
  root_message(std::move(message))
{}

void ControlPlaneNormalizedState::initialize()
{
    // For each message type, create a wrapper
    for (const auto& id : root_message->connections().ids())
    {
        connections.emplace(id, Connection(this->shared_from_this(), root_message->connections().entities().at(id)));
    }

    for (const auto& id : root_message->workers().ids())
    {
        workers.emplace(id, Worker(this->shared_from_this(), root_message->workers().entities().at(id)));
    }

    for (const auto& id : root_message->pipeline_definitions().ids())
    {
        pipeline_definitions.emplace(
            id,
            PipelineDefinition(this->shared_from_this(), root_message->pipeline_definitions().entities().at(id)));
    }

    for (const auto& id : root_message->pipeline_instances().ids())
    {
        pipeline_instances.emplace(
            id,
            PipelineInstance(this->shared_from_this(), root_message->pipeline_instances().entities().at(id)));
    }

    // for (const auto& id : root_message->segment_definitions().ids())
    // {
    //     segment_definitions.emplace(
    //         id,
    //         SegmentDefinition(this->shared_from_this(), root_message->segment_definitions().entities().at(id)));
    // }

    for (const auto& id : root_message->segment_instances().ids())
    {
        segment_instances.emplace(
            id,
            SegmentInstance(this->shared_from_this(), root_message->segment_instances().entities().at(id)));
    }
}

std::shared_ptr<ControlPlaneNormalizedState> ControlPlaneNormalizedState::create(
    std::unique_ptr<protos::ControlPlaneState> root_message)
{
    // Use new for the private constructor
    auto obj = std::shared_ptr<ControlPlaneNormalizedState>(new ControlPlaneNormalizedState(std::move(root_message)));

    // Must initialize as soon as object is created
    obj->initialize();

    return obj;
}

ControlPlaneState::ControlPlaneState(std::unique_ptr<protos::ControlPlaneState> message) :
  m_root_state(ControlPlaneNormalizedState::create(std::move(message)))
{}

const std::map<uint64_t, Connection>& ControlPlaneState::connections() const
{
    return m_root_state->connections;
}

const std::map<uint64_t, Worker>& ControlPlaneState::workers() const
{
    return m_root_state->workers;
}

const std::map<uint64_t, PipelineDefinition>& ControlPlaneState::pipeline_definitions() const
{
    return m_root_state->pipeline_definitions;
}

const std::map<uint64_t, PipelineInstance>& ControlPlaneState::pipeline_instances() const
{
    return m_root_state->pipeline_instances;
}

// const std::map<uint64_t, SegmentDefinition>& ControlPlaneState::segment_definitions() const
// {
//     return m_state->segment_definitions;
// }

const std::map<uint64_t, SegmentInstance>& ControlPlaneState::segment_instances() const
{
    return m_root_state->segment_instances;
}

ResourceState::ResourceState(const protos::ResourceState& message) : ControlPlaneStateBase(message), m_message(message)
{}

ResourceStatus ResourceState::status() const
{
    return static_cast<ResourceStatus>(m_message.status());
}

int32_t ResourceState::ref_count() const
{
    return m_message.ref_count();
}

Connection::Connection(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Connection& message) :
  ControlPlaneStateBase(message),
  m_root_state(std::move(state)),
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
        child_objs.emplace(id, m_root_state->workers.at(id));
    }

    return child_objs;
}

std::map<uint64_t, const PipelineInstance&> Connection::assigned_pipelines() const
{
    std::map<uint64_t, const PipelineInstance&> child_objs;

    for (const auto& id : m_message.assigned_pipeline_ids())
    {
        child_objs.emplace(id, m_root_state->pipeline_instances.at(id));
    }

    return child_objs;
}

Worker::Worker(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Worker& message) :
  ControlPlaneStateBase(message),
  m_root_state(std::move(state)),
  m_message(message),
  m_state(message.state())
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

const ResourceState& Worker::state() const
{
    return m_state;
}

std::map<uint64_t, const SegmentInstance&> Worker::assigned_segments() const
{
    std::map<uint64_t, const SegmentInstance&> child_objs;

    for (const auto& id : m_message.assigned_segment_ids())
    {
        child_objs.emplace(id, m_root_state->segment_instances.at(id));
    }

    return child_objs;
}

PipelineConfiguration::PipelineConfiguration(const protos::PipelineConfiguration& message) :
  ControlPlaneStateBase(message),
  m_message(message)
{}

PipelineDefinition::SegmentDefinition::SegmentDefinition(std::shared_ptr<ControlPlaneNormalizedState> state,
                                                         const protos::PipelineDefinition_SegmentDefinition& message) :
  ControlPlaneStateBase(message),
  m_message(message)
{}

uint64_t PipelineDefinition::SegmentDefinition::id() const
{
    return m_message.id();
}

const PipelineDefinition& PipelineDefinition::SegmentDefinition::parent() const
{
    return m_root_state->pipeline_definitions.at(m_message.parent_id());
}

std::string PipelineDefinition::SegmentDefinition::name() const
{
    return m_message.name();
}

std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> PipelineDefinition::SegmentDefinition::instances()
    const
{
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

    for (const auto& id : m_message.instance_ids())
    {
        child_objs.emplace(id, m_root_state->segment_instances.at(id));
    }

    return child_objs;
}

PipelineDefinition::PipelineDefinition(std::shared_ptr<ControlPlaneNormalizedState> state,
                                       const protos::PipelineDefinition& message) :
  ControlPlaneStateBase(message),
  m_root_state(std::move(state)),
  m_message(message),
  m_config(message.config())
{
    // Now setup the segments
    for (const auto& [seg_id, seg_def] : m_message.segments())
    {
        SegmentDefinition def(m_root_state, seg_def);
        // m_segments.emplace(seg_id, SegmentDefinition(m_state, seg_def));
    }
}

uint64_t PipelineDefinition::id() const
{
    return m_message.id();
}

const PipelineConfiguration& PipelineDefinition::config() const
{
    return m_config;
}

std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> PipelineDefinition::instances() const
{
    std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> child_objs;

    for (const auto& id : m_message.instance_ids())
    {
        child_objs.emplace(id, m_root_state->pipeline_instances.at(id));
    }

    return child_objs;
}

const std::map<uint64_t, PipelineDefinition::SegmentDefinition>& PipelineDefinition::segments() const
{
    return m_segments;
}

PipelineInstance::PipelineInstance(std::shared_ptr<ControlPlaneNormalizedState> state,
                                   const protos::PipelineInstance& message) :
  ControlPlaneStateBase(message),
  m_root_state(std::move(state)),
  m_message(message),
  m_state(message.state())
{}

uint64_t PipelineInstance::id() const
{
    return m_message.id();
}

const PipelineDefinition& PipelineInstance::definition() const
{
    return m_root_state->pipeline_definitions.at(m_message.definition_id());
}

uint64_t PipelineInstance::machine_id() const
{
    return m_message.machine_id();
}

const ResourceState& PipelineInstance::state() const
{
    return m_state;
}

std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> PipelineInstance::segments() const
{
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

    for (const auto& id : m_message.segment_ids())
    {
        child_objs.emplace(id, m_root_state->segment_instances.at(id));
    }

    return child_objs;
}

// SegmentDefinition::SegmentDefinition(std::shared_ptr<ControlPlaneNormalizedState> state,
//                                      const protos::SegmentDefinition& message) :
//   ControlPlaneStateBase(message),
//   m_state(std::move(state)),
//   m_message(message)
// {}

// uint64_t SegmentDefinition::id() const
// {
//     return m_message.id();
// }

// std::string SegmentDefinition::name() const
// {
//     return m_message.name();
// }

// const PipelineDefinition& SegmentDefinition::pipeline() const
// {
//     return m_state->pipeline_definitions.at(m_message.pipeline_id());
// }

// std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> SegmentDefinition::instances() const
// {
//     std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> child_objs;

//     for (const auto& id : m_message.instance_ids())
//     {
//         child_objs.emplace(id, m_state->segment_instances.at(id));
//     }

//     return child_objs;
// }

SegmentInstance::SegmentInstance(std::shared_ptr<ControlPlaneNormalizedState> state,
                                 const protos::SegmentInstance& message) :
  ControlPlaneStateBase(message),
  m_root_state(std::move(state)),
  m_message(message),
  m_state(message.state())
{}

uint64_t SegmentInstance::id() const
{
    return m_message.id();
}

const PipelineDefinition& SegmentInstance::pipeline_definition() const
{
    return m_root_state->pipeline_definitions.at(m_message.pipeline_definition_id());
}

std::string SegmentInstance::name() const
{
    return m_message.name();
}

uint32_t SegmentInstance::address() const
{
    return m_message.address();
}

const Worker& SegmentInstance::worker() const
{
    return m_root_state->workers.at(m_message.worker_id());
}

const PipelineInstance& SegmentInstance::pipeline_instance() const
{
    return m_root_state->pipeline_instances.at(m_message.pipeline_instance_id());
}

const ResourceState& SegmentInstance::state() const
{
    return m_state;
}

}  // namespace mrc::internal::control_plane::state
