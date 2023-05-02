#include <cstdint>
#include <map>
#include <string>
#include <utility>

#pragma once

namespace google::protobuf {
class Message;
}

namespace mrc::protos {
class ControlPlaneState;
class Connection;
class Worker;
class PipelineDefinition;
class PipelineInstance;
class SegmentDefinition;
class SegmentInstance;
}  // namespace mrc::protos

namespace mrc::internal::control_plane::state {

// Mirror this enum so we dont have to include all the proto classes
enum class WorkerStates : int
{
    Registered  = 0,
    Activated   = 1,
    Deactivated = 2,
    Destroyed   = 3,
};

struct ControlPlaneState;
struct Connection;
struct Worker;
struct PipelineDefinition;
struct PipelineInstance;
struct SegmentDefinition;
struct SegmentInstance;

struct ControlPlaneStateBase
{
    ControlPlaneStateBase(const google::protobuf::Message& message);

    bool operator==(const ControlPlaneStateBase& other) const;

  private:
    const google::protobuf::Message& m_internal_message;
};

struct ControlPlaneState : public ControlPlaneStateBase
{
    ControlPlaneState(protos::ControlPlaneState& message);

    const std::map<uint64_t, Connection>& connections() const;

    const std::map<uint64_t, Worker>& workers() const;

    const std::map<uint64_t, PipelineDefinition>& pipeline_definitions() const;

    const std::map<uint64_t, PipelineInstance>& pipeline_instances() const;

    const std::map<uint64_t, SegmentDefinition>& segment_definitions() const;

    const std::map<uint64_t, SegmentInstance>& segment_instances() const;

  protected:
  private:
    protos::ControlPlaneState& m_message;

    std::map<uint64_t, Connection> m_connections;
    std::map<uint64_t, Worker> m_workers;
    std::map<uint64_t, PipelineDefinition> m_pipeline_definitions;
    std::map<uint64_t, PipelineInstance> m_pipeline_instances;
    std::map<uint64_t, SegmentDefinition> m_segment_definitions;
    std::map<uint64_t, SegmentInstance> m_segment_instances;
};

struct Connection : public ControlPlaneStateBase
{
    Connection(ControlPlaneState& root, const protos::Connection& message);

    uint64_t id() const;

    std::string peer_info() const;

    std::map<uint64_t, const Worker&> workers() const;

    std::map<uint64_t, const PipelineInstance&> assigned_pipelines() const;

  private:
    ControlPlaneState& m_root;
    const protos::Connection& m_message;
};

struct Worker : public ControlPlaneStateBase
{
    Worker(ControlPlaneState& root, const protos::Worker& message);

    uint64_t id() const;

    std::string worker_address() const;

    uint64_t machine_id() const;

    WorkerStates state() const;

    std::map<uint64_t, const SegmentInstance&> assigned_segments() const;

  private:
    ControlPlaneState& m_root;
    const protos::Worker& m_message;
};

struct PipelineDefinition : public ControlPlaneStateBase
{
    PipelineDefinition(ControlPlaneState& root, const protos::PipelineDefinition& message);

    uint64_t id() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

    std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> instances() const;

  private:
    ControlPlaneState& m_root;
    const protos::PipelineDefinition& m_message;
};

struct PipelineInstance : public ControlPlaneStateBase
{
    PipelineInstance(ControlPlaneState& root, const protos::PipelineInstance& message);

    uint64_t id() const;

    const PipelineDefinition& definition() const;

    uint64_t machine_id() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

  private:
    ControlPlaneState& m_root;
    const protos::PipelineInstance& m_message;
};

struct SegmentDefinition : public ControlPlaneStateBase
{
    SegmentDefinition(ControlPlaneState& root, const protos::SegmentDefinition& message);

    uint64_t id() const;

    std::string name() const;

    const PipelineDefinition& pipeline() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> instances() const;

    // TBD on ingress_ports
    // TBD on egress_ports
    // TBD on options

  private:
    ControlPlaneState& m_root;
    const protos::SegmentDefinition& m_message;
};

struct SegmentInstance : public ControlPlaneStateBase
{
    SegmentInstance(ControlPlaneState& root, const protos::SegmentInstance& message);

    uint64_t id() const;

    uint32_t address() const;

    const SegmentDefinition& definition() const;

    const Worker& worker() const;

    const PipelineInstance& pipeline() const;

  private:
    ControlPlaneState& m_root;
    const protos::SegmentInstance& m_message;
};

}  // namespace mrc::internal::control_plane::state
