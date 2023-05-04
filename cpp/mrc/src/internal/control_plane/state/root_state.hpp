#include <cstdint>
#include <map>
#include <memory>
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

struct ControlPlaneNormalizedState : public std::enable_shared_from_this<ControlPlaneNormalizedState>
{
  private:
    ControlPlaneNormalizedState(std::unique_ptr<protos::ControlPlaneState> root_message);

    // Call right after constructor. Needed to allow for using `shared_from_this()`
    void initialize();

    static std::shared_ptr<ControlPlaneNormalizedState> create(std::unique_ptr<protos::ControlPlaneState> root_message);

  public:
    std::unique_ptr<protos::ControlPlaneState> root_message;

    std::map<uint64_t, Connection> connections;
    std::map<uint64_t, Worker> workers;
    std::map<uint64_t, PipelineDefinition> pipeline_definitions;
    std::map<uint64_t, PipelineInstance> pipeline_instances;
    std::map<uint64_t, SegmentDefinition> segment_definitions;
    std::map<uint64_t, SegmentInstance> segment_instances;

    friend struct ControlPlaneState;
};

struct ControlPlaneState : public ControlPlaneStateBase
{
    ControlPlaneState(std::unique_ptr<protos::ControlPlaneState> message);

    const std::map<uint64_t, Connection>& connections() const;

    const std::map<uint64_t, Worker>& workers() const;

    const std::map<uint64_t, PipelineDefinition>& pipeline_definitions() const;

    const std::map<uint64_t, PipelineInstance>& pipeline_instances() const;

    const std::map<uint64_t, SegmentDefinition>& segment_definitions() const;

    const std::map<uint64_t, SegmentInstance>& segment_instances() const;

  private:
    // Store as shared ptr to allow for trivial copies of this class without copying underlying data
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
};

struct Connection : public ControlPlaneStateBase
{
    Connection(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Connection& message);

    uint64_t id() const;

    std::string peer_info() const;

    std::map<uint64_t, const Worker&> workers() const;

    std::map<uint64_t, const PipelineInstance&> assigned_pipelines() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::Connection& m_message;
};

struct Worker : public ControlPlaneStateBase
{
    Worker(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Worker& message);

    uint64_t id() const;

    std::string worker_address() const;

    uint64_t machine_id() const;

    WorkerStates state() const;

    std::map<uint64_t, const SegmentInstance&> assigned_segments() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::Worker& m_message;
};

struct PipelineDefinition : public ControlPlaneStateBase
{
    PipelineDefinition(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::PipelineDefinition& message);

    uint64_t id() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

    std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> instances() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::PipelineDefinition& m_message;
};

struct PipelineInstance : public ControlPlaneStateBase
{
    PipelineInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::PipelineInstance& message);

    uint64_t id() const;

    const PipelineDefinition& definition() const;

    uint64_t machine_id() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::PipelineInstance& m_message;
};

struct SegmentDefinition : public ControlPlaneStateBase
{
    SegmentDefinition(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::SegmentDefinition& message);

    uint64_t id() const;

    std::string name() const;

    const PipelineDefinition& pipeline() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> instances() const;

    // TBD on ingress_ports
    // TBD on egress_ports
    // TBD on options

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::SegmentDefinition& m_message;
};

struct SegmentInstance : public ControlPlaneStateBase
{
    SegmentInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::SegmentInstance& message);

    uint64_t id() const;

    uint32_t address() const;

    const SegmentDefinition& definition() const;

    const Worker& worker() const;

    const PipelineInstance& pipeline() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_state;
    const protos::SegmentInstance& m_message;
};

}  // namespace mrc::internal::control_plane::state
