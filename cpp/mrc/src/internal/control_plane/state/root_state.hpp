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
class ResourceState;
class Connection;
class Worker;
class PipelineConfiguration;
class PipelineDefinition;
class PipelineDefinition_SegmentDefinition;
class PipelineInstance;
// class SegmentDefinition;
class SegmentInstance;
}  // namespace mrc::protos

namespace mrc::internal::control_plane::state {

// Mirror this enum so we dont have to include all the proto classes
enum class ResourceStatus : int
{
    // Control Plane indicates a resource should be created on the client
    Registered = 0,
    // Client has created resource but it is not ready
    Activated = 1,
    // Client and Control Plane can use the resource
    Ready = 2,
    // Control Plane has indicated the resource should be destroyed on the client. All users of the resource should stop
    // using it and decrement the ref count. Object is still running
    Deactivating = 3,
    // All ref counts have been decremented. Owner of the object on the client can begin destroying object
    Deactivated = 4,
    // Client owner of resource has begun destroying the object
    Unregistered = 5,
    // Object has been destroyed on the client (and may be removed from the server)
    Destroyed = 6,
};

enum class SegmentStates : int
{
    Initialized = 0,
    Running     = 1,
    Stopped     = 2,
    Completed   = 3,
};

struct ControlPlaneState;
struct Connection;
struct Worker;
struct PipelineConfiguration;
struct PipelineDefinition;
struct PipelineInstance;
// struct SegmentDefinition;
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
    // std::map<uint64_t, SegmentDefinition> segment_definitions;
    std::map<uint64_t, SegmentInstance> segment_instances;

    friend struct ControlPlaneState;
};

struct ControlPlaneState
{
    ControlPlaneState(std::unique_ptr<protos::ControlPlaneState> message);

    const std::map<uint64_t, Connection>& connections() const;

    const std::map<uint64_t, Worker>& workers() const;

    const std::map<uint64_t, PipelineDefinition>& pipeline_definitions() const;

    const std::map<uint64_t, PipelineInstance>& pipeline_instances() const;

    // const std::map<uint64_t, SegmentDefinition>& segment_definitions() const;

    const std::map<uint64_t, SegmentInstance>& segment_instances() const;

  private:
    // Store as shared ptr to allow for trivial copies of this class without copying underlying data
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
};

struct ResourceState : public ControlPlaneStateBase
{
    ResourceState(const protos::ResourceState& message);

    ResourceStatus status() const;

    int32_t ref_count() const;

  private:
    const protos::ResourceState& m_message;
};

struct Connection : public ControlPlaneStateBase
{
    Connection(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Connection& message);

    uint64_t id() const;

    std::string peer_info() const;

    std::map<uint64_t, const Worker&> workers() const;

    std::map<uint64_t, const PipelineInstance&> assigned_pipelines() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const protos::Connection& m_message;
};

struct Worker : public ControlPlaneStateBase
{
    Worker(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Worker& message);

    uint64_t id() const;

    std::string worker_address() const;

    uint64_t machine_id() const;

    const ResourceState& state() const;

    std::map<uint64_t, const SegmentInstance&> assigned_segments() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const protos::Worker& m_message;

    ResourceState m_state;
};

struct PipelineConfiguration : public ControlPlaneStateBase
{
    PipelineConfiguration(const protos::PipelineConfiguration& message);

  private:
    const protos::PipelineConfiguration& m_message;
};

struct PipelineDefinition : public ControlPlaneStateBase
{
    struct SegmentDefinition : public ControlPlaneStateBase
    {
        SegmentDefinition(std::shared_ptr<ControlPlaneNormalizedState> state,
                          const protos::PipelineDefinition_SegmentDefinition& message);

        uint64_t id() const;

        const PipelineDefinition& parent() const;

        std::string name() const;

        std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> instances() const;

      private:
        std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
        const protos::PipelineDefinition_SegmentDefinition& m_message;
    };

    PipelineDefinition(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::PipelineDefinition& message);

    uint64_t id() const;

    const PipelineConfiguration& config() const;

    std::map<uint64_t, std::reference_wrapper<const PipelineInstance>> instances() const;

    const std::map<uint64_t, SegmentDefinition>& segments() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const protos::PipelineDefinition& m_message;

    // Child messages
    PipelineConfiguration m_config;
    std::map<uint64_t, SegmentDefinition> m_segments;
};

struct PipelineInstance : public ControlPlaneStateBase
{
    PipelineInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::PipelineInstance& message);

    uint64_t id() const;

    const PipelineDefinition& definition() const;

    uint64_t machine_id() const;

    const ResourceState& state() const;

    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const protos::PipelineInstance& m_message;

    ResourceState m_state;
};

// struct SegmentDefinition : public ControlPlaneStateBase
// {
//     SegmentDefinition(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::SegmentDefinition& message);

//     uint64_t id() const;

//     std::string name() const;

//     const PipelineDefinition& pipeline() const;

//     std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> instances() const;

//     // TBD on ingress_ports
//     // TBD on egress_ports
//     // TBD on options

//   private:
//     std::shared_ptr<ControlPlaneNormalizedState> m_state;
//     const protos::SegmentDefinition& m_message;
// };

struct SegmentInstance : public ControlPlaneStateBase
{
    SegmentInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::SegmentInstance& message);

    uint64_t id() const;

    const PipelineDefinition& pipeline_definition() const;

    std::string name() const;

    uint32_t address() const;

    const Worker& worker() const;

    const PipelineInstance& pipeline_instance() const;

    SegmentStates state() const;

  private:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const protos::SegmentInstance& m_message;
};

}  // namespace mrc::internal::control_plane::state
