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
class PipelineDefinition_ManifoldDefinition;
class PipelineInstance;
class SegmentInstance;
class ManifoldInstance;
}  // namespace mrc::protos

namespace mrc::control_plane::state {

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

enum class ResourceRequestedStatus : int
{
    // Should never be used. Must start with 0
    Unknown = 0,
    // Requested that a placeholder be reserved for this resource
    Initialized = 1,
    // Requested that the resource be created but not started
    Created = 3,
    // Requested that the resource run to completion
    Completed = 5,
    // Requested that the resource be stopped
    Stopped = 7,
    // Requested that the resource be destroyed (and removed from the control plane)
    Destroyed = 9,
};

enum ResourceActualStatus : int
{
    // Resource has not informed its status
    Unknown = 0,
    // Owner of resource has acknowledged it should be created
    Initialized = 1,
    // Resource has acknowledged it should be created and has begun the process
    Creating = 2,
    // Resource is created and can be moved to ready when requested
    Created = 3,
    // Resource is running and will be moved to completed when finished
    Running = 4,
    // Resource is done running and ready to be torn down
    Completed = 5,
    // Resource has acknowledged it should be stopped and has begun the process
    Stopping = 6,
    // Resource has completed the stopped process
    Stopped = 7,
    // Owner of resource has begun destroying the object
    Destroying = 8,
    // Owner of resource has destroyed the object. Can be removed from control plane
    Destroyed = 9,
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
struct ManifoldInstance;
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

    uint64_t nonce;
    std::map<uint64_t, Connection> connections;
    std::map<uint64_t, Worker> workers;
    std::map<uint64_t, PipelineDefinition> pipeline_definitions;
    std::map<uint64_t, PipelineInstance> pipeline_instances;
    std::map<uint64_t, ManifoldInstance> manifold_instances;
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

    const std::map<uint64_t, ManifoldInstance>& manifold_instances() const;

    const std::map<uint64_t, SegmentInstance>& segment_instances() const;

  private:
    // Store as shared ptr to allow for trivial copies of this class without copying underlying data
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
};

template <typename ProtoT>
struct ControlPlaneTopLevelMessage : public ControlPlaneStateBase
{
    ControlPlaneTopLevelMessage(std::shared_ptr<ControlPlaneNormalizedState> state, const ProtoT& message) :
      ControlPlaneStateBase(message),
      m_root_state(std::move(state)),
      m_message(message)
    {}

  protected:
    std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    const ProtoT& m_message;
};

struct ResourceState : public ControlPlaneStateBase
{
    ResourceState(const protos::ResourceState& message);

    ResourceRequestedStatus requested_status() const;

    ResourceActualStatus actual_status() const;

    int32_t ref_count() const;

  private:
    const protos::ResourceState& m_message;
};

template <typename ProtoT>
struct ResourceTopLevelMessage : public ControlPlaneTopLevelMessage<ProtoT>
{
    ResourceTopLevelMessage(std::shared_ptr<ControlPlaneNormalizedState> state, const ProtoT& message) :
      ControlPlaneTopLevelMessage<ProtoT>(std::move(state), message),
      m_state(message.state())
    {}

    const ResourceState& state() const
    {
        return m_state;
    }

  private:
    ResourceState m_state;
};

struct Connection : public ResourceTopLevelMessage<protos::Connection>
{
    using ResourceTopLevelMessage::ResourceTopLevelMessage;
    // Connection(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Connection& message);

    uint64_t id() const;

    std::string peer_info() const;

    std::map<uint64_t, const Worker&> workers() const;

    std::map<uint64_t, const PipelineInstance&> assigned_pipelines() const;

    //   private:
    //     std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    //     const protos::Connection& m_message;
};

struct Worker : public ResourceTopLevelMessage<protos::Worker>
{
    using ResourceTopLevelMessage::ResourceTopLevelMessage;
    // Worker(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::Worker& message);

    uint64_t id() const;

    std::string worker_address() const;

    uint64_t machine_id() const;

    std::map<uint64_t, const SegmentInstance&> assigned_segments() const;
};

struct PipelineConfiguration : public ControlPlaneStateBase
{
    PipelineConfiguration(const protos::PipelineConfiguration& message);

  private:
    const protos::PipelineConfiguration& m_message;
};

struct PipelineDefinition : public ControlPlaneTopLevelMessage<protos::PipelineDefinition>
{
    struct ManifoldDefinition : public ControlPlaneStateBase
    {
        ManifoldDefinition(std::shared_ptr<ControlPlaneNormalizedState> state,
                           const protos::PipelineDefinition_ManifoldDefinition& message);

        uint64_t id() const;

        const PipelineDefinition& parent() const;

        std::string port_name() const;

        std::map<uint64_t, std::reference_wrapper<const ManifoldInstance>> instances() const;

      private:
        std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
        const protos::PipelineDefinition_ManifoldDefinition& m_message;
    };

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

    const std::map<std::string, ManifoldDefinition>& manifolds() const;
    const std::map<std::string, SegmentDefinition>& segments() const;

  private:
    // std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    // const protos::PipelineDefinition& m_message;

    // Child messages
    PipelineConfiguration m_config;
    std::map<std::string, ManifoldDefinition> m_manifolds;
    std::map<std::string, SegmentDefinition> m_segments;
};

struct PipelineInstance : public ResourceTopLevelMessage<protos::PipelineInstance>
{
    using ResourceTopLevelMessage::ResourceTopLevelMessage;
    // PipelineInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::PipelineInstance& message);

    uint64_t id() const;

    const PipelineDefinition& definition() const;

    uint64_t machine_id() const;

    std::map<uint64_t, std::reference_wrapper<const ManifoldInstance>> manifolds() const;
    std::map<uint64_t, std::reference_wrapper<const SegmentInstance>> segments() const;

  private:
    // std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    // const protos::PipelineInstance& m_message;

    // ResourceState m_state;
};

struct ManifoldInstance : public ResourceTopLevelMessage<protos::ManifoldInstance>
{
    using ResourceTopLevelMessage::ResourceTopLevelMessage;

    // ManifoldInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::ManifoldInstance& message);

    uint64_t id() const;

    const PipelineDefinition& pipeline_definition() const;

    std::string port_name() const;

    uint64_t machine_id() const;

    const PipelineInstance& pipeline_instance() const;

    std::map<uint32_t, bool> requested_output_segments() const;

    std::map<uint32_t, bool> requested_input_segments() const;

  private:
    // std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    // const protos::ManifoldInstance& m_message;

    // ResourceState m_state;
};

struct SegmentInstance : public ResourceTopLevelMessage<protos::SegmentInstance>
{
    using ResourceTopLevelMessage::ResourceTopLevelMessage;

    // SegmentInstance(std::shared_ptr<ControlPlaneNormalizedState> state, const protos::SegmentInstance& message);

    uint64_t id() const;

    const PipelineDefinition& pipeline_definition() const;

    std::string name() const;

    uint32_t address() const;

    const Worker& worker() const;

    const PipelineInstance& pipeline_instance() const;

  private:
    // std::shared_ptr<ControlPlaneNormalizedState> m_root_state;
    // const protos::SegmentInstance& m_message;
};

}  // namespace mrc::control_plane::state
