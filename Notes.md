- System
  - Needed by all
- Runnable
  - One per host partition
  - Needs ref to system and index
- UCX
  - One per flattened partition
  - Needs network task queue from System host partition
- Host
  - One per host partition
  - Needs matching runnable
  - Needs `ucx::RegistrationCallbackBuilder` which gets added to matching UCX resource
- Device
  - One per device partition
  - Needs matching runnable
  - Optional
- Network
  - One per flattened partition
  - Needs matching runnable
  - Needs matching UCX
  - Needs matching host partition
- Partition Resources
  - One per flattened partition
  - Needs matching runnable
  - Needs matching host resources
  - Needs matching device resources
  - Needs matching network resources


1. Runtime object will do the following
   1. Construct `Runtime` from `System` object
   2. Holds onto `System` object
   3. Creates internal `RuntimeResources` object (aka `resources::Manager`)
   4. Creates Control Plane service if needed
   5. Acts as root manager object
2. Runtime initialization process
   1. Create Control plane server if needed (not part of the resources object anymore)
   2. Creates the Control plane client
      1. This may need a root Fiber queue
   3. [Optional] Can establish connection to client to get cloud topology information
      1. TBD, lets just use the functionality there now
   4. Initializes the `RuntimeResources`
      1. This builds the host/device/network resources as needed for each partition
   5. Creates the `PartitionManager` for each partition
      1. Each `PartitionManager` managest the lifetime of each partition
      2. Including registering, activating, unregistering each worker
      3. `PartitionManager` holds onto the data plane pieces


Hierarchy
- Executor
  - Runtime/Connection
    - Partition/Worker
      - Segments
    - Pipelines
      - Manifolds


Other Notes:
- Pipelines/Manifolds will be created on-demand
  - Meaning, they will only be created once there is a segment to start that requests it
- Egress ports
  - Will be converted to pull-based queues
    - Downstream ingress ports will need to have a progress engine to pull from


Resources:
- `system::Options`
  - Configurable options that can be specified by the user
- `system::System`
  - Uses the `system::Options` to configure the system
  - Describes the available hardware
  - Configures the default partition layout
- `system::SystemProvider`
  - Returns a `system::System` object
- `system::ThreadingResources`
  - Created with a `system::SystemProvider`
  - Manages creating all threads/fibers
    - Stores any thread initializers/finalizers
  - Holds onto the ThreadManager/FiberManager
  - `FiberManager`
    - For system cpuset, create a thread on each cpu and set up `FiberTaskQueue` on each
- `resources::SystemResources`
  - Created with a `system::SystemProvider`
  - Creates or uses `system::ThreadingResources`
  - Generates resource objects from the partition info in `system::System`
    - Create resources for runnable, host, device, ucx, and network
    - Creates a root `IResources` object with the full cpuset
      - Adds the root `RunnableResources`
    - Create child partition `IResources` object called `PartitionResources`
- `resources::PartitionResources`
  - Inherits from `IResources`
  - Created using
- `runtime::Runtime`
  - Created with `resources::SystemResources`
  - Begins service startup procedure
    - Creates `control_plane::Server`
    - Creates `control_plane::Client`
    - Creates `PipelineManager` <== Seems out of place
  - Creats the child `runtime::PartitionRuntime` object
    - Adds each as a child service
