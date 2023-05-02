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
