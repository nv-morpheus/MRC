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
   5. Creates the `SegmentsManager` for each partition
      1. Each `SegmentsManager` managest the lifetime of each partition
      2. Including registering, activating, unregistering each worker
      3. `SegmentsManager` holds onto the data plane pieces


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

Resources functionality:
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

Resources:
- `IResources`
  - `runnable()`
    - `main()`
      - For enqueuing fiber work
    -


## Data Plane

Existing Objects:
- `IStorage`
  - Allows access to chunks of memory
  - Holds memory as "objects"
- `IEncodableStorage`
  - Interface which allows you to write memory into the `IStorage`
- `IDecodableStorage`
  - Interface which allows you to read memory from `IStorage`
- `ICodableStorage` : `IDecodableStorage`, `IEncodableStorage`
  - Both readable and writable from `IStorage`
- `EncodedStorage`
  - Object which holds onto an `IDecodableStorage` so you can read from it
  - Usually, this is created via an encoding process and is read-only
- `EncodedObject<T>` : `EncodedStorage`
  - Typed version of `EncodedStorage`.
  - Handles calling `mrc::codable::encode(object, *storage);` for you with the supplied object type

New Object Ideas:
- `IDescriptor`
  - Descriptor is an object which holds a piece of data. That data can either be:
    - In memory already as a type T, ready to be retrieved
    - Can be encoded into a storage object (i.e. just pulled from UCX)
  - Purpose of this object is to facilitate the lifetime of data as it moves between segments
    - When you create a `IDescriptor` you are transferring ownership of that piece of data
    - Ownership of the object can be transferred out of the `IDescriptor` by retrieving the storage object backing the
      data
      - Retrieving the storage can be triggered by either accessing the value locally, or by accessing the storage
        object (for remote)
  - `std::unique_ptr<IDecodableStorage> get_storage() = 0`
    - Gets the storage associated with this descriptor.
    - When called, the storage is moved out of the `IDescriptor` and it no longer has any data associated with it
  - `T decode<T>()`
    - Allows you to retrieve the value from the storage on a type erased object
    - calls `get_storage()` internally
- `RemoteDescriptor: IDescriptor`
  - Came from UCX. Holds onto some `DecodableStorageView` for accessing data
- `Descriptor<T>: IDescriptor`
  - Typed version of `IDescriptor
  - `T get_value() = 0`
    - Falls back to `this->decode<T>()`
- `ResidentDescriptor<T>: Descriptor<T>`
  - Used for local segment->segment connections
  - Holds onto a piece of data `T` and returns it directly when `get_value` is called
  - Can also serialize it on demand if `get_storage` is called
- `CodedDescriptor<T>: Descriptor<T>`
  - Used for remote segment->segment connections
  - Transferrs ownership of T immediately to a storage object


Steps for the Publisher:
1. Encoding from `T` -> `EncodedStorage`
   1. Requires getting a `CodableStorage` object before encoding
   2. `CodableStorage` object requires a partition runtime to get the UCX `registration_cache()`
   3. For `Publisher<T>` this runs without a progress engine using the thread of the caller
   4. Pushes output into a channel on the `PublisherService`
2. From `EncdedStorage` -> `RemoteDescriptor`
   1. Uses the `RemoteDescriptorManager` to handle the conversion
   2. Runs on the `main` engine of a partition
3. From `RemoteDescriptor` -> `RemoteDescriptorMessage`
   1. Combining a `RemoteDescriptor` with a tag and a UCX `Endpoint`
   2. Runs on the `main` engine of a partition in the same thread as the previous step
   3. Pushes output into a channel on the `DataPlaneClient`
4. From `RemoteDescriptorMessage` -> `Request`
   1. Takes the message and actually writes it to UCX
   2. Runs on the `mrc_network` engine of a partition
   3. Awaits for completion before returning
   4. The progress engine for this step usually has 16+ progress engines running on 1 thread

Steps for the Subscriber:
1. UCX Worker creates a `TransientBuffer` with the data
   1. Pushes output to a router a `std::pair<tag, TransientBuffer>` called `deserialize_source()` on `DataPlaneServer`
   2. Runs on the `mrc_network` engine of a partition
   3. Output lands in a channel on the `SubscriberService`
2. From `TransientBuffer` -> `RemoteDescriptor`
   1. Converts the buffer into a RemoteDescriptor proto
   2. Releases the buffer to be reused
   3. Converts the RemoteDescriptor proto into a `RemoteDescriptor` via the `RemoteDescriptorManager`
   4. Runs on the `main` engine of a partition
3. From `RemoteDescriptor` -> `T`
   1. Via `RemoteDescriptor::decode<T>()`
   2. Runs on the calling thread from the `Subscriber<T>`


Publisher<T> Data Path:
 * `T` -> `CodedDescriptor<T>` -> `LocalDescriptorHandle` -> `DescriptorMessage` -> `protos::RemoteDescriptor` -> `TransientBuffer` -> Data Plane Tagged Send
 * `T` -> `CodedDescriptor<T>`
   * Creates the encoding on the egress thread
 * `CodedDescriptor<T>` -> `LocalDescriptorHandle`
   * Registers the `CodedDescriptor<T>` with the Remote Descriptor Manager to track tokens and lifetime
   * Returns a `LocalDescriptorHandle` which tracks an internal `Storage` and can be turned into a Protobuf
 * `LocalDescriptorHandle` -> `DescriptorMessage`
   * Pairs the descriptor handle with a destination
 * `DescriptorMessage` -> `protos::RemoteDescriptor`
   * extracts the protobuf from the `LocalDescriptorHandle` (releasing it from tracking the `Storage` lifetime)
 *
 *
 * Subscriber<T> Data Path:
 * Data Plane Tagged Received -> Transient Buffer -> RemoteDescriptor -> Subscriber/Source<T> ->
