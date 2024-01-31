/* eslint-disable */
import Long from "long";
import _m0 from "protobufjs/minimal";
import { messageTypeRegistry } from "../../typeRegistry";

export const protobufPackage = "mrc.protos";

export enum ResourceStatus {
  /** Registered - Control Plane indicates a resource should be created on the client */
  Registered = "Registered",
  /** Activated - Client has created resource but it is not ready */
  Activated = "Activated",
  /** Ready - Client and Control Plane can use the resource */
  Ready = "Ready",
  /**
   * Deactivating - Control Plane has indicated the resource should be destroyed on the client. All users of the resource should stop
   * using it and decrement the ref count. Object is still running
   */
  Deactivating = "Deactivating",
  /** Deactivated - All ref counts have been decremented. Owner of the object on the client can begin destroying object */
  Deactivated = "Deactivated",
  /** Unregistered - Client owner of resource has begun destroying the object */
  Unregistered = "Unregistered",
  /** Destroyed - Object has been destroyed on the client (and may be removed from the server) */
  Destroyed = "Destroyed",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function resourceStatusFromJSON(object: any): ResourceStatus {
  switch (object) {
    case 0:
    case "Registered":
      return ResourceStatus.Registered;
    case 1:
    case "Activated":
      return ResourceStatus.Activated;
    case 2:
    case "Ready":
      return ResourceStatus.Ready;
    case 3:
    case "Deactivating":
      return ResourceStatus.Deactivating;
    case 4:
    case "Deactivated":
      return ResourceStatus.Deactivated;
    case 5:
    case "Unregistered":
      return ResourceStatus.Unregistered;
    case 6:
    case "Destroyed":
      return ResourceStatus.Destroyed;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ResourceStatus.UNRECOGNIZED;
  }
}

export function resourceStatusToJSON(object: ResourceStatus): string {
  switch (object) {
    case ResourceStatus.Registered:
      return "Registered";
    case ResourceStatus.Activated:
      return "Activated";
    case ResourceStatus.Ready:
      return "Ready";
    case ResourceStatus.Deactivating:
      return "Deactivating";
    case ResourceStatus.Deactivated:
      return "Deactivated";
    case ResourceStatus.Unregistered:
      return "Unregistered";
    case ResourceStatus.Destroyed:
      return "Destroyed";
    case ResourceStatus.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function resourceStatusToNumber(object: ResourceStatus): number {
  switch (object) {
    case ResourceStatus.Registered:
      return 0;
    case ResourceStatus.Activated:
      return 1;
    case ResourceStatus.Ready:
      return 2;
    case ResourceStatus.Deactivating:
      return 3;
    case ResourceStatus.Deactivated:
      return 4;
    case ResourceStatus.Unregistered:
      return 5;
    case ResourceStatus.Destroyed:
      return 6;
    case ResourceStatus.UNRECOGNIZED:
    default:
      return -1;
  }
}

export enum ResourceRequestedStatus {
  /** Requested_Unknown - Should never be used. Must start with 0 */
  Requested_Unknown = "Requested_Unknown",
  /** Requested_Initialized - Requested that a placeholder be reserved for this resource */
  Requested_Initialized = "Requested_Initialized",
  /** Requested_Created - Requested that the resource be created but not started */
  Requested_Created = "Requested_Created",
  /** Requested_Completed - Requested that the resource run to completion */
  Requested_Completed = "Requested_Completed",
  /** Requested_Stopped - Requested that the resource be stopped */
  Requested_Stopped = "Requested_Stopped",
  /** Requested_Destroyed - Requested that the resource be destroyed (and removed from the control plane) */
  Requested_Destroyed = "Requested_Destroyed",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function resourceRequestedStatusFromJSON(object: any): ResourceRequestedStatus {
  switch (object) {
    case 0:
    case "Requested_Unknown":
      return ResourceRequestedStatus.Requested_Unknown;
    case 1:
    case "Requested_Initialized":
      return ResourceRequestedStatus.Requested_Initialized;
    case 3:
    case "Requested_Created":
      return ResourceRequestedStatus.Requested_Created;
    case 5:
    case "Requested_Completed":
      return ResourceRequestedStatus.Requested_Completed;
    case 7:
    case "Requested_Stopped":
      return ResourceRequestedStatus.Requested_Stopped;
    case 9:
    case "Requested_Destroyed":
      return ResourceRequestedStatus.Requested_Destroyed;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ResourceRequestedStatus.UNRECOGNIZED;
  }
}

export function resourceRequestedStatusToJSON(object: ResourceRequestedStatus): string {
  switch (object) {
    case ResourceRequestedStatus.Requested_Unknown:
      return "Requested_Unknown";
    case ResourceRequestedStatus.Requested_Initialized:
      return "Requested_Initialized";
    case ResourceRequestedStatus.Requested_Created:
      return "Requested_Created";
    case ResourceRequestedStatus.Requested_Completed:
      return "Requested_Completed";
    case ResourceRequestedStatus.Requested_Stopped:
      return "Requested_Stopped";
    case ResourceRequestedStatus.Requested_Destroyed:
      return "Requested_Destroyed";
    case ResourceRequestedStatus.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function resourceRequestedStatusToNumber(object: ResourceRequestedStatus): number {
  switch (object) {
    case ResourceRequestedStatus.Requested_Unknown:
      return 0;
    case ResourceRequestedStatus.Requested_Initialized:
      return 1;
    case ResourceRequestedStatus.Requested_Created:
      return 3;
    case ResourceRequestedStatus.Requested_Completed:
      return 5;
    case ResourceRequestedStatus.Requested_Stopped:
      return 7;
    case ResourceRequestedStatus.Requested_Destroyed:
      return 9;
    case ResourceRequestedStatus.UNRECOGNIZED:
    default:
      return -1;
  }
}

export enum ResourceActualStatus {
  /** Actual_Unknown - Resource has not informed its status */
  Actual_Unknown = "Actual_Unknown",
  /** Actual_Initialized - Owner of resource has acknowledged it should be created */
  Actual_Initialized = "Actual_Initialized",
  /** Actual_Creating - Resource has acknowledged it should be created and has begun the process */
  Actual_Creating = "Actual_Creating",
  /** Actual_Created - Resource is created and can be moved to ready when requested */
  Actual_Created = "Actual_Created",
  /** Actual_Running - Resource is running and will be moved to completed when finished */
  Actual_Running = "Actual_Running",
  /** Actual_Completed - Resource is done running and ready to be torn down */
  Actual_Completed = "Actual_Completed",
  /** Actual_Stopping - Resource has acknowledged it should be stopped and has begun the process */
  Actual_Stopping = "Actual_Stopping",
  /** Actual_Stopped - Resource has completed the stopped process */
  Actual_Stopped = "Actual_Stopped",
  /** Actual_Destroying - Owner of resource has begun destroying the object */
  Actual_Destroying = "Actual_Destroying",
  /** Actual_Destroyed - Owner of resource has destroyed the object. Can be removed from control plane */
  Actual_Destroyed = "Actual_Destroyed",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function resourceActualStatusFromJSON(object: any): ResourceActualStatus {
  switch (object) {
    case 0:
    case "Actual_Unknown":
      return ResourceActualStatus.Actual_Unknown;
    case 1:
    case "Actual_Initialized":
      return ResourceActualStatus.Actual_Initialized;
    case 2:
    case "Actual_Creating":
      return ResourceActualStatus.Actual_Creating;
    case 3:
    case "Actual_Created":
      return ResourceActualStatus.Actual_Created;
    case 4:
    case "Actual_Running":
      return ResourceActualStatus.Actual_Running;
    case 5:
    case "Actual_Completed":
      return ResourceActualStatus.Actual_Completed;
    case 6:
    case "Actual_Stopping":
      return ResourceActualStatus.Actual_Stopping;
    case 7:
    case "Actual_Stopped":
      return ResourceActualStatus.Actual_Stopped;
    case 8:
    case "Actual_Destroying":
      return ResourceActualStatus.Actual_Destroying;
    case 9:
    case "Actual_Destroyed":
      return ResourceActualStatus.Actual_Destroyed;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ResourceActualStatus.UNRECOGNIZED;
  }
}

export function resourceActualStatusToJSON(object: ResourceActualStatus): string {
  switch (object) {
    case ResourceActualStatus.Actual_Unknown:
      return "Actual_Unknown";
    case ResourceActualStatus.Actual_Initialized:
      return "Actual_Initialized";
    case ResourceActualStatus.Actual_Creating:
      return "Actual_Creating";
    case ResourceActualStatus.Actual_Created:
      return "Actual_Created";
    case ResourceActualStatus.Actual_Running:
      return "Actual_Running";
    case ResourceActualStatus.Actual_Completed:
      return "Actual_Completed";
    case ResourceActualStatus.Actual_Stopping:
      return "Actual_Stopping";
    case ResourceActualStatus.Actual_Stopped:
      return "Actual_Stopped";
    case ResourceActualStatus.Actual_Destroying:
      return "Actual_Destroying";
    case ResourceActualStatus.Actual_Destroyed:
      return "Actual_Destroyed";
    case ResourceActualStatus.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function resourceActualStatusToNumber(object: ResourceActualStatus): number {
  switch (object) {
    case ResourceActualStatus.Actual_Unknown:
      return 0;
    case ResourceActualStatus.Actual_Initialized:
      return 1;
    case ResourceActualStatus.Actual_Creating:
      return 2;
    case ResourceActualStatus.Actual_Created:
      return 3;
    case ResourceActualStatus.Actual_Running:
      return 4;
    case ResourceActualStatus.Actual_Completed:
      return 5;
    case ResourceActualStatus.Actual_Stopping:
      return 6;
    case ResourceActualStatus.Actual_Stopped:
      return 7;
    case ResourceActualStatus.Actual_Destroying:
      return 8;
    case ResourceActualStatus.Actual_Destroyed:
      return 9;
    case ResourceActualStatus.UNRECOGNIZED:
    default:
      return -1;
  }
}

export enum SegmentMappingPolicies {
  /** Disabled - Do not run this segment for the specified machine ID */
  Disabled = "Disabled",
  /** OnePerWorker - Run one instance of this segment per worker on the specified machine ID */
  OnePerWorker = "OnePerWorker",
  /** OnePerExecutor - Run one instance of this segment per executor on the specified machine ID */
  OnePerExecutor = "OnePerExecutor",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function segmentMappingPoliciesFromJSON(object: any): SegmentMappingPolicies {
  switch (object) {
    case 0:
    case "Disabled":
      return SegmentMappingPolicies.Disabled;
    case 1:
    case "OnePerWorker":
      return SegmentMappingPolicies.OnePerWorker;
    case 2:
    case "OnePerExecutor":
      return SegmentMappingPolicies.OnePerExecutor;
    case -1:
    case "UNRECOGNIZED":
    default:
      return SegmentMappingPolicies.UNRECOGNIZED;
  }
}

export function segmentMappingPoliciesToJSON(object: SegmentMappingPolicies): string {
  switch (object) {
    case SegmentMappingPolicies.Disabled:
      return "Disabled";
    case SegmentMappingPolicies.OnePerWorker:
      return "OnePerWorker";
    case SegmentMappingPolicies.OnePerExecutor:
      return "OnePerExecutor";
    case SegmentMappingPolicies.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function segmentMappingPoliciesToNumber(object: SegmentMappingPolicies): number {
  switch (object) {
    case SegmentMappingPolicies.Disabled:
      return 0;
    case SegmentMappingPolicies.OnePerWorker:
      return 1;
    case SegmentMappingPolicies.OnePerExecutor:
      return 2;
    case SegmentMappingPolicies.UNRECOGNIZED:
    default:
      return -1;
  }
}

/**
 * // Current status of the resource
 * ResourceStatus status = 1;
 */
export interface ResourceState {
  $type: "mrc.protos.ResourceState";
  /** What the control plane would like the resource to be */
  requestedStatus: ResourceRequestedStatus;
  /** What the local resource has reported its state as */
  actualStatus: ResourceActualStatus;
  /** Number of users besides the owner of this resource */
  refCount: number;
}

export interface Executor {
  $type: "mrc.protos.Executor";
  /** The generated ExecutorID for this instance */
  id: string;
  /** The ExecutorAddress of this worker. 16 bit unused + 16 bit ExecutorID */
  executorAddress: number;
  /** Info about the client (IP/Port) */
  peerInfo: string;
  /** Serialized UCX worker address */
  ucxAddress: Uint8Array;
  /** The pipeline instances that are assigned to this machine */
  assignedPipelineIds: string[];
  /** The pipeline definitions that are assigned to this machine */
  mappedPipelineDefinitions: string[];
  /** The segment instances that are assigned to this worker */
  assignedSegmentIds: string[];
  /** Current state */
  state:
    | ResourceState
    | undefined;
  /**
   * TODO(MDD): Remove when removing partitions
   * Workers that belong to this machine
   */
  workerIds: string[];
}

/** TODO(MDD): Remove when removing partitions */
export interface Worker {
  $type: "mrc.protos.Worker";
  /** The generated PartitionID for this instance */
  id: string;
  /** The executor ID associated with this instance */
  executorId: string;
  /** The PartitionAddress of this worker. 16 bit unused + 16 bit PartitionID */
  partitionAddress: number;
  /** Serialized UCX worker address */
  ucxAddress: Uint8Array;
  /** Current state of the worker */
  state:
    | ResourceState
    | undefined;
  /** The segment instances that are assigned to this worker */
  assignedSegmentIds: string[];
}

export interface PipelineConfiguration {
  $type: "mrc.protos.PipelineConfiguration";
  segments: { [key: string]: PipelineConfiguration_SegmentConfiguration };
  manifolds: { [key: string]: PipelineConfiguration_ManifoldConfiguration };
}

export interface PipelineConfiguration_SegmentConfiguration {
  $type: "mrc.protos.PipelineConfiguration.SegmentConfiguration";
  /** Name of the segment */
  name: string;
  /** Hashed name of the segment. Only the lower 16 bits are used */
  nameHash: number;
  /** Ingress ports for this segment */
  ingressPorts: string[];
  /** Egress ports for this segment */
  egressPorts: string[];
  /** Segment options */
  options: SegmentOptions | undefined;
}

export interface PipelineConfiguration_ManifoldConfiguration {
  $type: "mrc.protos.PipelineConfiguration.ManifoldConfiguration";
  /** Name of the manifold */
  portName: string;
  /** Hashed name of the port. Only the lower 16 bits are used */
  portHash: number;
  /** TypeID for this manifold */
  typeId: number;
  /** Friendly type string */
  typeString: string;
  /** All options for this config */
  options: ManifoldOptions | undefined;
}

export interface PipelineConfiguration_SegmentsEntry {
  $type: "mrc.protos.PipelineConfiguration.SegmentsEntry";
  key: string;
  value: PipelineConfiguration_SegmentConfiguration | undefined;
}

export interface PipelineConfiguration_ManifoldsEntry {
  $type: "mrc.protos.PipelineConfiguration.ManifoldsEntry";
  key: string;
  value: PipelineConfiguration_ManifoldConfiguration | undefined;
}

export interface PipelineMapping {
  $type: "mrc.protos.PipelineMapping";
  executorId: string;
  segments: { [key: string]: PipelineMapping_SegmentMapping };
}

export interface PipelineMapping_SegmentMapping {
  $type: "mrc.protos.PipelineMapping.SegmentMapping";
  /** The segment definition ID */
  segmentName: string;
  /** General policy */
  byPolicy?:
    | PipelineMapping_SegmentMapping_ByPolicy
    | undefined;
  /** Manually specified */
  byWorker?:
    | PipelineMapping_SegmentMapping_ByWorker
    | undefined;
  /** Manually specified */
  byExecutor?: PipelineMapping_SegmentMapping_ByExecutor | undefined;
}

export interface PipelineMapping_SegmentMapping_ByPolicy {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByPolicy";
  /** Specify a general policy */
  value: SegmentMappingPolicies;
}

export interface PipelineMapping_SegmentMapping_ByWorker {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByWorker";
  /** The workers to assign this segment to */
  workerIds: string[];
}

export interface PipelineMapping_SegmentMapping_ByExecutor {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByExecutor";
  /** The executors to assign this segment to */
  executorIds: string[];
}

export interface PipelineMapping_SegmentsEntry {
  $type: "mrc.protos.PipelineMapping.SegmentsEntry";
  key: string;
  value: PipelineMapping_SegmentMapping | undefined;
}

export interface PipelineDefinition {
  $type: "mrc.protos.PipelineDefinition";
  /** Generated ID of the definition (int64 because the hash algorithms can give negative values) */
  id: string;
  /** Object that holds all of the configurable properties */
  config:
    | PipelineConfiguration
    | undefined;
  /** Machine IDs to mappings for all executors */
  mappings: { [key: string]: PipelineMapping };
  /** Running Pipeline Instance IDs */
  instanceIds: string[];
  /** Segment Info */
  segments: { [key: string]: PipelineDefinition_SegmentDefinition };
  /** Manifold Info */
  manifolds: { [key: string]: PipelineDefinition_ManifoldDefinition };
}

export interface PipelineDefinition_SegmentDefinition {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition";
  /** Generated ID of the definition */
  id: string;
  /** ID of the parent for back referencing */
  parentId: string;
  /** Name of the segment */
  name: string;
  /** Hashed name of the segment. Only the lower 16 bits are used */
  nameHash: number;
  /** Manifold definition IDs for attached ingress ports. Manifold Name/ID pair for cross referencing */
  ingressManifoldIds: { [key: string]: string };
  /** Manifold definition IDs for attached egress ports. Manifold Name/ID pair for cross referencing */
  egressManifoldIds: { [key: string]: string };
  /** Segment options */
  options:
    | SegmentOptions
    | undefined;
  /** Running Segment Instance IDs */
  instanceIds: string[];
}

export interface PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition.IngressManifoldIdsEntry";
  key: string;
  value: string;
}

export interface PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition.EgressManifoldIdsEntry";
  key: string;
  value: string;
}

export interface PipelineDefinition_ManifoldDefinition {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition";
  /** Generated ID of the definition */
  id: string;
  /** ID of the parent for back referencing */
  parentId: string;
  /** Port name for matching ingress/egress nodes */
  portName: string;
  /** Hashed name of the port. Only the lower 16 bits are used */
  portHash: number;
  /** Segment definition IDs for attached input segments. Segment Name/ID pair for cross referencing */
  inputSegmentIds: { [key: string]: string };
  /** Segment definition IDs for attached egress segments. Segment Name/ID pair for cross referencing */
  outputSegmentIds: { [key: string]: string };
  /** All options for this config */
  options:
    | ManifoldOptions
    | undefined;
  /** Running ManifoldInstance IDs */
  instanceIds: string[];
}

export interface PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.InputSegmentIdsEntry";
  key: string;
  value: string;
}

export interface PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.OutputSegmentIdsEntry";
  key: string;
  value: string;
}

export interface PipelineDefinition_MappingsEntry {
  $type: "mrc.protos.PipelineDefinition.MappingsEntry";
  key: string;
  value: PipelineMapping | undefined;
}

export interface PipelineDefinition_SegmentsEntry {
  $type: "mrc.protos.PipelineDefinition.SegmentsEntry";
  key: string;
  value: PipelineDefinition_SegmentDefinition | undefined;
}

export interface PipelineDefinition_ManifoldsEntry {
  $type: "mrc.protos.PipelineDefinition.ManifoldsEntry";
  key: string;
  value: PipelineDefinition_ManifoldDefinition | undefined;
}

export interface PipelineInstance {
  $type: "mrc.protos.PipelineInstance";
  /** Generated ID of the instance */
  id: string;
  /** The parent ExecutorID this belongs to */
  executorId: string;
  /** The PipelineAddress of this instance. 16 bit ExecutorID + 16 bit PipelineID. */
  pipelineAddress: number;
  /** Deinition this belongs to */
  definitionId: string;
  /** The current state of this resource */
  state:
    | ResourceState
    | undefined;
  /** Running Segment Instance IDs */
  segmentIds: string[];
  /** Running Manifold Instance IDs */
  manifoldIds: string[];
}

/**
 * message Properties{
 *    // Name of the segment
 *    string name = 1;
 */
export interface SegmentDefinition {
  $type: "mrc.protos.SegmentDefinition";
  /** Generated ID of the definition */
  id: string;
  /** Running Segment Instance IDs */
  instanceIds: string[];
}

export interface SegmentInstance {
  $type: "mrc.protos.SegmentInstance";
  /** Generated ID of the instance */
  id: string;
  /** The parent ExecutorID this belongs to */
  executorId: string;
  /** The parent PipelineID this belongs to */
  pipelineInstanceId: string;
  /** The hash of the segment name. Only the lower 16 bits are used. Matches the config */
  nameHash: number;
  /** 16 bit ExecutorID + 16 bit PipelineID + 16 bit SegmentHash + 16 bit SegmentID */
  segmentAddress: string;
  /** Pipeline Deinition this belongs to */
  pipelineDefinitionId: string;
  /** Segment name (Lookup segment config from pipeline def ID and name) */
  name: string;
  /** The current state of this resource */
  state:
    | ResourceState
    | undefined;
  /** Local running manifold instance IDs for egress ports */
  egressManifoldInstanceIds: string[];
  /** Local running manifold instance IDs for ingress ports */
  ingressManifoldInstanceIds: string[];
  /**
   * TODO(MDD): Remove when removing partitions
   * The worker ID associated with this instance
   */
  workerId: string;
}

export interface ManifoldInstance {
  $type: "mrc.protos.ManifoldInstance";
  /** Generated ID of the instance */
  id: string;
  /** The parent ExecutorID this belongs to */
  executorId: string;
  /** The parent PipelineID this belongs to */
  pipelineInstanceId: string;
  /** The hash of the port name. Only the lower 16 bits are used. Matches the config */
  portHash: number;
  /** The ManifoldAddress of this instance. 16 bit ExecutorID + 16 bit PipelineID + 16 bit ManifoldHash2 + 16 bit ManifoldID */
  manifoldAddress: string;
  /** Pipeline Deinition this belongs to */
  pipelineDefinitionId: string;
  /** Port name (Lookup manifold config from pipeline def ID and name) */
  portName: string;
  /** The current state of this resource */
  state:
    | ResourceState
    | undefined;
  /** The requested input segments. True = Local, False = Remote */
  requestedInputSegments: { [key: string]: boolean };
  /** The requested output segments. True = Local, False = Remote */
  requestedOutputSegments: { [key: string]: boolean };
  /** The actual input segments. True = Local, False = Remote */
  actualInputSegments: { [key: string]: boolean };
  /** The actual output segments. True = Local, False = Remote */
  actualOutputSegments: { [key: string]: boolean };
}

export interface ManifoldInstance_RequestedInputSegmentsEntry {
  $type: "mrc.protos.ManifoldInstance.RequestedInputSegmentsEntry";
  key: string;
  value: boolean;
}

export interface ManifoldInstance_RequestedOutputSegmentsEntry {
  $type: "mrc.protos.ManifoldInstance.RequestedOutputSegmentsEntry";
  key: string;
  value: boolean;
}

export interface ManifoldInstance_ActualInputSegmentsEntry {
  $type: "mrc.protos.ManifoldInstance.ActualInputSegmentsEntry";
  key: string;
  value: boolean;
}

export interface ManifoldInstance_ActualOutputSegmentsEntry {
  $type: "mrc.protos.ManifoldInstance.ActualOutputSegmentsEntry";
  key: string;
  value: boolean;
}

export interface ControlPlaneState {
  $type: "mrc.protos.ControlPlaneState";
  nonce: string;
  executors: ControlPlaneState_ExecutorsState | undefined;
  workers: ControlPlaneState_WorkerssState | undefined;
  pipelineDefinitions: ControlPlaneState_PipelineDefinitionsState | undefined;
  pipelineInstances: ControlPlaneState_PipelineInstancesState | undefined;
  segmentDefinitions: ControlPlaneState_SegmentDefinitionsState | undefined;
  segmentInstances: ControlPlaneState_SegmentInstancesState | undefined;
  manifoldInstances: ControlPlaneState_ManifoldInstancesState | undefined;
}

export interface ControlPlaneState_ExecutorsState {
  $type: "mrc.protos.ControlPlaneState.ExecutorsState";
  ids: string[];
  entities: { [key: string]: Executor };
}

export interface ControlPlaneState_ExecutorsState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.ExecutorsState.EntitiesEntry";
  key: string;
  value: Executor | undefined;
}

export interface ControlPlaneState_WorkerssState {
  $type: "mrc.protos.ControlPlaneState.WorkerssState";
  ids: string[];
  entities: { [key: string]: Worker };
}

export interface ControlPlaneState_WorkerssState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.WorkerssState.EntitiesEntry";
  key: string;
  value: Worker | undefined;
}

export interface ControlPlaneState_PipelineDefinitionsState {
  $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState";
  ids: string[];
  entities: { [key: string]: PipelineDefinition };
}

export interface ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState.EntitiesEntry";
  key: string;
  value: PipelineDefinition | undefined;
}

export interface ControlPlaneState_PipelineInstancesState {
  $type: "mrc.protos.ControlPlaneState.PipelineInstancesState";
  ids: string[];
  entities: { [key: string]: PipelineInstance };
}

export interface ControlPlaneState_PipelineInstancesState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.PipelineInstancesState.EntitiesEntry";
  key: string;
  value: PipelineInstance | undefined;
}

export interface ControlPlaneState_SegmentDefinitionsState {
  $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState";
  ids: string[];
  entities: { [key: string]: SegmentDefinition };
}

export interface ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState.EntitiesEntry";
  key: string;
  value: SegmentDefinition | undefined;
}

export interface ControlPlaneState_SegmentInstancesState {
  $type: "mrc.protos.ControlPlaneState.SegmentInstancesState";
  ids: string[];
  entities: { [key: string]: SegmentInstance };
}

export interface ControlPlaneState_SegmentInstancesState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.SegmentInstancesState.EntitiesEntry";
  key: string;
  value: SegmentInstance | undefined;
}

export interface ControlPlaneState_ManifoldInstancesState {
  $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState";
  ids: string[];
  entities: { [key: string]: ManifoldInstance };
}

export interface ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState.EntitiesEntry";
  key: string;
  value: ManifoldInstance | undefined;
}

export interface SegmentOptions {
  $type: "mrc.protos.SegmentOptions";
  placementStrategy: SegmentOptions_PlacementStrategy;
  scalingOptions: ScalingOptions | undefined;
}

export enum SegmentOptions_PlacementStrategy {
  ResourceGroup = "ResourceGroup",
  PhysicalMachine = "PhysicalMachine",
  Global = "Global",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function segmentOptions_PlacementStrategyFromJSON(object: any): SegmentOptions_PlacementStrategy {
  switch (object) {
    case 0:
    case "ResourceGroup":
      return SegmentOptions_PlacementStrategy.ResourceGroup;
    case 1:
    case "PhysicalMachine":
      return SegmentOptions_PlacementStrategy.PhysicalMachine;
    case 2:
    case "Global":
      return SegmentOptions_PlacementStrategy.Global;
    case -1:
    case "UNRECOGNIZED":
    default:
      return SegmentOptions_PlacementStrategy.UNRECOGNIZED;
  }
}

export function segmentOptions_PlacementStrategyToJSON(object: SegmentOptions_PlacementStrategy): string {
  switch (object) {
    case SegmentOptions_PlacementStrategy.ResourceGroup:
      return "ResourceGroup";
    case SegmentOptions_PlacementStrategy.PhysicalMachine:
      return "PhysicalMachine";
    case SegmentOptions_PlacementStrategy.Global:
      return "Global";
    case SegmentOptions_PlacementStrategy.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function segmentOptions_PlacementStrategyToNumber(object: SegmentOptions_PlacementStrategy): number {
  switch (object) {
    case SegmentOptions_PlacementStrategy.ResourceGroup:
      return 0;
    case SegmentOptions_PlacementStrategy.PhysicalMachine:
      return 1;
    case SegmentOptions_PlacementStrategy.Global:
      return 2;
    case SegmentOptions_PlacementStrategy.UNRECOGNIZED:
    default:
      return -1;
  }
}

export interface ScalingOptions {
  $type: "mrc.protos.ScalingOptions";
  strategy: ScalingOptions_ScalingStrategy;
  initialCount: number;
}

export enum ScalingOptions_ScalingStrategy {
  Static = "Static",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function scalingOptions_ScalingStrategyFromJSON(object: any): ScalingOptions_ScalingStrategy {
  switch (object) {
    case 0:
    case "Static":
      return ScalingOptions_ScalingStrategy.Static;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ScalingOptions_ScalingStrategy.UNRECOGNIZED;
  }
}

export function scalingOptions_ScalingStrategyToJSON(object: ScalingOptions_ScalingStrategy): string {
  switch (object) {
    case ScalingOptions_ScalingStrategy.Static:
      return "Static";
    case ScalingOptions_ScalingStrategy.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function scalingOptions_ScalingStrategyToNumber(object: ScalingOptions_ScalingStrategy): number {
  switch (object) {
    case ScalingOptions_ScalingStrategy.Static:
      return 0;
    case ScalingOptions_ScalingStrategy.UNRECOGNIZED:
    default:
      return -1;
  }
}

export interface ManifoldOptions {
  $type: "mrc.protos.ManifoldOptions";
  policy: ManifoldOptions_Policy;
}

export enum ManifoldOptions_Policy {
  LoadBalance = "LoadBalance",
  Broadcast = "Broadcast",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function manifoldOptions_PolicyFromJSON(object: any): ManifoldOptions_Policy {
  switch (object) {
    case 0:
    case "LoadBalance":
      return ManifoldOptions_Policy.LoadBalance;
    case 1:
    case "Broadcast":
      return ManifoldOptions_Policy.Broadcast;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ManifoldOptions_Policy.UNRECOGNIZED;
  }
}

export function manifoldOptions_PolicyToJSON(object: ManifoldOptions_Policy): string {
  switch (object) {
    case ManifoldOptions_Policy.LoadBalance:
      return "LoadBalance";
    case ManifoldOptions_Policy.Broadcast:
      return "Broadcast";
    case ManifoldOptions_Policy.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function manifoldOptions_PolicyToNumber(object: ManifoldOptions_Policy): number {
  switch (object) {
    case ManifoldOptions_Policy.LoadBalance:
      return 0;
    case ManifoldOptions_Policy.Broadcast:
      return 1;
    case ManifoldOptions_Policy.UNRECOGNIZED:
    default:
      return -1;
  }
}

export interface PortInfo {
  $type: "mrc.protos.PortInfo";
  portName: string;
  typeId: number;
  typeString: string;
}

export interface IngressPort {
  $type: "mrc.protos.IngressPort";
  name: string;
  id: number;
}

export interface EgressPort {
  $type: "mrc.protos.EgressPort";
  name: string;
  id: number;
  policyType: EgressPort_PolicyType;
}

export enum EgressPort_PolicyType {
  PolicyDefined = "PolicyDefined",
  UserDefined = "UserDefined",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function egressPort_PolicyTypeFromJSON(object: any): EgressPort_PolicyType {
  switch (object) {
    case 0:
    case "PolicyDefined":
      return EgressPort_PolicyType.PolicyDefined;
    case 1:
    case "UserDefined":
      return EgressPort_PolicyType.UserDefined;
    case -1:
    case "UNRECOGNIZED":
    default:
      return EgressPort_PolicyType.UNRECOGNIZED;
  }
}

export function egressPort_PolicyTypeToJSON(object: EgressPort_PolicyType): string {
  switch (object) {
    case EgressPort_PolicyType.PolicyDefined:
      return "PolicyDefined";
    case EgressPort_PolicyType.UserDefined:
      return "UserDefined";
    case EgressPort_PolicyType.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function egressPort_PolicyTypeToNumber(object: EgressPort_PolicyType): number {
  switch (object) {
    case EgressPort_PolicyType.PolicyDefined:
      return 0;
    case EgressPort_PolicyType.UserDefined:
      return 1;
    case EgressPort_PolicyType.UNRECOGNIZED:
    default:
      return -1;
  }
}

export interface IngressPolicy {
  $type: "mrc.protos.IngressPolicy";
  networkEnabled: boolean;
}

export interface EgressPolicy {
  $type: "mrc.protos.EgressPolicy";
  policy: EgressPolicy_Policy;
  /** list of allowed pol */
  segmentAddresses: number[];
}

export enum EgressPolicy_Policy {
  LoadBalance = "LoadBalance",
  Broadcast = "Broadcast",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function egressPolicy_PolicyFromJSON(object: any): EgressPolicy_Policy {
  switch (object) {
    case 0:
    case "LoadBalance":
      return EgressPolicy_Policy.LoadBalance;
    case 1:
    case "Broadcast":
      return EgressPolicy_Policy.Broadcast;
    case -1:
    case "UNRECOGNIZED":
    default:
      return EgressPolicy_Policy.UNRECOGNIZED;
  }
}

export function egressPolicy_PolicyToJSON(object: EgressPolicy_Policy): string {
  switch (object) {
    case EgressPolicy_Policy.LoadBalance:
      return "LoadBalance";
    case EgressPolicy_Policy.Broadcast:
      return "Broadcast";
    case EgressPolicy_Policy.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export function egressPolicy_PolicyToNumber(object: EgressPolicy_Policy): number {
  switch (object) {
    case EgressPolicy_Policy.LoadBalance:
      return 0;
    case EgressPolicy_Policy.Broadcast:
      return 1;
    case EgressPolicy_Policy.UNRECOGNIZED:
    default:
      return -1;
  }
}

function createBaseResourceState(): ResourceState {
  return {
    $type: "mrc.protos.ResourceState",
    requestedStatus: ResourceRequestedStatus.Requested_Unknown,
    actualStatus: ResourceActualStatus.Actual_Unknown,
    refCount: 0,
  };
}

export const ResourceState = {
  $type: "mrc.protos.ResourceState" as const,

  encode(message: ResourceState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.requestedStatus !== ResourceRequestedStatus.Requested_Unknown) {
      writer.uint32(16).int32(resourceRequestedStatusToNumber(message.requestedStatus));
    }
    if (message.actualStatus !== ResourceActualStatus.Actual_Unknown) {
      writer.uint32(24).int32(resourceActualStatusToNumber(message.actualStatus));
    }
    if (message.refCount !== 0) {
      writer.uint32(32).int32(message.refCount);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ResourceState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseResourceState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 2:
          if (tag !== 16) {
            break;
          }

          message.requestedStatus = resourceRequestedStatusFromJSON(reader.int32());
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.actualStatus = resourceActualStatusFromJSON(reader.int32());
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.refCount = reader.int32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ResourceState {
    return {
      $type: ResourceState.$type,
      requestedStatus: isSet(object.requestedStatus)
        ? resourceRequestedStatusFromJSON(object.requestedStatus)
        : ResourceRequestedStatus.Requested_Unknown,
      actualStatus: isSet(object.actualStatus)
        ? resourceActualStatusFromJSON(object.actualStatus)
        : ResourceActualStatus.Actual_Unknown,
      refCount: isSet(object.refCount) ? Number(object.refCount) : 0,
    };
  },

  toJSON(message: ResourceState): unknown {
    const obj: any = {};
    message.requestedStatus !== undefined &&
      (obj.requestedStatus = resourceRequestedStatusToJSON(message.requestedStatus));
    message.actualStatus !== undefined && (obj.actualStatus = resourceActualStatusToJSON(message.actualStatus));
    message.refCount !== undefined && (obj.refCount = Math.round(message.refCount));
    return obj;
  },

  create(base?: DeepPartial<ResourceState>): ResourceState {
    return ResourceState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ResourceState>): ResourceState {
    const message = createBaseResourceState();
    message.requestedStatus = object.requestedStatus ?? ResourceRequestedStatus.Requested_Unknown;
    message.actualStatus = object.actualStatus ?? ResourceActualStatus.Actual_Unknown;
    message.refCount = object.refCount ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ResourceState.$type, ResourceState);

function createBaseExecutor(): Executor {
  return {
    $type: "mrc.protos.Executor",
    id: "0",
    executorAddress: 0,
    peerInfo: "",
    ucxAddress: new Uint8Array(0),
    assignedPipelineIds: [],
    mappedPipelineDefinitions: [],
    assignedSegmentIds: [],
    state: undefined,
    workerIds: [],
  };
}

export const Executor = {
  $type: "mrc.protos.Executor" as const,

  encode(message: Executor, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.executorAddress !== 0) {
      writer.uint32(16).uint32(message.executorAddress);
    }
    if (message.peerInfo !== "") {
      writer.uint32(26).string(message.peerInfo);
    }
    if (message.ucxAddress.length !== 0) {
      writer.uint32(34).bytes(message.ucxAddress);
    }
    writer.uint32(42).fork();
    for (const v of message.assignedPipelineIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    writer.uint32(50).fork();
    for (const v of message.mappedPipelineDefinitions) {
      writer.uint64(v);
    }
    writer.ldelim();
    writer.uint32(58).fork();
    for (const v of message.assignedSegmentIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(66).fork()).ldelim();
    }
    writer.uint32(74).fork();
    for (const v of message.workerIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Executor {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseExecutor();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.executorAddress = reader.uint32();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.peerInfo = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.ucxAddress = reader.bytes();
          continue;
        case 5:
          if (tag === 40) {
            message.assignedPipelineIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 42) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.assignedPipelineIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 6:
          if (tag === 48) {
            message.mappedPipelineDefinitions.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 50) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.mappedPipelineDefinitions.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 7:
          if (tag === 56) {
            message.assignedSegmentIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 58) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.assignedSegmentIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.state = ResourceState.decode(reader, reader.uint32());
          continue;
        case 9:
          if (tag === 72) {
            message.workerIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 74) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.workerIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): Executor {
    return {
      $type: Executor.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      executorAddress: isSet(object.executorAddress) ? Number(object.executorAddress) : 0,
      peerInfo: isSet(object.peerInfo) ? String(object.peerInfo) : "",
      ucxAddress: isSet(object.ucxAddress) ? bytesFromBase64(object.ucxAddress) : new Uint8Array(0),
      assignedPipelineIds: Array.isArray(object?.assignedPipelineIds)
        ? object.assignedPipelineIds.map((e: any) => String(e))
        : [],
      mappedPipelineDefinitions: Array.isArray(object?.mappedPipelineDefinitions)
        ? object.mappedPipelineDefinitions.map((e: any) => String(e))
        : [],
      assignedSegmentIds: Array.isArray(object?.assignedSegmentIds)
        ? object.assignedSegmentIds.map((e: any) => String(e))
        : [],
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      workerIds: Array.isArray(object?.workerIds) ? object.workerIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: Executor): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.executorAddress !== undefined && (obj.executorAddress = Math.round(message.executorAddress));
    message.peerInfo !== undefined && (obj.peerInfo = message.peerInfo);
    message.ucxAddress !== undefined &&
      (obj.ucxAddress = base64FromBytes(message.ucxAddress !== undefined ? message.ucxAddress : new Uint8Array(0)));
    if (message.assignedPipelineIds) {
      obj.assignedPipelineIds = message.assignedPipelineIds.map((e) => e);
    } else {
      obj.assignedPipelineIds = [];
    }
    if (message.mappedPipelineDefinitions) {
      obj.mappedPipelineDefinitions = message.mappedPipelineDefinitions.map((e) => e);
    } else {
      obj.mappedPipelineDefinitions = [];
    }
    if (message.assignedSegmentIds) {
      obj.assignedSegmentIds = message.assignedSegmentIds.map((e) => e);
    } else {
      obj.assignedSegmentIds = [];
    }
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    if (message.workerIds) {
      obj.workerIds = message.workerIds.map((e) => e);
    } else {
      obj.workerIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<Executor>): Executor {
    return Executor.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Executor>): Executor {
    const message = createBaseExecutor();
    message.id = object.id ?? "0";
    message.executorAddress = object.executorAddress ?? 0;
    message.peerInfo = object.peerInfo ?? "";
    message.ucxAddress = object.ucxAddress ?? new Uint8Array(0);
    message.assignedPipelineIds = object.assignedPipelineIds?.map((e) => e) || [];
    message.mappedPipelineDefinitions = object.mappedPipelineDefinitions?.map((e) => e) || [];
    message.assignedSegmentIds = object.assignedSegmentIds?.map((e) => e) || [];
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.workerIds = object.workerIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(Executor.$type, Executor);

function createBaseWorker(): Worker {
  return {
    $type: "mrc.protos.Worker",
    id: "0",
    executorId: "0",
    partitionAddress: 0,
    ucxAddress: new Uint8Array(0),
    state: undefined,
    assignedSegmentIds: [],
  };
}

export const Worker = {
  $type: "mrc.protos.Worker" as const,

  encode(message: Worker, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.executorId !== "0") {
      writer.uint32(16).uint64(message.executorId);
    }
    if (message.partitionAddress !== 0) {
      writer.uint32(24).uint32(message.partitionAddress);
    }
    if (message.ucxAddress.length !== 0) {
      writer.uint32(34).bytes(message.ucxAddress);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(42).fork()).ldelim();
    }
    writer.uint32(50).fork();
    for (const v of message.assignedSegmentIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Worker {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorker();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.executorId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.partitionAddress = reader.uint32();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.ucxAddress = reader.bytes();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.state = ResourceState.decode(reader, reader.uint32());
          continue;
        case 6:
          if (tag === 48) {
            message.assignedSegmentIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 50) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.assignedSegmentIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): Worker {
    return {
      $type: Worker.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      executorId: isSet(object.executorId) ? String(object.executorId) : "0",
      partitionAddress: isSet(object.partitionAddress) ? Number(object.partitionAddress) : 0,
      ucxAddress: isSet(object.ucxAddress) ? bytesFromBase64(object.ucxAddress) : new Uint8Array(0),
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      assignedSegmentIds: Array.isArray(object?.assignedSegmentIds)
        ? object.assignedSegmentIds.map((e: any) => String(e))
        : [],
    };
  },

  toJSON(message: Worker): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.executorId !== undefined && (obj.executorId = message.executorId);
    message.partitionAddress !== undefined && (obj.partitionAddress = Math.round(message.partitionAddress));
    message.ucxAddress !== undefined &&
      (obj.ucxAddress = base64FromBytes(message.ucxAddress !== undefined ? message.ucxAddress : new Uint8Array(0)));
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    if (message.assignedSegmentIds) {
      obj.assignedSegmentIds = message.assignedSegmentIds.map((e) => e);
    } else {
      obj.assignedSegmentIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<Worker>): Worker {
    return Worker.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Worker>): Worker {
    const message = createBaseWorker();
    message.id = object.id ?? "0";
    message.executorId = object.executorId ?? "0";
    message.partitionAddress = object.partitionAddress ?? 0;
    message.ucxAddress = object.ucxAddress ?? new Uint8Array(0);
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.assignedSegmentIds = object.assignedSegmentIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(Worker.$type, Worker);

function createBasePipelineConfiguration(): PipelineConfiguration {
  return { $type: "mrc.protos.PipelineConfiguration", segments: {}, manifolds: {} };
}

export const PipelineConfiguration = {
  $type: "mrc.protos.PipelineConfiguration" as const,

  encode(message: PipelineConfiguration, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    Object.entries(message.segments).forEach(([key, value]) => {
      PipelineConfiguration_SegmentsEntry.encode({
        $type: "mrc.protos.PipelineConfiguration.SegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(10).fork()).ldelim();
    });
    Object.entries(message.manifolds).forEach(([key, value]) => {
      PipelineConfiguration_ManifoldsEntry.encode({
        $type: "mrc.protos.PipelineConfiguration.ManifoldsEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          const entry1 = PipelineConfiguration_SegmentsEntry.decode(reader, reader.uint32());
          if (entry1.value !== undefined) {
            message.segments[entry1.key] = entry1.value;
          }
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = PipelineConfiguration_ManifoldsEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.manifolds[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration {
    return {
      $type: PipelineConfiguration.$type,
      segments: isObject(object.segments)
        ? Object.entries(object.segments).reduce<{ [key: string]: PipelineConfiguration_SegmentConfiguration }>(
          (acc, [key, value]) => {
            acc[key] = PipelineConfiguration_SegmentConfiguration.fromJSON(value);
            return acc;
          },
          {},
        )
        : {},
      manifolds: isObject(object.manifolds)
        ? Object.entries(object.manifolds).reduce<{ [key: string]: PipelineConfiguration_ManifoldConfiguration }>(
          (acc, [key, value]) => {
            acc[key] = PipelineConfiguration_ManifoldConfiguration.fromJSON(value);
            return acc;
          },
          {},
        )
        : {},
    };
  },

  toJSON(message: PipelineConfiguration): unknown {
    const obj: any = {};
    obj.segments = {};
    if (message.segments) {
      Object.entries(message.segments).forEach(([k, v]) => {
        obj.segments[k] = PipelineConfiguration_SegmentConfiguration.toJSON(v);
      });
    }
    obj.manifolds = {};
    if (message.manifolds) {
      Object.entries(message.manifolds).forEach(([k, v]) => {
        obj.manifolds[k] = PipelineConfiguration_ManifoldConfiguration.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration>): PipelineConfiguration {
    return PipelineConfiguration.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineConfiguration>): PipelineConfiguration {
    const message = createBasePipelineConfiguration();
    message.segments = Object.entries(object.segments ?? {}).reduce<
      { [key: string]: PipelineConfiguration_SegmentConfiguration }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = PipelineConfiguration_SegmentConfiguration.fromPartial(value);
      }
      return acc;
    }, {});
    message.manifolds = Object.entries(object.manifolds ?? {}).reduce<
      { [key: string]: PipelineConfiguration_ManifoldConfiguration }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = PipelineConfiguration_ManifoldConfiguration.fromPartial(value);
      }
      return acc;
    }, {});
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration.$type, PipelineConfiguration);

function createBasePipelineConfiguration_SegmentConfiguration(): PipelineConfiguration_SegmentConfiguration {
  return {
    $type: "mrc.protos.PipelineConfiguration.SegmentConfiguration",
    name: "",
    nameHash: 0,
    ingressPorts: [],
    egressPorts: [],
    options: undefined,
  };
}

export const PipelineConfiguration_SegmentConfiguration = {
  $type: "mrc.protos.PipelineConfiguration.SegmentConfiguration" as const,

  encode(message: PipelineConfiguration_SegmentConfiguration, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.nameHash !== 0) {
      writer.uint32(16).uint32(message.nameHash);
    }
    for (const v of message.ingressPorts) {
      writer.uint32(26).string(v!);
    }
    for (const v of message.egressPorts) {
      writer.uint32(34).string(v!);
    }
    if (message.options !== undefined) {
      SegmentOptions.encode(message.options, writer.uint32(42).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration_SegmentConfiguration {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_SegmentConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.name = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.nameHash = reader.uint32();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.ingressPorts.push(reader.string());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.egressPorts.push(reader.string());
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.options = SegmentOptions.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration_SegmentConfiguration {
    return {
      $type: PipelineConfiguration_SegmentConfiguration.$type,
      name: isSet(object.name) ? String(object.name) : "",
      nameHash: isSet(object.nameHash) ? Number(object.nameHash) : 0,
      ingressPorts: Array.isArray(object?.ingressPorts) ? object.ingressPorts.map((e: any) => String(e)) : [],
      egressPorts: Array.isArray(object?.egressPorts) ? object.egressPorts.map((e: any) => String(e)) : [],
      options: isSet(object.options) ? SegmentOptions.fromJSON(object.options) : undefined,
    };
  },

  toJSON(message: PipelineConfiguration_SegmentConfiguration): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.nameHash !== undefined && (obj.nameHash = Math.round(message.nameHash));
    if (message.ingressPorts) {
      obj.ingressPorts = message.ingressPorts.map((e) => e);
    } else {
      obj.ingressPorts = [];
    }
    if (message.egressPorts) {
      obj.egressPorts = message.egressPorts.map((e) => e);
    } else {
      obj.egressPorts = [];
    }
    message.options !== undefined &&
      (obj.options = message.options ? SegmentOptions.toJSON(message.options) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration_SegmentConfiguration>): PipelineConfiguration_SegmentConfiguration {
    return PipelineConfiguration_SegmentConfiguration.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineConfiguration_SegmentConfiguration>,
  ): PipelineConfiguration_SegmentConfiguration {
    const message = createBasePipelineConfiguration_SegmentConfiguration();
    message.name = object.name ?? "";
    message.nameHash = object.nameHash ?? 0;
    message.ingressPorts = object.ingressPorts?.map((e) => e) || [];
    message.egressPorts = object.egressPorts?.map((e) => e) || [];
    message.options = (object.options !== undefined && object.options !== null)
      ? SegmentOptions.fromPartial(object.options)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration_SegmentConfiguration.$type, PipelineConfiguration_SegmentConfiguration);

function createBasePipelineConfiguration_ManifoldConfiguration(): PipelineConfiguration_ManifoldConfiguration {
  return {
    $type: "mrc.protos.PipelineConfiguration.ManifoldConfiguration",
    portName: "",
    portHash: 0,
    typeId: 0,
    typeString: "",
    options: undefined,
  };
}

export const PipelineConfiguration_ManifoldConfiguration = {
  $type: "mrc.protos.PipelineConfiguration.ManifoldConfiguration" as const,

  encode(message: PipelineConfiguration_ManifoldConfiguration, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.portName !== "") {
      writer.uint32(10).string(message.portName);
    }
    if (message.portHash !== 0) {
      writer.uint32(16).uint32(message.portHash);
    }
    if (message.typeId !== 0) {
      writer.uint32(24).uint32(message.typeId);
    }
    if (message.typeString !== "") {
      writer.uint32(34).string(message.typeString);
    }
    if (message.options !== undefined) {
      ManifoldOptions.encode(message.options, writer.uint32(42).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration_ManifoldConfiguration {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_ManifoldConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.portName = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.portHash = reader.uint32();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.typeId = reader.uint32();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.typeString = reader.string();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.options = ManifoldOptions.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration_ManifoldConfiguration {
    return {
      $type: PipelineConfiguration_ManifoldConfiguration.$type,
      portName: isSet(object.portName) ? String(object.portName) : "",
      portHash: isSet(object.portHash) ? Number(object.portHash) : 0,
      typeId: isSet(object.typeId) ? Number(object.typeId) : 0,
      typeString: isSet(object.typeString) ? String(object.typeString) : "",
      options: isSet(object.options) ? ManifoldOptions.fromJSON(object.options) : undefined,
    };
  },

  toJSON(message: PipelineConfiguration_ManifoldConfiguration): unknown {
    const obj: any = {};
    message.portName !== undefined && (obj.portName = message.portName);
    message.portHash !== undefined && (obj.portHash = Math.round(message.portHash));
    message.typeId !== undefined && (obj.typeId = Math.round(message.typeId));
    message.typeString !== undefined && (obj.typeString = message.typeString);
    message.options !== undefined &&
      (obj.options = message.options ? ManifoldOptions.toJSON(message.options) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration_ManifoldConfiguration>): PipelineConfiguration_ManifoldConfiguration {
    return PipelineConfiguration_ManifoldConfiguration.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineConfiguration_ManifoldConfiguration>,
  ): PipelineConfiguration_ManifoldConfiguration {
    const message = createBasePipelineConfiguration_ManifoldConfiguration();
    message.portName = object.portName ?? "";
    message.portHash = object.portHash ?? 0;
    message.typeId = object.typeId ?? 0;
    message.typeString = object.typeString ?? "";
    message.options = (object.options !== undefined && object.options !== null)
      ? ManifoldOptions.fromPartial(object.options)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration_ManifoldConfiguration.$type, PipelineConfiguration_ManifoldConfiguration);

function createBasePipelineConfiguration_SegmentsEntry(): PipelineConfiguration_SegmentsEntry {
  return { $type: "mrc.protos.PipelineConfiguration.SegmentsEntry", key: "", value: undefined };
}

export const PipelineConfiguration_SegmentsEntry = {
  $type: "mrc.protos.PipelineConfiguration.SegmentsEntry" as const,

  encode(message: PipelineConfiguration_SegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      PipelineConfiguration_SegmentConfiguration.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration_SegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineConfiguration_SegmentConfiguration.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration_SegmentsEntry {
    return {
      $type: PipelineConfiguration_SegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? PipelineConfiguration_SegmentConfiguration.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineConfiguration_SegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined &&
      (obj.value = message.value ? PipelineConfiguration_SegmentConfiguration.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration_SegmentsEntry>): PipelineConfiguration_SegmentsEntry {
    return PipelineConfiguration_SegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineConfiguration_SegmentsEntry>): PipelineConfiguration_SegmentsEntry {
    const message = createBasePipelineConfiguration_SegmentsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineConfiguration_SegmentConfiguration.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration_SegmentsEntry.$type, PipelineConfiguration_SegmentsEntry);

function createBasePipelineConfiguration_ManifoldsEntry(): PipelineConfiguration_ManifoldsEntry {
  return { $type: "mrc.protos.PipelineConfiguration.ManifoldsEntry", key: "", value: undefined };
}

export const PipelineConfiguration_ManifoldsEntry = {
  $type: "mrc.protos.PipelineConfiguration.ManifoldsEntry" as const,

  encode(message: PipelineConfiguration_ManifoldsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      PipelineConfiguration_ManifoldConfiguration.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration_ManifoldsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_ManifoldsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineConfiguration_ManifoldConfiguration.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration_ManifoldsEntry {
    return {
      $type: PipelineConfiguration_ManifoldsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? PipelineConfiguration_ManifoldConfiguration.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineConfiguration_ManifoldsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined &&
      (obj.value = message.value ? PipelineConfiguration_ManifoldConfiguration.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration_ManifoldsEntry>): PipelineConfiguration_ManifoldsEntry {
    return PipelineConfiguration_ManifoldsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineConfiguration_ManifoldsEntry>): PipelineConfiguration_ManifoldsEntry {
    const message = createBasePipelineConfiguration_ManifoldsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineConfiguration_ManifoldConfiguration.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration_ManifoldsEntry.$type, PipelineConfiguration_ManifoldsEntry);

function createBasePipelineMapping(): PipelineMapping {
  return { $type: "mrc.protos.PipelineMapping", executorId: "0", segments: {} };
}

export const PipelineMapping = {
  $type: "mrc.protos.PipelineMapping" as const,

  encode(message: PipelineMapping, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.executorId !== "0") {
      writer.uint32(8).uint64(message.executorId);
    }
    Object.entries(message.segments).forEach(([key, value]) => {
      PipelineMapping_SegmentsEntry.encode({
        $type: "mrc.protos.PipelineMapping.SegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.executorId = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = PipelineMapping_SegmentsEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.segments[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping {
    return {
      $type: PipelineMapping.$type,
      executorId: isSet(object.executorId) ? String(object.executorId) : "0",
      segments: isObject(object.segments)
        ? Object.entries(object.segments).reduce<{ [key: string]: PipelineMapping_SegmentMapping }>(
          (acc, [key, value]) => {
            acc[key] = PipelineMapping_SegmentMapping.fromJSON(value);
            return acc;
          },
          {},
        )
        : {},
    };
  },

  toJSON(message: PipelineMapping): unknown {
    const obj: any = {};
    message.executorId !== undefined && (obj.executorId = message.executorId);
    obj.segments = {};
    if (message.segments) {
      Object.entries(message.segments).forEach(([k, v]) => {
        obj.segments[k] = PipelineMapping_SegmentMapping.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping>): PipelineMapping {
    return PipelineMapping.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineMapping>): PipelineMapping {
    const message = createBasePipelineMapping();
    message.executorId = object.executorId ?? "0";
    message.segments = Object.entries(object.segments ?? {}).reduce<{ [key: string]: PipelineMapping_SegmentMapping }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = PipelineMapping_SegmentMapping.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping.$type, PipelineMapping);

function createBasePipelineMapping_SegmentMapping(): PipelineMapping_SegmentMapping {
  return {
    $type: "mrc.protos.PipelineMapping.SegmentMapping",
    segmentName: "",
    byPolicy: undefined,
    byWorker: undefined,
    byExecutor: undefined,
  };
}

export const PipelineMapping_SegmentMapping = {
  $type: "mrc.protos.PipelineMapping.SegmentMapping" as const,

  encode(message: PipelineMapping_SegmentMapping, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.segmentName !== "") {
      writer.uint32(10).string(message.segmentName);
    }
    if (message.byPolicy !== undefined) {
      PipelineMapping_SegmentMapping_ByPolicy.encode(message.byPolicy, writer.uint32(18).fork()).ldelim();
    }
    if (message.byWorker !== undefined) {
      PipelineMapping_SegmentMapping_ByWorker.encode(message.byWorker, writer.uint32(26).fork()).ldelim();
    }
    if (message.byExecutor !== undefined) {
      PipelineMapping_SegmentMapping_ByExecutor.encode(message.byExecutor, writer.uint32(34).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.segmentName = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.byPolicy = PipelineMapping_SegmentMapping_ByPolicy.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.byWorker = PipelineMapping_SegmentMapping_ByWorker.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.byExecutor = PipelineMapping_SegmentMapping_ByExecutor.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping {
    return {
      $type: PipelineMapping_SegmentMapping.$type,
      segmentName: isSet(object.segmentName) ? String(object.segmentName) : "",
      byPolicy: isSet(object.byPolicy) ? PipelineMapping_SegmentMapping_ByPolicy.fromJSON(object.byPolicy) : undefined,
      byWorker: isSet(object.byWorker) ? PipelineMapping_SegmentMapping_ByWorker.fromJSON(object.byWorker) : undefined,
      byExecutor: isSet(object.byExecutor)
        ? PipelineMapping_SegmentMapping_ByExecutor.fromJSON(object.byExecutor)
        : undefined,
    };
  },

  toJSON(message: PipelineMapping_SegmentMapping): unknown {
    const obj: any = {};
    message.segmentName !== undefined && (obj.segmentName = message.segmentName);
    message.byPolicy !== undefined &&
      (obj.byPolicy = message.byPolicy ? PipelineMapping_SegmentMapping_ByPolicy.toJSON(message.byPolicy) : undefined);
    message.byWorker !== undefined &&
      (obj.byWorker = message.byWorker ? PipelineMapping_SegmentMapping_ByWorker.toJSON(message.byWorker) : undefined);
    message.byExecutor !== undefined && (obj.byExecutor = message.byExecutor
      ? PipelineMapping_SegmentMapping_ByExecutor.toJSON(message.byExecutor)
      : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping_SegmentMapping>): PipelineMapping_SegmentMapping {
    return PipelineMapping_SegmentMapping.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineMapping_SegmentMapping>): PipelineMapping_SegmentMapping {
    const message = createBasePipelineMapping_SegmentMapping();
    message.segmentName = object.segmentName ?? "";
    message.byPolicy = (object.byPolicy !== undefined && object.byPolicy !== null)
      ? PipelineMapping_SegmentMapping_ByPolicy.fromPartial(object.byPolicy)
      : undefined;
    message.byWorker = (object.byWorker !== undefined && object.byWorker !== null)
      ? PipelineMapping_SegmentMapping_ByWorker.fromPartial(object.byWorker)
      : undefined;
    message.byExecutor = (object.byExecutor !== undefined && object.byExecutor !== null)
      ? PipelineMapping_SegmentMapping_ByExecutor.fromPartial(object.byExecutor)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentMapping.$type, PipelineMapping_SegmentMapping);

function createBasePipelineMapping_SegmentMapping_ByPolicy(): PipelineMapping_SegmentMapping_ByPolicy {
  return { $type: "mrc.protos.PipelineMapping.SegmentMapping.ByPolicy", value: SegmentMappingPolicies.Disabled };
}

export const PipelineMapping_SegmentMapping_ByPolicy = {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByPolicy" as const,

  encode(message: PipelineMapping_SegmentMapping_ByPolicy, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.value !== SegmentMappingPolicies.Disabled) {
      writer.uint32(8).int32(segmentMappingPoliciesToNumber(message.value));
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping_ByPolicy {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping_ByPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.value = segmentMappingPoliciesFromJSON(reader.int32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping_ByPolicy {
    return {
      $type: PipelineMapping_SegmentMapping_ByPolicy.$type,
      value: isSet(object.value) ? segmentMappingPoliciesFromJSON(object.value) : SegmentMappingPolicies.Disabled,
    };
  },

  toJSON(message: PipelineMapping_SegmentMapping_ByPolicy): unknown {
    const obj: any = {};
    message.value !== undefined && (obj.value = segmentMappingPoliciesToJSON(message.value));
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping_SegmentMapping_ByPolicy>): PipelineMapping_SegmentMapping_ByPolicy {
    return PipelineMapping_SegmentMapping_ByPolicy.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineMapping_SegmentMapping_ByPolicy>): PipelineMapping_SegmentMapping_ByPolicy {
    const message = createBasePipelineMapping_SegmentMapping_ByPolicy();
    message.value = object.value ?? SegmentMappingPolicies.Disabled;
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentMapping_ByPolicy.$type, PipelineMapping_SegmentMapping_ByPolicy);

function createBasePipelineMapping_SegmentMapping_ByWorker(): PipelineMapping_SegmentMapping_ByWorker {
  return { $type: "mrc.protos.PipelineMapping.SegmentMapping.ByWorker", workerIds: [] };
}

export const PipelineMapping_SegmentMapping_ByWorker = {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByWorker" as const,

  encode(message: PipelineMapping_SegmentMapping_ByWorker, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.workerIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping_ByWorker {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping_ByWorker();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.workerIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.workerIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping_ByWorker {
    return {
      $type: PipelineMapping_SegmentMapping_ByWorker.$type,
      workerIds: Array.isArray(object?.workerIds) ? object.workerIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineMapping_SegmentMapping_ByWorker): unknown {
    const obj: any = {};
    if (message.workerIds) {
      obj.workerIds = message.workerIds.map((e) => e);
    } else {
      obj.workerIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping_SegmentMapping_ByWorker>): PipelineMapping_SegmentMapping_ByWorker {
    return PipelineMapping_SegmentMapping_ByWorker.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineMapping_SegmentMapping_ByWorker>): PipelineMapping_SegmentMapping_ByWorker {
    const message = createBasePipelineMapping_SegmentMapping_ByWorker();
    message.workerIds = object.workerIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentMapping_ByWorker.$type, PipelineMapping_SegmentMapping_ByWorker);

function createBasePipelineMapping_SegmentMapping_ByExecutor(): PipelineMapping_SegmentMapping_ByExecutor {
  return { $type: "mrc.protos.PipelineMapping.SegmentMapping.ByExecutor", executorIds: [] };
}

export const PipelineMapping_SegmentMapping_ByExecutor = {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByExecutor" as const,

  encode(message: PipelineMapping_SegmentMapping_ByExecutor, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.executorIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping_ByExecutor {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping_ByExecutor();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.executorIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.executorIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping_ByExecutor {
    return {
      $type: PipelineMapping_SegmentMapping_ByExecutor.$type,
      executorIds: Array.isArray(object?.executorIds) ? object.executorIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineMapping_SegmentMapping_ByExecutor): unknown {
    const obj: any = {};
    if (message.executorIds) {
      obj.executorIds = message.executorIds.map((e) => e);
    } else {
      obj.executorIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping_SegmentMapping_ByExecutor>): PipelineMapping_SegmentMapping_ByExecutor {
    return PipelineMapping_SegmentMapping_ByExecutor.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineMapping_SegmentMapping_ByExecutor>,
  ): PipelineMapping_SegmentMapping_ByExecutor {
    const message = createBasePipelineMapping_SegmentMapping_ByExecutor();
    message.executorIds = object.executorIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentMapping_ByExecutor.$type, PipelineMapping_SegmentMapping_ByExecutor);

function createBasePipelineMapping_SegmentsEntry(): PipelineMapping_SegmentsEntry {
  return { $type: "mrc.protos.PipelineMapping.SegmentsEntry", key: "", value: undefined };
}

export const PipelineMapping_SegmentsEntry = {
  $type: "mrc.protos.PipelineMapping.SegmentsEntry" as const,

  encode(message: PipelineMapping_SegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      PipelineMapping_SegmentMapping.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineMapping_SegmentMapping.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentsEntry {
    return {
      $type: PipelineMapping_SegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? PipelineMapping_SegmentMapping.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineMapping_SegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined &&
      (obj.value = message.value ? PipelineMapping_SegmentMapping.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineMapping_SegmentsEntry>): PipelineMapping_SegmentsEntry {
    return PipelineMapping_SegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineMapping_SegmentsEntry>): PipelineMapping_SegmentsEntry {
    const message = createBasePipelineMapping_SegmentsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineMapping_SegmentMapping.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentsEntry.$type, PipelineMapping_SegmentsEntry);

function createBasePipelineDefinition(): PipelineDefinition {
  return {
    $type: "mrc.protos.PipelineDefinition",
    id: "0",
    config: undefined,
    mappings: {},
    instanceIds: [],
    segments: {},
    manifolds: {},
  };
}

export const PipelineDefinition = {
  $type: "mrc.protos.PipelineDefinition" as const,

  encode(message: PipelineDefinition, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).int64(message.id);
    }
    if (message.config !== undefined) {
      PipelineConfiguration.encode(message.config, writer.uint32(18).fork()).ldelim();
    }
    Object.entries(message.mappings).forEach(([key, value]) => {
      PipelineDefinition_MappingsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.MappingsEntry",
        key: key as any,
        value,
      }, writer.uint32(26).fork()).ldelim();
    });
    writer.uint32(34).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.segments).forEach(([key, value]) => {
      PipelineDefinition_SegmentsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.SegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(42).fork()).ldelim();
    });
    Object.entries(message.manifolds).forEach(([key, value]) => {
      PipelineDefinition_ManifoldsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.ManifoldsEntry",
        key: key as any,
        value,
      }, writer.uint32(50).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.int64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.config = PipelineConfiguration.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          const entry3 = PipelineDefinition_MappingsEntry.decode(reader, reader.uint32());
          if (entry3.value !== undefined) {
            message.mappings[entry3.key] = entry3.value;
          }
          continue;
        case 4:
          if (tag === 32) {
            message.instanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 34) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 5:
          if (tag !== 42) {
            break;
          }

          const entry5 = PipelineDefinition_SegmentsEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.segments[entry5.key] = entry5.value;
          }
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          const entry6 = PipelineDefinition_ManifoldsEntry.decode(reader, reader.uint32());
          if (entry6.value !== undefined) {
            message.manifolds[entry6.key] = entry6.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition {
    return {
      $type: PipelineDefinition.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      config: isSet(object.config) ? PipelineConfiguration.fromJSON(object.config) : undefined,
      mappings: isObject(object.mappings)
        ? Object.entries(object.mappings).reduce<{ [key: string]: PipelineMapping }>((acc, [key, value]) => {
          acc[key] = PipelineMapping.fromJSON(value);
          return acc;
        }, {})
        : {},
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => String(e)) : [],
      segments: isObject(object.segments)
        ? Object.entries(object.segments).reduce<{ [key: string]: PipelineDefinition_SegmentDefinition }>(
          (acc, [key, value]) => {
            acc[key] = PipelineDefinition_SegmentDefinition.fromJSON(value);
            return acc;
          },
          {},
        )
        : {},
      manifolds: isObject(object.manifolds)
        ? Object.entries(object.manifolds).reduce<{ [key: string]: PipelineDefinition_ManifoldDefinition }>(
          (acc, [key, value]) => {
            acc[key] = PipelineDefinition_ManifoldDefinition.fromJSON(value);
            return acc;
          },
          {},
        )
        : {},
    };
  },

  toJSON(message: PipelineDefinition): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.config !== undefined &&
      (obj.config = message.config ? PipelineConfiguration.toJSON(message.config) : undefined);
    obj.mappings = {};
    if (message.mappings) {
      Object.entries(message.mappings).forEach(([k, v]) => {
        obj.mappings[k] = PipelineMapping.toJSON(v);
      });
    }
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => e);
    } else {
      obj.instanceIds = [];
    }
    obj.segments = {};
    if (message.segments) {
      Object.entries(message.segments).forEach(([k, v]) => {
        obj.segments[k] = PipelineDefinition_SegmentDefinition.toJSON(v);
      });
    }
    obj.manifolds = {};
    if (message.manifolds) {
      Object.entries(message.manifolds).forEach(([k, v]) => {
        obj.manifolds[k] = PipelineDefinition_ManifoldDefinition.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition>): PipelineDefinition {
    return PipelineDefinition.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition>): PipelineDefinition {
    const message = createBasePipelineDefinition();
    message.id = object.id ?? "0";
    message.config = (object.config !== undefined && object.config !== null)
      ? PipelineConfiguration.fromPartial(object.config)
      : undefined;
    message.mappings = Object.entries(object.mappings ?? {}).reduce<{ [key: string]: PipelineMapping }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = PipelineMapping.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    message.segments = Object.entries(object.segments ?? {}).reduce<
      { [key: string]: PipelineDefinition_SegmentDefinition }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = PipelineDefinition_SegmentDefinition.fromPartial(value);
      }
      return acc;
    }, {});
    message.manifolds = Object.entries(object.manifolds ?? {}).reduce<
      { [key: string]: PipelineDefinition_ManifoldDefinition }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = PipelineDefinition_ManifoldDefinition.fromPartial(value);
      }
      return acc;
    }, {});
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition.$type, PipelineDefinition);

function createBasePipelineDefinition_SegmentDefinition(): PipelineDefinition_SegmentDefinition {
  return {
    $type: "mrc.protos.PipelineDefinition.SegmentDefinition",
    id: "0",
    parentId: "0",
    name: "",
    nameHash: 0,
    ingressManifoldIds: {},
    egressManifoldIds: {},
    options: undefined,
    instanceIds: [],
  };
}

export const PipelineDefinition_SegmentDefinition = {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition" as const,

  encode(message: PipelineDefinition_SegmentDefinition, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.parentId !== "0") {
      writer.uint32(16).uint64(message.parentId);
    }
    if (message.name !== "") {
      writer.uint32(26).string(message.name);
    }
    if (message.nameHash !== 0) {
      writer.uint32(32).uint32(message.nameHash);
    }
    Object.entries(message.ingressManifoldIds).forEach(([key, value]) => {
      PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.SegmentDefinition.IngressManifoldIdsEntry",
        key: key as any,
        value,
      }, writer.uint32(42).fork()).ldelim();
    });
    Object.entries(message.egressManifoldIds).forEach(([key, value]) => {
      PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.SegmentDefinition.EgressManifoldIdsEntry",
        key: key as any,
        value,
      }, writer.uint32(50).fork()).ldelim();
    });
    if (message.options !== undefined) {
      SegmentOptions.encode(message.options, writer.uint32(58).fork()).ldelim();
    }
    writer.uint32(66).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_SegmentDefinition {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.parentId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.name = reader.string();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.nameHash = reader.uint32();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          const entry5 = PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.ingressManifoldIds[entry5.key] = entry5.value;
          }
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          const entry6 = PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry.decode(reader, reader.uint32());
          if (entry6.value !== undefined) {
            message.egressManifoldIds[entry6.key] = entry6.value;
          }
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.options = SegmentOptions.decode(reader, reader.uint32());
          continue;
        case 8:
          if (tag === 64) {
            message.instanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 66) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_SegmentDefinition {
    return {
      $type: PipelineDefinition_SegmentDefinition.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      parentId: isSet(object.parentId) ? String(object.parentId) : "0",
      name: isSet(object.name) ? String(object.name) : "",
      nameHash: isSet(object.nameHash) ? Number(object.nameHash) : 0,
      ingressManifoldIds: isObject(object.ingressManifoldIds)
        ? Object.entries(object.ingressManifoldIds).reduce<{ [key: string]: string }>((acc, [key, value]) => {
          acc[key] = String(value);
          return acc;
        }, {})
        : {},
      egressManifoldIds: isObject(object.egressManifoldIds)
        ? Object.entries(object.egressManifoldIds).reduce<{ [key: string]: string }>((acc, [key, value]) => {
          acc[key] = String(value);
          return acc;
        }, {})
        : {},
      options: isSet(object.options) ? SegmentOptions.fromJSON(object.options) : undefined,
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineDefinition_SegmentDefinition): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.parentId !== undefined && (obj.parentId = message.parentId);
    message.name !== undefined && (obj.name = message.name);
    message.nameHash !== undefined && (obj.nameHash = Math.round(message.nameHash));
    obj.ingressManifoldIds = {};
    if (message.ingressManifoldIds) {
      Object.entries(message.ingressManifoldIds).forEach(([k, v]) => {
        obj.ingressManifoldIds[k] = v;
      });
    }
    obj.egressManifoldIds = {};
    if (message.egressManifoldIds) {
      Object.entries(message.egressManifoldIds).forEach(([k, v]) => {
        obj.egressManifoldIds[k] = v;
      });
    }
    message.options !== undefined &&
      (obj.options = message.options ? SegmentOptions.toJSON(message.options) : undefined);
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => e);
    } else {
      obj.instanceIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition_SegmentDefinition>): PipelineDefinition_SegmentDefinition {
    return PipelineDefinition_SegmentDefinition.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition_SegmentDefinition>): PipelineDefinition_SegmentDefinition {
    const message = createBasePipelineDefinition_SegmentDefinition();
    message.id = object.id ?? "0";
    message.parentId = object.parentId ?? "0";
    message.name = object.name ?? "";
    message.nameHash = object.nameHash ?? 0;
    message.ingressManifoldIds = Object.entries(object.ingressManifoldIds ?? {}).reduce<{ [key: string]: string }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = String(value);
        }
        return acc;
      },
      {},
    );
    message.egressManifoldIds = Object.entries(object.egressManifoldIds ?? {}).reduce<{ [key: string]: string }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = String(value);
        }
        return acc;
      },
      {},
    );
    message.options = (object.options !== undefined && object.options !== null)
      ? SegmentOptions.fromPartial(object.options)
      : undefined;
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_SegmentDefinition.$type, PipelineDefinition_SegmentDefinition);

function createBasePipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry(): PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
  return { $type: "mrc.protos.PipelineDefinition.SegmentDefinition.IngressManifoldIdsEntry", key: "", value: "0" };
}

export const PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry = {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition.IngressManifoldIdsEntry" as const,

  encode(
    message: PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== "0") {
      writer.uint32(16).uint64(message.value);
    }
    return writer;
  },

  decode(
    input: _m0.Reader | Uint8Array,
    length?: number,
  ): PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = longToString(reader.uint64() as Long);
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
    return {
      $type: PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? String(object.value) : "0",
    };
  },

  toJSON(message: PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry>,
  ): PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
    return PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry>,
  ): PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry {
    const message = createBasePipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry();
    message.key = object.key ?? "";
    message.value = object.value ?? "0";
    return message;
  },
};

messageTypeRegistry.set(
  PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry.$type,
  PipelineDefinition_SegmentDefinition_IngressManifoldIdsEntry,
);

function createBasePipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry(): PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
  return { $type: "mrc.protos.PipelineDefinition.SegmentDefinition.EgressManifoldIdsEntry", key: "", value: "0" };
}

export const PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry = {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition.EgressManifoldIdsEntry" as const,

  encode(
    message: PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== "0") {
      writer.uint32(16).uint64(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = longToString(reader.uint64() as Long);
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
    return {
      $type: PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? String(object.value) : "0",
    };
  },

  toJSON(message: PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry>,
  ): PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
    return PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry>,
  ): PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry {
    const message = createBasePipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry();
    message.key = object.key ?? "";
    message.value = object.value ?? "0";
    return message;
  },
};

messageTypeRegistry.set(
  PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry.$type,
  PipelineDefinition_SegmentDefinition_EgressManifoldIdsEntry,
);

function createBasePipelineDefinition_ManifoldDefinition(): PipelineDefinition_ManifoldDefinition {
  return {
    $type: "mrc.protos.PipelineDefinition.ManifoldDefinition",
    id: "0",
    parentId: "0",
    portName: "",
    portHash: 0,
    inputSegmentIds: {},
    outputSegmentIds: {},
    options: undefined,
    instanceIds: [],
  };
}

export const PipelineDefinition_ManifoldDefinition = {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition" as const,

  encode(message: PipelineDefinition_ManifoldDefinition, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.parentId !== "0") {
      writer.uint32(16).uint64(message.parentId);
    }
    if (message.portName !== "") {
      writer.uint32(26).string(message.portName);
    }
    if (message.portHash !== 0) {
      writer.uint32(32).uint32(message.portHash);
    }
    Object.entries(message.inputSegmentIds).forEach(([key, value]) => {
      PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.InputSegmentIdsEntry",
        key: key as any,
        value,
      }, writer.uint32(42).fork()).ldelim();
    });
    Object.entries(message.outputSegmentIds).forEach(([key, value]) => {
      PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry.encode({
        $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.OutputSegmentIdsEntry",
        key: key as any,
        value,
      }, writer.uint32(50).fork()).ldelim();
    });
    if (message.options !== undefined) {
      ManifoldOptions.encode(message.options, writer.uint32(58).fork()).ldelim();
    }
    writer.uint32(66).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_ManifoldDefinition {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_ManifoldDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.parentId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.portName = reader.string();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.portHash = reader.uint32();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          const entry5 = PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.inputSegmentIds[entry5.key] = entry5.value;
          }
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          const entry6 = PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry.decode(reader, reader.uint32());
          if (entry6.value !== undefined) {
            message.outputSegmentIds[entry6.key] = entry6.value;
          }
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.options = ManifoldOptions.decode(reader, reader.uint32());
          continue;
        case 8:
          if (tag === 64) {
            message.instanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 66) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_ManifoldDefinition {
    return {
      $type: PipelineDefinition_ManifoldDefinition.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      parentId: isSet(object.parentId) ? String(object.parentId) : "0",
      portName: isSet(object.portName) ? String(object.portName) : "",
      portHash: isSet(object.portHash) ? Number(object.portHash) : 0,
      inputSegmentIds: isObject(object.inputSegmentIds)
        ? Object.entries(object.inputSegmentIds).reduce<{ [key: string]: string }>((acc, [key, value]) => {
          acc[key] = String(value);
          return acc;
        }, {})
        : {},
      outputSegmentIds: isObject(object.outputSegmentIds)
        ? Object.entries(object.outputSegmentIds).reduce<{ [key: string]: string }>((acc, [key, value]) => {
          acc[key] = String(value);
          return acc;
        }, {})
        : {},
      options: isSet(object.options) ? ManifoldOptions.fromJSON(object.options) : undefined,
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineDefinition_ManifoldDefinition): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.parentId !== undefined && (obj.parentId = message.parentId);
    message.portName !== undefined && (obj.portName = message.portName);
    message.portHash !== undefined && (obj.portHash = Math.round(message.portHash));
    obj.inputSegmentIds = {};
    if (message.inputSegmentIds) {
      Object.entries(message.inputSegmentIds).forEach(([k, v]) => {
        obj.inputSegmentIds[k] = v;
      });
    }
    obj.outputSegmentIds = {};
    if (message.outputSegmentIds) {
      Object.entries(message.outputSegmentIds).forEach(([k, v]) => {
        obj.outputSegmentIds[k] = v;
      });
    }
    message.options !== undefined &&
      (obj.options = message.options ? ManifoldOptions.toJSON(message.options) : undefined);
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => e);
    } else {
      obj.instanceIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition_ManifoldDefinition>): PipelineDefinition_ManifoldDefinition {
    return PipelineDefinition_ManifoldDefinition.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition_ManifoldDefinition>): PipelineDefinition_ManifoldDefinition {
    const message = createBasePipelineDefinition_ManifoldDefinition();
    message.id = object.id ?? "0";
    message.parentId = object.parentId ?? "0";
    message.portName = object.portName ?? "";
    message.portHash = object.portHash ?? 0;
    message.inputSegmentIds = Object.entries(object.inputSegmentIds ?? {}).reduce<{ [key: string]: string }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = String(value);
        }
        return acc;
      },
      {},
    );
    message.outputSegmentIds = Object.entries(object.outputSegmentIds ?? {}).reduce<{ [key: string]: string }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = String(value);
        }
        return acc;
      },
      {},
    );
    message.options = (object.options !== undefined && object.options !== null)
      ? ManifoldOptions.fromPartial(object.options)
      : undefined;
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_ManifoldDefinition.$type, PipelineDefinition_ManifoldDefinition);

function createBasePipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry(): PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
  return { $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.InputSegmentIdsEntry", key: "", value: "0" };
}

export const PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry = {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.InputSegmentIdsEntry" as const,

  encode(
    message: PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== "0") {
      writer.uint32(16).uint64(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = longToString(reader.uint64() as Long);
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
    return {
      $type: PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? String(object.value) : "0",
    };
  },

  toJSON(message: PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry>,
  ): PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
    return PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry>,
  ): PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry {
    const message = createBasePipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry();
    message.key = object.key ?? "";
    message.value = object.value ?? "0";
    return message;
  },
};

messageTypeRegistry.set(
  PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry.$type,
  PipelineDefinition_ManifoldDefinition_InputSegmentIdsEntry,
);

function createBasePipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry(): PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
  return { $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.OutputSegmentIdsEntry", key: "", value: "0" };
}

export const PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry = {
  $type: "mrc.protos.PipelineDefinition.ManifoldDefinition.OutputSegmentIdsEntry" as const,

  encode(
    message: PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== "0") {
      writer.uint32(16).uint64(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = longToString(reader.uint64() as Long);
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
    return {
      $type: PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? String(object.value) : "0",
    };
  },

  toJSON(message: PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry>,
  ): PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
    return PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry>,
  ): PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry {
    const message = createBasePipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry();
    message.key = object.key ?? "";
    message.value = object.value ?? "0";
    return message;
  },
};

messageTypeRegistry.set(
  PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry.$type,
  PipelineDefinition_ManifoldDefinition_OutputSegmentIdsEntry,
);

function createBasePipelineDefinition_MappingsEntry(): PipelineDefinition_MappingsEntry {
  return { $type: "mrc.protos.PipelineDefinition.MappingsEntry", key: "0", value: undefined };
}

export const PipelineDefinition_MappingsEntry = {
  $type: "mrc.protos.PipelineDefinition.MappingsEntry" as const,

  encode(message: PipelineDefinition_MappingsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      PipelineMapping.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_MappingsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_MappingsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineMapping.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_MappingsEntry {
    return {
      $type: PipelineDefinition_MappingsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? PipelineMapping.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineDefinition_MappingsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? PipelineMapping.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition_MappingsEntry>): PipelineDefinition_MappingsEntry {
    return PipelineDefinition_MappingsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition_MappingsEntry>): PipelineDefinition_MappingsEntry {
    const message = createBasePipelineDefinition_MappingsEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineMapping.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_MappingsEntry.$type, PipelineDefinition_MappingsEntry);

function createBasePipelineDefinition_SegmentsEntry(): PipelineDefinition_SegmentsEntry {
  return { $type: "mrc.protos.PipelineDefinition.SegmentsEntry", key: "", value: undefined };
}

export const PipelineDefinition_SegmentsEntry = {
  $type: "mrc.protos.PipelineDefinition.SegmentsEntry" as const,

  encode(message: PipelineDefinition_SegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      PipelineDefinition_SegmentDefinition.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_SegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineDefinition_SegmentDefinition.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_SegmentsEntry {
    return {
      $type: PipelineDefinition_SegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? PipelineDefinition_SegmentDefinition.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineDefinition_SegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined &&
      (obj.value = message.value ? PipelineDefinition_SegmentDefinition.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition_SegmentsEntry>): PipelineDefinition_SegmentsEntry {
    return PipelineDefinition_SegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition_SegmentsEntry>): PipelineDefinition_SegmentsEntry {
    const message = createBasePipelineDefinition_SegmentsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineDefinition_SegmentDefinition.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_SegmentsEntry.$type, PipelineDefinition_SegmentsEntry);

function createBasePipelineDefinition_ManifoldsEntry(): PipelineDefinition_ManifoldsEntry {
  return { $type: "mrc.protos.PipelineDefinition.ManifoldsEntry", key: "", value: undefined };
}

export const PipelineDefinition_ManifoldsEntry = {
  $type: "mrc.protos.PipelineDefinition.ManifoldsEntry" as const,

  encode(message: PipelineDefinition_ManifoldsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      PipelineDefinition_ManifoldDefinition.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_ManifoldsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_ManifoldsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineDefinition_ManifoldDefinition.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_ManifoldsEntry {
    return {
      $type: PipelineDefinition_ManifoldsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "",
      value: isSet(object.value) ? PipelineDefinition_ManifoldDefinition.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: PipelineDefinition_ManifoldsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined &&
      (obj.value = message.value ? PipelineDefinition_ManifoldDefinition.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<PipelineDefinition_ManifoldsEntry>): PipelineDefinition_ManifoldsEntry {
    return PipelineDefinition_ManifoldsEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineDefinition_ManifoldsEntry>): PipelineDefinition_ManifoldsEntry {
    const message = createBasePipelineDefinition_ManifoldsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineDefinition_ManifoldDefinition.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_ManifoldsEntry.$type, PipelineDefinition_ManifoldsEntry);

function createBasePipelineInstance(): PipelineInstance {
  return {
    $type: "mrc.protos.PipelineInstance",
    id: "0",
    executorId: "0",
    pipelineAddress: 0,
    definitionId: "0",
    state: undefined,
    segmentIds: [],
    manifoldIds: [],
  };
}

export const PipelineInstance = {
  $type: "mrc.protos.PipelineInstance" as const,

  encode(message: PipelineInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.executorId !== "0") {
      writer.uint32(16).uint64(message.executorId);
    }
    if (message.pipelineAddress !== 0) {
      writer.uint32(24).uint32(message.pipelineAddress);
    }
    if (message.definitionId !== "0") {
      writer.uint32(32).int64(message.definitionId);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(42).fork()).ldelim();
    }
    writer.uint32(50).fork();
    for (const v of message.segmentIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    writer.uint32(58).fork();
    for (const v of message.manifoldIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineInstance {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.executorId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.pipelineAddress = reader.uint32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.definitionId = longToString(reader.int64() as Long);
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.state = ResourceState.decode(reader, reader.uint32());
          continue;
        case 6:
          if (tag === 48) {
            message.segmentIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 50) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.segmentIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 7:
          if (tag === 56) {
            message.manifoldIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 58) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.manifoldIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PipelineInstance {
    return {
      $type: PipelineInstance.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      executorId: isSet(object.executorId) ? String(object.executorId) : "0",
      pipelineAddress: isSet(object.pipelineAddress) ? Number(object.pipelineAddress) : 0,
      definitionId: isSet(object.definitionId) ? String(object.definitionId) : "0",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      segmentIds: Array.isArray(object?.segmentIds) ? object.segmentIds.map((e: any) => String(e)) : [],
      manifoldIds: Array.isArray(object?.manifoldIds) ? object.manifoldIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineInstance): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.executorId !== undefined && (obj.executorId = message.executorId);
    message.pipelineAddress !== undefined && (obj.pipelineAddress = Math.round(message.pipelineAddress));
    message.definitionId !== undefined && (obj.definitionId = message.definitionId);
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    if (message.segmentIds) {
      obj.segmentIds = message.segmentIds.map((e) => e);
    } else {
      obj.segmentIds = [];
    }
    if (message.manifoldIds) {
      obj.manifoldIds = message.manifoldIds.map((e) => e);
    } else {
      obj.manifoldIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineInstance>): PipelineInstance {
    return PipelineInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineInstance>): PipelineInstance {
    const message = createBasePipelineInstance();
    message.id = object.id ?? "0";
    message.executorId = object.executorId ?? "0";
    message.pipelineAddress = object.pipelineAddress ?? 0;
    message.definitionId = object.definitionId ?? "0";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.segmentIds = object.segmentIds?.map((e) => e) || [];
    message.manifoldIds = object.manifoldIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineInstance.$type, PipelineInstance);

function createBaseSegmentDefinition(): SegmentDefinition {
  return { $type: "mrc.protos.SegmentDefinition", id: "0", instanceIds: [] };
}

export const SegmentDefinition = {
  $type: "mrc.protos.SegmentDefinition" as const,

  encode(message: SegmentDefinition, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    writer.uint32(26).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentDefinition {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag === 24) {
            message.instanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 26) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): SegmentDefinition {
    return {
      $type: SegmentDefinition.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: SegmentDefinition): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => e);
    } else {
      obj.instanceIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<SegmentDefinition>): SegmentDefinition {
    return SegmentDefinition.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentDefinition>): SegmentDefinition {
    const message = createBaseSegmentDefinition();
    message.id = object.id ?? "0";
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(SegmentDefinition.$type, SegmentDefinition);

function createBaseSegmentInstance(): SegmentInstance {
  return {
    $type: "mrc.protos.SegmentInstance",
    id: "0",
    executorId: "0",
    pipelineInstanceId: "0",
    nameHash: 0,
    segmentAddress: "0",
    pipelineDefinitionId: "0",
    name: "",
    state: undefined,
    egressManifoldInstanceIds: [],
    ingressManifoldInstanceIds: [],
    workerId: "0",
  };
}

export const SegmentInstance = {
  $type: "mrc.protos.SegmentInstance" as const,

  encode(message: SegmentInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.executorId !== "0") {
      writer.uint32(16).uint64(message.executorId);
    }
    if (message.pipelineInstanceId !== "0") {
      writer.uint32(24).uint64(message.pipelineInstanceId);
    }
    if (message.nameHash !== 0) {
      writer.uint32(32).uint32(message.nameHash);
    }
    if (message.segmentAddress !== "0") {
      writer.uint32(40).uint64(message.segmentAddress);
    }
    if (message.pipelineDefinitionId !== "0") {
      writer.uint32(48).int64(message.pipelineDefinitionId);
    }
    if (message.name !== "") {
      writer.uint32(58).string(message.name);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(66).fork()).ldelim();
    }
    writer.uint32(74).fork();
    for (const v of message.egressManifoldInstanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    writer.uint32(82).fork();
    for (const v of message.ingressManifoldInstanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    if (message.workerId !== "0") {
      writer.uint32(88).uint64(message.workerId);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentInstance {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.executorId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.pipelineInstanceId = longToString(reader.uint64() as Long);
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.nameHash = reader.uint32();
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.segmentAddress = longToString(reader.uint64() as Long);
          continue;
        case 6:
          if (tag !== 48) {
            break;
          }

          message.pipelineDefinitionId = longToString(reader.int64() as Long);
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.name = reader.string();
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.state = ResourceState.decode(reader, reader.uint32());
          continue;
        case 9:
          if (tag === 72) {
            message.egressManifoldInstanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 74) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.egressManifoldInstanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 10:
          if (tag === 80) {
            message.ingressManifoldInstanceIds.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 82) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ingressManifoldInstanceIds.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 11:
          if (tag !== 88) {
            break;
          }

          message.workerId = longToString(reader.uint64() as Long);
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): SegmentInstance {
    return {
      $type: SegmentInstance.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      executorId: isSet(object.executorId) ? String(object.executorId) : "0",
      pipelineInstanceId: isSet(object.pipelineInstanceId) ? String(object.pipelineInstanceId) : "0",
      nameHash: isSet(object.nameHash) ? Number(object.nameHash) : 0,
      segmentAddress: isSet(object.segmentAddress) ? String(object.segmentAddress) : "0",
      pipelineDefinitionId: isSet(object.pipelineDefinitionId) ? String(object.pipelineDefinitionId) : "0",
      name: isSet(object.name) ? String(object.name) : "",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      egressManifoldInstanceIds: Array.isArray(object?.egressManifoldInstanceIds)
        ? object.egressManifoldInstanceIds.map((e: any) => String(e))
        : [],
      ingressManifoldInstanceIds: Array.isArray(object?.ingressManifoldInstanceIds)
        ? object.ingressManifoldInstanceIds.map((e: any) => String(e))
        : [],
      workerId: isSet(object.workerId) ? String(object.workerId) : "0",
    };
  },

  toJSON(message: SegmentInstance): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.executorId !== undefined && (obj.executorId = message.executorId);
    message.pipelineInstanceId !== undefined && (obj.pipelineInstanceId = message.pipelineInstanceId);
    message.nameHash !== undefined && (obj.nameHash = Math.round(message.nameHash));
    message.segmentAddress !== undefined && (obj.segmentAddress = message.segmentAddress);
    message.pipelineDefinitionId !== undefined && (obj.pipelineDefinitionId = message.pipelineDefinitionId);
    message.name !== undefined && (obj.name = message.name);
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    if (message.egressManifoldInstanceIds) {
      obj.egressManifoldInstanceIds = message.egressManifoldInstanceIds.map((e) => e);
    } else {
      obj.egressManifoldInstanceIds = [];
    }
    if (message.ingressManifoldInstanceIds) {
      obj.ingressManifoldInstanceIds = message.ingressManifoldInstanceIds.map((e) => e);
    } else {
      obj.ingressManifoldInstanceIds = [];
    }
    message.workerId !== undefined && (obj.workerId = message.workerId);
    return obj;
  },

  create(base?: DeepPartial<SegmentInstance>): SegmentInstance {
    return SegmentInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentInstance>): SegmentInstance {
    const message = createBaseSegmentInstance();
    message.id = object.id ?? "0";
    message.executorId = object.executorId ?? "0";
    message.pipelineInstanceId = object.pipelineInstanceId ?? "0";
    message.nameHash = object.nameHash ?? 0;
    message.segmentAddress = object.segmentAddress ?? "0";
    message.pipelineDefinitionId = object.pipelineDefinitionId ?? "0";
    message.name = object.name ?? "";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.egressManifoldInstanceIds = object.egressManifoldInstanceIds?.map((e) => e) || [];
    message.ingressManifoldInstanceIds = object.ingressManifoldInstanceIds?.map((e) => e) || [];
    message.workerId = object.workerId ?? "0";
    return message;
  },
};

messageTypeRegistry.set(SegmentInstance.$type, SegmentInstance);

function createBaseManifoldInstance(): ManifoldInstance {
  return {
    $type: "mrc.protos.ManifoldInstance",
    id: "0",
    executorId: "0",
    pipelineInstanceId: "0",
    portHash: 0,
    manifoldAddress: "0",
    pipelineDefinitionId: "0",
    portName: "",
    state: undefined,
    requestedInputSegments: {},
    requestedOutputSegments: {},
    actualInputSegments: {},
    actualOutputSegments: {},
  };
}

export const ManifoldInstance = {
  $type: "mrc.protos.ManifoldInstance" as const,

  encode(message: ManifoldInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.executorId !== "0") {
      writer.uint32(16).uint64(message.executorId);
    }
    if (message.pipelineInstanceId !== "0") {
      writer.uint32(24).uint64(message.pipelineInstanceId);
    }
    if (message.portHash !== 0) {
      writer.uint32(32).uint32(message.portHash);
    }
    if (message.manifoldAddress !== "0") {
      writer.uint32(40).uint64(message.manifoldAddress);
    }
    if (message.pipelineDefinitionId !== "0") {
      writer.uint32(48).int64(message.pipelineDefinitionId);
    }
    if (message.portName !== "") {
      writer.uint32(58).string(message.portName);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(66).fork()).ldelim();
    }
    Object.entries(message.requestedInputSegments).forEach(([key, value]) => {
      ManifoldInstance_RequestedInputSegmentsEntry.encode({
        $type: "mrc.protos.ManifoldInstance.RequestedInputSegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(74).fork()).ldelim();
    });
    Object.entries(message.requestedOutputSegments).forEach(([key, value]) => {
      ManifoldInstance_RequestedOutputSegmentsEntry.encode({
        $type: "mrc.protos.ManifoldInstance.RequestedOutputSegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(82).fork()).ldelim();
    });
    Object.entries(message.actualInputSegments).forEach(([key, value]) => {
      ManifoldInstance_ActualInputSegmentsEntry.encode({
        $type: "mrc.protos.ManifoldInstance.ActualInputSegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(90).fork()).ldelim();
    });
    Object.entries(message.actualOutputSegments).forEach(([key, value]) => {
      ManifoldInstance_ActualOutputSegmentsEntry.encode({
        $type: "mrc.protos.ManifoldInstance.ActualOutputSegmentsEntry",
        key: key as any,
        value,
      }, writer.uint32(98).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldInstance {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.executorId = longToString(reader.uint64() as Long);
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.pipelineInstanceId = longToString(reader.uint64() as Long);
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.portHash = reader.uint32();
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.manifoldAddress = longToString(reader.uint64() as Long);
          continue;
        case 6:
          if (tag !== 48) {
            break;
          }

          message.pipelineDefinitionId = longToString(reader.int64() as Long);
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.portName = reader.string();
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.state = ResourceState.decode(reader, reader.uint32());
          continue;
        case 9:
          if (tag !== 74) {
            break;
          }

          const entry9 = ManifoldInstance_RequestedInputSegmentsEntry.decode(reader, reader.uint32());
          if (entry9.value !== undefined) {
            message.requestedInputSegments[entry9.key] = entry9.value;
          }
          continue;
        case 10:
          if (tag !== 82) {
            break;
          }

          const entry10 = ManifoldInstance_RequestedOutputSegmentsEntry.decode(reader, reader.uint32());
          if (entry10.value !== undefined) {
            message.requestedOutputSegments[entry10.key] = entry10.value;
          }
          continue;
        case 11:
          if (tag !== 90) {
            break;
          }

          const entry11 = ManifoldInstance_ActualInputSegmentsEntry.decode(reader, reader.uint32());
          if (entry11.value !== undefined) {
            message.actualInputSegments[entry11.key] = entry11.value;
          }
          continue;
        case 12:
          if (tag !== 98) {
            break;
          }

          const entry12 = ManifoldInstance_ActualOutputSegmentsEntry.decode(reader, reader.uint32());
          if (entry12.value !== undefined) {
            message.actualOutputSegments[entry12.key] = entry12.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldInstance {
    return {
      $type: ManifoldInstance.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      executorId: isSet(object.executorId) ? String(object.executorId) : "0",
      pipelineInstanceId: isSet(object.pipelineInstanceId) ? String(object.pipelineInstanceId) : "0",
      portHash: isSet(object.portHash) ? Number(object.portHash) : 0,
      manifoldAddress: isSet(object.manifoldAddress) ? String(object.manifoldAddress) : "0",
      pipelineDefinitionId: isSet(object.pipelineDefinitionId) ? String(object.pipelineDefinitionId) : "0",
      portName: isSet(object.portName) ? String(object.portName) : "",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      requestedInputSegments: isObject(object.requestedInputSegments)
        ? Object.entries(object.requestedInputSegments).reduce<{ [key: string]: boolean }>((acc, [key, value]) => {
          acc[key] = Boolean(value);
          return acc;
        }, {})
        : {},
      requestedOutputSegments: isObject(object.requestedOutputSegments)
        ? Object.entries(object.requestedOutputSegments).reduce<{ [key: string]: boolean }>((acc, [key, value]) => {
          acc[key] = Boolean(value);
          return acc;
        }, {})
        : {},
      actualInputSegments: isObject(object.actualInputSegments)
        ? Object.entries(object.actualInputSegments).reduce<{ [key: string]: boolean }>((acc, [key, value]) => {
          acc[key] = Boolean(value);
          return acc;
        }, {})
        : {},
      actualOutputSegments: isObject(object.actualOutputSegments)
        ? Object.entries(object.actualOutputSegments).reduce<{ [key: string]: boolean }>((acc, [key, value]) => {
          acc[key] = Boolean(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ManifoldInstance): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.executorId !== undefined && (obj.executorId = message.executorId);
    message.pipelineInstanceId !== undefined && (obj.pipelineInstanceId = message.pipelineInstanceId);
    message.portHash !== undefined && (obj.portHash = Math.round(message.portHash));
    message.manifoldAddress !== undefined && (obj.manifoldAddress = message.manifoldAddress);
    message.pipelineDefinitionId !== undefined && (obj.pipelineDefinitionId = message.pipelineDefinitionId);
    message.portName !== undefined && (obj.portName = message.portName);
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    obj.requestedInputSegments = {};
    if (message.requestedInputSegments) {
      Object.entries(message.requestedInputSegments).forEach(([k, v]) => {
        obj.requestedInputSegments[k] = v;
      });
    }
    obj.requestedOutputSegments = {};
    if (message.requestedOutputSegments) {
      Object.entries(message.requestedOutputSegments).forEach(([k, v]) => {
        obj.requestedOutputSegments[k] = v;
      });
    }
    obj.actualInputSegments = {};
    if (message.actualInputSegments) {
      Object.entries(message.actualInputSegments).forEach(([k, v]) => {
        obj.actualInputSegments[k] = v;
      });
    }
    obj.actualOutputSegments = {};
    if (message.actualOutputSegments) {
      Object.entries(message.actualOutputSegments).forEach(([k, v]) => {
        obj.actualOutputSegments[k] = v;
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ManifoldInstance>): ManifoldInstance {
    return ManifoldInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ManifoldInstance>): ManifoldInstance {
    const message = createBaseManifoldInstance();
    message.id = object.id ?? "0";
    message.executorId = object.executorId ?? "0";
    message.pipelineInstanceId = object.pipelineInstanceId ?? "0";
    message.portHash = object.portHash ?? 0;
    message.manifoldAddress = object.manifoldAddress ?? "0";
    message.pipelineDefinitionId = object.pipelineDefinitionId ?? "0";
    message.portName = object.portName ?? "";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.requestedInputSegments = Object.entries(object.requestedInputSegments ?? {}).reduce<
      { [key: string]: boolean }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = Boolean(value);
      }
      return acc;
    }, {});
    message.requestedOutputSegments = Object.entries(object.requestedOutputSegments ?? {}).reduce<
      { [key: string]: boolean }
    >((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = Boolean(value);
      }
      return acc;
    }, {});
    message.actualInputSegments = Object.entries(object.actualInputSegments ?? {}).reduce<{ [key: string]: boolean }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = Boolean(value);
        }
        return acc;
      },
      {},
    );
    message.actualOutputSegments = Object.entries(object.actualOutputSegments ?? {}).reduce<{ [key: string]: boolean }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = Boolean(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ManifoldInstance.$type, ManifoldInstance);

function createBaseManifoldInstance_RequestedInputSegmentsEntry(): ManifoldInstance_RequestedInputSegmentsEntry {
  return { $type: "mrc.protos.ManifoldInstance.RequestedInputSegmentsEntry", key: "0", value: false };
}

export const ManifoldInstance_RequestedInputSegmentsEntry = {
  $type: "mrc.protos.ManifoldInstance.RequestedInputSegmentsEntry" as const,

  encode(message: ManifoldInstance_RequestedInputSegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value === true) {
      writer.uint32(16).bool(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldInstance_RequestedInputSegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldInstance_RequestedInputSegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldInstance_RequestedInputSegmentsEntry {
    return {
      $type: ManifoldInstance_RequestedInputSegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Boolean(object.value) : false,
    };
  },

  toJSON(message: ManifoldInstance_RequestedInputSegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<ManifoldInstance_RequestedInputSegmentsEntry>,
  ): ManifoldInstance_RequestedInputSegmentsEntry {
    return ManifoldInstance_RequestedInputSegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ManifoldInstance_RequestedInputSegmentsEntry>,
  ): ManifoldInstance_RequestedInputSegmentsEntry {
    const message = createBaseManifoldInstance_RequestedInputSegmentsEntry();
    message.key = object.key ?? "0";
    message.value = object.value ?? false;
    return message;
  },
};

messageTypeRegistry.set(
  ManifoldInstance_RequestedInputSegmentsEntry.$type,
  ManifoldInstance_RequestedInputSegmentsEntry,
);

function createBaseManifoldInstance_RequestedOutputSegmentsEntry(): ManifoldInstance_RequestedOutputSegmentsEntry {
  return { $type: "mrc.protos.ManifoldInstance.RequestedOutputSegmentsEntry", key: "0", value: false };
}

export const ManifoldInstance_RequestedOutputSegmentsEntry = {
  $type: "mrc.protos.ManifoldInstance.RequestedOutputSegmentsEntry" as const,

  encode(message: ManifoldInstance_RequestedOutputSegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value === true) {
      writer.uint32(16).bool(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldInstance_RequestedOutputSegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldInstance_RequestedOutputSegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldInstance_RequestedOutputSegmentsEntry {
    return {
      $type: ManifoldInstance_RequestedOutputSegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Boolean(object.value) : false,
    };
  },

  toJSON(message: ManifoldInstance_RequestedOutputSegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(
    base?: DeepPartial<ManifoldInstance_RequestedOutputSegmentsEntry>,
  ): ManifoldInstance_RequestedOutputSegmentsEntry {
    return ManifoldInstance_RequestedOutputSegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ManifoldInstance_RequestedOutputSegmentsEntry>,
  ): ManifoldInstance_RequestedOutputSegmentsEntry {
    const message = createBaseManifoldInstance_RequestedOutputSegmentsEntry();
    message.key = object.key ?? "0";
    message.value = object.value ?? false;
    return message;
  },
};

messageTypeRegistry.set(
  ManifoldInstance_RequestedOutputSegmentsEntry.$type,
  ManifoldInstance_RequestedOutputSegmentsEntry,
);

function createBaseManifoldInstance_ActualInputSegmentsEntry(): ManifoldInstance_ActualInputSegmentsEntry {
  return { $type: "mrc.protos.ManifoldInstance.ActualInputSegmentsEntry", key: "0", value: false };
}

export const ManifoldInstance_ActualInputSegmentsEntry = {
  $type: "mrc.protos.ManifoldInstance.ActualInputSegmentsEntry" as const,

  encode(message: ManifoldInstance_ActualInputSegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value === true) {
      writer.uint32(16).bool(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldInstance_ActualInputSegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldInstance_ActualInputSegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldInstance_ActualInputSegmentsEntry {
    return {
      $type: ManifoldInstance_ActualInputSegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Boolean(object.value) : false,
    };
  },

  toJSON(message: ManifoldInstance_ActualInputSegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(base?: DeepPartial<ManifoldInstance_ActualInputSegmentsEntry>): ManifoldInstance_ActualInputSegmentsEntry {
    return ManifoldInstance_ActualInputSegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ManifoldInstance_ActualInputSegmentsEntry>,
  ): ManifoldInstance_ActualInputSegmentsEntry {
    const message = createBaseManifoldInstance_ActualInputSegmentsEntry();
    message.key = object.key ?? "0";
    message.value = object.value ?? false;
    return message;
  },
};

messageTypeRegistry.set(ManifoldInstance_ActualInputSegmentsEntry.$type, ManifoldInstance_ActualInputSegmentsEntry);

function createBaseManifoldInstance_ActualOutputSegmentsEntry(): ManifoldInstance_ActualOutputSegmentsEntry {
  return { $type: "mrc.protos.ManifoldInstance.ActualOutputSegmentsEntry", key: "0", value: false };
}

export const ManifoldInstance_ActualOutputSegmentsEntry = {
  $type: "mrc.protos.ManifoldInstance.ActualOutputSegmentsEntry" as const,

  encode(message: ManifoldInstance_ActualOutputSegmentsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value === true) {
      writer.uint32(16).bool(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldInstance_ActualOutputSegmentsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldInstance_ActualOutputSegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldInstance_ActualOutputSegmentsEntry {
    return {
      $type: ManifoldInstance_ActualOutputSegmentsEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Boolean(object.value) : false,
    };
  },

  toJSON(message: ManifoldInstance_ActualOutputSegmentsEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value);
    return obj;
  },

  create(base?: DeepPartial<ManifoldInstance_ActualOutputSegmentsEntry>): ManifoldInstance_ActualOutputSegmentsEntry {
    return ManifoldInstance_ActualOutputSegmentsEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ManifoldInstance_ActualOutputSegmentsEntry>,
  ): ManifoldInstance_ActualOutputSegmentsEntry {
    const message = createBaseManifoldInstance_ActualOutputSegmentsEntry();
    message.key = object.key ?? "0";
    message.value = object.value ?? false;
    return message;
  },
};

messageTypeRegistry.set(ManifoldInstance_ActualOutputSegmentsEntry.$type, ManifoldInstance_ActualOutputSegmentsEntry);

function createBaseControlPlaneState(): ControlPlaneState {
  return {
    $type: "mrc.protos.ControlPlaneState",
    nonce: "0",
    executors: undefined,
    workers: undefined,
    pipelineDefinitions: undefined,
    pipelineInstances: undefined,
    segmentDefinitions: undefined,
    segmentInstances: undefined,
    manifoldInstances: undefined,
  };
}

export const ControlPlaneState = {
  $type: "mrc.protos.ControlPlaneState" as const,

  encode(message: ControlPlaneState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.nonce !== "0") {
      writer.uint32(8).uint64(message.nonce);
    }
    if (message.executors !== undefined) {
      ControlPlaneState_ExecutorsState.encode(message.executors, writer.uint32(18).fork()).ldelim();
    }
    if (message.workers !== undefined) {
      ControlPlaneState_WorkerssState.encode(message.workers, writer.uint32(26).fork()).ldelim();
    }
    if (message.pipelineDefinitions !== undefined) {
      ControlPlaneState_PipelineDefinitionsState.encode(message.pipelineDefinitions, writer.uint32(34).fork()).ldelim();
    }
    if (message.pipelineInstances !== undefined) {
      ControlPlaneState_PipelineInstancesState.encode(message.pipelineInstances, writer.uint32(42).fork()).ldelim();
    }
    if (message.segmentDefinitions !== undefined) {
      ControlPlaneState_SegmentDefinitionsState.encode(message.segmentDefinitions, writer.uint32(50).fork()).ldelim();
    }
    if (message.segmentInstances !== undefined) {
      ControlPlaneState_SegmentInstancesState.encode(message.segmentInstances, writer.uint32(58).fork()).ldelim();
    }
    if (message.manifoldInstances !== undefined) {
      ControlPlaneState_ManifoldInstancesState.encode(message.manifoldInstances, writer.uint32(66).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.nonce = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.executors = ControlPlaneState_ExecutorsState.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.workers = ControlPlaneState_WorkerssState.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.pipelineDefinitions = ControlPlaneState_PipelineDefinitionsState.decode(reader, reader.uint32());
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.pipelineInstances = ControlPlaneState_PipelineInstancesState.decode(reader, reader.uint32());
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.segmentDefinitions = ControlPlaneState_SegmentDefinitionsState.decode(reader, reader.uint32());
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.segmentInstances = ControlPlaneState_SegmentInstancesState.decode(reader, reader.uint32());
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.manifoldInstances = ControlPlaneState_ManifoldInstancesState.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState {
    return {
      $type: ControlPlaneState.$type,
      nonce: isSet(object.nonce) ? String(object.nonce) : "0",
      executors: isSet(object.executors) ? ControlPlaneState_ExecutorsState.fromJSON(object.executors) : undefined,
      workers: isSet(object.workers) ? ControlPlaneState_WorkerssState.fromJSON(object.workers) : undefined,
      pipelineDefinitions: isSet(object.pipelineDefinitions)
        ? ControlPlaneState_PipelineDefinitionsState.fromJSON(object.pipelineDefinitions)
        : undefined,
      pipelineInstances: isSet(object.pipelineInstances)
        ? ControlPlaneState_PipelineInstancesState.fromJSON(object.pipelineInstances)
        : undefined,
      segmentDefinitions: isSet(object.segmentDefinitions)
        ? ControlPlaneState_SegmentDefinitionsState.fromJSON(object.segmentDefinitions)
        : undefined,
      segmentInstances: isSet(object.segmentInstances)
        ? ControlPlaneState_SegmentInstancesState.fromJSON(object.segmentInstances)
        : undefined,
      manifoldInstances: isSet(object.manifoldInstances)
        ? ControlPlaneState_ManifoldInstancesState.fromJSON(object.manifoldInstances)
        : undefined,
    };
  },

  toJSON(message: ControlPlaneState): unknown {
    const obj: any = {};
    message.nonce !== undefined && (obj.nonce = message.nonce);
    message.executors !== undefined &&
      (obj.executors = message.executors ? ControlPlaneState_ExecutorsState.toJSON(message.executors) : undefined);
    message.workers !== undefined &&
      (obj.workers = message.workers ? ControlPlaneState_WorkerssState.toJSON(message.workers) : undefined);
    message.pipelineDefinitions !== undefined && (obj.pipelineDefinitions = message.pipelineDefinitions
      ? ControlPlaneState_PipelineDefinitionsState.toJSON(message.pipelineDefinitions)
      : undefined);
    message.pipelineInstances !== undefined && (obj.pipelineInstances = message.pipelineInstances
      ? ControlPlaneState_PipelineInstancesState.toJSON(message.pipelineInstances)
      : undefined);
    message.segmentDefinitions !== undefined && (obj.segmentDefinitions = message.segmentDefinitions
      ? ControlPlaneState_SegmentDefinitionsState.toJSON(message.segmentDefinitions)
      : undefined);
    message.segmentInstances !== undefined && (obj.segmentInstances = message.segmentInstances
      ? ControlPlaneState_SegmentInstancesState.toJSON(message.segmentInstances)
      : undefined);
    message.manifoldInstances !== undefined && (obj.manifoldInstances = message.manifoldInstances
      ? ControlPlaneState_ManifoldInstancesState.toJSON(message.manifoldInstances)
      : undefined);
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState>): ControlPlaneState {
    return ControlPlaneState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState>): ControlPlaneState {
    const message = createBaseControlPlaneState();
    message.nonce = object.nonce ?? "0";
    message.executors = (object.executors !== undefined && object.executors !== null)
      ? ControlPlaneState_ExecutorsState.fromPartial(object.executors)
      : undefined;
    message.workers = (object.workers !== undefined && object.workers !== null)
      ? ControlPlaneState_WorkerssState.fromPartial(object.workers)
      : undefined;
    message.pipelineDefinitions = (object.pipelineDefinitions !== undefined && object.pipelineDefinitions !== null)
      ? ControlPlaneState_PipelineDefinitionsState.fromPartial(object.pipelineDefinitions)
      : undefined;
    message.pipelineInstances = (object.pipelineInstances !== undefined && object.pipelineInstances !== null)
      ? ControlPlaneState_PipelineInstancesState.fromPartial(object.pipelineInstances)
      : undefined;
    message.segmentDefinitions = (object.segmentDefinitions !== undefined && object.segmentDefinitions !== null)
      ? ControlPlaneState_SegmentDefinitionsState.fromPartial(object.segmentDefinitions)
      : undefined;
    message.segmentInstances = (object.segmentInstances !== undefined && object.segmentInstances !== null)
      ? ControlPlaneState_SegmentInstancesState.fromPartial(object.segmentInstances)
      : undefined;
    message.manifoldInstances = (object.manifoldInstances !== undefined && object.manifoldInstances !== null)
      ? ControlPlaneState_ManifoldInstancesState.fromPartial(object.manifoldInstances)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState.$type, ControlPlaneState);

function createBaseControlPlaneState_ExecutorsState(): ControlPlaneState_ExecutorsState {
  return { $type: "mrc.protos.ControlPlaneState.ExecutorsState", ids: [], entities: {} };
}

export const ControlPlaneState_ExecutorsState = {
  $type: "mrc.protos.ControlPlaneState.ExecutorsState" as const,

  encode(message: ControlPlaneState_ExecutorsState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_ExecutorsState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.ExecutorsState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ExecutorsState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ExecutorsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_ExecutorsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ExecutorsState {
    return {
      $type: ControlPlaneState_ExecutorsState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: Executor }>((acc, [key, value]) => {
          acc[key] = Executor.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_ExecutorsState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = Executor.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_ExecutorsState>): ControlPlaneState_ExecutorsState {
    return ControlPlaneState_ExecutorsState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_ExecutorsState>): ControlPlaneState_ExecutorsState {
    const message = createBaseControlPlaneState_ExecutorsState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: Executor }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = Executor.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_ExecutorsState.$type, ControlPlaneState_ExecutorsState);

function createBaseControlPlaneState_ExecutorsState_EntitiesEntry(): ControlPlaneState_ExecutorsState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.ExecutorsState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_ExecutorsState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.ExecutorsState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_ExecutorsState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      Executor.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ExecutorsState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ExecutorsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = Executor.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ExecutorsState_EntitiesEntry {
    return {
      $type: ControlPlaneState_ExecutorsState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Executor.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_ExecutorsState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? Executor.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_ExecutorsState_EntitiesEntry>,
  ): ControlPlaneState_ExecutorsState_EntitiesEntry {
    return ControlPlaneState_ExecutorsState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_ExecutorsState_EntitiesEntry>,
  ): ControlPlaneState_ExecutorsState_EntitiesEntry {
    const message = createBaseControlPlaneState_ExecutorsState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? Executor.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_ExecutorsState_EntitiesEntry.$type,
  ControlPlaneState_ExecutorsState_EntitiesEntry,
);

function createBaseControlPlaneState_WorkerssState(): ControlPlaneState_WorkerssState {
  return { $type: "mrc.protos.ControlPlaneState.WorkerssState", ids: [], entities: {} };
}

export const ControlPlaneState_WorkerssState = {
  $type: "mrc.protos.ControlPlaneState.WorkerssState" as const,

  encode(message: ControlPlaneState_WorkerssState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_WorkerssState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.WorkerssState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_WorkerssState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_WorkerssState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_WorkerssState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_WorkerssState {
    return {
      $type: ControlPlaneState_WorkerssState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: Worker }>((acc, [key, value]) => {
          acc[key] = Worker.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_WorkerssState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = Worker.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_WorkerssState>): ControlPlaneState_WorkerssState {
    return ControlPlaneState_WorkerssState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_WorkerssState>): ControlPlaneState_WorkerssState {
    const message = createBaseControlPlaneState_WorkerssState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: Worker }>((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = Worker.fromPartial(value);
      }
      return acc;
    }, {});
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_WorkerssState.$type, ControlPlaneState_WorkerssState);

function createBaseControlPlaneState_WorkerssState_EntitiesEntry(): ControlPlaneState_WorkerssState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.WorkerssState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_WorkerssState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.WorkerssState.EntitiesEntry" as const,

  encode(message: ControlPlaneState_WorkerssState_EntitiesEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      Worker.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_WorkerssState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_WorkerssState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = Worker.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_WorkerssState_EntitiesEntry {
    return {
      $type: ControlPlaneState_WorkerssState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Worker.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_WorkerssState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? Worker.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_WorkerssState_EntitiesEntry>,
  ): ControlPlaneState_WorkerssState_EntitiesEntry {
    return ControlPlaneState_WorkerssState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_WorkerssState_EntitiesEntry>,
  ): ControlPlaneState_WorkerssState_EntitiesEntry {
    const message = createBaseControlPlaneState_WorkerssState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? Worker.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_WorkerssState_EntitiesEntry.$type,
  ControlPlaneState_WorkerssState_EntitiesEntry,
);

function createBaseControlPlaneState_PipelineDefinitionsState(): ControlPlaneState_PipelineDefinitionsState {
  return { $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState", ids: [], entities: {} };
}

export const ControlPlaneState_PipelineDefinitionsState = {
  $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState" as const,

  encode(message: ControlPlaneState_PipelineDefinitionsState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_PipelineDefinitionsState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineDefinitionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_PipelineDefinitionsState {
    return {
      $type: ControlPlaneState_PipelineDefinitionsState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: PipelineDefinition }>((acc, [key, value]) => {
          acc[key] = PipelineDefinition.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_PipelineDefinitionsState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = PipelineDefinition.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_PipelineDefinitionsState>): ControlPlaneState_PipelineDefinitionsState {
    return ControlPlaneState_PipelineDefinitionsState.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_PipelineDefinitionsState>,
  ): ControlPlaneState_PipelineDefinitionsState {
    const message = createBaseControlPlaneState_PipelineDefinitionsState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: PipelineDefinition }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = PipelineDefinition.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_PipelineDefinitionsState.$type, ControlPlaneState_PipelineDefinitionsState);

function createBaseControlPlaneState_PipelineDefinitionsState_EntitiesEntry(): ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_PipelineDefinitionsState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.PipelineDefinitionsState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_PipelineDefinitionsState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      PipelineDefinition.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineDefinitionsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineDefinition.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
    return {
      $type: ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? PipelineDefinition.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_PipelineDefinitionsState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? PipelineDefinition.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_PipelineDefinitionsState_EntitiesEntry>,
  ): ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
    return ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_PipelineDefinitionsState_EntitiesEntry>,
  ): ControlPlaneState_PipelineDefinitionsState_EntitiesEntry {
    const message = createBaseControlPlaneState_PipelineDefinitionsState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineDefinition.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.$type,
  ControlPlaneState_PipelineDefinitionsState_EntitiesEntry,
);

function createBaseControlPlaneState_PipelineInstancesState(): ControlPlaneState_PipelineInstancesState {
  return { $type: "mrc.protos.ControlPlaneState.PipelineInstancesState", ids: [], entities: {} };
}

export const ControlPlaneState_PipelineInstancesState = {
  $type: "mrc.protos.ControlPlaneState.PipelineInstancesState" as const,

  encode(message: ControlPlaneState_PipelineInstancesState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_PipelineInstancesState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.PipelineInstancesState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_PipelineInstancesState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineInstancesState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_PipelineInstancesState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_PipelineInstancesState {
    return {
      $type: ControlPlaneState_PipelineInstancesState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: PipelineInstance }>((acc, [key, value]) => {
          acc[key] = PipelineInstance.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_PipelineInstancesState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = PipelineInstance.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_PipelineInstancesState>): ControlPlaneState_PipelineInstancesState {
    return ControlPlaneState_PipelineInstancesState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_PipelineInstancesState>): ControlPlaneState_PipelineInstancesState {
    const message = createBaseControlPlaneState_PipelineInstancesState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: PipelineInstance }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = PipelineInstance.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_PipelineInstancesState.$type, ControlPlaneState_PipelineInstancesState);

function createBaseControlPlaneState_PipelineInstancesState_EntitiesEntry(): ControlPlaneState_PipelineInstancesState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.PipelineInstancesState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_PipelineInstancesState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.PipelineInstancesState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_PipelineInstancesState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      PipelineInstance.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_PipelineInstancesState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineInstancesState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = PipelineInstance.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_PipelineInstancesState_EntitiesEntry {
    return {
      $type: ControlPlaneState_PipelineInstancesState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? PipelineInstance.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_PipelineInstancesState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? PipelineInstance.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_PipelineInstancesState_EntitiesEntry>,
  ): ControlPlaneState_PipelineInstancesState_EntitiesEntry {
    return ControlPlaneState_PipelineInstancesState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_PipelineInstancesState_EntitiesEntry>,
  ): ControlPlaneState_PipelineInstancesState_EntitiesEntry {
    const message = createBaseControlPlaneState_PipelineInstancesState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? PipelineInstance.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_PipelineInstancesState_EntitiesEntry.$type,
  ControlPlaneState_PipelineInstancesState_EntitiesEntry,
);

function createBaseControlPlaneState_SegmentDefinitionsState(): ControlPlaneState_SegmentDefinitionsState {
  return { $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState", ids: [], entities: {} };
}

export const ControlPlaneState_SegmentDefinitionsState = {
  $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState" as const,

  encode(message: ControlPlaneState_SegmentDefinitionsState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_SegmentDefinitionsState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentDefinitionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_SegmentDefinitionsState {
    return {
      $type: ControlPlaneState_SegmentDefinitionsState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: SegmentDefinition }>((acc, [key, value]) => {
          acc[key] = SegmentDefinition.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_SegmentDefinitionsState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = SegmentDefinition.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_SegmentDefinitionsState>): ControlPlaneState_SegmentDefinitionsState {
    return ControlPlaneState_SegmentDefinitionsState.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_SegmentDefinitionsState>,
  ): ControlPlaneState_SegmentDefinitionsState {
    const message = createBaseControlPlaneState_SegmentDefinitionsState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: SegmentDefinition }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = SegmentDefinition.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_SegmentDefinitionsState.$type, ControlPlaneState_SegmentDefinitionsState);

function createBaseControlPlaneState_SegmentDefinitionsState_EntitiesEntry(): ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_SegmentDefinitionsState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.SegmentDefinitionsState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_SegmentDefinitionsState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      SegmentDefinition.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentDefinitionsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = SegmentDefinition.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
    return {
      $type: ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? SegmentDefinition.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_SegmentDefinitionsState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? SegmentDefinition.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_SegmentDefinitionsState_EntitiesEntry>,
  ): ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
    return ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_SegmentDefinitionsState_EntitiesEntry>,
  ): ControlPlaneState_SegmentDefinitionsState_EntitiesEntry {
    const message = createBaseControlPlaneState_SegmentDefinitionsState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? SegmentDefinition.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.$type,
  ControlPlaneState_SegmentDefinitionsState_EntitiesEntry,
);

function createBaseControlPlaneState_SegmentInstancesState(): ControlPlaneState_SegmentInstancesState {
  return { $type: "mrc.protos.ControlPlaneState.SegmentInstancesState", ids: [], entities: {} };
}

export const ControlPlaneState_SegmentInstancesState = {
  $type: "mrc.protos.ControlPlaneState.SegmentInstancesState" as const,

  encode(message: ControlPlaneState_SegmentInstancesState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_SegmentInstancesState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.SegmentInstancesState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_SegmentInstancesState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentInstancesState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_SegmentInstancesState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_SegmentInstancesState {
    return {
      $type: ControlPlaneState_SegmentInstancesState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: SegmentInstance }>((acc, [key, value]) => {
          acc[key] = SegmentInstance.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_SegmentInstancesState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = SegmentInstance.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_SegmentInstancesState>): ControlPlaneState_SegmentInstancesState {
    return ControlPlaneState_SegmentInstancesState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_SegmentInstancesState>): ControlPlaneState_SegmentInstancesState {
    const message = createBaseControlPlaneState_SegmentInstancesState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: SegmentInstance }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = SegmentInstance.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_SegmentInstancesState.$type, ControlPlaneState_SegmentInstancesState);

function createBaseControlPlaneState_SegmentInstancesState_EntitiesEntry(): ControlPlaneState_SegmentInstancesState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.SegmentInstancesState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_SegmentInstancesState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.SegmentInstancesState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_SegmentInstancesState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      SegmentInstance.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_SegmentInstancesState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentInstancesState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = SegmentInstance.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_SegmentInstancesState_EntitiesEntry {
    return {
      $type: ControlPlaneState_SegmentInstancesState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? SegmentInstance.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_SegmentInstancesState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? SegmentInstance.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_SegmentInstancesState_EntitiesEntry>,
  ): ControlPlaneState_SegmentInstancesState_EntitiesEntry {
    return ControlPlaneState_SegmentInstancesState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_SegmentInstancesState_EntitiesEntry>,
  ): ControlPlaneState_SegmentInstancesState_EntitiesEntry {
    const message = createBaseControlPlaneState_SegmentInstancesState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? SegmentInstance.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_SegmentInstancesState_EntitiesEntry.$type,
  ControlPlaneState_SegmentInstancesState_EntitiesEntry,
);

function createBaseControlPlaneState_ManifoldInstancesState(): ControlPlaneState_ManifoldInstancesState {
  return { $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState", ids: [], entities: {} };
}

export const ControlPlaneState_ManifoldInstancesState = {
  $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState" as const,

  encode(message: ControlPlaneState_ManifoldInstancesState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_ManifoldInstancesState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ManifoldInstancesState {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ManifoldInstancesState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag === 8) {
            message.ids.push(longToString(reader.uint64() as Long));

            continue;
          }

          if (tag === 10) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }

            continue;
          }

          break;
        case 2:
          if (tag !== 18) {
            break;
          }

          const entry2 = ControlPlaneState_ManifoldInstancesState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ManifoldInstancesState {
    return {
      $type: ControlPlaneState_ManifoldInstancesState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: ManifoldInstance }>((acc, [key, value]) => {
          acc[key] = ManifoldInstance.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_ManifoldInstancesState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = ManifoldInstance.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_ManifoldInstancesState>): ControlPlaneState_ManifoldInstancesState {
    return ControlPlaneState_ManifoldInstancesState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_ManifoldInstancesState>): ControlPlaneState_ManifoldInstancesState {
    const message = createBaseControlPlaneState_ManifoldInstancesState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: ManifoldInstance }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = ManifoldInstance.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_ManifoldInstancesState.$type, ControlPlaneState_ManifoldInstancesState);

function createBaseControlPlaneState_ManifoldInstancesState_EntitiesEntry(): ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_ManifoldInstancesState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.ManifoldInstancesState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_ManifoldInstancesState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      ManifoldInstance.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ManifoldInstancesState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.key = longToString(reader.uint64() as Long);
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = ManifoldInstance.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
    return {
      $type: ControlPlaneState_ManifoldInstancesState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? ManifoldInstance.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_ManifoldInstancesState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? ManifoldInstance.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_ManifoldInstancesState_EntitiesEntry>,
  ): ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
    return ControlPlaneState_ManifoldInstancesState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_ManifoldInstancesState_EntitiesEntry>,
  ): ControlPlaneState_ManifoldInstancesState_EntitiesEntry {
    const message = createBaseControlPlaneState_ManifoldInstancesState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? ManifoldInstance.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_ManifoldInstancesState_EntitiesEntry.$type,
  ControlPlaneState_ManifoldInstancesState_EntitiesEntry,
);

function createBaseSegmentOptions(): SegmentOptions {
  return {
    $type: "mrc.protos.SegmentOptions",
    placementStrategy: SegmentOptions_PlacementStrategy.ResourceGroup,
    scalingOptions: undefined,
  };
}

export const SegmentOptions = {
  $type: "mrc.protos.SegmentOptions" as const,

  encode(message: SegmentOptions, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.placementStrategy !== SegmentOptions_PlacementStrategy.ResourceGroup) {
      writer.uint32(8).int32(segmentOptions_PlacementStrategyToNumber(message.placementStrategy));
    }
    if (message.scalingOptions !== undefined) {
      ScalingOptions.encode(message.scalingOptions, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentOptions {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentOptions();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.placementStrategy = segmentOptions_PlacementStrategyFromJSON(reader.int32());
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.scalingOptions = ScalingOptions.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): SegmentOptions {
    return {
      $type: SegmentOptions.$type,
      placementStrategy: isSet(object.placementStrategy)
        ? segmentOptions_PlacementStrategyFromJSON(object.placementStrategy)
        : SegmentOptions_PlacementStrategy.ResourceGroup,
      scalingOptions: isSet(object.scalingOptions) ? ScalingOptions.fromJSON(object.scalingOptions) : undefined,
    };
  },

  toJSON(message: SegmentOptions): unknown {
    const obj: any = {};
    message.placementStrategy !== undefined &&
      (obj.placementStrategy = segmentOptions_PlacementStrategyToJSON(message.placementStrategy));
    message.scalingOptions !== undefined &&
      (obj.scalingOptions = message.scalingOptions ? ScalingOptions.toJSON(message.scalingOptions) : undefined);
    return obj;
  },

  create(base?: DeepPartial<SegmentOptions>): SegmentOptions {
    return SegmentOptions.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentOptions>): SegmentOptions {
    const message = createBaseSegmentOptions();
    message.placementStrategy = object.placementStrategy ?? SegmentOptions_PlacementStrategy.ResourceGroup;
    message.scalingOptions = (object.scalingOptions !== undefined && object.scalingOptions !== null)
      ? ScalingOptions.fromPartial(object.scalingOptions)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentOptions.$type, SegmentOptions);

function createBaseScalingOptions(): ScalingOptions {
  return { $type: "mrc.protos.ScalingOptions", strategy: ScalingOptions_ScalingStrategy.Static, initialCount: 0 };
}

export const ScalingOptions = {
  $type: "mrc.protos.ScalingOptions" as const,

  encode(message: ScalingOptions, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.strategy !== ScalingOptions_ScalingStrategy.Static) {
      writer.uint32(8).int32(scalingOptions_ScalingStrategyToNumber(message.strategy));
    }
    if (message.initialCount !== 0) {
      writer.uint32(16).uint32(message.initialCount);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ScalingOptions {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseScalingOptions();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.strategy = scalingOptions_ScalingStrategyFromJSON(reader.int32());
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.initialCount = reader.uint32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ScalingOptions {
    return {
      $type: ScalingOptions.$type,
      strategy: isSet(object.strategy)
        ? scalingOptions_ScalingStrategyFromJSON(object.strategy)
        : ScalingOptions_ScalingStrategy.Static,
      initialCount: isSet(object.initialCount) ? Number(object.initialCount) : 0,
    };
  },

  toJSON(message: ScalingOptions): unknown {
    const obj: any = {};
    message.strategy !== undefined && (obj.strategy = scalingOptions_ScalingStrategyToJSON(message.strategy));
    message.initialCount !== undefined && (obj.initialCount = Math.round(message.initialCount));
    return obj;
  },

  create(base?: DeepPartial<ScalingOptions>): ScalingOptions {
    return ScalingOptions.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ScalingOptions>): ScalingOptions {
    const message = createBaseScalingOptions();
    message.strategy = object.strategy ?? ScalingOptions_ScalingStrategy.Static;
    message.initialCount = object.initialCount ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ScalingOptions.$type, ScalingOptions);

function createBaseManifoldOptions(): ManifoldOptions {
  return { $type: "mrc.protos.ManifoldOptions", policy: ManifoldOptions_Policy.LoadBalance };
}

export const ManifoldOptions = {
  $type: "mrc.protos.ManifoldOptions" as const,

  encode(message: ManifoldOptions, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.policy !== ManifoldOptions_Policy.LoadBalance) {
      writer.uint32(8).int32(manifoldOptions_PolicyToNumber(message.policy));
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ManifoldOptions {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseManifoldOptions();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.policy = manifoldOptions_PolicyFromJSON(reader.int32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): ManifoldOptions {
    return {
      $type: ManifoldOptions.$type,
      policy: isSet(object.policy) ? manifoldOptions_PolicyFromJSON(object.policy) : ManifoldOptions_Policy.LoadBalance,
    };
  },

  toJSON(message: ManifoldOptions): unknown {
    const obj: any = {};
    message.policy !== undefined && (obj.policy = manifoldOptions_PolicyToJSON(message.policy));
    return obj;
  },

  create(base?: DeepPartial<ManifoldOptions>): ManifoldOptions {
    return ManifoldOptions.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ManifoldOptions>): ManifoldOptions {
    const message = createBaseManifoldOptions();
    message.policy = object.policy ?? ManifoldOptions_Policy.LoadBalance;
    return message;
  },
};

messageTypeRegistry.set(ManifoldOptions.$type, ManifoldOptions);

function createBasePortInfo(): PortInfo {
  return { $type: "mrc.protos.PortInfo", portName: "", typeId: 0, typeString: "" };
}

export const PortInfo = {
  $type: "mrc.protos.PortInfo" as const,

  encode(message: PortInfo, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.portName !== "") {
      writer.uint32(10).string(message.portName);
    }
    if (message.typeId !== 0) {
      writer.uint32(16).uint32(message.typeId);
    }
    if (message.typeString !== "") {
      writer.uint32(26).string(message.typeString);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PortInfo {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePortInfo();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.portName = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.typeId = reader.uint32();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.typeString = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): PortInfo {
    return {
      $type: PortInfo.$type,
      portName: isSet(object.portName) ? String(object.portName) : "",
      typeId: isSet(object.typeId) ? Number(object.typeId) : 0,
      typeString: isSet(object.typeString) ? String(object.typeString) : "",
    };
  },

  toJSON(message: PortInfo): unknown {
    const obj: any = {};
    message.portName !== undefined && (obj.portName = message.portName);
    message.typeId !== undefined && (obj.typeId = Math.round(message.typeId));
    message.typeString !== undefined && (obj.typeString = message.typeString);
    return obj;
  },

  create(base?: DeepPartial<PortInfo>): PortInfo {
    return PortInfo.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PortInfo>): PortInfo {
    const message = createBasePortInfo();
    message.portName = object.portName ?? "";
    message.typeId = object.typeId ?? 0;
    message.typeString = object.typeString ?? "";
    return message;
  },
};

messageTypeRegistry.set(PortInfo.$type, PortInfo);

function createBaseIngressPort(): IngressPort {
  return { $type: "mrc.protos.IngressPort", name: "", id: 0 };
}

export const IngressPort = {
  $type: "mrc.protos.IngressPort" as const,

  encode(message: IngressPort, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.id !== 0) {
      writer.uint32(16).uint32(message.id);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): IngressPort {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseIngressPort();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.name = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.id = reader.uint32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): IngressPort {
    return {
      $type: IngressPort.$type,
      name: isSet(object.name) ? String(object.name) : "",
      id: isSet(object.id) ? Number(object.id) : 0,
    };
  },

  toJSON(message: IngressPort): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.id !== undefined && (obj.id = Math.round(message.id));
    return obj;
  },

  create(base?: DeepPartial<IngressPort>): IngressPort {
    return IngressPort.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<IngressPort>): IngressPort {
    const message = createBaseIngressPort();
    message.name = object.name ?? "";
    message.id = object.id ?? 0;
    return message;
  },
};

messageTypeRegistry.set(IngressPort.$type, IngressPort);

function createBaseEgressPort(): EgressPort {
  return { $type: "mrc.protos.EgressPort", name: "", id: 0, policyType: EgressPort_PolicyType.PolicyDefined };
}

export const EgressPort = {
  $type: "mrc.protos.EgressPort" as const,

  encode(message: EgressPort, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.id !== 0) {
      writer.uint32(16).uint32(message.id);
    }
    if (message.policyType !== EgressPort_PolicyType.PolicyDefined) {
      writer.uint32(24).int32(egressPort_PolicyTypeToNumber(message.policyType));
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): EgressPort {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseEgressPort();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.name = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.id = reader.uint32();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.policyType = egressPort_PolicyTypeFromJSON(reader.int32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): EgressPort {
    return {
      $type: EgressPort.$type,
      name: isSet(object.name) ? String(object.name) : "",
      id: isSet(object.id) ? Number(object.id) : 0,
      policyType: isSet(object.policyType)
        ? egressPort_PolicyTypeFromJSON(object.policyType)
        : EgressPort_PolicyType.PolicyDefined,
    };
  },

  toJSON(message: EgressPort): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.id !== undefined && (obj.id = Math.round(message.id));
    message.policyType !== undefined && (obj.policyType = egressPort_PolicyTypeToJSON(message.policyType));
    return obj;
  },

  create(base?: DeepPartial<EgressPort>): EgressPort {
    return EgressPort.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<EgressPort>): EgressPort {
    const message = createBaseEgressPort();
    message.name = object.name ?? "";
    message.id = object.id ?? 0;
    message.policyType = object.policyType ?? EgressPort_PolicyType.PolicyDefined;
    return message;
  },
};

messageTypeRegistry.set(EgressPort.$type, EgressPort);

function createBaseIngressPolicy(): IngressPolicy {
  return { $type: "mrc.protos.IngressPolicy", networkEnabled: false };
}

export const IngressPolicy = {
  $type: "mrc.protos.IngressPolicy" as const,

  encode(message: IngressPolicy, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.networkEnabled === true) {
      writer.uint32(8).bool(message.networkEnabled);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): IngressPolicy {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseIngressPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.networkEnabled = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): IngressPolicy {
    return {
      $type: IngressPolicy.$type,
      networkEnabled: isSet(object.networkEnabled) ? Boolean(object.networkEnabled) : false,
    };
  },

  toJSON(message: IngressPolicy): unknown {
    const obj: any = {};
    message.networkEnabled !== undefined && (obj.networkEnabled = message.networkEnabled);
    return obj;
  },

  create(base?: DeepPartial<IngressPolicy>): IngressPolicy {
    return IngressPolicy.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<IngressPolicy>): IngressPolicy {
    const message = createBaseIngressPolicy();
    message.networkEnabled = object.networkEnabled ?? false;
    return message;
  },
};

messageTypeRegistry.set(IngressPolicy.$type, IngressPolicy);

function createBaseEgressPolicy(): EgressPolicy {
  return { $type: "mrc.protos.EgressPolicy", policy: EgressPolicy_Policy.LoadBalance, segmentAddresses: [] };
}

export const EgressPolicy = {
  $type: "mrc.protos.EgressPolicy" as const,

  encode(message: EgressPolicy, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.policy !== EgressPolicy_Policy.LoadBalance) {
      writer.uint32(24).int32(egressPolicy_PolicyToNumber(message.policy));
    }
    writer.uint32(34).fork();
    for (const v of message.segmentAddresses) {
      writer.uint32(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): EgressPolicy {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseEgressPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 3:
          if (tag !== 24) {
            break;
          }

          message.policy = egressPolicy_PolicyFromJSON(reader.int32());
          continue;
        case 4:
          if (tag === 32) {
            message.segmentAddresses.push(reader.uint32());

            continue;
          }

          if (tag === 34) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.segmentAddresses.push(reader.uint32());
            }

            continue;
          }

          break;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  fromJSON(object: any): EgressPolicy {
    return {
      $type: EgressPolicy.$type,
      policy: isSet(object.policy) ? egressPolicy_PolicyFromJSON(object.policy) : EgressPolicy_Policy.LoadBalance,
      segmentAddresses: Array.isArray(object?.segmentAddresses)
        ? object.segmentAddresses.map((e: any) => Number(e))
        : [],
    };
  },

  toJSON(message: EgressPolicy): unknown {
    const obj: any = {};
    message.policy !== undefined && (obj.policy = egressPolicy_PolicyToJSON(message.policy));
    if (message.segmentAddresses) {
      obj.segmentAddresses = message.segmentAddresses.map((e) => Math.round(e));
    } else {
      obj.segmentAddresses = [];
    }
    return obj;
  },

  create(base?: DeepPartial<EgressPolicy>): EgressPolicy {
    return EgressPolicy.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<EgressPolicy>): EgressPolicy {
    const message = createBaseEgressPolicy();
    message.policy = object.policy ?? EgressPolicy_Policy.LoadBalance;
    message.segmentAddresses = object.segmentAddresses?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(EgressPolicy.$type, EgressPolicy);

declare var self: any | undefined;
declare var window: any | undefined;
declare var global: any | undefined;
var tsProtoGlobalThis: any = (() => {
  if (typeof globalThis !== "undefined") {
    return globalThis;
  }
  if (typeof self !== "undefined") {
    return self;
  }
  if (typeof window !== "undefined") {
    return window;
  }
  if (typeof global !== "undefined") {
    return global;
  }
  throw "Unable to locate global object";
})();

function bytesFromBase64(b64: string): Uint8Array {
  if (tsProtoGlobalThis.Buffer) {
    return Uint8Array.from(tsProtoGlobalThis.Buffer.from(b64, "base64"));
  } else {
    const bin = tsProtoGlobalThis.atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) {
      arr[i] = bin.charCodeAt(i);
    }
    return arr;
  }
}

function base64FromBytes(arr: Uint8Array): string {
  if (tsProtoGlobalThis.Buffer) {
    return tsProtoGlobalThis.Buffer.from(arr).toString("base64");
  } else {
    const bin: string[] = [];
    arr.forEach((byte) => {
      bin.push(String.fromCharCode(byte));
    });
    return tsProtoGlobalThis.btoa(bin.join(""));
  }
}

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

export type DeepPartial<T> = T extends Builtin ? T
  : T extends Array<infer U> ? Array<DeepPartial<U>> : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in Exclude<keyof T, "$type">]?: DeepPartial<T[K]> }
  : Partial<T>;

function longToString(long: Long) {
  return long.toString();
}

if (_m0.util.Long !== Long) {
  _m0.util.Long = Long as any;
  _m0.configure();
}

function isObject(value: any): boolean {
  return typeof value === "object" && value !== null;
}

function isSet(value: any): boolean {
  return value !== null && value !== undefined;
}
