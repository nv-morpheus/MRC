/* eslint-disable */
import Long from "long";
import _m0 from "protobufjs/minimal";
import { messageTypeRegistry } from "../../typeRegistry";

export const protobufPackage = "mrc.protos";

export enum ResourceStatus {
  /** Registered - Control Plane indicates a resource should be created on the client */
  Registered = 0,
  /** Activated - Client has created resource but it is not ready */
  Activated = 1,
  /** Ready - Client and Control Plane can use the resource */
  Ready = 2,
  /**
   * Deactivating - Control Plane has indicated the resource should be destroyed on the client. All users of the resource should stop
   * using it and decrement the ref count. Object is still running
   */
  Deactivating = 3,
  /** Deactivated - All ref counts have been decremented. Owner of the object on the client can begin destroying object */
  Deactivated = 4,
  /** Unregistered - Client owner of resource has begun destroying the object */
  Unregistered = 5,
  /** Destroyed - Object has been destroyed on the client (and may be removed from the server) */
  Destroyed = 6,
  UNRECOGNIZED = -1,
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

export enum SegmentStates {
  Initialized = 0,
  Running = 1,
  Stopped = 2,
  Completed = 3,
  UNRECOGNIZED = -1,
}

export function segmentStatesFromJSON(object: any): SegmentStates {
  switch (object) {
    case 0:
    case "Initialized":
      return SegmentStates.Initialized;
    case 1:
    case "Running":
      return SegmentStates.Running;
    case 2:
    case "Stopped":
      return SegmentStates.Stopped;
    case 3:
    case "Completed":
      return SegmentStates.Completed;
    case -1:
    case "UNRECOGNIZED":
    default:
      return SegmentStates.UNRECOGNIZED;
  }
}

export function segmentStatesToJSON(object: SegmentStates): string {
  switch (object) {
    case SegmentStates.Initialized:
      return "Initialized";
    case SegmentStates.Running:
      return "Running";
    case SegmentStates.Stopped:
      return "Stopped";
    case SegmentStates.Completed:
      return "Completed";
    case SegmentStates.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export enum SegmentMappingPolicies {
  OnePerWorker = 0,
  UNRECOGNIZED = -1,
}

export function segmentMappingPoliciesFromJSON(object: any): SegmentMappingPolicies {
  switch (object) {
    case 0:
    case "OnePerWorker":
      return SegmentMappingPolicies.OnePerWorker;
    case -1:
    case "UNRECOGNIZED":
    default:
      return SegmentMappingPolicies.UNRECOGNIZED;
  }
}

export function segmentMappingPoliciesToJSON(object: SegmentMappingPolicies): string {
  switch (object) {
    case SegmentMappingPolicies.OnePerWorker:
      return "OnePerWorker";
    case SegmentMappingPolicies.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export interface ResourceState {
  $type: "mrc.protos.ResourceState";
  /** Current status of the resource */
  status: ResourceStatus;
  /** Number of users besides the owner of this resource */
  refCount: number;
}

export interface Connection {
  $type: "mrc.protos.Connection";
  id: string;
  /** Info about the client (IP/Port) */
  peerInfo: string;
  /** Workers that belong to this machine */
  workerIds: string[];
  /** The pipeline instances that are assigned to this machine */
  assignedPipelineIds: string[];
}

export interface Worker {
  $type: "mrc.protos.Worker";
  id: string;
  /** Serialized worker address */
  workerAddress: Uint8Array;
  /** Parent machine this worker belongs to */
  machineId: string;
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
}

export interface PipelineConfiguration_SegmentConfiguration {
  $type: "mrc.protos.PipelineConfiguration.SegmentConfiguration";
  /** Name of the segment */
  name: string;
  /** Ingress ports for this segment */
  ingressPorts: IngressPort[];
  /** Egress ports for this segment */
  egressPorts: EgressPort[];
  /** Segment options */
  options: SegmentOptions | undefined;
}

export interface PipelineConfiguration_SegmentsEntry {
  $type: "mrc.protos.PipelineConfiguration.SegmentsEntry";
  key: string;
  value: PipelineConfiguration_SegmentConfiguration | undefined;
}

export interface PipelineMapping {
  $type: "mrc.protos.PipelineMapping";
  machineId: string;
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
  byWorker?: PipelineMapping_SegmentMapping_ByWorker | undefined;
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
  /** Machine IDs to mappings for all connections */
  mappings: { [key: string]: PipelineMapping };
  /** Running Pipeline Instance IDs */
  instanceIds: string[];
  /** Running Segment Info */
  segments: { [key: string]: PipelineDefinition_SegmentDefinition };
}

export interface PipelineDefinition_SegmentDefinition {
  $type: "mrc.protos.PipelineDefinition.SegmentDefinition";
  /** Generated ID of the definition */
  id: string;
  /** ID of the parent for back referencing */
  parentId: string;
  /** Name of the segment */
  name: string;
  /** Running Segment Instance IDs */
  instanceIds: string[];
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

export interface PipelineInstance {
  $type: "mrc.protos.PipelineInstance";
  id: string;
  /** Deinition this belongs to */
  definitionId: string;
  /** The machine this instance is running on */
  machineId: string;
  /** The current state of this resource */
  state:
    | ResourceState
    | undefined;
  /** Running Segment Instance IDs */
  segmentIds: string[];
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
  id: string;
  /** Pipeline Deinition this belongs to */
  pipelineDefinitionId: string;
  /** Segment name (Lookup segment config from pipeline def ID and name) */
  name: string;
  /** The encoded address of this instance */
  address: number;
  /** The worker/partition that this belongs to */
  workerId: string;
  /** The running pipeline instance id */
  pipelineInstanceId: string;
  /** The current state of this resource */
  state: ResourceState | undefined;
}

export interface ControlPlaneState {
  $type: "mrc.protos.ControlPlaneState";
  connections: ControlPlaneState_ConnectionsState | undefined;
  workers: ControlPlaneState_WorkerssState | undefined;
  pipelineDefinitions: ControlPlaneState_PipelineDefinitionsState | undefined;
  pipelineInstances: ControlPlaneState_PipelineInstancesState | undefined;
  segmentDefinitions: ControlPlaneState_SegmentDefinitionsState | undefined;
  segmentInstances: ControlPlaneState_SegmentInstancesState | undefined;
}

export interface ControlPlaneState_ConnectionsState {
  $type: "mrc.protos.ControlPlaneState.ConnectionsState";
  ids: string[];
  entities: { [key: string]: Connection };
}

export interface ControlPlaneState_ConnectionsState_EntitiesEntry {
  $type: "mrc.protos.ControlPlaneState.ConnectionsState.EntitiesEntry";
  key: string;
  value: Connection | undefined;
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

export interface SegmentOptions {
  $type: "mrc.protos.SegmentOptions";
  placementStrategy: SegmentOptions_PlacementStrategy;
  scalingOptions: ScalingOptions | undefined;
}

export enum SegmentOptions_PlacementStrategy {
  ResourceGroup = 0,
  PhysicalMachine = 1,
  Global = 2,
  UNRECOGNIZED = -1,
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

export interface ScalingOptions {
  $type: "mrc.protos.ScalingOptions";
  strategy: ScalingOptions_ScalingStrategy;
  initialCount: number;
}

export enum ScalingOptions_ScalingStrategy {
  Static = 0,
  UNRECOGNIZED = -1,
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
  PolicyDefined = 0,
  UserDefined = 1,
  UNRECOGNIZED = -1,
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
  LoadBalance = 0,
  Broadcast = 1,
  UNRECOGNIZED = -1,
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

function createBaseResourceState(): ResourceState {
  return { $type: "mrc.protos.ResourceState", status: 0, refCount: 0 };
}

export const ResourceState = {
  $type: "mrc.protos.ResourceState" as const,

  encode(message: ResourceState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.status !== 0) {
      writer.uint32(8).int32(message.status);
    }
    if (message.refCount !== 0) {
      writer.uint32(16).int32(message.refCount);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ResourceState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseResourceState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.status = reader.int32() as any;
          break;
        case 2:
          message.refCount = reader.int32();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ResourceState {
    return {
      $type: ResourceState.$type,
      status: isSet(object.status) ? resourceStatusFromJSON(object.status) : 0,
      refCount: isSet(object.refCount) ? Number(object.refCount) : 0,
    };
  },

  toJSON(message: ResourceState): unknown {
    const obj: any = {};
    message.status !== undefined && (obj.status = resourceStatusToJSON(message.status));
    message.refCount !== undefined && (obj.refCount = Math.round(message.refCount));
    return obj;
  },

  create(base?: DeepPartial<ResourceState>): ResourceState {
    return ResourceState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ResourceState>): ResourceState {
    const message = createBaseResourceState();
    message.status = object.status ?? 0;
    message.refCount = object.refCount ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ResourceState.$type, ResourceState);

function createBaseConnection(): Connection {
  return { $type: "mrc.protos.Connection", id: "0", peerInfo: "", workerIds: [], assignedPipelineIds: [] };
}

export const Connection = {
  $type: "mrc.protos.Connection" as const,

  encode(message: Connection, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.peerInfo !== "") {
      writer.uint32(18).string(message.peerInfo);
    }
    writer.uint32(26).fork();
    for (const v of message.workerIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    writer.uint32(34).fork();
    for (const v of message.assignedPipelineIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Connection {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseConnection();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.peerInfo = reader.string();
          break;
        case 3:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.workerIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.workerIds.push(longToString(reader.uint64() as Long));
          }
          break;
        case 4:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.assignedPipelineIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.assignedPipelineIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Connection {
    return {
      $type: Connection.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      peerInfo: isSet(object.peerInfo) ? String(object.peerInfo) : "",
      workerIds: Array.isArray(object?.workerIds) ? object.workerIds.map((e: any) => String(e)) : [],
      assignedPipelineIds: Array.isArray(object?.assignedPipelineIds)
        ? object.assignedPipelineIds.map((e: any) => String(e))
        : [],
    };
  },

  toJSON(message: Connection): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.peerInfo !== undefined && (obj.peerInfo = message.peerInfo);
    if (message.workerIds) {
      obj.workerIds = message.workerIds.map((e) => e);
    } else {
      obj.workerIds = [];
    }
    if (message.assignedPipelineIds) {
      obj.assignedPipelineIds = message.assignedPipelineIds.map((e) => e);
    } else {
      obj.assignedPipelineIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<Connection>): Connection {
    return Connection.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Connection>): Connection {
    const message = createBaseConnection();
    message.id = object.id ?? "0";
    message.peerInfo = object.peerInfo ?? "";
    message.workerIds = object.workerIds?.map((e) => e) || [];
    message.assignedPipelineIds = object.assignedPipelineIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(Connection.$type, Connection);

function createBaseWorker(): Worker {
  return {
    $type: "mrc.protos.Worker",
    id: "0",
    workerAddress: new Uint8Array(),
    machineId: "0",
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
    if (message.workerAddress.length !== 0) {
      writer.uint32(18).bytes(message.workerAddress);
    }
    if (message.machineId !== "0") {
      writer.uint32(24).uint64(message.machineId);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(34).fork()).ldelim();
    }
    writer.uint32(42).fork();
    for (const v of message.assignedSegmentIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Worker {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorker();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.workerAddress = reader.bytes();
          break;
        case 3:
          message.machineId = longToString(reader.uint64() as Long);
          break;
        case 4:
          message.state = ResourceState.decode(reader, reader.uint32());
          break;
        case 5:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.assignedSegmentIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.assignedSegmentIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Worker {
    return {
      $type: Worker.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      workerAddress: isSet(object.workerAddress) ? bytesFromBase64(object.workerAddress) : new Uint8Array(),
      machineId: isSet(object.machineId) ? String(object.machineId) : "0",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      assignedSegmentIds: Array.isArray(object?.assignedSegmentIds)
        ? object.assignedSegmentIds.map((e: any) => String(e))
        : [],
    };
  },

  toJSON(message: Worker): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.workerAddress !== undefined &&
      (obj.workerAddress = base64FromBytes(
        message.workerAddress !== undefined ? message.workerAddress : new Uint8Array(),
      ));
    message.machineId !== undefined && (obj.machineId = message.machineId);
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
    message.workerAddress = object.workerAddress ?? new Uint8Array();
    message.machineId = object.machineId ?? "0";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.assignedSegmentIds = object.assignedSegmentIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(Worker.$type, Worker);

function createBasePipelineConfiguration(): PipelineConfiguration {
  return { $type: "mrc.protos.PipelineConfiguration", segments: {} };
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
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          const entry1 = PipelineConfiguration_SegmentsEntry.decode(reader, reader.uint32());
          if (entry1.value !== undefined) {
            message.segments[entry1.key] = entry1.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration.$type, PipelineConfiguration);

function createBasePipelineConfiguration_SegmentConfiguration(): PipelineConfiguration_SegmentConfiguration {
  return {
    $type: "mrc.protos.PipelineConfiguration.SegmentConfiguration",
    name: "",
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
    for (const v of message.ingressPorts) {
      IngressPort.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    for (const v of message.egressPorts) {
      EgressPort.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    if (message.options !== undefined) {
      SegmentOptions.encode(message.options, writer.uint32(34).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineConfiguration_SegmentConfiguration {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_SegmentConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.ingressPorts.push(IngressPort.decode(reader, reader.uint32()));
          break;
        case 3:
          message.egressPorts.push(EgressPort.decode(reader, reader.uint32()));
          break;
        case 4:
          message.options = SegmentOptions.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineConfiguration_SegmentConfiguration {
    return {
      $type: PipelineConfiguration_SegmentConfiguration.$type,
      name: isSet(object.name) ? String(object.name) : "",
      ingressPorts: Array.isArray(object?.ingressPorts)
        ? object.ingressPorts.map((e: any) => IngressPort.fromJSON(e))
        : [],
      egressPorts: Array.isArray(object?.egressPorts) ? object.egressPorts.map((e: any) => EgressPort.fromJSON(e)) : [],
      options: isSet(object.options) ? SegmentOptions.fromJSON(object.options) : undefined,
    };
  },

  toJSON(message: PipelineConfiguration_SegmentConfiguration): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    if (message.ingressPorts) {
      obj.ingressPorts = message.ingressPorts.map((e) => e ? IngressPort.toJSON(e) : undefined);
    } else {
      obj.ingressPorts = [];
    }
    if (message.egressPorts) {
      obj.egressPorts = message.egressPorts.map((e) => e ? EgressPort.toJSON(e) : undefined);
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
    message.ingressPorts = object.ingressPorts?.map((e) => IngressPort.fromPartial(e)) || [];
    message.egressPorts = object.egressPorts?.map((e) => EgressPort.fromPartial(e)) || [];
    message.options = (object.options !== undefined && object.options !== null)
      ? SegmentOptions.fromPartial(object.options)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration_SegmentConfiguration.$type, PipelineConfiguration_SegmentConfiguration);

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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineConfiguration_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.string();
          break;
        case 2:
          message.value = PipelineConfiguration_SegmentConfiguration.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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

function createBasePipelineMapping(): PipelineMapping {
  return { $type: "mrc.protos.PipelineMapping", machineId: "0", segments: {} };
}

export const PipelineMapping = {
  $type: "mrc.protos.PipelineMapping" as const,

  encode(message: PipelineMapping, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.machineId !== "0") {
      writer.uint32(8).uint64(message.machineId);
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.machineId = longToString(reader.uint64() as Long);
          break;
        case 2:
          const entry2 = PipelineMapping_SegmentsEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.segments[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping {
    return {
      $type: PipelineMapping.$type,
      machineId: isSet(object.machineId) ? String(object.machineId) : "0",
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
    message.machineId !== undefined && (obj.machineId = message.machineId);
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
    message.machineId = object.machineId ?? "0";
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
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.segmentName = reader.string();
          break;
        case 2:
          message.byPolicy = PipelineMapping_SegmentMapping_ByPolicy.decode(reader, reader.uint32());
          break;
        case 3:
          message.byWorker = PipelineMapping_SegmentMapping_ByWorker.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping {
    return {
      $type: PipelineMapping_SegmentMapping.$type,
      segmentName: isSet(object.segmentName) ? String(object.segmentName) : "",
      byPolicy: isSet(object.byPolicy) ? PipelineMapping_SegmentMapping_ByPolicy.fromJSON(object.byPolicy) : undefined,
      byWorker: isSet(object.byWorker) ? PipelineMapping_SegmentMapping_ByWorker.fromJSON(object.byWorker) : undefined,
    };
  },

  toJSON(message: PipelineMapping_SegmentMapping): unknown {
    const obj: any = {};
    message.segmentName !== undefined && (obj.segmentName = message.segmentName);
    message.byPolicy !== undefined &&
      (obj.byPolicy = message.byPolicy ? PipelineMapping_SegmentMapping_ByPolicy.toJSON(message.byPolicy) : undefined);
    message.byWorker !== undefined &&
      (obj.byWorker = message.byWorker ? PipelineMapping_SegmentMapping_ByWorker.toJSON(message.byWorker) : undefined);
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
    return message;
  },
};

messageTypeRegistry.set(PipelineMapping_SegmentMapping.$type, PipelineMapping_SegmentMapping);

function createBasePipelineMapping_SegmentMapping_ByPolicy(): PipelineMapping_SegmentMapping_ByPolicy {
  return { $type: "mrc.protos.PipelineMapping.SegmentMapping.ByPolicy", value: 0 };
}

export const PipelineMapping_SegmentMapping_ByPolicy = {
  $type: "mrc.protos.PipelineMapping.SegmentMapping.ByPolicy" as const,

  encode(message: PipelineMapping_SegmentMapping_ByPolicy, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.value !== 0) {
      writer.uint32(8).int32(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineMapping_SegmentMapping_ByPolicy {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping_ByPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.value = reader.int32() as any;
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineMapping_SegmentMapping_ByPolicy {
    return {
      $type: PipelineMapping_SegmentMapping_ByPolicy.$type,
      value: isSet(object.value) ? segmentMappingPoliciesFromJSON(object.value) : 0,
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
    message.value = object.value ?? 0;
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentMapping_ByWorker();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.workerIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.workerIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineMapping_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.string();
          break;
        case 2:
          message.value = PipelineMapping_SegmentMapping.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.int64() as Long);
          break;
        case 2:
          message.config = PipelineConfiguration.decode(reader, reader.uint32());
          break;
        case 3:
          const entry3 = PipelineDefinition_MappingsEntry.decode(reader, reader.uint32());
          if (entry3.value !== undefined) {
            message.mappings[entry3.key] = entry3.value;
          }
          break;
        case 4:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.instanceIds.push(longToString(reader.uint64() as Long));
          }
          break;
        case 5:
          const entry5 = PipelineDefinition_SegmentsEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.segments[entry5.key] = entry5.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    writer.uint32(34).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineDefinition_SegmentDefinition {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.parentId = longToString(reader.uint64() as Long);
          break;
        case 3:
          message.name = reader.string();
          break;
        case 4:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.instanceIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineDefinition_SegmentDefinition {
    return {
      $type: PipelineDefinition_SegmentDefinition.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      parentId: isSet(object.parentId) ? String(object.parentId) : "0",
      name: isSet(object.name) ? String(object.name) : "",
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineDefinition_SegmentDefinition): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.parentId !== undefined && (obj.parentId = message.parentId);
    message.name !== undefined && (obj.name = message.name);
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
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineDefinition_SegmentDefinition.$type, PipelineDefinition_SegmentDefinition);

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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_MappingsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = PipelineMapping.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineDefinition_SegmentsEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.string();
          break;
        case 2:
          message.value = PipelineDefinition_SegmentDefinition.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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

function createBasePipelineInstance(): PipelineInstance {
  return {
    $type: "mrc.protos.PipelineInstance",
    id: "0",
    definitionId: "0",
    machineId: "0",
    state: undefined,
    segmentIds: [],
  };
}

export const PipelineInstance = {
  $type: "mrc.protos.PipelineInstance" as const,

  encode(message: PipelineInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.definitionId !== "0") {
      writer.uint32(16).int64(message.definitionId);
    }
    if (message.machineId !== "0") {
      writer.uint32(24).uint64(message.machineId);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(34).fork()).ldelim();
    }
    writer.uint32(42).fork();
    for (const v of message.segmentIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PipelineInstance {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipelineInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.definitionId = longToString(reader.int64() as Long);
          break;
        case 3:
          message.machineId = longToString(reader.uint64() as Long);
          break;
        case 4:
          message.state = ResourceState.decode(reader, reader.uint32());
          break;
        case 5:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.segmentIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.segmentIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PipelineInstance {
    return {
      $type: PipelineInstance.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      definitionId: isSet(object.definitionId) ? String(object.definitionId) : "0",
      machineId: isSet(object.machineId) ? String(object.machineId) : "0",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
      segmentIds: Array.isArray(object?.segmentIds) ? object.segmentIds.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: PipelineInstance): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.definitionId !== undefined && (obj.definitionId = message.definitionId);
    message.machineId !== undefined && (obj.machineId = message.machineId);
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    if (message.segmentIds) {
      obj.segmentIds = message.segmentIds.map((e) => e);
    } else {
      obj.segmentIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineInstance>): PipelineInstance {
    return PipelineInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineInstance>): PipelineInstance {
    const message = createBasePipelineInstance();
    message.id = object.id ?? "0";
    message.definitionId = object.definitionId ?? "0";
    message.machineId = object.machineId ?? "0";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    message.segmentIds = object.segmentIds?.map((e) => e) || [];
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentDefinition();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 3:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.instanceIds.push(longToString(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    pipelineDefinitionId: "0",
    name: "",
    address: 0,
    workerId: "0",
    pipelineInstanceId: "0",
    state: undefined,
  };
}

export const SegmentInstance = {
  $type: "mrc.protos.SegmentInstance" as const,

  encode(message: SegmentInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== "0") {
      writer.uint32(8).uint64(message.id);
    }
    if (message.pipelineDefinitionId !== "0") {
      writer.uint32(16).int64(message.pipelineDefinitionId);
    }
    if (message.name !== "") {
      writer.uint32(26).string(message.name);
    }
    if (message.address !== 0) {
      writer.uint32(32).uint32(message.address);
    }
    if (message.workerId !== "0") {
      writer.uint32(40).uint64(message.workerId);
    }
    if (message.pipelineInstanceId !== "0") {
      writer.uint32(48).uint64(message.pipelineInstanceId);
    }
    if (message.state !== undefined) {
      ResourceState.encode(message.state, writer.uint32(58).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentInstance {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.id = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.pipelineDefinitionId = longToString(reader.int64() as Long);
          break;
        case 3:
          message.name = reader.string();
          break;
        case 4:
          message.address = reader.uint32();
          break;
        case 5:
          message.workerId = longToString(reader.uint64() as Long);
          break;
        case 6:
          message.pipelineInstanceId = longToString(reader.uint64() as Long);
          break;
        case 7:
          message.state = ResourceState.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentInstance {
    return {
      $type: SegmentInstance.$type,
      id: isSet(object.id) ? String(object.id) : "0",
      pipelineDefinitionId: isSet(object.pipelineDefinitionId) ? String(object.pipelineDefinitionId) : "0",
      name: isSet(object.name) ? String(object.name) : "",
      address: isSet(object.address) ? Number(object.address) : 0,
      workerId: isSet(object.workerId) ? String(object.workerId) : "0",
      pipelineInstanceId: isSet(object.pipelineInstanceId) ? String(object.pipelineInstanceId) : "0",
      state: isSet(object.state) ? ResourceState.fromJSON(object.state) : undefined,
    };
  },

  toJSON(message: SegmentInstance): unknown {
    const obj: any = {};
    message.id !== undefined && (obj.id = message.id);
    message.pipelineDefinitionId !== undefined && (obj.pipelineDefinitionId = message.pipelineDefinitionId);
    message.name !== undefined && (obj.name = message.name);
    message.address !== undefined && (obj.address = Math.round(message.address));
    message.workerId !== undefined && (obj.workerId = message.workerId);
    message.pipelineInstanceId !== undefined && (obj.pipelineInstanceId = message.pipelineInstanceId);
    message.state !== undefined && (obj.state = message.state ? ResourceState.toJSON(message.state) : undefined);
    return obj;
  },

  create(base?: DeepPartial<SegmentInstance>): SegmentInstance {
    return SegmentInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentInstance>): SegmentInstance {
    const message = createBaseSegmentInstance();
    message.id = object.id ?? "0";
    message.pipelineDefinitionId = object.pipelineDefinitionId ?? "0";
    message.name = object.name ?? "";
    message.address = object.address ?? 0;
    message.workerId = object.workerId ?? "0";
    message.pipelineInstanceId = object.pipelineInstanceId ?? "0";
    message.state = (object.state !== undefined && object.state !== null)
      ? ResourceState.fromPartial(object.state)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentInstance.$type, SegmentInstance);

function createBaseControlPlaneState(): ControlPlaneState {
  return {
    $type: "mrc.protos.ControlPlaneState",
    connections: undefined,
    workers: undefined,
    pipelineDefinitions: undefined,
    pipelineInstances: undefined,
    segmentDefinitions: undefined,
    segmentInstances: undefined,
  };
}

export const ControlPlaneState = {
  $type: "mrc.protos.ControlPlaneState" as const,

  encode(message: ControlPlaneState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.connections !== undefined) {
      ControlPlaneState_ConnectionsState.encode(message.connections, writer.uint32(10).fork()).ldelim();
    }
    if (message.workers !== undefined) {
      ControlPlaneState_WorkerssState.encode(message.workers, writer.uint32(18).fork()).ldelim();
    }
    if (message.pipelineDefinitions !== undefined) {
      ControlPlaneState_PipelineDefinitionsState.encode(message.pipelineDefinitions, writer.uint32(26).fork()).ldelim();
    }
    if (message.pipelineInstances !== undefined) {
      ControlPlaneState_PipelineInstancesState.encode(message.pipelineInstances, writer.uint32(34).fork()).ldelim();
    }
    if (message.segmentDefinitions !== undefined) {
      ControlPlaneState_SegmentDefinitionsState.encode(message.segmentDefinitions, writer.uint32(42).fork()).ldelim();
    }
    if (message.segmentInstances !== undefined) {
      ControlPlaneState_SegmentInstancesState.encode(message.segmentInstances, writer.uint32(50).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.connections = ControlPlaneState_ConnectionsState.decode(reader, reader.uint32());
          break;
        case 2:
          message.workers = ControlPlaneState_WorkerssState.decode(reader, reader.uint32());
          break;
        case 3:
          message.pipelineDefinitions = ControlPlaneState_PipelineDefinitionsState.decode(reader, reader.uint32());
          break;
        case 4:
          message.pipelineInstances = ControlPlaneState_PipelineInstancesState.decode(reader, reader.uint32());
          break;
        case 5:
          message.segmentDefinitions = ControlPlaneState_SegmentDefinitionsState.decode(reader, reader.uint32());
          break;
        case 6:
          message.segmentInstances = ControlPlaneState_SegmentInstancesState.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState {
    return {
      $type: ControlPlaneState.$type,
      connections: isSet(object.connections)
        ? ControlPlaneState_ConnectionsState.fromJSON(object.connections)
        : undefined,
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
    };
  },

  toJSON(message: ControlPlaneState): unknown {
    const obj: any = {};
    message.connections !== undefined && (obj.connections = message.connections
      ? ControlPlaneState_ConnectionsState.toJSON(message.connections)
      : undefined);
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
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState>): ControlPlaneState {
    return ControlPlaneState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState>): ControlPlaneState {
    const message = createBaseControlPlaneState();
    message.connections = (object.connections !== undefined && object.connections !== null)
      ? ControlPlaneState_ConnectionsState.fromPartial(object.connections)
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
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState.$type, ControlPlaneState);

function createBaseControlPlaneState_ConnectionsState(): ControlPlaneState_ConnectionsState {
  return { $type: "mrc.protos.ControlPlaneState.ConnectionsState", ids: [], entities: {} };
}

export const ControlPlaneState_ConnectionsState = {
  $type: "mrc.protos.ControlPlaneState.ConnectionsState" as const,

  encode(message: ControlPlaneState_ConnectionsState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.ids) {
      writer.uint64(v);
    }
    writer.ldelim();
    Object.entries(message.entities).forEach(([key, value]) => {
      ControlPlaneState_ConnectionsState_EntitiesEntry.encode({
        $type: "mrc.protos.ControlPlaneState.ConnectionsState.EntitiesEntry",
        key: key as any,
        value,
      }, writer.uint32(18).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ConnectionsState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ConnectionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_ConnectionsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ConnectionsState {
    return {
      $type: ControlPlaneState_ConnectionsState.$type,
      ids: Array.isArray(object?.ids) ? object.ids.map((e: any) => String(e)) : [],
      entities: isObject(object.entities)
        ? Object.entries(object.entities).reduce<{ [key: string]: Connection }>((acc, [key, value]) => {
          acc[key] = Connection.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: ControlPlaneState_ConnectionsState): unknown {
    const obj: any = {};
    if (message.ids) {
      obj.ids = message.ids.map((e) => e);
    } else {
      obj.ids = [];
    }
    obj.entities = {};
    if (message.entities) {
      Object.entries(message.entities).forEach(([k, v]) => {
        obj.entities[k] = Connection.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<ControlPlaneState_ConnectionsState>): ControlPlaneState_ConnectionsState {
    return ControlPlaneState_ConnectionsState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ControlPlaneState_ConnectionsState>): ControlPlaneState_ConnectionsState {
    const message = createBaseControlPlaneState_ConnectionsState();
    message.ids = object.ids?.map((e) => e) || [];
    message.entities = Object.entries(object.entities ?? {}).reduce<{ [key: string]: Connection }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = Connection.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(ControlPlaneState_ConnectionsState.$type, ControlPlaneState_ConnectionsState);

function createBaseControlPlaneState_ConnectionsState_EntitiesEntry(): ControlPlaneState_ConnectionsState_EntitiesEntry {
  return { $type: "mrc.protos.ControlPlaneState.ConnectionsState.EntitiesEntry", key: "0", value: undefined };
}

export const ControlPlaneState_ConnectionsState_EntitiesEntry = {
  $type: "mrc.protos.ControlPlaneState.ConnectionsState.EntitiesEntry" as const,

  encode(
    message: ControlPlaneState_ConnectionsState_EntitiesEntry,
    writer: _m0.Writer = _m0.Writer.create(),
  ): _m0.Writer {
    if (message.key !== "0") {
      writer.uint32(8).uint64(message.key);
    }
    if (message.value !== undefined) {
      Connection.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlPlaneState_ConnectionsState_EntitiesEntry {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_ConnectionsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = Connection.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ControlPlaneState_ConnectionsState_EntitiesEntry {
    return {
      $type: ControlPlaneState_ConnectionsState_EntitiesEntry.$type,
      key: isSet(object.key) ? String(object.key) : "0",
      value: isSet(object.value) ? Connection.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: ControlPlaneState_ConnectionsState_EntitiesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = message.key);
    message.value !== undefined && (obj.value = message.value ? Connection.toJSON(message.value) : undefined);
    return obj;
  },

  create(
    base?: DeepPartial<ControlPlaneState_ConnectionsState_EntitiesEntry>,
  ): ControlPlaneState_ConnectionsState_EntitiesEntry {
    return ControlPlaneState_ConnectionsState_EntitiesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<ControlPlaneState_ConnectionsState_EntitiesEntry>,
  ): ControlPlaneState_ConnectionsState_EntitiesEntry {
    const message = createBaseControlPlaneState_ConnectionsState_EntitiesEntry();
    message.key = object.key ?? "0";
    message.value = (object.value !== undefined && object.value !== null)
      ? Connection.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(
  ControlPlaneState_ConnectionsState_EntitiesEntry.$type,
  ControlPlaneState_ConnectionsState_EntitiesEntry,
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_WorkerssState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_WorkerssState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_WorkerssState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = Worker.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineDefinitionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_PipelineDefinitionsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineDefinitionsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = PipelineDefinition.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineInstancesState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_PipelineInstancesState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_PipelineInstancesState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = PipelineInstance.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentDefinitionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_SegmentDefinitionsState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentDefinitionsState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = SegmentDefinition.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentInstancesState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.ids.push(longToString(reader.uint64() as Long));
            }
          } else {
            message.ids.push(longToString(reader.uint64() as Long));
          }
          break;
        case 2:
          const entry2 = ControlPlaneState_SegmentInstancesState_EntitiesEntry.decode(reader, reader.uint32());
          if (entry2.value !== undefined) {
            message.entities[entry2.key] = entry2.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlPlaneState_SegmentInstancesState_EntitiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = longToString(reader.uint64() as Long);
          break;
        case 2:
          message.value = SegmentInstance.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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

function createBaseSegmentOptions(): SegmentOptions {
  return { $type: "mrc.protos.SegmentOptions", placementStrategy: 0, scalingOptions: undefined };
}

export const SegmentOptions = {
  $type: "mrc.protos.SegmentOptions" as const,

  encode(message: SegmentOptions, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.placementStrategy !== 0) {
      writer.uint32(8).int32(message.placementStrategy);
    }
    if (message.scalingOptions !== undefined) {
      ScalingOptions.encode(message.scalingOptions, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentOptions {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentOptions();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.placementStrategy = reader.int32() as any;
          break;
        case 2:
          message.scalingOptions = ScalingOptions.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentOptions {
    return {
      $type: SegmentOptions.$type,
      placementStrategy: isSet(object.placementStrategy)
        ? segmentOptions_PlacementStrategyFromJSON(object.placementStrategy)
        : 0,
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
    message.placementStrategy = object.placementStrategy ?? 0;
    message.scalingOptions = (object.scalingOptions !== undefined && object.scalingOptions !== null)
      ? ScalingOptions.fromPartial(object.scalingOptions)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentOptions.$type, SegmentOptions);

function createBaseScalingOptions(): ScalingOptions {
  return { $type: "mrc.protos.ScalingOptions", strategy: 0, initialCount: 0 };
}

export const ScalingOptions = {
  $type: "mrc.protos.ScalingOptions" as const,

  encode(message: ScalingOptions, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.strategy !== 0) {
      writer.uint32(8).int32(message.strategy);
    }
    if (message.initialCount !== 0) {
      writer.uint32(16).uint32(message.initialCount);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ScalingOptions {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseScalingOptions();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.strategy = reader.int32() as any;
          break;
        case 2:
          message.initialCount = reader.uint32();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ScalingOptions {
    return {
      $type: ScalingOptions.$type,
      strategy: isSet(object.strategy) ? scalingOptions_ScalingStrategyFromJSON(object.strategy) : 0,
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
    message.strategy = object.strategy ?? 0;
    message.initialCount = object.initialCount ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ScalingOptions.$type, ScalingOptions);

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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseIngressPort();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.id = reader.uint32();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
  return { $type: "mrc.protos.EgressPort", name: "", id: 0, policyType: 0 };
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
    if (message.policyType !== 0) {
      writer.uint32(24).int32(message.policyType);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): EgressPort {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseEgressPort();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.id = reader.uint32();
          break;
        case 3:
          message.policyType = reader.int32() as any;
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): EgressPort {
    return {
      $type: EgressPort.$type,
      name: isSet(object.name) ? String(object.name) : "",
      id: isSet(object.id) ? Number(object.id) : 0,
      policyType: isSet(object.policyType) ? egressPort_PolicyTypeFromJSON(object.policyType) : 0,
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
    message.policyType = object.policyType ?? 0;
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
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseIngressPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.networkEnabled = reader.bool();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
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
  return { $type: "mrc.protos.EgressPolicy", policy: 0, segmentAddresses: [] };
}

export const EgressPolicy = {
  $type: "mrc.protos.EgressPolicy" as const,

  encode(message: EgressPolicy, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.policy !== 0) {
      writer.uint32(24).int32(message.policy);
    }
    writer.uint32(34).fork();
    for (const v of message.segmentAddresses) {
      writer.uint32(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): EgressPolicy {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseEgressPolicy();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 3:
          message.policy = reader.int32() as any;
          break;
        case 4:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.segmentAddresses.push(reader.uint32());
            }
          } else {
            message.segmentAddresses.push(reader.uint32());
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): EgressPolicy {
    return {
      $type: EgressPolicy.$type,
      policy: isSet(object.policy) ? egressPolicy_PolicyFromJSON(object.policy) : 0,
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
    message.policy = object.policy ?? 0;
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
