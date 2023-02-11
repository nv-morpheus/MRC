/* eslint-disable */
import Long from "long";
import type { CallContext, CallOptions } from "nice-grpc-common";
import _m0 from "protobufjs/minimal";
import { Any } from "../../google/protobuf/any";
import { messageTypeRegistry } from "../../typeRegistry";

export enum EventType {
  Unused = 0,
  Response = 1,
  ControlStop = 2,
  /** ClientEventRequestStateUpdate - Client Events - No Response */
  ClientEventRequestStateUpdate = 100,
  /** ClientUnaryRegisterWorkers - Connection Management */
  ClientUnaryRegisterWorkers = 201,
  ClientUnaryActivateStream = 202,
  ClientUnaryLookupWorkerAddresses = 203,
  ClientUnaryDropWorker = 204,
  /** ClientUnaryCreateSubscriptionService - SubscriptionService */
  ClientUnaryCreateSubscriptionService = 301,
  ClientUnaryRegisterSubscriptionService = 302,
  ClientUnaryActivateSubscriptionService = 303,
  ClientUnaryDropSubscriptionService = 304,
  ClientEventUpdateSubscriptionService = 305,
  /** ServerEvent - Server Event issues to Client(s) */
  ServerEvent = 1000,
  ServerStateUpdate = 1001,
  UNRECOGNIZED = -1,
}

export function eventTypeFromJSON(object: any): EventType {
  switch (object) {
    case 0:
    case "Unused":
      return EventType.Unused;
    case 1:
    case "Response":
      return EventType.Response;
    case 2:
    case "ControlStop":
      return EventType.ControlStop;
    case 100:
    case "ClientEventRequestStateUpdate":
      return EventType.ClientEventRequestStateUpdate;
    case 201:
    case "ClientUnaryRegisterWorkers":
      return EventType.ClientUnaryRegisterWorkers;
    case 202:
    case "ClientUnaryActivateStream":
      return EventType.ClientUnaryActivateStream;
    case 203:
    case "ClientUnaryLookupWorkerAddresses":
      return EventType.ClientUnaryLookupWorkerAddresses;
    case 204:
    case "ClientUnaryDropWorker":
      return EventType.ClientUnaryDropWorker;
    case 301:
    case "ClientUnaryCreateSubscriptionService":
      return EventType.ClientUnaryCreateSubscriptionService;
    case 302:
    case "ClientUnaryRegisterSubscriptionService":
      return EventType.ClientUnaryRegisterSubscriptionService;
    case 303:
    case "ClientUnaryActivateSubscriptionService":
      return EventType.ClientUnaryActivateSubscriptionService;
    case 304:
    case "ClientUnaryDropSubscriptionService":
      return EventType.ClientUnaryDropSubscriptionService;
    case 305:
    case "ClientEventUpdateSubscriptionService":
      return EventType.ClientEventUpdateSubscriptionService;
    case 1000:
    case "ServerEvent":
      return EventType.ServerEvent;
    case 1001:
    case "ServerStateUpdate":
      return EventType.ServerStateUpdate;
    case -1:
    case "UNRECOGNIZED":
    default:
      return EventType.UNRECOGNIZED;
  }
}

export function eventTypeToJSON(object: EventType): string {
  switch (object) {
    case EventType.Unused:
      return "Unused";
    case EventType.Response:
      return "Response";
    case EventType.ControlStop:
      return "ControlStop";
    case EventType.ClientEventRequestStateUpdate:
      return "ClientEventRequestStateUpdate";
    case EventType.ClientUnaryRegisterWorkers:
      return "ClientUnaryRegisterWorkers";
    case EventType.ClientUnaryActivateStream:
      return "ClientUnaryActivateStream";
    case EventType.ClientUnaryLookupWorkerAddresses:
      return "ClientUnaryLookupWorkerAddresses";
    case EventType.ClientUnaryDropWorker:
      return "ClientUnaryDropWorker";
    case EventType.ClientUnaryCreateSubscriptionService:
      return "ClientUnaryCreateSubscriptionService";
    case EventType.ClientUnaryRegisterSubscriptionService:
      return "ClientUnaryRegisterSubscriptionService";
    case EventType.ClientUnaryActivateSubscriptionService:
      return "ClientUnaryActivateSubscriptionService";
    case EventType.ClientUnaryDropSubscriptionService:
      return "ClientUnaryDropSubscriptionService";
    case EventType.ClientEventUpdateSubscriptionService:
      return "ClientEventUpdateSubscriptionService";
    case EventType.ServerEvent:
      return "ServerEvent";
    case EventType.ServerStateUpdate:
      return "ServerStateUpdate";
    case EventType.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export enum ErrorCode {
  Success = 0,
  ServerError = 1,
  ClientError = 2,
  InstanceError = 3,
  UNRECOGNIZED = -1,
}

export function errorCodeFromJSON(object: any): ErrorCode {
  switch (object) {
    case 0:
    case "Success":
      return ErrorCode.Success;
    case 1:
    case "ServerError":
      return ErrorCode.ServerError;
    case 2:
    case "ClientError":
      return ErrorCode.ClientError;
    case 3:
    case "InstanceError":
      return ErrorCode.InstanceError;
    case -1:
    case "UNRECOGNIZED":
    default:
      return ErrorCode.UNRECOGNIZED;
  }
}

export function errorCodeToJSON(object: ErrorCode): string {
  switch (object) {
    case ErrorCode.Success:
      return "Success";
    case ErrorCode.ServerError:
      return "ServerError";
    case ErrorCode.ClientError:
      return "ClientError";
    case ErrorCode.InstanceError:
      return "InstanceError";
    case ErrorCode.UNRECOGNIZED:
    default:
      return "UNRECOGNIZED";
  }
}

export interface PingRequest {
  $type: "mrc.protos.PingRequest";
  tag: number;
}

export interface PingResponse {
  $type: "mrc.protos.PingResponse";
  tag: number;
}

export interface ShutdownRequest {
  $type: "mrc.protos.ShutdownRequest";
  tag: number;
}

export interface ShutdownResponse {
  $type: "mrc.protos.ShutdownResponse";
  tag: number;
}

export interface Event {
  $type: "mrc.protos.Event";
  event: EventType;
  tag: number;
  message?: Any | undefined;
  error?: Error | undefined;
}

export interface Error {
  $type: "mrc.protos.Error";
  code: ErrorCode;
  message: string;
}

export interface Ack {
  $type: "mrc.protos.Ack";
}

export interface RegisterWorkersRequest {
  $type: "mrc.protos.RegisterWorkersRequest";
  ucxWorkerAddresses: Uint8Array[];
  pipeline?: Pipeline;
}

export interface RegisterWorkersResponse {
  $type: "mrc.protos.RegisterWorkersResponse";
  machineId: number;
  instanceIds: number[];
}

export interface RegisterPipelineRequest {
  $type: "mrc.protos.RegisterPipelineRequest";
  /** uint32 machine_id = 1; */
  pipeline?: Pipeline;
  requestedConfig: PipelineConfiguration[];
}

export interface RegisterPipelineResponse {
  $type: "mrc.protos.RegisterPipelineResponse";
}

export interface LookupWorkersRequest {
  $type: "mrc.protos.LookupWorkersRequest";
  instanceIds: number[];
}

export interface LookupWorkersResponse {
  $type: "mrc.protos.LookupWorkersResponse";
  workerAddresses: WorkerAddress[];
}

export interface CreateSubscriptionServiceRequest {
  $type: "mrc.protos.CreateSubscriptionServiceRequest";
  serviceName: string;
  roles: string[];
}

export interface RegisterSubscriptionServiceRequest {
  $type: "mrc.protos.RegisterSubscriptionServiceRequest";
  serviceName: string;
  role: string;
  subscribeToRoles: string[];
  instanceId: number;
}

export interface RegisterSubscriptionServiceResponse {
  $type: "mrc.protos.RegisterSubscriptionServiceResponse";
  serviceName: string;
  role: string;
  tag: number;
}

export interface ActivateSubscriptionServiceRequest {
  $type: "mrc.protos.ActivateSubscriptionServiceRequest";
  serviceName: string;
  role: string;
  subscribeToRoles: string[];
  instanceId: number;
  tag: number;
}

export interface DropSubscriptionServiceRequest {
  $type: "mrc.protos.DropSubscriptionServiceRequest";
  serviceName: string;
  instanceId: number;
  tag: number;
}

export interface UpdateSubscriptionServiceRequest {
  $type: "mrc.protos.UpdateSubscriptionServiceRequest";
  serviceName: string;
  role: string;
  nonce: number;
  tags: number[];
}

export interface TaggedInstance {
  $type: "mrc.protos.TaggedInstance";
  instanceId: number;
  tag: number;
}

/** message sent by an UpdateManager */
export interface StateUpdate {
  $type: "mrc.protos.StateUpdate";
  serviceName: string;
  nonce: number;
  instanceId: number;
  connections?: UpdateConnectionsState | undefined;
  updateSubscriptionService?: UpdateSubscriptionServiceState | undefined;
  dropSubscriptionService?: DropSubscriptionServiceState | undefined;
}

export interface UpdateConnectionsState {
  $type: "mrc.protos.UpdateConnectionsState";
  taggedInstances: TaggedInstance[];
}

export interface UpdateSubscriptionServiceState {
  $type: "mrc.protos.UpdateSubscriptionServiceState";
  role: string;
  taggedInstances: TaggedInstance[];
}

export interface DropSubscriptionServiceState {
  $type: "mrc.protos.DropSubscriptionServiceState";
  role: string;
  tag: number;
}

export interface ControlMessage {
  $type: "mrc.protos.ControlMessage";
}

export interface OnComplete {
  $type: "mrc.protos.OnComplete";
  segmentAddresses: number[];
}

export interface UpdateAssignments {
  $type: "mrc.protos.UpdateAssignments";
  assignments: SegmentAssignment[];
}

export interface SegmentAssignment {
  $type: "mrc.protos.SegmentAssignment";
  machineId: number;
  instanceId: number;
  address: number;
  egressPolices: { [key: number]: EgressPolicy };
  issueEventOnComplete: boolean;
  networkIngressPorts: number[];
}

export interface SegmentAssignment_EgressPolicesEntry {
  $type: "mrc.protos.SegmentAssignment.EgressPolicesEntry";
  key: number;
  value?: EgressPolicy;
}

export interface Topology {
  $type: "mrc.protos.Topology";
  hwlocXmlString: string;
  cpuSet: string;
  gpuInfo: GpuInfo[];
}

export interface GpuInfo {
  $type: "mrc.protos.GpuInfo";
  cpuSet: string;
  name: string;
  uuid: string;
  pcieBusId: string;
  memoryCapacity: number;
  cudaDeviceId: number;
}

export interface Pipeline {
  $type: "mrc.protos.Pipeline";
  name: string;
  segments: SegmentDefinition[];
}

export interface SegmentDefinition {
  $type: "mrc.protos.SegmentDefinition";
  name: string;
  id: number;
  ingressPorts: IngressPort[];
  egressPorts: EgressPort[];
  options?: SegmentOptions;
}

export interface SegmentOptions {
  $type: "mrc.protos.SegmentOptions";
  placementStrategy: SegmentOptions_PlacementStrategy;
  scalingOptions?: ScalingOptions;
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

export interface PipelineConfiguration {
  $type: "mrc.protos.PipelineConfiguration";
  instanceId: number;
  segments: SegmentConfiguration[];
}

export interface SegmentConfiguration {
  $type: "mrc.protos.SegmentConfiguration";
  name: string;
  concurrency: number;
  rank: number;
  egressPolices: { [key: number]: EgressPolicy };
  ingressPolicies: { [key: number]: IngressPolicy };
}

export interface SegmentConfiguration_EgressPolicesEntry {
  $type: "mrc.protos.SegmentConfiguration.EgressPolicesEntry";
  key: number;
  value?: EgressPolicy;
}

export interface SegmentConfiguration_IngressPoliciesEntry {
  $type: "mrc.protos.SegmentConfiguration.IngressPoliciesEntry";
  key: number;
  value?: IngressPolicy;
}

export interface WorkerAddress {
  $type: "mrc.protos.WorkerAddress";
  machineId: number;
  instanceId: number;
  workerAddress: Uint8Array;
}

export interface InstancesResources {
  $type: "mrc.protos.InstancesResources";
  hostMemory: number;
  cpus: CPU[];
  gpus: GPU[];
  /**
   * todo - topology - assign cpu/numa_nodes, gpus and nics into optimized groups
   * use topology groups as the default unit of placement
   */
  nics: NIC[];
}

export interface CPU {
  $type: "mrc.protos.CPU";
  cores: number;
  /** numa_node_masks - which cores are assigned each numa_node */
  numaNodes: number;
}

export interface GPU {
  $type: "mrc.protos.GPU";
  name: string;
  cores: number;
  memory: number;
  computeCapability: number;
}

export interface NIC {
  $type: "mrc.protos.NIC";
}

function createBasePingRequest(): PingRequest {
  return { $type: "mrc.protos.PingRequest", tag: 0 };
}

export const PingRequest = {
  $type: "mrc.protos.PingRequest" as const,

  encode(message: PingRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.tag !== 0) {
      writer.uint32(8).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PingRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePingRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PingRequest {
    return { $type: PingRequest.$type, tag: isSet(object.tag) ? Number(object.tag) : 0 };
  },

  toJSON(message: PingRequest): unknown {
    const obj: any = {};
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<PingRequest>): PingRequest {
    return PingRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PingRequest>): PingRequest {
    const message = createBasePingRequest();
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(PingRequest.$type, PingRequest);

function createBasePingResponse(): PingResponse {
  return { $type: "mrc.protos.PingResponse", tag: 0 };
}

export const PingResponse = {
  $type: "mrc.protos.PingResponse" as const,

  encode(message: PingResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.tag !== 0) {
      writer.uint32(8).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PingResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePingResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): PingResponse {
    return { $type: PingResponse.$type, tag: isSet(object.tag) ? Number(object.tag) : 0 };
  },

  toJSON(message: PingResponse): unknown {
    const obj: any = {};
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<PingResponse>): PingResponse {
    return PingResponse.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PingResponse>): PingResponse {
    const message = createBasePingResponse();
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(PingResponse.$type, PingResponse);

function createBaseShutdownRequest(): ShutdownRequest {
  return { $type: "mrc.protos.ShutdownRequest", tag: 0 };
}

export const ShutdownRequest = {
  $type: "mrc.protos.ShutdownRequest" as const,

  encode(message: ShutdownRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.tag !== 0) {
      writer.uint32(8).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ShutdownRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseShutdownRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ShutdownRequest {
    return { $type: ShutdownRequest.$type, tag: isSet(object.tag) ? Number(object.tag) : 0 };
  },

  toJSON(message: ShutdownRequest): unknown {
    const obj: any = {};
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<ShutdownRequest>): ShutdownRequest {
    return ShutdownRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ShutdownRequest>): ShutdownRequest {
    const message = createBaseShutdownRequest();
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ShutdownRequest.$type, ShutdownRequest);

function createBaseShutdownResponse(): ShutdownResponse {
  return { $type: "mrc.protos.ShutdownResponse", tag: 0 };
}

export const ShutdownResponse = {
  $type: "mrc.protos.ShutdownResponse" as const,

  encode(message: ShutdownResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.tag !== 0) {
      writer.uint32(8).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ShutdownResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseShutdownResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ShutdownResponse {
    return { $type: ShutdownResponse.$type, tag: isSet(object.tag) ? Number(object.tag) : 0 };
  },

  toJSON(message: ShutdownResponse): unknown {
    const obj: any = {};
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<ShutdownResponse>): ShutdownResponse {
    return ShutdownResponse.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ShutdownResponse>): ShutdownResponse {
    const message = createBaseShutdownResponse();
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ShutdownResponse.$type, ShutdownResponse);

function createBaseEvent(): Event {
  return { $type: "mrc.protos.Event", event: 0, tag: 0, message: undefined, error: undefined };
}

export const Event = {
  $type: "mrc.protos.Event" as const,

  encode(message: Event, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.event !== 0) {
      writer.uint32(8).int32(message.event);
    }
    if (message.tag !== 0) {
      writer.uint32(16).uint64(message.tag);
    }
    if (message.message !== undefined) {
      Any.encode(message.message, writer.uint32(26).fork()).ldelim();
    }
    if (message.error !== undefined) {
      Error.encode(message.error, writer.uint32(34).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Event {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseEvent();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.event = reader.int32() as any;
          break;
        case 2:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        case 3:
          message.message = Any.decode(reader, reader.uint32());
          break;
        case 4:
          message.error = Error.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Event {
    return {
      $type: Event.$type,
      event: isSet(object.event) ? eventTypeFromJSON(object.event) : 0,
      tag: isSet(object.tag) ? Number(object.tag) : 0,
      message: isSet(object.message) ? Any.fromJSON(object.message) : undefined,
      error: isSet(object.error) ? Error.fromJSON(object.error) : undefined,
    };
  },

  toJSON(message: Event): unknown {
    const obj: any = {};
    message.event !== undefined && (obj.event = eventTypeToJSON(message.event));
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    message.message !== undefined && (obj.message = message.message ? Any.toJSON(message.message) : undefined);
    message.error !== undefined && (obj.error = message.error ? Error.toJSON(message.error) : undefined);
    return obj;
  },

  create(base?: DeepPartial<Event>): Event {
    return Event.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Event>): Event {
    const message = createBaseEvent();
    message.event = object.event ?? 0;
    message.tag = object.tag ?? 0;
    message.message = (object.message !== undefined && object.message !== null)
      ? Any.fromPartial(object.message)
      : undefined;
    message.error = (object.error !== undefined && object.error !== null) ? Error.fromPartial(object.error) : undefined;
    return message;
  },
};

messageTypeRegistry.set(Event.$type, Event);

function createBaseError(): Error {
  return { $type: "mrc.protos.Error", code: 0, message: "" };
}

export const Error = {
  $type: "mrc.protos.Error" as const,

  encode(message: Error, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.code !== 0) {
      writer.uint32(8).int32(message.code);
    }
    if (message.message !== "") {
      writer.uint32(18).string(message.message);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Error {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseError();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.code = reader.int32() as any;
          break;
        case 2:
          message.message = reader.string();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Error {
    return {
      $type: Error.$type,
      code: isSet(object.code) ? errorCodeFromJSON(object.code) : 0,
      message: isSet(object.message) ? String(object.message) : "",
    };
  },

  toJSON(message: Error): unknown {
    const obj: any = {};
    message.code !== undefined && (obj.code = errorCodeToJSON(message.code));
    message.message !== undefined && (obj.message = message.message);
    return obj;
  },

  create(base?: DeepPartial<Error>): Error {
    return Error.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Error>): Error {
    const message = createBaseError();
    message.code = object.code ?? 0;
    message.message = object.message ?? "";
    return message;
  },
};

messageTypeRegistry.set(Error.$type, Error);

function createBaseAck(): Ack {
  return { $type: "mrc.protos.Ack" };
}

export const Ack = {
  $type: "mrc.protos.Ack" as const,

  encode(_: Ack, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Ack {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseAck();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(_: any): Ack {
    return { $type: Ack.$type };
  },

  toJSON(_: Ack): unknown {
    const obj: any = {};
    return obj;
  },

  create(base?: DeepPartial<Ack>): Ack {
    return Ack.fromPartial(base ?? {});
  },

  fromPartial(_: DeepPartial<Ack>): Ack {
    const message = createBaseAck();
    return message;
  },
};

messageTypeRegistry.set(Ack.$type, Ack);

function createBaseRegisterWorkersRequest(): RegisterWorkersRequest {
  return { $type: "mrc.protos.RegisterWorkersRequest", ucxWorkerAddresses: [], pipeline: undefined };
}

export const RegisterWorkersRequest = {
  $type: "mrc.protos.RegisterWorkersRequest" as const,

  encode(message: RegisterWorkersRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.ucxWorkerAddresses) {
      writer.uint32(10).bytes(v!);
    }
    if (message.pipeline !== undefined) {
      Pipeline.encode(message.pipeline, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterWorkersRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterWorkersRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.ucxWorkerAddresses.push(reader.bytes());
          break;
        case 2:
          message.pipeline = Pipeline.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): RegisterWorkersRequest {
    return {
      $type: RegisterWorkersRequest.$type,
      ucxWorkerAddresses: Array.isArray(object?.ucxWorkerAddresses)
        ? object.ucxWorkerAddresses.map((e: any) => bytesFromBase64(e))
        : [],
      pipeline: isSet(object.pipeline) ? Pipeline.fromJSON(object.pipeline) : undefined,
    };
  },

  toJSON(message: RegisterWorkersRequest): unknown {
    const obj: any = {};
    if (message.ucxWorkerAddresses) {
      obj.ucxWorkerAddresses = message.ucxWorkerAddresses.map((e) =>
        base64FromBytes(e !== undefined ? e : new Uint8Array())
      );
    } else {
      obj.ucxWorkerAddresses = [];
    }
    message.pipeline !== undefined && (obj.pipeline = message.pipeline ? Pipeline.toJSON(message.pipeline) : undefined);
    return obj;
  },

  create(base?: DeepPartial<RegisterWorkersRequest>): RegisterWorkersRequest {
    return RegisterWorkersRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<RegisterWorkersRequest>): RegisterWorkersRequest {
    const message = createBaseRegisterWorkersRequest();
    message.ucxWorkerAddresses = object.ucxWorkerAddresses?.map((e) => e) || [];
    message.pipeline = (object.pipeline !== undefined && object.pipeline !== null)
      ? Pipeline.fromPartial(object.pipeline)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(RegisterWorkersRequest.$type, RegisterWorkersRequest);

function createBaseRegisterWorkersResponse(): RegisterWorkersResponse {
  return { $type: "mrc.protos.RegisterWorkersResponse", machineId: 0, instanceIds: [] };
}

export const RegisterWorkersResponse = {
  $type: "mrc.protos.RegisterWorkersResponse" as const,

  encode(message: RegisterWorkersResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.machineId !== 0) {
      writer.uint32(8).uint64(message.machineId);
    }
    writer.uint32(18).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterWorkersResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterWorkersResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.machineId = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToNumber(reader.uint64() as Long));
            }
          } else {
            message.instanceIds.push(longToNumber(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): RegisterWorkersResponse {
    return {
      $type: RegisterWorkersResponse.$type,
      machineId: isSet(object.machineId) ? Number(object.machineId) : 0,
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => Number(e)) : [],
    };
  },

  toJSON(message: RegisterWorkersResponse): unknown {
    const obj: any = {};
    message.machineId !== undefined && (obj.machineId = Math.round(message.machineId));
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => Math.round(e));
    } else {
      obj.instanceIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<RegisterWorkersResponse>): RegisterWorkersResponse {
    return RegisterWorkersResponse.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<RegisterWorkersResponse>): RegisterWorkersResponse {
    const message = createBaseRegisterWorkersResponse();
    message.machineId = object.machineId ?? 0;
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(RegisterWorkersResponse.$type, RegisterWorkersResponse);

function createBaseRegisterPipelineRequest(): RegisterPipelineRequest {
  return { $type: "mrc.protos.RegisterPipelineRequest", pipeline: undefined, requestedConfig: [] };
}

export const RegisterPipelineRequest = {
  $type: "mrc.protos.RegisterPipelineRequest" as const,

  encode(message: RegisterPipelineRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.pipeline !== undefined) {
      Pipeline.encode(message.pipeline, writer.uint32(18).fork()).ldelim();
    }
    for (const v of message.requestedConfig) {
      PipelineConfiguration.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterPipelineRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterPipelineRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 2:
          message.pipeline = Pipeline.decode(reader, reader.uint32());
          break;
        case 3:
          message.requestedConfig.push(PipelineConfiguration.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): RegisterPipelineRequest {
    return {
      $type: RegisterPipelineRequest.$type,
      pipeline: isSet(object.pipeline) ? Pipeline.fromJSON(object.pipeline) : undefined,
      requestedConfig: Array.isArray(object?.requestedConfig)
        ? object.requestedConfig.map((e: any) => PipelineConfiguration.fromJSON(e))
        : [],
    };
  },

  toJSON(message: RegisterPipelineRequest): unknown {
    const obj: any = {};
    message.pipeline !== undefined && (obj.pipeline = message.pipeline ? Pipeline.toJSON(message.pipeline) : undefined);
    if (message.requestedConfig) {
      obj.requestedConfig = message.requestedConfig.map((e) => e ? PipelineConfiguration.toJSON(e) : undefined);
    } else {
      obj.requestedConfig = [];
    }
    return obj;
  },

  create(base?: DeepPartial<RegisterPipelineRequest>): RegisterPipelineRequest {
    return RegisterPipelineRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<RegisterPipelineRequest>): RegisterPipelineRequest {
    const message = createBaseRegisterPipelineRequest();
    message.pipeline = (object.pipeline !== undefined && object.pipeline !== null)
      ? Pipeline.fromPartial(object.pipeline)
      : undefined;
    message.requestedConfig = object.requestedConfig?.map((e) => PipelineConfiguration.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(RegisterPipelineRequest.$type, RegisterPipelineRequest);

function createBaseRegisterPipelineResponse(): RegisterPipelineResponse {
  return { $type: "mrc.protos.RegisterPipelineResponse" };
}

export const RegisterPipelineResponse = {
  $type: "mrc.protos.RegisterPipelineResponse" as const,

  encode(_: RegisterPipelineResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterPipelineResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterPipelineResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(_: any): RegisterPipelineResponse {
    return { $type: RegisterPipelineResponse.$type };
  },

  toJSON(_: RegisterPipelineResponse): unknown {
    const obj: any = {};
    return obj;
  },

  create(base?: DeepPartial<RegisterPipelineResponse>): RegisterPipelineResponse {
    return RegisterPipelineResponse.fromPartial(base ?? {});
  },

  fromPartial(_: DeepPartial<RegisterPipelineResponse>): RegisterPipelineResponse {
    const message = createBaseRegisterPipelineResponse();
    return message;
  },
};

messageTypeRegistry.set(RegisterPipelineResponse.$type, RegisterPipelineResponse);

function createBaseLookupWorkersRequest(): LookupWorkersRequest {
  return { $type: "mrc.protos.LookupWorkersRequest", instanceIds: [] };
}

export const LookupWorkersRequest = {
  $type: "mrc.protos.LookupWorkersRequest" as const,

  encode(message: LookupWorkersRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.instanceIds) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): LookupWorkersRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseLookupWorkersRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.instanceIds.push(longToNumber(reader.uint64() as Long));
            }
          } else {
            message.instanceIds.push(longToNumber(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): LookupWorkersRequest {
    return {
      $type: LookupWorkersRequest.$type,
      instanceIds: Array.isArray(object?.instanceIds) ? object.instanceIds.map((e: any) => Number(e)) : [],
    };
  },

  toJSON(message: LookupWorkersRequest): unknown {
    const obj: any = {};
    if (message.instanceIds) {
      obj.instanceIds = message.instanceIds.map((e) => Math.round(e));
    } else {
      obj.instanceIds = [];
    }
    return obj;
  },

  create(base?: DeepPartial<LookupWorkersRequest>): LookupWorkersRequest {
    return LookupWorkersRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<LookupWorkersRequest>): LookupWorkersRequest {
    const message = createBaseLookupWorkersRequest();
    message.instanceIds = object.instanceIds?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(LookupWorkersRequest.$type, LookupWorkersRequest);

function createBaseLookupWorkersResponse(): LookupWorkersResponse {
  return { $type: "mrc.protos.LookupWorkersResponse", workerAddresses: [] };
}

export const LookupWorkersResponse = {
  $type: "mrc.protos.LookupWorkersResponse" as const,

  encode(message: LookupWorkersResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.workerAddresses) {
      WorkerAddress.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): LookupWorkersResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseLookupWorkersResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 2:
          message.workerAddresses.push(WorkerAddress.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): LookupWorkersResponse {
    return {
      $type: LookupWorkersResponse.$type,
      workerAddresses: Array.isArray(object?.workerAddresses)
        ? object.workerAddresses.map((e: any) => WorkerAddress.fromJSON(e))
        : [],
    };
  },

  toJSON(message: LookupWorkersResponse): unknown {
    const obj: any = {};
    if (message.workerAddresses) {
      obj.workerAddresses = message.workerAddresses.map((e) => e ? WorkerAddress.toJSON(e) : undefined);
    } else {
      obj.workerAddresses = [];
    }
    return obj;
  },

  create(base?: DeepPartial<LookupWorkersResponse>): LookupWorkersResponse {
    return LookupWorkersResponse.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<LookupWorkersResponse>): LookupWorkersResponse {
    const message = createBaseLookupWorkersResponse();
    message.workerAddresses = object.workerAddresses?.map((e) => WorkerAddress.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(LookupWorkersResponse.$type, LookupWorkersResponse);

function createBaseCreateSubscriptionServiceRequest(): CreateSubscriptionServiceRequest {
  return { $type: "mrc.protos.CreateSubscriptionServiceRequest", serviceName: "", roles: [] };
}

export const CreateSubscriptionServiceRequest = {
  $type: "mrc.protos.CreateSubscriptionServiceRequest" as const,

  encode(message: CreateSubscriptionServiceRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    for (const v of message.roles) {
      writer.uint32(18).string(v!);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): CreateSubscriptionServiceRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseCreateSubscriptionServiceRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.roles.push(reader.string());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): CreateSubscriptionServiceRequest {
    return {
      $type: CreateSubscriptionServiceRequest.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      roles: Array.isArray(object?.roles) ? object.roles.map((e: any) => String(e)) : [],
    };
  },

  toJSON(message: CreateSubscriptionServiceRequest): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    if (message.roles) {
      obj.roles = message.roles.map((e) => e);
    } else {
      obj.roles = [];
    }
    return obj;
  },

  create(base?: DeepPartial<CreateSubscriptionServiceRequest>): CreateSubscriptionServiceRequest {
    return CreateSubscriptionServiceRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<CreateSubscriptionServiceRequest>): CreateSubscriptionServiceRequest {
    const message = createBaseCreateSubscriptionServiceRequest();
    message.serviceName = object.serviceName ?? "";
    message.roles = object.roles?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(CreateSubscriptionServiceRequest.$type, CreateSubscriptionServiceRequest);

function createBaseRegisterSubscriptionServiceRequest(): RegisterSubscriptionServiceRequest {
  return {
    $type: "mrc.protos.RegisterSubscriptionServiceRequest",
    serviceName: "",
    role: "",
    subscribeToRoles: [],
    instanceId: 0,
  };
}

export const RegisterSubscriptionServiceRequest = {
  $type: "mrc.protos.RegisterSubscriptionServiceRequest" as const,

  encode(message: RegisterSubscriptionServiceRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.role !== "") {
      writer.uint32(18).string(message.role);
    }
    for (const v of message.subscribeToRoles) {
      writer.uint32(26).string(v!);
    }
    if (message.instanceId !== 0) {
      writer.uint32(32).uint64(message.instanceId);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterSubscriptionServiceRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterSubscriptionServiceRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.role = reader.string();
          break;
        case 3:
          message.subscribeToRoles.push(reader.string());
          break;
        case 4:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): RegisterSubscriptionServiceRequest {
    return {
      $type: RegisterSubscriptionServiceRequest.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      role: isSet(object.role) ? String(object.role) : "",
      subscribeToRoles: Array.isArray(object?.subscribeToRoles)
        ? object.subscribeToRoles.map((e: any) => String(e))
        : [],
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
    };
  },

  toJSON(message: RegisterSubscriptionServiceRequest): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.role !== undefined && (obj.role = message.role);
    if (message.subscribeToRoles) {
      obj.subscribeToRoles = message.subscribeToRoles.map((e) => e);
    } else {
      obj.subscribeToRoles = [];
    }
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    return obj;
  },

  create(base?: DeepPartial<RegisterSubscriptionServiceRequest>): RegisterSubscriptionServiceRequest {
    return RegisterSubscriptionServiceRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<RegisterSubscriptionServiceRequest>): RegisterSubscriptionServiceRequest {
    const message = createBaseRegisterSubscriptionServiceRequest();
    message.serviceName = object.serviceName ?? "";
    message.role = object.role ?? "";
    message.subscribeToRoles = object.subscribeToRoles?.map((e) => e) || [];
    message.instanceId = object.instanceId ?? 0;
    return message;
  },
};

messageTypeRegistry.set(RegisterSubscriptionServiceRequest.$type, RegisterSubscriptionServiceRequest);

function createBaseRegisterSubscriptionServiceResponse(): RegisterSubscriptionServiceResponse {
  return { $type: "mrc.protos.RegisterSubscriptionServiceResponse", serviceName: "", role: "", tag: 0 };
}

export const RegisterSubscriptionServiceResponse = {
  $type: "mrc.protos.RegisterSubscriptionServiceResponse" as const,

  encode(message: RegisterSubscriptionServiceResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.role !== "") {
      writer.uint32(18).string(message.role);
    }
    if (message.tag !== 0) {
      writer.uint32(24).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RegisterSubscriptionServiceResponse {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRegisterSubscriptionServiceResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.role = reader.string();
          break;
        case 3:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): RegisterSubscriptionServiceResponse {
    return {
      $type: RegisterSubscriptionServiceResponse.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      role: isSet(object.role) ? String(object.role) : "",
      tag: isSet(object.tag) ? Number(object.tag) : 0,
    };
  },

  toJSON(message: RegisterSubscriptionServiceResponse): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.role !== undefined && (obj.role = message.role);
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<RegisterSubscriptionServiceResponse>): RegisterSubscriptionServiceResponse {
    return RegisterSubscriptionServiceResponse.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<RegisterSubscriptionServiceResponse>): RegisterSubscriptionServiceResponse {
    const message = createBaseRegisterSubscriptionServiceResponse();
    message.serviceName = object.serviceName ?? "";
    message.role = object.role ?? "";
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(RegisterSubscriptionServiceResponse.$type, RegisterSubscriptionServiceResponse);

function createBaseActivateSubscriptionServiceRequest(): ActivateSubscriptionServiceRequest {
  return {
    $type: "mrc.protos.ActivateSubscriptionServiceRequest",
    serviceName: "",
    role: "",
    subscribeToRoles: [],
    instanceId: 0,
    tag: 0,
  };
}

export const ActivateSubscriptionServiceRequest = {
  $type: "mrc.protos.ActivateSubscriptionServiceRequest" as const,

  encode(message: ActivateSubscriptionServiceRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.role !== "") {
      writer.uint32(18).string(message.role);
    }
    for (const v of message.subscribeToRoles) {
      writer.uint32(26).string(v!);
    }
    if (message.instanceId !== 0) {
      writer.uint32(32).uint64(message.instanceId);
    }
    if (message.tag !== 0) {
      writer.uint32(40).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ActivateSubscriptionServiceRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseActivateSubscriptionServiceRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.role = reader.string();
          break;
        case 3:
          message.subscribeToRoles.push(reader.string());
          break;
        case 4:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 5:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): ActivateSubscriptionServiceRequest {
    return {
      $type: ActivateSubscriptionServiceRequest.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      role: isSet(object.role) ? String(object.role) : "",
      subscribeToRoles: Array.isArray(object?.subscribeToRoles)
        ? object.subscribeToRoles.map((e: any) => String(e))
        : [],
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      tag: isSet(object.tag) ? Number(object.tag) : 0,
    };
  },

  toJSON(message: ActivateSubscriptionServiceRequest): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.role !== undefined && (obj.role = message.role);
    if (message.subscribeToRoles) {
      obj.subscribeToRoles = message.subscribeToRoles.map((e) => e);
    } else {
      obj.subscribeToRoles = [];
    }
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<ActivateSubscriptionServiceRequest>): ActivateSubscriptionServiceRequest {
    return ActivateSubscriptionServiceRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<ActivateSubscriptionServiceRequest>): ActivateSubscriptionServiceRequest {
    const message = createBaseActivateSubscriptionServiceRequest();
    message.serviceName = object.serviceName ?? "";
    message.role = object.role ?? "";
    message.subscribeToRoles = object.subscribeToRoles?.map((e) => e) || [];
    message.instanceId = object.instanceId ?? 0;
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(ActivateSubscriptionServiceRequest.$type, ActivateSubscriptionServiceRequest);

function createBaseDropSubscriptionServiceRequest(): DropSubscriptionServiceRequest {
  return { $type: "mrc.protos.DropSubscriptionServiceRequest", serviceName: "", instanceId: 0, tag: 0 };
}

export const DropSubscriptionServiceRequest = {
  $type: "mrc.protos.DropSubscriptionServiceRequest" as const,

  encode(message: DropSubscriptionServiceRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.instanceId !== 0) {
      writer.uint32(16).uint64(message.instanceId);
    }
    if (message.tag !== 0) {
      writer.uint32(24).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): DropSubscriptionServiceRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseDropSubscriptionServiceRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 3:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): DropSubscriptionServiceRequest {
    return {
      $type: DropSubscriptionServiceRequest.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      tag: isSet(object.tag) ? Number(object.tag) : 0,
    };
  },

  toJSON(message: DropSubscriptionServiceRequest): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<DropSubscriptionServiceRequest>): DropSubscriptionServiceRequest {
    return DropSubscriptionServiceRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<DropSubscriptionServiceRequest>): DropSubscriptionServiceRequest {
    const message = createBaseDropSubscriptionServiceRequest();
    message.serviceName = object.serviceName ?? "";
    message.instanceId = object.instanceId ?? 0;
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(DropSubscriptionServiceRequest.$type, DropSubscriptionServiceRequest);

function createBaseUpdateSubscriptionServiceRequest(): UpdateSubscriptionServiceRequest {
  return { $type: "mrc.protos.UpdateSubscriptionServiceRequest", serviceName: "", role: "", nonce: 0, tags: [] };
}

export const UpdateSubscriptionServiceRequest = {
  $type: "mrc.protos.UpdateSubscriptionServiceRequest" as const,

  encode(message: UpdateSubscriptionServiceRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.role !== "") {
      writer.uint32(18).string(message.role);
    }
    if (message.nonce !== 0) {
      writer.uint32(24).uint64(message.nonce);
    }
    writer.uint32(34).fork();
    for (const v of message.tags) {
      writer.uint64(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): UpdateSubscriptionServiceRequest {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseUpdateSubscriptionServiceRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.role = reader.string();
          break;
        case 3:
          message.nonce = longToNumber(reader.uint64() as Long);
          break;
        case 4:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.tags.push(longToNumber(reader.uint64() as Long));
            }
          } else {
            message.tags.push(longToNumber(reader.uint64() as Long));
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): UpdateSubscriptionServiceRequest {
    return {
      $type: UpdateSubscriptionServiceRequest.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      role: isSet(object.role) ? String(object.role) : "",
      nonce: isSet(object.nonce) ? Number(object.nonce) : 0,
      tags: Array.isArray(object?.tags) ? object.tags.map((e: any) => Number(e)) : [],
    };
  },

  toJSON(message: UpdateSubscriptionServiceRequest): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.role !== undefined && (obj.role = message.role);
    message.nonce !== undefined && (obj.nonce = Math.round(message.nonce));
    if (message.tags) {
      obj.tags = message.tags.map((e) => Math.round(e));
    } else {
      obj.tags = [];
    }
    return obj;
  },

  create(base?: DeepPartial<UpdateSubscriptionServiceRequest>): UpdateSubscriptionServiceRequest {
    return UpdateSubscriptionServiceRequest.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<UpdateSubscriptionServiceRequest>): UpdateSubscriptionServiceRequest {
    const message = createBaseUpdateSubscriptionServiceRequest();
    message.serviceName = object.serviceName ?? "";
    message.role = object.role ?? "";
    message.nonce = object.nonce ?? 0;
    message.tags = object.tags?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(UpdateSubscriptionServiceRequest.$type, UpdateSubscriptionServiceRequest);

function createBaseTaggedInstance(): TaggedInstance {
  return { $type: "mrc.protos.TaggedInstance", instanceId: 0, tag: 0 };
}

export const TaggedInstance = {
  $type: "mrc.protos.TaggedInstance" as const,

  encode(message: TaggedInstance, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.instanceId !== 0) {
      writer.uint32(8).uint64(message.instanceId);
    }
    if (message.tag !== 0) {
      writer.uint32(16).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): TaggedInstance {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseTaggedInstance();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): TaggedInstance {
    return {
      $type: TaggedInstance.$type,
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      tag: isSet(object.tag) ? Number(object.tag) : 0,
    };
  },

  toJSON(message: TaggedInstance): unknown {
    const obj: any = {};
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<TaggedInstance>): TaggedInstance {
    return TaggedInstance.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<TaggedInstance>): TaggedInstance {
    const message = createBaseTaggedInstance();
    message.instanceId = object.instanceId ?? 0;
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(TaggedInstance.$type, TaggedInstance);

function createBaseStateUpdate(): StateUpdate {
  return {
    $type: "mrc.protos.StateUpdate",
    serviceName: "",
    nonce: 0,
    instanceId: 0,
    connections: undefined,
    updateSubscriptionService: undefined,
    dropSubscriptionService: undefined,
  };
}

export const StateUpdate = {
  $type: "mrc.protos.StateUpdate" as const,

  encode(message: StateUpdate, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.serviceName !== "") {
      writer.uint32(10).string(message.serviceName);
    }
    if (message.nonce !== 0) {
      writer.uint32(16).uint64(message.nonce);
    }
    if (message.instanceId !== 0) {
      writer.uint32(24).uint64(message.instanceId);
    }
    if (message.connections !== undefined) {
      UpdateConnectionsState.encode(message.connections, writer.uint32(34).fork()).ldelim();
    }
    if (message.updateSubscriptionService !== undefined) {
      UpdateSubscriptionServiceState.encode(message.updateSubscriptionService, writer.uint32(42).fork()).ldelim();
    }
    if (message.dropSubscriptionService !== undefined) {
      DropSubscriptionServiceState.encode(message.dropSubscriptionService, writer.uint32(50).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): StateUpdate {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseStateUpdate();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.serviceName = reader.string();
          break;
        case 2:
          message.nonce = longToNumber(reader.uint64() as Long);
          break;
        case 3:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 4:
          message.connections = UpdateConnectionsState.decode(reader, reader.uint32());
          break;
        case 5:
          message.updateSubscriptionService = UpdateSubscriptionServiceState.decode(reader, reader.uint32());
          break;
        case 6:
          message.dropSubscriptionService = DropSubscriptionServiceState.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): StateUpdate {
    return {
      $type: StateUpdate.$type,
      serviceName: isSet(object.serviceName) ? String(object.serviceName) : "",
      nonce: isSet(object.nonce) ? Number(object.nonce) : 0,
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      connections: isSet(object.connections) ? UpdateConnectionsState.fromJSON(object.connections) : undefined,
      updateSubscriptionService: isSet(object.updateSubscriptionService)
        ? UpdateSubscriptionServiceState.fromJSON(object.updateSubscriptionService)
        : undefined,
      dropSubscriptionService: isSet(object.dropSubscriptionService)
        ? DropSubscriptionServiceState.fromJSON(object.dropSubscriptionService)
        : undefined,
    };
  },

  toJSON(message: StateUpdate): unknown {
    const obj: any = {};
    message.serviceName !== undefined && (obj.serviceName = message.serviceName);
    message.nonce !== undefined && (obj.nonce = Math.round(message.nonce));
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.connections !== undefined &&
      (obj.connections = message.connections ? UpdateConnectionsState.toJSON(message.connections) : undefined);
    message.updateSubscriptionService !== undefined &&
      (obj.updateSubscriptionService = message.updateSubscriptionService
        ? UpdateSubscriptionServiceState.toJSON(message.updateSubscriptionService)
        : undefined);
    message.dropSubscriptionService !== undefined && (obj.dropSubscriptionService = message.dropSubscriptionService
      ? DropSubscriptionServiceState.toJSON(message.dropSubscriptionService)
      : undefined);
    return obj;
  },

  create(base?: DeepPartial<StateUpdate>): StateUpdate {
    return StateUpdate.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<StateUpdate>): StateUpdate {
    const message = createBaseStateUpdate();
    message.serviceName = object.serviceName ?? "";
    message.nonce = object.nonce ?? 0;
    message.instanceId = object.instanceId ?? 0;
    message.connections = (object.connections !== undefined && object.connections !== null)
      ? UpdateConnectionsState.fromPartial(object.connections)
      : undefined;
    message.updateSubscriptionService =
      (object.updateSubscriptionService !== undefined && object.updateSubscriptionService !== null)
        ? UpdateSubscriptionServiceState.fromPartial(object.updateSubscriptionService)
        : undefined;
    message.dropSubscriptionService =
      (object.dropSubscriptionService !== undefined && object.dropSubscriptionService !== null)
        ? DropSubscriptionServiceState.fromPartial(object.dropSubscriptionService)
        : undefined;
    return message;
  },
};

messageTypeRegistry.set(StateUpdate.$type, StateUpdate);

function createBaseUpdateConnectionsState(): UpdateConnectionsState {
  return { $type: "mrc.protos.UpdateConnectionsState", taggedInstances: [] };
}

export const UpdateConnectionsState = {
  $type: "mrc.protos.UpdateConnectionsState" as const,

  encode(message: UpdateConnectionsState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.taggedInstances) {
      TaggedInstance.encode(v!, writer.uint32(10).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): UpdateConnectionsState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseUpdateConnectionsState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.taggedInstances.push(TaggedInstance.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): UpdateConnectionsState {
    return {
      $type: UpdateConnectionsState.$type,
      taggedInstances: Array.isArray(object?.taggedInstances)
        ? object.taggedInstances.map((e: any) => TaggedInstance.fromJSON(e))
        : [],
    };
  },

  toJSON(message: UpdateConnectionsState): unknown {
    const obj: any = {};
    if (message.taggedInstances) {
      obj.taggedInstances = message.taggedInstances.map((e) => e ? TaggedInstance.toJSON(e) : undefined);
    } else {
      obj.taggedInstances = [];
    }
    return obj;
  },

  create(base?: DeepPartial<UpdateConnectionsState>): UpdateConnectionsState {
    return UpdateConnectionsState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<UpdateConnectionsState>): UpdateConnectionsState {
    const message = createBaseUpdateConnectionsState();
    message.taggedInstances = object.taggedInstances?.map((e) => TaggedInstance.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(UpdateConnectionsState.$type, UpdateConnectionsState);

function createBaseUpdateSubscriptionServiceState(): UpdateSubscriptionServiceState {
  return { $type: "mrc.protos.UpdateSubscriptionServiceState", role: "", taggedInstances: [] };
}

export const UpdateSubscriptionServiceState = {
  $type: "mrc.protos.UpdateSubscriptionServiceState" as const,

  encode(message: UpdateSubscriptionServiceState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.role !== "") {
      writer.uint32(10).string(message.role);
    }
    for (const v of message.taggedInstances) {
      TaggedInstance.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): UpdateSubscriptionServiceState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseUpdateSubscriptionServiceState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.role = reader.string();
          break;
        case 2:
          message.taggedInstances.push(TaggedInstance.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): UpdateSubscriptionServiceState {
    return {
      $type: UpdateSubscriptionServiceState.$type,
      role: isSet(object.role) ? String(object.role) : "",
      taggedInstances: Array.isArray(object?.taggedInstances)
        ? object.taggedInstances.map((e: any) => TaggedInstance.fromJSON(e))
        : [],
    };
  },

  toJSON(message: UpdateSubscriptionServiceState): unknown {
    const obj: any = {};
    message.role !== undefined && (obj.role = message.role);
    if (message.taggedInstances) {
      obj.taggedInstances = message.taggedInstances.map((e) => e ? TaggedInstance.toJSON(e) : undefined);
    } else {
      obj.taggedInstances = [];
    }
    return obj;
  },

  create(base?: DeepPartial<UpdateSubscriptionServiceState>): UpdateSubscriptionServiceState {
    return UpdateSubscriptionServiceState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<UpdateSubscriptionServiceState>): UpdateSubscriptionServiceState {
    const message = createBaseUpdateSubscriptionServiceState();
    message.role = object.role ?? "";
    message.taggedInstances = object.taggedInstances?.map((e) => TaggedInstance.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(UpdateSubscriptionServiceState.$type, UpdateSubscriptionServiceState);

function createBaseDropSubscriptionServiceState(): DropSubscriptionServiceState {
  return { $type: "mrc.protos.DropSubscriptionServiceState", role: "", tag: 0 };
}

export const DropSubscriptionServiceState = {
  $type: "mrc.protos.DropSubscriptionServiceState" as const,

  encode(message: DropSubscriptionServiceState, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.role !== "") {
      writer.uint32(10).string(message.role);
    }
    if (message.tag !== 0) {
      writer.uint32(16).uint64(message.tag);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): DropSubscriptionServiceState {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseDropSubscriptionServiceState();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.role = reader.string();
          break;
        case 2:
          message.tag = longToNumber(reader.uint64() as Long);
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): DropSubscriptionServiceState {
    return {
      $type: DropSubscriptionServiceState.$type,
      role: isSet(object.role) ? String(object.role) : "",
      tag: isSet(object.tag) ? Number(object.tag) : 0,
    };
  },

  toJSON(message: DropSubscriptionServiceState): unknown {
    const obj: any = {};
    message.role !== undefined && (obj.role = message.role);
    message.tag !== undefined && (obj.tag = Math.round(message.tag));
    return obj;
  },

  create(base?: DeepPartial<DropSubscriptionServiceState>): DropSubscriptionServiceState {
    return DropSubscriptionServiceState.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<DropSubscriptionServiceState>): DropSubscriptionServiceState {
    const message = createBaseDropSubscriptionServiceState();
    message.role = object.role ?? "";
    message.tag = object.tag ?? 0;
    return message;
  },
};

messageTypeRegistry.set(DropSubscriptionServiceState.$type, DropSubscriptionServiceState);

function createBaseControlMessage(): ControlMessage {
  return { $type: "mrc.protos.ControlMessage" };
}

export const ControlMessage = {
  $type: "mrc.protos.ControlMessage" as const,

  encode(_: ControlMessage, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ControlMessage {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseControlMessage();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(_: any): ControlMessage {
    return { $type: ControlMessage.$type };
  },

  toJSON(_: ControlMessage): unknown {
    const obj: any = {};
    return obj;
  },

  create(base?: DeepPartial<ControlMessage>): ControlMessage {
    return ControlMessage.fromPartial(base ?? {});
  },

  fromPartial(_: DeepPartial<ControlMessage>): ControlMessage {
    const message = createBaseControlMessage();
    return message;
  },
};

messageTypeRegistry.set(ControlMessage.$type, ControlMessage);

function createBaseOnComplete(): OnComplete {
  return { $type: "mrc.protos.OnComplete", segmentAddresses: [] };
}

export const OnComplete = {
  $type: "mrc.protos.OnComplete" as const,

  encode(message: OnComplete, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    writer.uint32(10).fork();
    for (const v of message.segmentAddresses) {
      writer.uint32(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): OnComplete {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseOnComplete();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
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

  fromJSON(object: any): OnComplete {
    return {
      $type: OnComplete.$type,
      segmentAddresses: Array.isArray(object?.segmentAddresses)
        ? object.segmentAddresses.map((e: any) => Number(e))
        : [],
    };
  },

  toJSON(message: OnComplete): unknown {
    const obj: any = {};
    if (message.segmentAddresses) {
      obj.segmentAddresses = message.segmentAddresses.map((e) => Math.round(e));
    } else {
      obj.segmentAddresses = [];
    }
    return obj;
  },

  create(base?: DeepPartial<OnComplete>): OnComplete {
    return OnComplete.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<OnComplete>): OnComplete {
    const message = createBaseOnComplete();
    message.segmentAddresses = object.segmentAddresses?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(OnComplete.$type, OnComplete);

function createBaseUpdateAssignments(): UpdateAssignments {
  return { $type: "mrc.protos.UpdateAssignments", assignments: [] };
}

export const UpdateAssignments = {
  $type: "mrc.protos.UpdateAssignments" as const,

  encode(message: UpdateAssignments, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.assignments) {
      SegmentAssignment.encode(v!, writer.uint32(10).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): UpdateAssignments {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseUpdateAssignments();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.assignments.push(SegmentAssignment.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): UpdateAssignments {
    return {
      $type: UpdateAssignments.$type,
      assignments: Array.isArray(object?.assignments)
        ? object.assignments.map((e: any) => SegmentAssignment.fromJSON(e))
        : [],
    };
  },

  toJSON(message: UpdateAssignments): unknown {
    const obj: any = {};
    if (message.assignments) {
      obj.assignments = message.assignments.map((e) => e ? SegmentAssignment.toJSON(e) : undefined);
    } else {
      obj.assignments = [];
    }
    return obj;
  },

  create(base?: DeepPartial<UpdateAssignments>): UpdateAssignments {
    return UpdateAssignments.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<UpdateAssignments>): UpdateAssignments {
    const message = createBaseUpdateAssignments();
    message.assignments = object.assignments?.map((e) => SegmentAssignment.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(UpdateAssignments.$type, UpdateAssignments);

function createBaseSegmentAssignment(): SegmentAssignment {
  return {
    $type: "mrc.protos.SegmentAssignment",
    machineId: 0,
    instanceId: 0,
    address: 0,
    egressPolices: {},
    issueEventOnComplete: false,
    networkIngressPorts: [],
  };
}

export const SegmentAssignment = {
  $type: "mrc.protos.SegmentAssignment" as const,

  encode(message: SegmentAssignment, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.machineId !== 0) {
      writer.uint32(8).uint64(message.machineId);
    }
    if (message.instanceId !== 0) {
      writer.uint32(16).uint64(message.instanceId);
    }
    if (message.address !== 0) {
      writer.uint32(24).uint32(message.address);
    }
    Object.entries(message.egressPolices).forEach(([key, value]) => {
      SegmentAssignment_EgressPolicesEntry.encode({
        $type: "mrc.protos.SegmentAssignment.EgressPolicesEntry",
        key: key as any,
        value,
      }, writer.uint32(42).fork()).ldelim();
    });
    if (message.issueEventOnComplete === true) {
      writer.uint32(48).bool(message.issueEventOnComplete);
    }
    writer.uint32(58).fork();
    for (const v of message.networkIngressPorts) {
      writer.uint32(v);
    }
    writer.ldelim();
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentAssignment {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentAssignment();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.machineId = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 3:
          message.address = reader.uint32();
          break;
        case 5:
          const entry5 = SegmentAssignment_EgressPolicesEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.egressPolices[entry5.key] = entry5.value;
          }
          break;
        case 6:
          message.issueEventOnComplete = reader.bool();
          break;
        case 7:
          if ((tag & 7) === 2) {
            const end2 = reader.uint32() + reader.pos;
            while (reader.pos < end2) {
              message.networkIngressPorts.push(reader.uint32());
            }
          } else {
            message.networkIngressPorts.push(reader.uint32());
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentAssignment {
    return {
      $type: SegmentAssignment.$type,
      machineId: isSet(object.machineId) ? Number(object.machineId) : 0,
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      address: isSet(object.address) ? Number(object.address) : 0,
      egressPolices: isObject(object.egressPolices)
        ? Object.entries(object.egressPolices).reduce<{ [key: number]: EgressPolicy }>((acc, [key, value]) => {
          acc[Number(key)] = EgressPolicy.fromJSON(value);
          return acc;
        }, {})
        : {},
      issueEventOnComplete: isSet(object.issueEventOnComplete) ? Boolean(object.issueEventOnComplete) : false,
      networkIngressPorts: Array.isArray(object?.networkIngressPorts)
        ? object.networkIngressPorts.map((e: any) => Number(e))
        : [],
    };
  },

  toJSON(message: SegmentAssignment): unknown {
    const obj: any = {};
    message.machineId !== undefined && (obj.machineId = Math.round(message.machineId));
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.address !== undefined && (obj.address = Math.round(message.address));
    obj.egressPolices = {};
    if (message.egressPolices) {
      Object.entries(message.egressPolices).forEach(([k, v]) => {
        obj.egressPolices[k] = EgressPolicy.toJSON(v);
      });
    }
    message.issueEventOnComplete !== undefined && (obj.issueEventOnComplete = message.issueEventOnComplete);
    if (message.networkIngressPorts) {
      obj.networkIngressPorts = message.networkIngressPorts.map((e) => Math.round(e));
    } else {
      obj.networkIngressPorts = [];
    }
    return obj;
  },

  create(base?: DeepPartial<SegmentAssignment>): SegmentAssignment {
    return SegmentAssignment.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentAssignment>): SegmentAssignment {
    const message = createBaseSegmentAssignment();
    message.machineId = object.machineId ?? 0;
    message.instanceId = object.instanceId ?? 0;
    message.address = object.address ?? 0;
    message.egressPolices = Object.entries(object.egressPolices ?? {}).reduce<{ [key: number]: EgressPolicy }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[Number(key)] = EgressPolicy.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    message.issueEventOnComplete = object.issueEventOnComplete ?? false;
    message.networkIngressPorts = object.networkIngressPorts?.map((e) => e) || [];
    return message;
  },
};

messageTypeRegistry.set(SegmentAssignment.$type, SegmentAssignment);

function createBaseSegmentAssignment_EgressPolicesEntry(): SegmentAssignment_EgressPolicesEntry {
  return { $type: "mrc.protos.SegmentAssignment.EgressPolicesEntry", key: 0, value: undefined };
}

export const SegmentAssignment_EgressPolicesEntry = {
  $type: "mrc.protos.SegmentAssignment.EgressPolicesEntry" as const,

  encode(message: SegmentAssignment_EgressPolicesEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== 0) {
      writer.uint32(8).uint32(message.key);
    }
    if (message.value !== undefined) {
      EgressPolicy.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentAssignment_EgressPolicesEntry {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentAssignment_EgressPolicesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.uint32();
          break;
        case 2:
          message.value = EgressPolicy.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentAssignment_EgressPolicesEntry {
    return {
      $type: SegmentAssignment_EgressPolicesEntry.$type,
      key: isSet(object.key) ? Number(object.key) : 0,
      value: isSet(object.value) ? EgressPolicy.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: SegmentAssignment_EgressPolicesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = Math.round(message.key));
    message.value !== undefined && (obj.value = message.value ? EgressPolicy.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<SegmentAssignment_EgressPolicesEntry>): SegmentAssignment_EgressPolicesEntry {
    return SegmentAssignment_EgressPolicesEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentAssignment_EgressPolicesEntry>): SegmentAssignment_EgressPolicesEntry {
    const message = createBaseSegmentAssignment_EgressPolicesEntry();
    message.key = object.key ?? 0;
    message.value = (object.value !== undefined && object.value !== null)
      ? EgressPolicy.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentAssignment_EgressPolicesEntry.$type, SegmentAssignment_EgressPolicesEntry);

function createBaseTopology(): Topology {
  return { $type: "mrc.protos.Topology", hwlocXmlString: "", cpuSet: "", gpuInfo: [] };
}

export const Topology = {
  $type: "mrc.protos.Topology" as const,

  encode(message: Topology, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.hwlocXmlString !== "") {
      writer.uint32(10).string(message.hwlocXmlString);
    }
    if (message.cpuSet !== "") {
      writer.uint32(18).string(message.cpuSet);
    }
    for (const v of message.gpuInfo) {
      GpuInfo.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Topology {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseTopology();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.hwlocXmlString = reader.string();
          break;
        case 2:
          message.cpuSet = reader.string();
          break;
        case 3:
          message.gpuInfo.push(GpuInfo.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Topology {
    return {
      $type: Topology.$type,
      hwlocXmlString: isSet(object.hwlocXmlString) ? String(object.hwlocXmlString) : "",
      cpuSet: isSet(object.cpuSet) ? String(object.cpuSet) : "",
      gpuInfo: Array.isArray(object?.gpuInfo) ? object.gpuInfo.map((e: any) => GpuInfo.fromJSON(e)) : [],
    };
  },

  toJSON(message: Topology): unknown {
    const obj: any = {};
    message.hwlocXmlString !== undefined && (obj.hwlocXmlString = message.hwlocXmlString);
    message.cpuSet !== undefined && (obj.cpuSet = message.cpuSet);
    if (message.gpuInfo) {
      obj.gpuInfo = message.gpuInfo.map((e) => e ? GpuInfo.toJSON(e) : undefined);
    } else {
      obj.gpuInfo = [];
    }
    return obj;
  },

  create(base?: DeepPartial<Topology>): Topology {
    return Topology.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Topology>): Topology {
    const message = createBaseTopology();
    message.hwlocXmlString = object.hwlocXmlString ?? "";
    message.cpuSet = object.cpuSet ?? "";
    message.gpuInfo = object.gpuInfo?.map((e) => GpuInfo.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(Topology.$type, Topology);

function createBaseGpuInfo(): GpuInfo {
  return {
    $type: "mrc.protos.GpuInfo",
    cpuSet: "",
    name: "",
    uuid: "",
    pcieBusId: "",
    memoryCapacity: 0,
    cudaDeviceId: 0,
  };
}

export const GpuInfo = {
  $type: "mrc.protos.GpuInfo" as const,

  encode(message: GpuInfo, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.cpuSet !== "") {
      writer.uint32(10).string(message.cpuSet);
    }
    if (message.name !== "") {
      writer.uint32(18).string(message.name);
    }
    if (message.uuid !== "") {
      writer.uint32(26).string(message.uuid);
    }
    if (message.pcieBusId !== "") {
      writer.uint32(34).string(message.pcieBusId);
    }
    if (message.memoryCapacity !== 0) {
      writer.uint32(40).uint64(message.memoryCapacity);
    }
    if (message.cudaDeviceId !== 0) {
      writer.uint32(48).int32(message.cudaDeviceId);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): GpuInfo {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseGpuInfo();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.cpuSet = reader.string();
          break;
        case 2:
          message.name = reader.string();
          break;
        case 3:
          message.uuid = reader.string();
          break;
        case 4:
          message.pcieBusId = reader.string();
          break;
        case 5:
          message.memoryCapacity = longToNumber(reader.uint64() as Long);
          break;
        case 6:
          message.cudaDeviceId = reader.int32();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): GpuInfo {
    return {
      $type: GpuInfo.$type,
      cpuSet: isSet(object.cpuSet) ? String(object.cpuSet) : "",
      name: isSet(object.name) ? String(object.name) : "",
      uuid: isSet(object.uuid) ? String(object.uuid) : "",
      pcieBusId: isSet(object.pcieBusId) ? String(object.pcieBusId) : "",
      memoryCapacity: isSet(object.memoryCapacity) ? Number(object.memoryCapacity) : 0,
      cudaDeviceId: isSet(object.cudaDeviceId) ? Number(object.cudaDeviceId) : 0,
    };
  },

  toJSON(message: GpuInfo): unknown {
    const obj: any = {};
    message.cpuSet !== undefined && (obj.cpuSet = message.cpuSet);
    message.name !== undefined && (obj.name = message.name);
    message.uuid !== undefined && (obj.uuid = message.uuid);
    message.pcieBusId !== undefined && (obj.pcieBusId = message.pcieBusId);
    message.memoryCapacity !== undefined && (obj.memoryCapacity = Math.round(message.memoryCapacity));
    message.cudaDeviceId !== undefined && (obj.cudaDeviceId = Math.round(message.cudaDeviceId));
    return obj;
  },

  create(base?: DeepPartial<GpuInfo>): GpuInfo {
    return GpuInfo.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<GpuInfo>): GpuInfo {
    const message = createBaseGpuInfo();
    message.cpuSet = object.cpuSet ?? "";
    message.name = object.name ?? "";
    message.uuid = object.uuid ?? "";
    message.pcieBusId = object.pcieBusId ?? "";
    message.memoryCapacity = object.memoryCapacity ?? 0;
    message.cudaDeviceId = object.cudaDeviceId ?? 0;
    return message;
  },
};

messageTypeRegistry.set(GpuInfo.$type, GpuInfo);

function createBasePipeline(): Pipeline {
  return { $type: "mrc.protos.Pipeline", name: "", segments: [] };
}

export const Pipeline = {
  $type: "mrc.protos.Pipeline" as const,

  encode(message: Pipeline, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    for (const v of message.segments) {
      SegmentDefinition.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Pipeline {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePipeline();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.segments.push(SegmentDefinition.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): Pipeline {
    return {
      $type: Pipeline.$type,
      name: isSet(object.name) ? String(object.name) : "",
      segments: Array.isArray(object?.segments) ? object.segments.map((e: any) => SegmentDefinition.fromJSON(e)) : [],
    };
  },

  toJSON(message: Pipeline): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    if (message.segments) {
      obj.segments = message.segments.map((e) => e ? SegmentDefinition.toJSON(e) : undefined);
    } else {
      obj.segments = [];
    }
    return obj;
  },

  create(base?: DeepPartial<Pipeline>): Pipeline {
    return Pipeline.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<Pipeline>): Pipeline {
    const message = createBasePipeline();
    message.name = object.name ?? "";
    message.segments = object.segments?.map((e) => SegmentDefinition.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(Pipeline.$type, Pipeline);

function createBaseSegmentDefinition(): SegmentDefinition {
  return {
    $type: "mrc.protos.SegmentDefinition",
    name: "",
    id: 0,
    ingressPorts: [],
    egressPorts: [],
    options: undefined,
  };
}

export const SegmentDefinition = {
  $type: "mrc.protos.SegmentDefinition" as const,

  encode(message: SegmentDefinition, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.id !== 0) {
      writer.uint32(16).uint32(message.id);
    }
    for (const v of message.ingressPorts) {
      IngressPort.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    for (const v of message.egressPorts) {
      EgressPort.encode(v!, writer.uint32(34).fork()).ldelim();
    }
    if (message.options !== undefined) {
      SegmentOptions.encode(message.options, writer.uint32(42).fork()).ldelim();
    }
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
          message.name = reader.string();
          break;
        case 2:
          message.id = reader.uint32();
          break;
        case 3:
          message.ingressPorts.push(IngressPort.decode(reader, reader.uint32()));
          break;
        case 4:
          message.egressPorts.push(EgressPort.decode(reader, reader.uint32()));
          break;
        case 5:
          message.options = SegmentOptions.decode(reader, reader.uint32());
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
      name: isSet(object.name) ? String(object.name) : "",
      id: isSet(object.id) ? Number(object.id) : 0,
      ingressPorts: Array.isArray(object?.ingressPorts)
        ? object.ingressPorts.map((e: any) => IngressPort.fromJSON(e))
        : [],
      egressPorts: Array.isArray(object?.egressPorts) ? object.egressPorts.map((e: any) => EgressPort.fromJSON(e)) : [],
      options: isSet(object.options) ? SegmentOptions.fromJSON(object.options) : undefined,
    };
  },

  toJSON(message: SegmentDefinition): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.id !== undefined && (obj.id = Math.round(message.id));
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

  create(base?: DeepPartial<SegmentDefinition>): SegmentDefinition {
    return SegmentDefinition.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentDefinition>): SegmentDefinition {
    const message = createBaseSegmentDefinition();
    message.name = object.name ?? "";
    message.id = object.id ?? 0;
    message.ingressPorts = object.ingressPorts?.map((e) => IngressPort.fromPartial(e)) || [];
    message.egressPorts = object.egressPorts?.map((e) => EgressPort.fromPartial(e)) || [];
    message.options = (object.options !== undefined && object.options !== null)
      ? SegmentOptions.fromPartial(object.options)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentDefinition.$type, SegmentDefinition);

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

function createBasePipelineConfiguration(): PipelineConfiguration {
  return { $type: "mrc.protos.PipelineConfiguration", instanceId: 0, segments: [] };
}

export const PipelineConfiguration = {
  $type: "mrc.protos.PipelineConfiguration" as const,

  encode(message: PipelineConfiguration, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.instanceId !== 0) {
      writer.uint32(8).uint64(message.instanceId);
    }
    for (const v of message.segments) {
      SegmentConfiguration.encode(v!, writer.uint32(18).fork()).ldelim();
    }
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
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          message.segments.push(SegmentConfiguration.decode(reader, reader.uint32()));
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
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      segments: Array.isArray(object?.segments)
        ? object.segments.map((e: any) => SegmentConfiguration.fromJSON(e))
        : [],
    };
  },

  toJSON(message: PipelineConfiguration): unknown {
    const obj: any = {};
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    if (message.segments) {
      obj.segments = message.segments.map((e) => e ? SegmentConfiguration.toJSON(e) : undefined);
    } else {
      obj.segments = [];
    }
    return obj;
  },

  create(base?: DeepPartial<PipelineConfiguration>): PipelineConfiguration {
    return PipelineConfiguration.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<PipelineConfiguration>): PipelineConfiguration {
    const message = createBasePipelineConfiguration();
    message.instanceId = object.instanceId ?? 0;
    message.segments = object.segments?.map((e) => SegmentConfiguration.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(PipelineConfiguration.$type, PipelineConfiguration);

function createBaseSegmentConfiguration(): SegmentConfiguration {
  return {
    $type: "mrc.protos.SegmentConfiguration",
    name: "",
    concurrency: 0,
    rank: 0,
    egressPolices: {},
    ingressPolicies: {},
  };
}

export const SegmentConfiguration = {
  $type: "mrc.protos.SegmentConfiguration" as const,

  encode(message: SegmentConfiguration, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.concurrency !== 0) {
      writer.uint32(16).uint32(message.concurrency);
    }
    if (message.rank !== 0) {
      writer.uint32(24).uint32(message.rank);
    }
    Object.entries(message.egressPolices).forEach(([key, value]) => {
      SegmentConfiguration_EgressPolicesEntry.encode({
        $type: "mrc.protos.SegmentConfiguration.EgressPolicesEntry",
        key: key as any,
        value,
      }, writer.uint32(34).fork()).ldelim();
    });
    Object.entries(message.ingressPolicies).forEach(([key, value]) => {
      SegmentConfiguration_IngressPoliciesEntry.encode({
        $type: "mrc.protos.SegmentConfiguration.IngressPoliciesEntry",
        key: key as any,
        value,
      }, writer.uint32(42).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentConfiguration {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentConfiguration();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.concurrency = reader.uint32();
          break;
        case 3:
          message.rank = reader.uint32();
          break;
        case 4:
          const entry4 = SegmentConfiguration_EgressPolicesEntry.decode(reader, reader.uint32());
          if (entry4.value !== undefined) {
            message.egressPolices[entry4.key] = entry4.value;
          }
          break;
        case 5:
          const entry5 = SegmentConfiguration_IngressPoliciesEntry.decode(reader, reader.uint32());
          if (entry5.value !== undefined) {
            message.ingressPolicies[entry5.key] = entry5.value;
          }
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentConfiguration {
    return {
      $type: SegmentConfiguration.$type,
      name: isSet(object.name) ? String(object.name) : "",
      concurrency: isSet(object.concurrency) ? Number(object.concurrency) : 0,
      rank: isSet(object.rank) ? Number(object.rank) : 0,
      egressPolices: isObject(object.egressPolices)
        ? Object.entries(object.egressPolices).reduce<{ [key: number]: EgressPolicy }>((acc, [key, value]) => {
          acc[Number(key)] = EgressPolicy.fromJSON(value);
          return acc;
        }, {})
        : {},
      ingressPolicies: isObject(object.ingressPolicies)
        ? Object.entries(object.ingressPolicies).reduce<{ [key: number]: IngressPolicy }>((acc, [key, value]) => {
          acc[Number(key)] = IngressPolicy.fromJSON(value);
          return acc;
        }, {})
        : {},
    };
  },

  toJSON(message: SegmentConfiguration): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.concurrency !== undefined && (obj.concurrency = Math.round(message.concurrency));
    message.rank !== undefined && (obj.rank = Math.round(message.rank));
    obj.egressPolices = {};
    if (message.egressPolices) {
      Object.entries(message.egressPolices).forEach(([k, v]) => {
        obj.egressPolices[k] = EgressPolicy.toJSON(v);
      });
    }
    obj.ingressPolicies = {};
    if (message.ingressPolicies) {
      Object.entries(message.ingressPolicies).forEach(([k, v]) => {
        obj.ingressPolicies[k] = IngressPolicy.toJSON(v);
      });
    }
    return obj;
  },

  create(base?: DeepPartial<SegmentConfiguration>): SegmentConfiguration {
    return SegmentConfiguration.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentConfiguration>): SegmentConfiguration {
    const message = createBaseSegmentConfiguration();
    message.name = object.name ?? "";
    message.concurrency = object.concurrency ?? 0;
    message.rank = object.rank ?? 0;
    message.egressPolices = Object.entries(object.egressPolices ?? {}).reduce<{ [key: number]: EgressPolicy }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[Number(key)] = EgressPolicy.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    message.ingressPolicies = Object.entries(object.ingressPolicies ?? {}).reduce<{ [key: number]: IngressPolicy }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[Number(key)] = IngressPolicy.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    return message;
  },
};

messageTypeRegistry.set(SegmentConfiguration.$type, SegmentConfiguration);

function createBaseSegmentConfiguration_EgressPolicesEntry(): SegmentConfiguration_EgressPolicesEntry {
  return { $type: "mrc.protos.SegmentConfiguration.EgressPolicesEntry", key: 0, value: undefined };
}

export const SegmentConfiguration_EgressPolicesEntry = {
  $type: "mrc.protos.SegmentConfiguration.EgressPolicesEntry" as const,

  encode(message: SegmentConfiguration_EgressPolicesEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== 0) {
      writer.uint32(8).uint32(message.key);
    }
    if (message.value !== undefined) {
      EgressPolicy.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentConfiguration_EgressPolicesEntry {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentConfiguration_EgressPolicesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.uint32();
          break;
        case 2:
          message.value = EgressPolicy.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentConfiguration_EgressPolicesEntry {
    return {
      $type: SegmentConfiguration_EgressPolicesEntry.$type,
      key: isSet(object.key) ? Number(object.key) : 0,
      value: isSet(object.value) ? EgressPolicy.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: SegmentConfiguration_EgressPolicesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = Math.round(message.key));
    message.value !== undefined && (obj.value = message.value ? EgressPolicy.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<SegmentConfiguration_EgressPolicesEntry>): SegmentConfiguration_EgressPolicesEntry {
    return SegmentConfiguration_EgressPolicesEntry.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<SegmentConfiguration_EgressPolicesEntry>): SegmentConfiguration_EgressPolicesEntry {
    const message = createBaseSegmentConfiguration_EgressPolicesEntry();
    message.key = object.key ?? 0;
    message.value = (object.value !== undefined && object.value !== null)
      ? EgressPolicy.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentConfiguration_EgressPolicesEntry.$type, SegmentConfiguration_EgressPolicesEntry);

function createBaseSegmentConfiguration_IngressPoliciesEntry(): SegmentConfiguration_IngressPoliciesEntry {
  return { $type: "mrc.protos.SegmentConfiguration.IngressPoliciesEntry", key: 0, value: undefined };
}

export const SegmentConfiguration_IngressPoliciesEntry = {
  $type: "mrc.protos.SegmentConfiguration.IngressPoliciesEntry" as const,

  encode(message: SegmentConfiguration_IngressPoliciesEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== 0) {
      writer.uint32(8).uint32(message.key);
    }
    if (message.value !== undefined) {
      IngressPolicy.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SegmentConfiguration_IngressPoliciesEntry {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSegmentConfiguration_IngressPoliciesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.key = reader.uint32();
          break;
        case 2:
          message.value = IngressPolicy.decode(reader, reader.uint32());
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): SegmentConfiguration_IngressPoliciesEntry {
    return {
      $type: SegmentConfiguration_IngressPoliciesEntry.$type,
      key: isSet(object.key) ? Number(object.key) : 0,
      value: isSet(object.value) ? IngressPolicy.fromJSON(object.value) : undefined,
    };
  },

  toJSON(message: SegmentConfiguration_IngressPoliciesEntry): unknown {
    const obj: any = {};
    message.key !== undefined && (obj.key = Math.round(message.key));
    message.value !== undefined && (obj.value = message.value ? IngressPolicy.toJSON(message.value) : undefined);
    return obj;
  },

  create(base?: DeepPartial<SegmentConfiguration_IngressPoliciesEntry>): SegmentConfiguration_IngressPoliciesEntry {
    return SegmentConfiguration_IngressPoliciesEntry.fromPartial(base ?? {});
  },

  fromPartial(
    object: DeepPartial<SegmentConfiguration_IngressPoliciesEntry>,
  ): SegmentConfiguration_IngressPoliciesEntry {
    const message = createBaseSegmentConfiguration_IngressPoliciesEntry();
    message.key = object.key ?? 0;
    message.value = (object.value !== undefined && object.value !== null)
      ? IngressPolicy.fromPartial(object.value)
      : undefined;
    return message;
  },
};

messageTypeRegistry.set(SegmentConfiguration_IngressPoliciesEntry.$type, SegmentConfiguration_IngressPoliciesEntry);

function createBaseWorkerAddress(): WorkerAddress {
  return { $type: "mrc.protos.WorkerAddress", machineId: 0, instanceId: 0, workerAddress: new Uint8Array() };
}

export const WorkerAddress = {
  $type: "mrc.protos.WorkerAddress" as const,

  encode(message: WorkerAddress, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.machineId !== 0) {
      writer.uint32(8).uint64(message.machineId);
    }
    if (message.instanceId !== 0) {
      writer.uint32(16).uint64(message.instanceId);
    }
    if (message.workerAddress.length !== 0) {
      writer.uint32(26).bytes(message.workerAddress);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): WorkerAddress {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkerAddress();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.machineId = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          message.instanceId = longToNumber(reader.uint64() as Long);
          break;
        case 3:
          message.workerAddress = reader.bytes();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): WorkerAddress {
    return {
      $type: WorkerAddress.$type,
      machineId: isSet(object.machineId) ? Number(object.machineId) : 0,
      instanceId: isSet(object.instanceId) ? Number(object.instanceId) : 0,
      workerAddress: isSet(object.workerAddress) ? bytesFromBase64(object.workerAddress) : new Uint8Array(),
    };
  },

  toJSON(message: WorkerAddress): unknown {
    const obj: any = {};
    message.machineId !== undefined && (obj.machineId = Math.round(message.machineId));
    message.instanceId !== undefined && (obj.instanceId = Math.round(message.instanceId));
    message.workerAddress !== undefined &&
      (obj.workerAddress = base64FromBytes(
        message.workerAddress !== undefined ? message.workerAddress : new Uint8Array(),
      ));
    return obj;
  },

  create(base?: DeepPartial<WorkerAddress>): WorkerAddress {
    return WorkerAddress.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<WorkerAddress>): WorkerAddress {
    const message = createBaseWorkerAddress();
    message.machineId = object.machineId ?? 0;
    message.instanceId = object.instanceId ?? 0;
    message.workerAddress = object.workerAddress ?? new Uint8Array();
    return message;
  },
};

messageTypeRegistry.set(WorkerAddress.$type, WorkerAddress);

function createBaseInstancesResources(): InstancesResources {
  return { $type: "mrc.protos.InstancesResources", hostMemory: 0, cpus: [], gpus: [], nics: [] };
}

export const InstancesResources = {
  $type: "mrc.protos.InstancesResources" as const,

  encode(message: InstancesResources, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.hostMemory !== 0) {
      writer.uint32(8).uint64(message.hostMemory);
    }
    for (const v of message.cpus) {
      CPU.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    for (const v of message.gpus) {
      GPU.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    for (const v of message.nics) {
      NIC.encode(v!, writer.uint32(34).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): InstancesResources {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseInstancesResources();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.hostMemory = longToNumber(reader.uint64() as Long);
          break;
        case 2:
          message.cpus.push(CPU.decode(reader, reader.uint32()));
          break;
        case 3:
          message.gpus.push(GPU.decode(reader, reader.uint32()));
          break;
        case 4:
          message.nics.push(NIC.decode(reader, reader.uint32()));
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): InstancesResources {
    return {
      $type: InstancesResources.$type,
      hostMemory: isSet(object.hostMemory) ? Number(object.hostMemory) : 0,
      cpus: Array.isArray(object?.cpus) ? object.cpus.map((e: any) => CPU.fromJSON(e)) : [],
      gpus: Array.isArray(object?.gpus) ? object.gpus.map((e: any) => GPU.fromJSON(e)) : [],
      nics: Array.isArray(object?.nics) ? object.nics.map((e: any) => NIC.fromJSON(e)) : [],
    };
  },

  toJSON(message: InstancesResources): unknown {
    const obj: any = {};
    message.hostMemory !== undefined && (obj.hostMemory = Math.round(message.hostMemory));
    if (message.cpus) {
      obj.cpus = message.cpus.map((e) => e ? CPU.toJSON(e) : undefined);
    } else {
      obj.cpus = [];
    }
    if (message.gpus) {
      obj.gpus = message.gpus.map((e) => e ? GPU.toJSON(e) : undefined);
    } else {
      obj.gpus = [];
    }
    if (message.nics) {
      obj.nics = message.nics.map((e) => e ? NIC.toJSON(e) : undefined);
    } else {
      obj.nics = [];
    }
    return obj;
  },

  create(base?: DeepPartial<InstancesResources>): InstancesResources {
    return InstancesResources.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<InstancesResources>): InstancesResources {
    const message = createBaseInstancesResources();
    message.hostMemory = object.hostMemory ?? 0;
    message.cpus = object.cpus?.map((e) => CPU.fromPartial(e)) || [];
    message.gpus = object.gpus?.map((e) => GPU.fromPartial(e)) || [];
    message.nics = object.nics?.map((e) => NIC.fromPartial(e)) || [];
    return message;
  },
};

messageTypeRegistry.set(InstancesResources.$type, InstancesResources);

function createBaseCPU(): CPU {
  return { $type: "mrc.protos.CPU", cores: 0, numaNodes: 0 };
}

export const CPU = {
  $type: "mrc.protos.CPU" as const,

  encode(message: CPU, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.cores !== 0) {
      writer.uint32(8).uint32(message.cores);
    }
    if (message.numaNodes !== 0) {
      writer.uint32(16).uint32(message.numaNodes);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): CPU {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseCPU();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.cores = reader.uint32();
          break;
        case 2:
          message.numaNodes = reader.uint32();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): CPU {
    return {
      $type: CPU.$type,
      cores: isSet(object.cores) ? Number(object.cores) : 0,
      numaNodes: isSet(object.numaNodes) ? Number(object.numaNodes) : 0,
    };
  },

  toJSON(message: CPU): unknown {
    const obj: any = {};
    message.cores !== undefined && (obj.cores = Math.round(message.cores));
    message.numaNodes !== undefined && (obj.numaNodes = Math.round(message.numaNodes));
    return obj;
  },

  create(base?: DeepPartial<CPU>): CPU {
    return CPU.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<CPU>): CPU {
    const message = createBaseCPU();
    message.cores = object.cores ?? 0;
    message.numaNodes = object.numaNodes ?? 0;
    return message;
  },
};

messageTypeRegistry.set(CPU.$type, CPU);

function createBaseGPU(): GPU {
  return { $type: "mrc.protos.GPU", name: "", cores: 0, memory: 0, computeCapability: 0 };
}

export const GPU = {
  $type: "mrc.protos.GPU" as const,

  encode(message: GPU, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.name !== "") {
      writer.uint32(10).string(message.name);
    }
    if (message.cores !== 0) {
      writer.uint32(16).uint32(message.cores);
    }
    if (message.memory !== 0) {
      writer.uint32(24).uint64(message.memory);
    }
    if (message.computeCapability !== 0) {
      writer.uint32(37).float(message.computeCapability);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): GPU {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseGPU();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          message.name = reader.string();
          break;
        case 2:
          message.cores = reader.uint32();
          break;
        case 3:
          message.memory = longToNumber(reader.uint64() as Long);
          break;
        case 4:
          message.computeCapability = reader.float();
          break;
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(object: any): GPU {
    return {
      $type: GPU.$type,
      name: isSet(object.name) ? String(object.name) : "",
      cores: isSet(object.cores) ? Number(object.cores) : 0,
      memory: isSet(object.memory) ? Number(object.memory) : 0,
      computeCapability: isSet(object.computeCapability) ? Number(object.computeCapability) : 0,
    };
  },

  toJSON(message: GPU): unknown {
    const obj: any = {};
    message.name !== undefined && (obj.name = message.name);
    message.cores !== undefined && (obj.cores = Math.round(message.cores));
    message.memory !== undefined && (obj.memory = Math.round(message.memory));
    message.computeCapability !== undefined && (obj.computeCapability = message.computeCapability);
    return obj;
  },

  create(base?: DeepPartial<GPU>): GPU {
    return GPU.fromPartial(base ?? {});
  },

  fromPartial(object: DeepPartial<GPU>): GPU {
    const message = createBaseGPU();
    message.name = object.name ?? "";
    message.cores = object.cores ?? 0;
    message.memory = object.memory ?? 0;
    message.computeCapability = object.computeCapability ?? 0;
    return message;
  },
};

messageTypeRegistry.set(GPU.$type, GPU);

function createBaseNIC(): NIC {
  return { $type: "mrc.protos.NIC" };
}

export const NIC = {
  $type: "mrc.protos.NIC" as const,

  encode(_: NIC, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): NIC {
    const reader = input instanceof _m0.Reader ? input : new _m0.Reader(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseNIC();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        default:
          reader.skipType(tag & 7);
          break;
      }
    }
    return message;
  },

  fromJSON(_: any): NIC {
    return { $type: NIC.$type };
  },

  toJSON(_: NIC): unknown {
    const obj: any = {};
    return obj;
  },

  create(base?: DeepPartial<NIC>): NIC {
    return NIC.fromPartial(base ?? {});
  },

  fromPartial(_: DeepPartial<NIC>): NIC {
    const message = createBaseNIC();
    return message;
  },
};

messageTypeRegistry.set(NIC.$type, NIC);

export type ArchitectDefinition = typeof ArchitectDefinition;
export const ArchitectDefinition = {
  name: "Architect",
  fullName: "mrc.protos.Architect",
  methods: {
    eventStream: {
      name: "EventStream",
      requestType: Event,
      requestStream: true,
      responseType: Event,
      responseStream: true,
      options: {},
    },
    ping: {
      name: "Ping",
      requestType: PingRequest,
      requestStream: false,
      responseType: PingResponse,
      responseStream: false,
      options: {},
    },
    shutdown: {
      name: "Shutdown",
      requestType: ShutdownRequest,
      requestStream: false,
      responseType: ShutdownResponse,
      responseStream: false,
      options: {},
    },
  },
} as const;

export interface ArchitectServiceImplementation<CallContextExt = {}> {
  eventStream(
    request: AsyncIterable<Event>,
    context: CallContext & CallContextExt,
  ): ServerStreamingMethodResult<DeepPartial<Event>>;
  ping(request: PingRequest, context: CallContext & CallContextExt): Promise<DeepPartial<PingResponse>>;
  shutdown(request: ShutdownRequest, context: CallContext & CallContextExt): Promise<DeepPartial<ShutdownResponse>>;
}

export interface ArchitectClient<CallOptionsExt = {}> {
  eventStream(request: AsyncIterable<DeepPartial<Event>>, options?: CallOptions & CallOptionsExt): AsyncIterable<Event>;
  ping(request: DeepPartial<PingRequest>, options?: CallOptions & CallOptionsExt): Promise<PingResponse>;
  shutdown(request: DeepPartial<ShutdownRequest>, options?: CallOptions & CallOptionsExt): Promise<ShutdownResponse>;
}

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

type DeepPartial<T> = T extends Builtin ? T
  : T extends Array<infer U> ? Array<DeepPartial<U>> : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in Exclude<keyof T, "$type">]?: DeepPartial<T[K]> }
  : Partial<T>;

function longToNumber(long: Long): number {
  if (long.gt(Number.MAX_SAFE_INTEGER)) {
    throw new tsProtoGlobalThis.Error("Value is larger than Number.MAX_SAFE_INTEGER");
  }
  return long.toNumber();
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

export type ServerStreamingMethodResult<Response> = { [Symbol.asyncIterator](): AsyncIterator<Response, void> };
