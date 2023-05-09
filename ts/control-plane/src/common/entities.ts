import {PipelineRequestAssignmentRequest_SegmentMapping} from "@mrc/proto/mrc/protos/architect";
import {
   Connection,
   EgressPort,
   IngressPort,
   PipelineConfiguration,
   PipelineConfiguration_SegmentConfiguration,
   PipelineDefinition,
   PipelineDefinition_SegmentDefinition,
   PipelineInstance,
   ResourceState,
   ScalingOptions,
   SegmentInstance,
   SegmentOptions,
   Worker,
} from "@mrc/proto/mrc/protos/architect_state";

export type IResourceState = Omit<ResourceState, "$type">;

export type IConnection = Omit<Connection, "$type">;

export type IWorker = Omit<Worker, "$type"|"state">&{
   state: IResourceState,
};

export type IIngressPort    = Omit<IngressPort, "$type">;
export type IEgressPort     = Omit<EgressPort, "$type">;
export type IScalingOptions = Omit<ScalingOptions, "$type">;
export type ISegmentOptions = Omit<SegmentOptions, "$type">&{
   scalingOptions?: IScalingOptions,
};

export type ISegmentConfiguration =
    Omit<PipelineConfiguration_SegmentConfiguration, "$type"|"ingressPorts"|"egressPorts"|"options">&{
       ingressPorts: IIngressPort[],
       egressPorts: IEgressPort[],
       options?: ISegmentOptions,
    };

export type IPipelineConfiguration = Omit<PipelineConfiguration, "$type"|"segments">&{
   segments: {[key: string]: ISegmentConfiguration},
};

export type ISegmentDefinition = Omit<PipelineDefinition_SegmentDefinition, "$type">;

export type IPipelineDefinition = Omit<PipelineDefinition, "$type"|"config"|"segments">&{
   config: IPipelineConfiguration,
   segments: {[key: string]: ISegmentDefinition},
};

export type ISegmentMapping = Omit<PipelineRequestAssignmentRequest_SegmentMapping, "$type">;

export type IPipelineInstance = Omit<PipelineInstance, "$type"|"state">&{
   state: IResourceState,
};

export type ISegmentInstance = Omit<SegmentInstance, "$type">;
