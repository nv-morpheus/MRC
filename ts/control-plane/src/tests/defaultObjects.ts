import {
   IConnection,
   IPipelineConfiguration,
   IPipelineDefinition,
   IPipelineInstance,
   IPipelineMapping,
   ISegmentDefinition,
   ISegmentInstance,
   ISegmentMapping,
   IWorker,
} from "@mrc/common/entities";
import {generateSegmentHash, hashProtoMessage, stringToBytes} from "@mrc/common/utils";
import {
   PipelineConfiguration,
   PipelineConfiguration_SegmentConfiguration,
   ResourceStatus,
   SegmentStates,
} from "@mrc/proto/mrc/protos/architect_state";
import {generateId} from "@mrc/server/utils";

export const connection: IConnection = {
   id: generateId(),
   peerInfo: "localhost:1234",
   workerIds: [],
   assignedPipelineIds: [],
};

export const worker: IWorker = {
   id: generateId(),
   machineId: connection.id,
   workerAddress: stringToBytes("-----"),
   state: {
      status: ResourceStatus.Registered,
      refCount: 0,
   },
   assignedSegmentIds: [],
};

export const workers: IWorker[] = [worker];

export const pipeline_config: IPipelineConfiguration = {
   segments: {
      seg1: {
         name: "seg1",
         egressPorts: [],
         ingressPorts: [],
      },
      seg2: {
         name: "seg2",
         egressPorts: [],
         ingressPorts: [],
      },
   },
};

const pipeline_config_hash = hashProtoMessage(PipelineConfiguration.create(pipeline_config));

export const pipeline_mappings = Object.fromEntries(workers.map(
    (w) => {return [
       w.machineId,
       {
          machineId: w.machineId,
          segments: Object.fromEntries(Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
             return [
                seg_name,
                {segmentName: seg_name, byWorker: {workerIds: [worker.id]}} as ISegmentMapping,
             ];
          })),
       } as IPipelineMapping,
    ]}));

export const pipeline_def: IPipelineDefinition = {
   id: pipeline_config_hash,
   config: pipeline_config,
   mappings: pipeline_mappings,
   instanceIds: [],
   segments: Object.fromEntries(Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
      return [
         seg_name,
         {
            id: hashProtoMessage(PipelineConfiguration_SegmentConfiguration.create(seg_config)),
            parentId: pipeline_config_hash,
            name: seg_name,
            instanceIds: [],
         } as ISegmentDefinition,
      ];
   })),
};

export const pipeline: IPipelineInstance = {
   id: generateId(),
   definitionId: pipeline_def.id,
   machineId: connection.id,
   state: {
      status: ResourceStatus.Registered,
      refCount: 0,
   },
   segmentIds: [],
};

export const segments: ISegmentInstance[] = Object.entries(pipeline_def.segments).map(([seg_name, seg_def]) => {
   const address = generateSegmentHash(seg_name, worker.id);

   return {
      id: address.toString(),
      pipelineDefinitionId: seg_def.parentId,
      pipelineInstanceId: pipeline.id,
      name: seg_name,
      address: address,
      workerId: worker.id,
      pipelineId: pipeline.id,
      state: SegmentStates.Initialized,
   } as ISegmentInstance;
});

export const segments_map = Object.fromEntries(segments.map((s) => [s.name, s]));
