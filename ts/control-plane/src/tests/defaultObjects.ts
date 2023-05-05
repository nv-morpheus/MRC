import {hashProtoMessage, stringToBytes} from "@mrc/common/utils";
import {
   PipelineConfiguration,
   PipelineConfiguration_SegmentConfiguration,
   SegmentStates,
   WorkerStates,
} from "@mrc/proto/mrc/protos/architect_state";
import {IConnection} from "@mrc/server/store/slices/connectionsSlice";
import {
   IPipelineConfiguration,
   IPipelineDefinition,
   ISegmentDefinition,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {IPipelineInstance} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {ISegmentInstance} from "@mrc/server/store/slices/segmentInstancesSlice";
import {IWorker} from "@mrc/server/store/slices/workersSlice";
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
   state: WorkerStates.Registered,
   assignedSegmentIds: [],
};

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

export const pipeline_def: IPipelineDefinition = {
   id: pipeline_config_hash,
   instanceIds: [],
   config: pipeline_config,
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
   segmentIds: [],
};

export const segments: ISegmentInstance[] = Object.entries(pipeline_def.segments).map(([seg_name, seg_def]) => {
   return {
      id: generateId(),
      pipelineDefinitionId: seg_def.parentId,
      pipelineInstanceId: pipeline.id,
      name: seg_name,
      address: 2222,
      workerId: worker.id,
      pipelineId: pipeline.id,
      state: SegmentStates.Initialized,
   } as ISegmentInstance;
});

export const segments_map = Object.fromEntries(segments.map((s) => [s.name, s]));
