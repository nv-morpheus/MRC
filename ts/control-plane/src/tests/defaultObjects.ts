import {stringToBytes} from "@mrc/common/utils";
import {SegmentOptions_PlacementStrategy, SegmentStates, WorkerStates} from "@mrc/proto/mrc/protos/architect_state";
import {IConnection} from "@mrc/server/store/slices/connectionsSlice";
import {IPipelineDefinition} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {IPipelineInstance} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {ISegmentDefinition} from "@mrc/server/store/slices/segmentDefinitionsSlice";
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

export const pipeline_def: IPipelineDefinition = {
   id: generateId(),
   instanceIds: [],
   segmentIds: [],
};

export const pipeline: IPipelineInstance = {
   id: generateId(),
   definitionId: pipeline_def.id,
   machineId: connection.id,
   segmentIds: [],
};

export const segment_def: ISegmentDefinition = {
   id: generateId(),
   egressPorts: [],
   ingressPorts: [],
   instanceIds: [],
   name: "my_segment",
   pipelineId: pipeline_def.id,
};

export const segment: ISegmentInstance = {
   id: generateId(),
   address: 2222,
   definitionId: segment_def.id,
   pipelineId: pipeline.id,
   workerId: worker.id,
   state: SegmentStates.Initialized,
};
