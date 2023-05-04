import {stringToBytes} from "@mrc/common/utils";
import {SegmentStates, WorkerStates} from "@mrc/proto/mrc/protos/architect_state";
import {IConnection} from "@mrc/server/store/slices/connectionsSlice";
import {IPipelineInstance} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {ISegmentInstance} from "@mrc/server/store/slices/segmentInstancesSlice";
import {IWorker} from "@mrc/server/store/slices/workersSlice";

export const connection: IConnection = {
   id: 1111,
   peerInfo: "localhost:1234",
   workerIds: [],
   assignedPipelineIds: [],
};

export const worker: IWorker = {
   id: 1234,
   machineId: 1111,
   workerAddress: stringToBytes("-----"),
   state: WorkerStates.Registered,
   assignedSegmentIds: [],
};

export const pipeline: IPipelineInstance = {
   id: 1122,
   definitionId: 1133,
   machineId: connection.id,
   segmentIds: [],
};

export const segment: ISegmentInstance = {
   id: 1123,
   address: 2222,
   definitionId: 0,
   pipelineId: pipeline.id,
   workerId: worker.id,
   state: SegmentStates.Initialized,
};
