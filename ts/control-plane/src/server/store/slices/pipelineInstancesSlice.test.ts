import {expect} from "@jest/globals";
import {
   addPipelineInstance,
   IPipelineInstance,
   pipelineInstancesSelectAll,
   pipelineInstancesSelectById,
   pipelineInstancesSelectTotal,
   removePipelineInstance,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   addSegmentInstance,
   ISegmentInstance,
   removeSegmentInstance,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import assert from "assert";

import {stringToBytes} from "../../../common/utils";
import {RootStore, setupStore} from "../store";

import {
   addConnection,
   connectionsSelectAll,
   connectionsSelectById,
   connectionsSelectTotal,
   IConnection,
   removeConnection,
} from "./connectionsSlice";
import {
   activateWorkers,
   addWorker,
   IWorker,
   removeWorker,
   workersSelectAll,
   workersSelectById,
   workersSelectTotal,
} from "./workersSlice";

let store: RootStore;

const connection: IConnection = {
   id: 1111,
   peerInfo: "localhost:1234",
   workerIds: [],
   assignedPipelineIds: [],
};

const worker: IWorker = {
   id: 1234,
   activated: false,
   machineId: 1111,
   workerAddress: stringToBytes("-----"),
   assignedSegmentIds: [],
};

const pipeline: IPipelineInstance = {
   id: 1122,
   definitionId: 1133,
   machineId: connection.id,
   segmentIds: [],
};

const segment: ISegmentInstance = {
   id: 1123,
   address: 2222,
   definitionId: 0,
   pipelineId: pipeline.id,
   workerId: worker.id,
};

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(pipelineInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(pipelineInstancesSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(removePipelineInstance({id: pipeline.id, machineId: pipeline.machineId})));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(addPipelineInstance(pipeline));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(addConnection(connection));

      store.dispatch(addPipelineInstance(pipeline));
   });

   test("Select All", () => {
      const allPipelines = pipelineInstancesSelectAll(store.getState());

      expect(allPipelines).toHaveLength(1);

      expect(allPipelines[0]).toHaveProperty("id", pipeline.id);
      expect(allPipelines[0]).toHaveProperty("definitionId", pipeline.definitionId);
      expect(allPipelines[0]).toHaveProperty("machineId", pipeline.machineId);
      expect(allPipelines[0]).toHaveProperty("segmentIds", []);
   });

   test("Select One", () => {
      const foundPipeline = pipelineInstancesSelectById(store.getState(), pipeline.id);

      expect(foundPipeline).toHaveProperty("id", pipeline.id);
      expect(foundPipeline).toHaveProperty("definitionId", pipeline.definitionId);
      expect(foundPipeline).toHaveProperty("machineId", pipeline.machineId);
      expect(foundPipeline).toHaveProperty("segmentIds", []);
   });

   test("Total", () => {
      expect(pipelineInstancesSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(addPipelineInstance(pipeline)));
   });

   it("Remove Valid ID", () => {
      store.dispatch(removePipelineInstance({
         id: pipeline.id,
         machineId: pipeline.machineId,
      }));

      expect(pipelineInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() => store.dispatch(removePipelineInstance({
         id: -9999,
         machineId: 1,
      })));
   });

   test("Remove Incorrect Machine ID", () => {
      assert.throws(() => store.dispatch(removePipelineInstance({
         id: pipeline.id,
         machineId: 1,
      })));
   });

   describe("With Segment", () => {
      beforeEach(() => {
         // Add a worker first, then a segment
         store.dispatch(addWorker(worker));

         // Now add a segment
         store.dispatch(addSegmentInstance(segment));
      });

      test("Contains Segment ID", () => {
         const foundPipeline = pipelineInstancesSelectById(store.getState(), pipeline.id);

         expect(foundPipeline?.segmentIds).toContain(segment.id);
      });

      test("Remove Segment ID", () => {
         store.dispatch(removeSegmentInstance(segment));

         const foundPipeline = pipelineInstancesSelectById(store.getState(), pipeline.id);

         expect(foundPipeline?.segmentIds).not.toContain(segment.id);
      });
   });
});
