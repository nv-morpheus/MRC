import {expect} from "@jest/globals";
import {WorkerStates} from "@mrc/proto/mrc/protos/architect_state";
import {
   addPipelineInstance,
   IPipelineInstance,
   removePipelineInstance,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
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
   machineId: 1111,
   workerAddress: stringToBytes("-----"),
   state: WorkerStates.Registered,
   assignedSegmentIds: [],
};

const pipeline: IPipelineInstance = {
   id: 1122,
   definitionId: 1133,
   machineId: connection.id,
   segmentIds: [],
};

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(connectionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(connectionsSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(removeConnection({
         id: connection.id,
      })));
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(addConnection(connection));
   });

   test("Select All", () => {
      const allConnections = connectionsSelectAll(store.getState());

      expect(allConnections).toHaveLength(1);

      expect(allConnections[0]).toHaveProperty("id", connection.id);
      expect(allConnections[0]).toHaveProperty("peerInfo", connection.peerInfo);
      expect(allConnections[0]).toHaveProperty("workerIds", []);
      expect(allConnections[0]).toHaveProperty("assignedPipelineIds", []);
   });

   test("Select One", () => {
      const foundConnection = connectionsSelectById(store.getState(), connection.id);

      expect(foundConnection).toHaveProperty("id", connection.id);
      expect(foundConnection).toHaveProperty("peerInfo", connection.peerInfo);
      expect(foundConnection).toHaveProperty("workerIds", []);
      expect(foundConnection).toHaveProperty("assignedPipelineIds", []);
   });

   test("Total", () => {
      expect(connectionsSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(addConnection({
         id: connection.id,
         peerInfo: connection.peerInfo,
      })));
   });

   it("Remove Valid ID", () => {
      store.dispatch(removeConnection({
         id: connection.id,
      }));

      expect(connectionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() => store.dispatch(removeConnection({
         id: -9999,
      })));
   });

   describe("With Worker", () => {
      beforeEach(() => {
         store.dispatch(addWorker(worker));
      });

      test("Contains Worker ID", () => {
         expect(connectionsSelectById(store.getState(), connection.id)?.workerIds).toContain(worker.id);
      });

      test("Add Duplicate", () => {
         assert.throws(() => {
            store.dispatch(addWorker(worker));
         });
      });

      test("Remove Worker ID", () => {
         store.dispatch(removeWorker(worker));

         expect(connectionsSelectById(store.getState(), connection.id)?.workerIds).not.toContain(worker.id);
         expect(connectionsSelectById(store.getState(), connection.id)?.workerIds).toHaveLength(0);
      });
   });

   describe("With Pipeline", () => {
      beforeEach(() => {
         store.dispatch(addPipelineInstance(pipeline));
      });

      test("Contains Pipeline ID", () => {
         expect(connectionsSelectById(store.getState(), connection.id)?.assignedPipelineIds).toContain(pipeline.id);
      });

      test("Add Duplicate", () => {
         assert.throws(() => {
            store.dispatch(addPipelineInstance(pipeline));
         });
      });

      test("Remove Pipeline ID", () => {
         store.dispatch(removePipelineInstance(pipeline));

         expect(connectionsSelectById(store.getState(), connection.id)?.assignedPipelineIds).not.toContain(pipeline.id);
         expect(connectionsSelectById(store.getState(), connection.id)?.assignedPipelineIds).toHaveLength(0);
      });
   });
});
