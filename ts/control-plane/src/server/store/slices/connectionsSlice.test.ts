import { expect } from "@jest/globals";
import { pipelineDefinitionsAdd } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesAdd, pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { executor, pipeline, pipeline_def, worker } from "@mrc/tests/defaultObjects";
import assert from "assert";

import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import {
   connectionsSelectAll,
   connectionsSelectTotal,
   connectionsRemove,
   connectionsAdd,
   connectionsSelectById,
} from "@mrc/server/store/slices/connectionsSlice";
import { workersAdd, workersRemove } from "@mrc/server/store/slices/workersSlice";
import { RootStore, setupStore } from "@mrc/server/store/store";

let store: RootStore;

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
      assert.throws(() => store.dispatch(connectionsRemove(executor)));
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(executor));
   });

   test("Select All", () => {
      const allConnections = connectionsSelectAll(store.getState());

      expect(allConnections).toHaveLength(1);

      expect(allConnections[0]).toHaveProperty("id", executor.id);
      expect(allConnections[0]).toHaveProperty("peerInfo", executor.peerInfo);
      expect(allConnections[0]).toHaveProperty("workerIds", []);
      expect(allConnections[0]).toHaveProperty("assignedPipelineIds", []);
   });

   test("Select One", () => {
      const foundConnection = connectionsSelectById(store.getState(), executor.id);

      expect(foundConnection).toHaveProperty("id", executor.id);
      expect(foundConnection).toHaveProperty("peerInfo", executor.peerInfo);
      expect(foundConnection).toHaveProperty("workerIds", []);
      expect(foundConnection).toHaveProperty("assignedPipelineIds", []);
   });

   test("Total", () => {
      expect(connectionsSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(connectionsAdd(executor)));
   });

   it("Remove Valid ID", () => {
      store.dispatch(connectionsRemove(executor));

      expect(connectionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() =>
         store.dispatch(
            connectionsRemove({
               ...executor,
               id: "9999",
            })
         )
      );
   });

   describe("With Worker", () => {
      beforeEach(() => {
         store.dispatch(workersAdd(worker));
      });

      test("Contains Worker ID", () => {
         expect(connectionsSelectById(store.getState(), executor.id)?.workerIds).toContain(worker.id);
      });

      test("Add Duplicate", () => {
         assert.throws(() => {
            store.dispatch(workersAdd(worker));
         });
      });

      test("Remove Worker ID", () => {
         store.dispatch(workersRemove(worker));

         expect(connectionsSelectById(store.getState(), executor.id)?.workerIds).not.toContain(worker.id);
         expect(connectionsSelectById(store.getState(), executor.id)?.workerIds).toHaveLength(0);
      });

      test("Remove Connection First", () => {
         assert.throws(() => {
            store.dispatch(connectionsRemove(executor));
         });
      });
   });

   describe("With Pipeline", () => {
      beforeEach(() => {
         store.dispatch(pipelineDefinitionsAdd(pipeline_def));

         store.dispatch(pipelineInstancesAdd(pipeline));
      });

      test("Contains Pipeline ID", () => {
         expect(connectionsSelectById(store.getState(), executor.id)?.assignedPipelineIds).toContain(pipeline.id);
      });

      test("Add Duplicate", () => {
         assert.throws(() => {
            store.dispatch(pipelineInstancesAdd(pipeline));
         });
      });

      test("Remove Pipeline ID", () => {
         store.dispatch(pipelineInstancesRemove(pipeline));

         expect(connectionsSelectById(store.getState(), executor.id)?.assignedPipelineIds).not.toContain(pipeline.id);
         expect(connectionsSelectById(store.getState(), executor.id)?.assignedPipelineIds).toHaveLength(0);
      });

      test("Remove Connection First", () => {
         assert.throws(() => {
            store.dispatch(connectionsRemove(executor));
         });
      });
   });
});
