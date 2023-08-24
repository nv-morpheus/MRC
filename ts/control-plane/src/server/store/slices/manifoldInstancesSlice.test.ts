import { expect } from "@jest/globals";
import { ResourceActualStatus } from "@mrc/proto/mrc/protos/architect_state";
import {
   pipelineDefinitionsAdd,
   pipelineDefinitionsSelectById,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesAdd } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { workersAdd } from "@mrc/server/store/slices/workersSlice";
import {
   connection,
   manifolds,
   manifolds_map,
   pipeline,
   pipeline_def,
   segments,
   worker,
} from "@mrc/tests/defaultObjects";
import assert from "assert";

import {
   manifoldInstancesAdd,
   manifoldInstancesAddMany,
   manifoldInstancesRemove,
   manifoldInstancesSelectAll,
   manifoldInstancesSelectById,
   manifoldInstancesSelectTotal,
   manifoldInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/manifoldInstancesSlice";
import { segmentInstancesAddMany } from "@mrc/server/store/slices/segmentInstancesSlice";
import { connectionsAdd, connectionsDropOne } from "@mrc/server/store/slices/connectionsSlice";
import { RootStore, setupStore } from "@mrc/server/store/store";

let store: RootStore;

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(manifoldInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(manifoldInstancesSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(manifoldInstancesRemove(manifolds[0])));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(manifoldInstancesAdd(manifolds[0]));
      });
   });

   test("Before Pipeline", () => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      assert.throws(() => {
         store.dispatch(manifoldInstancesAdd(manifolds[0]));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      store.dispatch(pipelineDefinitionsAdd(pipeline_def));

      store.dispatch(pipelineInstancesAdd(pipeline));

      store.dispatch(manifoldInstancesAddMany(manifolds));
   });

   test("Select All", () => {
      const found = manifoldInstancesSelectAll(store.getState());

      expect(found).toHaveLength(manifolds.length);

      found.forEach((m) => {
         expect(m.actualInputSegments).toEqual(manifolds_map[m.portName].actualInputSegments);
         expect(m.actualOutputSegments).toEqual(manifolds_map[m.portName].actualOutputSegments);
         expect(m.id).toEqual(manifolds_map[m.portName].id);
         expect(m.machineId).toEqual(manifolds_map[m.portName].machineId);
         expect(m.pipelineDefinitionId).toEqual(pipeline_def.id);
         expect(m.pipelineInstanceId).toEqual(pipeline.id);
         expect(m.portName).toEqual(manifolds_map[m.portName].portName);
         expect(m.requestedInputSegments).toEqual(manifolds_map[m.portName].requestedInputSegments);
         expect(m.requestedOutputSegments).toEqual(manifolds_map[m.portName].requestedOutputSegments);
         expect(m.state.actualStatus).toEqual(ResourceActualStatus.Actual_Unknown);
      });
   });

   test("Total", () => {
      expect(manifoldInstancesSelectTotal(store.getState())).toBe(manifolds.length);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(manifoldInstancesAdd(manifolds[0])));
   });

   test("Update State", () => {
      for (const s of [
         ResourceActualStatus.Actual_Creating,
         ResourceActualStatus.Actual_Created,
         ResourceActualStatus.Actual_Running,
         ResourceActualStatus.Actual_Stopping,
         ResourceActualStatus.Actual_Stopped,
         ResourceActualStatus.Actual_Destroying,
         ResourceActualStatus.Actual_Destroyed,
      ]) {
         store.dispatch(manifoldInstancesUpdateResourceActualState({ resource: manifolds[0], status: s }));

         expect(manifoldInstancesSelectById(store.getState(), manifolds[0].id)?.state.actualStatus).toBe(s);
      }
   });

   test("Update State Backwards", () => {
      // Set the state running first
      store.dispatch(
         manifoldInstancesUpdateResourceActualState({
            resource: manifolds[0],
            status: ResourceActualStatus.Actual_Running,
         })
      );

      // Try to set it back to initialized
      assert.throws(() =>
         store.dispatch(
            manifoldInstancesUpdateResourceActualState({
               resource: manifolds[0],
               status: ResourceActualStatus.Actual_Creating,
            })
         )
      );
   });

   it("Remove Valid ID", () => {
      // Set the instance to completed first
      store.dispatch(
         manifoldInstancesUpdateResourceActualState({
            resource: manifolds[0],
            status: ResourceActualStatus.Actual_Destroyed,
         })
      );

      store.dispatch(manifoldInstancesRemove(manifolds[0]));

      expect(manifoldInstancesSelectAll(store.getState())).toHaveLength(manifolds.length - 1);
   });

   test("Remove Unknown ID", () => {
      // Set the instance to completed first
      store.dispatch(
         manifoldInstancesUpdateResourceActualState({
            resource: manifolds[0],
            status: ResourceActualStatus.Actual_Destroyed,
         })
      );

      assert.throws(() =>
         store.dispatch(
            manifoldInstancesRemove({
               ...manifolds[0],
               id: "9999",
            })
         )
      );
   });

   test("Drop Connection", async () => {
      await store.dispatch(connectionsDropOne({ id: connection.id }));

      expect(manifoldInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   describe("With SegmentInstances", () => {
      beforeEach(() => {
         // Add the segment instances
         store.dispatch(segmentInstancesAddMany(segments));
      });

      test("Contains Instance", () => {
         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expect(found?.instanceIds).toContain(pipeline.id);
      });
   });
});
