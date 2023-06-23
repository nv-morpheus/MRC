import { expect } from "@jest/globals";
import { ResourceActualStatus } from "@mrc/proto/mrc/protos/architect_state";
import { connectionsAdd, connectionsDropOne } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsAdd } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesAdd } from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesAdd,
   segmentInstancesAddMany,
   segmentInstancesRemove,
   segmentInstancesSelectAll,
   segmentInstancesSelectById,
   segmentInstancesSelectTotal,
   segmentInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { workersAdd } from "@mrc/server/store/slices/workersSlice";
import { RootStore, setupStore } from "@mrc/server/store/store";
import { connection, pipeline, pipeline_def, segments, segments_map, worker } from "@mrc/tests/defaultObjects";
import assert from "assert";

let store: RootStore;

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(segmentInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(segmentInstancesSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(segmentInstancesRemove(segments[0])));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segments[0]));
      });
   });

   test("Before Worker", () => {
      store.dispatch(connectionsAdd(connection));

      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segments[0]));
      });
   });

   test("Before Pipeline", () => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segments[0]));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      store.dispatch(pipelineDefinitionsAdd(pipeline_def));

      store.dispatch(pipelineInstancesAdd(pipeline));

      store.dispatch(segmentInstancesAddMany(segments));
   });

   test("Select All", () => {
      const found = segmentInstancesSelectAll(store.getState());

      expect(found).toHaveLength(segments.length);

      found.forEach((s) => {
         expect(s.id).toEqual(segments_map[s.name].id);
         expect(s.address).toEqual(segments_map[s.name].address);
         expect(s.name).toEqual(segments_map[s.name].name);
         expect(s.pipelineDefinitionId).toEqual(pipeline_def.id);
         expect(s.pipelineInstanceId).toEqual(pipeline.id);
         expect(s.state.actualStatus).toEqual(ResourceActualStatus.Actual_Unknown);
         expect(s.workerId).toEqual(worker.id);
      });
   });

   test("Total", () => {
      expect(segmentInstancesSelectTotal(store.getState())).toBe(segments.length);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(segmentInstancesAdd(segments[0])));
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
         store.dispatch(segmentInstancesUpdateResourceActualState({ resource: segments[0], status: s }));

         expect(segmentInstancesSelectById(store.getState(), segments[0].id)?.state.actualStatus).toBe(s);
      }
   });

   test("Update State Backwards", () => {
      // Set the state running first
      store.dispatch(
         segmentInstancesUpdateResourceActualState({
            resource: segments[0],
            status: ResourceActualStatus.Actual_Running,
         })
      );

      // Try to set it back to initialized
      assert.throws(() =>
         store.dispatch(
            segmentInstancesUpdateResourceActualState({
               resource: segments[0],
               status: ResourceActualStatus.Actual_Creating,
            })
         )
      );
   });

   it("Remove Valid ID", () => {
      // Set the instance to completed first
      store.dispatch(
         segmentInstancesUpdateResourceActualState({
            resource: segments[0],
            status: ResourceActualStatus.Actual_Destroyed,
         })
      );

      store.dispatch(segmentInstancesRemove(segments[0]));

      expect(segmentInstancesSelectAll(store.getState())).toHaveLength(segments.length - 1);
   });

   test("Remove Unknown ID", () => {
      // Set the instance to completed first
      store.dispatch(
         segmentInstancesUpdateResourceActualState({
            resource: segments[0],
            status: ResourceActualStatus.Actual_Destroyed,
         })
      );

      assert.throws(() =>
         store.dispatch(
            segmentInstancesRemove({
               ...segments[0],
               id: "9999",
            })
         )
      );
   });

   test("Drop Connection", async () => {
      await store.dispatch(connectionsDropOne({ id: connection.id }));

      expect(segmentInstancesSelectAll(store.getState())).toHaveLength(0);
   });
});
