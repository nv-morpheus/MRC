import { expect } from "@jest/globals";
import { ResourceActualStatus } from "@mrc/proto/mrc/protos/architect_state";
import { connectionsAdd, connectionsDropOne } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsAdd } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesAdd } from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesAddMany,
   segmentInstancesRemove,
   segmentInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import {
   workersSelectAll,
   workersSelectTotal,
   workersRemove,
   workersAdd,
   workersSelectById,
   workersUpdateResourceActualState,
} from "@mrc/server/store/slices/workersSlice";
import { RootStore, setupStore } from "@mrc/server/store/store";
import { connection, pipeline, pipeline_def, segments, worker } from "@mrc/tests/defaultObjects";
import assert from "assert";

let store: RootStore;

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(workersSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(workersSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(workersRemove(worker)));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(workersAdd(worker));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));
   });

   test("Select All", () => {
      const found = workersSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0].id).toEqual(worker.id);
      expect(found[0].assignedSegmentIds).toEqual([]);
      expect(found[0].machineId).toEqual(connection.id);
      expect(found[0].state.actualStatus).toEqual(ResourceActualStatus.Actual_Unknown);
      expect(found[0].workerAddress).toEqual(worker.workerAddress);
   });

   test("Select One", () => {
      const found = workersSelectById(store.getState(), worker.id);

      expect(found).toBeDefined();

      expect(found?.id).toEqual(worker.id);
      expect(found?.assignedSegmentIds).toEqual([]);
      expect(found?.machineId).toEqual(connection.id);
      expect(found?.state.actualStatus).toEqual(ResourceActualStatus.Actual_Unknown);
      expect(found?.workerAddress).toEqual(worker.workerAddress);
   });

   test("Total", () => {
      expect(workersSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(workersAdd(worker)));
   });

   test("Remove Valid ID", () => {
      store.dispatch(workersRemove(worker));

      expect(workersSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() =>
         store.dispatch(
            workersRemove({
               ...worker,
               id: "9999",
            })
         )
      );
   });

   test("Activate", () => {
      store.dispatch(
         workersUpdateResourceActualState({ resource: worker, status: ResourceActualStatus.Actual_Running })
      );

      expect(workersSelectById(store.getState(), worker.id)?.state.actualStatus).toEqual(
         ResourceActualStatus.Actual_Running
      );
   });

   test("Activate Twice", () => {
      store.dispatch(
         workersUpdateResourceActualState({ resource: worker, status: ResourceActualStatus.Actual_Running })
      );
      store.dispatch(
         workersUpdateResourceActualState({ resource: worker, status: ResourceActualStatus.Actual_Running })
      );

      expect(workersSelectById(store.getState(), worker.id)?.state.actualStatus).toEqual(
         ResourceActualStatus.Actual_Running
      );
   });

   test("Connection Dropped", async () => {
      await store.dispatch(connectionsDropOne({ id: connection.id }));

      expect(workersSelectAll(store.getState())).toHaveLength(0);
   });

   describe("With Segment", () => {
      beforeEach(() => {
         store.dispatch(pipelineDefinitionsAdd(pipeline_def));

         // Add a pipeline and then a segment
         store.dispatch(pipelineInstancesAdd(pipeline));

         // Now add a segment
         store.dispatch(segmentInstancesAddMany(segments));
      });

      test("Contains Segment", () => {
         const found = workersSelectById(store.getState(), worker.id);

         segments.forEach((s) => expect(found?.assignedSegmentIds).toContain(s.id));
      });

      test("Remove Segment", () => {
         segments.forEach((s) =>
            store.dispatch(
               segmentInstancesUpdateResourceActualState({ resource: s, status: ResourceActualStatus.Actual_Destroyed })
            )
         );
         segments.forEach((s) => store.dispatch(segmentInstancesRemove(s)));

         const found = workersSelectById(store.getState(), worker.id);

         segments.forEach((s) => expect(found?.assignedSegmentIds).not.toContain(s.id));

         // Then remove the object to check that we dont get an error
         store.dispatch(workersRemove(worker));

         expect(workersSelectAll(store.getState())).toHaveLength(0);
      });

      test("Remove Before Segment", () => {
         assert.throws(() => {
            // Remove the pipeline with running segments
            store.dispatch(workersRemove(worker));
         });
      });
   });
});
