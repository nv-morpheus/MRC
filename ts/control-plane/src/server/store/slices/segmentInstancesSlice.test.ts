import {expect} from "@jest/globals";
import {SegmentStates} from "@mrc/proto/mrc/protos/architect_state";
import {
   pipelineInstancesAdd,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesAdd,
   segmentInstancesRemove,
   segmentInstancesSelectAll,
   segmentInstancesSelectById,
   segmentInstancesSelectTotal,
   segmentInstancesUpdateState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import {workersAdd} from "@mrc/server/store/slices/workersSlice";
import {connection, pipeline, segment, worker} from "@mrc/tests/defaultObjects";
import assert from "assert";

import {RootStore, setupStore} from "../store";

import {
   connectionsAdd,
   connectionsDropOne,
} from "./connectionsSlice";

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
      assert.throws(() => store.dispatch(segmentInstancesRemove(segment)));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segment));
      });
   });

   test("Before Worker", () => {
      store.dispatch(connectionsAdd(connection));

      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segment));
      });
   });

   test("Before Pipeline", () => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      assert.throws(() => {
         store.dispatch(segmentInstancesAdd(segment));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      store.dispatch(pipelineInstancesAdd(pipeline));

      store.dispatch(segmentInstancesAdd(segment));
   });

   test("Select All", () => {
      const found = segmentInstancesSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0]).toHaveProperty("id", segment.id);
      expect(found[0]).toHaveProperty("address", segment.address);
      expect(found[0]).toHaveProperty("definitionId", 0);
      expect(found[0]).toHaveProperty("pipelineId", pipeline.id);
      expect(found[0]).toHaveProperty("workerId", worker.id);
   });

   test("Select One", () => {
      const found = segmentInstancesSelectById(store.getState(), segment.id);

      expect(found).toHaveProperty("id", segment.id);
      expect(found).toHaveProperty("address", segment.address);
      expect(found).toHaveProperty("definitionId", 0);
      expect(found).toHaveProperty("pipelineId", pipeline.id);
      expect(found).toHaveProperty("workerId", worker.id);
   });

   test("Total", () => {
      expect(segmentInstancesSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(segmentInstancesAdd(segment)));
   });

   test("Update State", () => {
      for (const s of [SegmentStates.Running, SegmentStates.Stopped, SegmentStates.Completed])
      {
         store.dispatch(segmentInstancesUpdateState({id: segment.id, state: s}));

         expect(segmentInstancesSelectById(store.getState(), segment.id)).toHaveProperty("state", s);
      }
   });

   test("Update State Backwards", () => {
      // Set the state running first
      store.dispatch(segmentInstancesUpdateState({id: segment.id, state: SegmentStates.Running}));

      // Try to set it back to initialized
      assert.throws(
          () => store.dispatch(segmentInstancesUpdateState({id: segment.id, state: SegmentStates.Initialized})));
   });

   it("Remove Valid ID", () => {
      // Set the instance to completed first
      store.dispatch(segmentInstancesUpdateState({id: segment.id, state: SegmentStates.Completed}));

      store.dispatch(segmentInstancesRemove(segment));

      expect(segmentInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      // Set the instance to completed first
      store.dispatch(segmentInstancesUpdateState({id: segment.id, state: SegmentStates.Completed}));

      assert.throws(() => store.dispatch(segmentInstancesRemove({
         ...segment,
         id: -9999,
      })));
   });

   test("Drop Connection", () => {
      store.dispatch(connectionsDropOne({id: connection.id}));

      expect(segmentInstancesSelectAll(store.getState())).toHaveLength(0);
   });
});
