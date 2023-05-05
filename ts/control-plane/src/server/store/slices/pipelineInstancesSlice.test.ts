import {expect} from "@jest/globals";
import {SegmentStates} from "@mrc/proto/mrc/protos/architect_state";
import {pipelineDefinitionsAdd} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {
   pipelineInstancesAdd,
   pipelineInstancesRemove,
   pipelineInstancesSelectAll,
   pipelineInstancesSelectById,
   pipelineInstancesSelectTotal,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesAdd,
   segmentInstancesAddMany,
   segmentInstancesRemove,
   segmentInstancesUpdateState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import {connection, pipeline, pipeline_def, segments, worker} from "@mrc/tests/defaultObjects";
import assert from "assert";

import {RootStore, setupStore} from "../store";

import {
   connectionsAdd,
   connectionsDropOne,
} from "./connectionsSlice";
import {
   workersAdd,
} from "./workersSlice";

let store: RootStore;

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
      assert.throws(() => store.dispatch(pipelineInstancesRemove(pipeline)));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(pipelineInstancesAdd(pipeline));
      });
   });

   test("Before Definition", () => {
      store.dispatch(connectionsAdd(connection));

      assert.throws(() => {
         store.dispatch(pipelineInstancesAdd(pipeline));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(pipelineDefinitionsAdd(pipeline_def));

      store.dispatch(pipelineInstancesAdd(pipeline));
   });

   test("Select All", () => {
      const found = pipelineInstancesSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0]).toHaveProperty("id", pipeline.id);
      expect(found[0]).toHaveProperty("definitionId", pipeline.definitionId);
      expect(found[0]).toHaveProperty("machineId", pipeline.machineId);
      expect(found[0]).toHaveProperty("segmentIds", []);
   });

   test("Select One", () => {
      const found = pipelineInstancesSelectById(store.getState(), pipeline.id);

      expect(found).toHaveProperty("id", pipeline.id);
      expect(found).toHaveProperty("definitionId", pipeline.definitionId);
      expect(found).toHaveProperty("machineId", pipeline.machineId);
      expect(found).toHaveProperty("segmentIds", []);
   });

   test("Total", () => {
      expect(pipelineInstancesSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(pipelineInstancesAdd(pipeline)));
   });

   it("Remove Valid ID", () => {
      store.dispatch(pipelineInstancesRemove(pipeline));

      expect(pipelineInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() => store.dispatch(pipelineInstancesRemove({
         ...pipeline,
         id: -9999,
      })));
   });

   test("Remove Incorrect Machine ID", () => {
      assert.throws(() => store.dispatch(pipelineInstancesRemove({
         ...pipeline,
         machineId: 1,
      })));
   });

   test("Drop Connection", () => {
      store.dispatch(connectionsDropOne({id: connection.id}));

      expect(pipelineInstancesSelectAll(store.getState())).toHaveLength(0);
   });

   describe("With Segment Instance", () => {
      beforeEach(() => {
         // Add a worker first, then a segment
         store.dispatch(workersAdd(worker));

         // Now add a segment
         store.dispatch(segmentInstancesAddMany(segments));
      });

      test("Contains Instance", () => {
         const found = pipelineInstancesSelectById(store.getState(), pipeline.id);

         segments.forEach((s) => expect(found?.segmentIds).toContain(s.id));
      });

      test("Remove Segment", () => {
         segments.forEach(
             (s) => store.dispatch(segmentInstancesUpdateState({id: s.id, state: SegmentStates.Completed})));
         segments.forEach((s) => store.dispatch(segmentInstancesRemove(s)));

         const found = pipelineInstancesSelectById(store.getState(), pipeline.id);

         segments.forEach((s) => expect(found?.segmentIds).not.toContain(s.id));

         // Then remove the pipeline
         store.dispatch(pipelineInstancesRemove(pipeline));

         expect(pipelineInstancesSelectAll(store.getState())).toHaveLength(0);
      });

      test("Remove Pipeline Before Segment", () => {
         assert.throws(() => {
            // Remove the pipeline with running segments
            store.dispatch(pipelineInstancesRemove(pipeline));
         });
      });
   });
});
