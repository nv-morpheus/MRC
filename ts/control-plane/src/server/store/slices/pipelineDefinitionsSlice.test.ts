import {expect} from "@jest/globals";
import {SegmentStates} from "@mrc/proto/mrc/protos/architect_state";
import {
   pipelineDefinitionsAdd,
   pipelineDefinitionsRemove,
   pipelineDefinitionsSelectAll,
   pipelineDefinitionsSelectById,
   pipelineDefinitionsSelectTotal,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {
   segmentDefinitionsAdd,
   segmentDefinitionsRemove,
} from "@mrc/server/store/slices/segmentDefinitionsSlice";
import {connection, pipeline, pipeline_def, segment, segment_def, worker} from "@mrc/tests/defaultObjects";
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
      expect(pipelineDefinitionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(pipelineDefinitionsSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(pipelineDefinitionsRemove(pipeline_def)));
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(pipelineDefinitionsAdd(pipeline_def));
   });

   test("Select All", () => {
      const found = pipelineDefinitionsSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0]).toHaveProperty("id", pipeline_def.id);
      expect(found[0]).toHaveProperty("instanceIds", []);
      expect(found[0]).toHaveProperty("segmentIds", []);
   });

   test("Select One", () => {
      const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

      expect(found).toHaveProperty("id", pipeline_def.id);
      expect(found).toHaveProperty("instanceIds", []);
      expect(found).toHaveProperty("segmentIds", []);
   });

   test("Total", () => {
      expect(pipelineDefinitionsSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(pipelineDefinitionsAdd(pipeline_def)));
   });

   it("Remove Valid ID", () => {
      store.dispatch(pipelineDefinitionsRemove(pipeline_def));

      expect(pipelineDefinitionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() => store.dispatch(pipelineDefinitionsRemove({
         ...pipeline_def,
         id: -9999,
      })));
   });

   describe("With Segment", () => {
      beforeEach(() => {
         // Now add a segment
         store.dispatch(segmentDefinitionsAdd(segment_def));
      });

      test("Contains Segment", () => {
         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expect(found?.segmentIds).toContain(segment_def.id);
      });

      test("Remove Segment", () => {
         store.dispatch(segmentDefinitionsRemove(segment_def));

         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expect(found?.segmentIds).not.toContain(segment_def.id);

         // Then remove the pipeline
         store.dispatch(pipelineDefinitionsRemove(pipeline_def));

         expect(pipelineDefinitionsSelectAll(store.getState())).toHaveLength(0);
      });

      test("Remove Pipeline Before Segment", () => {
         assert.throws(() => {
            // Remove the pipeline with running segments
            store.dispatch(pipelineDefinitionsRemove(pipeline_def));
         });
      });
   });
});
