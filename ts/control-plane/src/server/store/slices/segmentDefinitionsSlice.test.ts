import {expect} from "@jest/globals";
import {
   pipelineDefinitionsAdd,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {
   segmentDefinitionsAdd,
   segmentDefinitionsRemove,
   segmentDefinitionsSelectAll,
   segmentDefinitionsSelectById,
   segmentDefinitionsSelectTotal,
} from "@mrc/server/store/slices/segmentDefinitionsSlice";
import {workersAdd} from "@mrc/server/store/slices/workersSlice";
import {connection, pipeline_def, segment_def, worker} from "@mrc/tests/defaultObjects";
import assert from "assert";

import {RootStore, setupStore} from "../store";

import {
   connectionsAdd,
} from "./connectionsSlice";

let store: RootStore;

// Get a clean store each time
beforeEach(() => {
   store = setupStore();
});

describe("Empty", () => {
   test("Select All", () => {
      expect(segmentDefinitionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Total", () => {
      expect(segmentDefinitionsSelectTotal(store.getState())).toBe(0);
   });

   test("Remove", () => {
      assert.throws(() => store.dispatch(segmentDefinitionsRemove(segment_def)));
   });

   test("Before Connection", () => {
      assert.throws(() => {
         store.dispatch(segmentDefinitionsAdd(segment_def));
      });
   });

   test("Before Worker", () => {
      store.dispatch(connectionsAdd(connection));

      assert.throws(() => {
         store.dispatch(segmentDefinitionsAdd(segment_def));
      });
   });

   test("Before Pipeline", () => {
      store.dispatch(connectionsAdd(connection));

      store.dispatch(workersAdd(worker));

      assert.throws(() => {
         store.dispatch(segmentDefinitionsAdd(segment_def));
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(pipelineDefinitionsAdd(pipeline_def));

      store.dispatch(segmentDefinitionsAdd(segment_def));
   });

   test("Select All", () => {
      const found = segmentDefinitionsSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0]).toHaveProperty("id", segment_def.id);
      expect(found[0]).toHaveProperty("egressPorts", []);
      expect(found[0]).toHaveProperty("ingressPorts", []);
      expect(found[0]).toHaveProperty("instanceIds", []);
      expect(found[0]).toHaveProperty("name", segment_def.name);
      expect(found[0]).not.toHaveProperty("options");
      expect(found[0]).toHaveProperty("pipelineId", pipeline_def.id);
   });

   test("Select One", () => {
      const found = segmentDefinitionsSelectById(store.getState(), segment_def.id);

      expect(found).toHaveProperty("id", segment_def.id);
      expect(found).toHaveProperty("egressPorts", []);
      expect(found).toHaveProperty("ingressPorts", []);
      expect(found).toHaveProperty("instanceIds", []);
      expect(found).toHaveProperty("name", segment_def.name);
      expect(found).not.toHaveProperty("options");
      expect(found).toHaveProperty("pipelineId", pipeline_def.id);
   });

   test("Total", () => {
      expect(segmentDefinitionsSelectTotal(store.getState())).toBe(1);
   });

   test("Add Duplicate", () => {
      assert.throws(() => store.dispatch(segmentDefinitionsAdd(segment_def)));
   });

   it("Remove Valid ID", () => {
      store.dispatch(segmentDefinitionsRemove(segment_def));

      expect(segmentDefinitionsSelectAll(store.getState())).toHaveLength(0);
   });

   test("Remove Unknown ID", () => {
      assert.throws(() => store.dispatch(segmentDefinitionsRemove({
         ...segment_def,
         id: -9999,
      })));
   });
});
