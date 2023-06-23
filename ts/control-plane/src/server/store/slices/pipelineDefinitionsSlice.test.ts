/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { expect } from "@jest/globals";
import { ResourceActualStatus, ResourceStatus } from "@mrc/proto/mrc/protos/architect_state";
import { connectionsAdd } from "@mrc/server/store/slices/connectionsSlice";
import {
   pipelineDefinitionsAdd,
   pipelineDefinitionsCreateOrUpdate,
   pipelineDefinitionsRemove,
   pipelineDefinitionsSelectAll,
   pipelineDefinitionsSelectById,
   pipelineDefinitionsSelectTotal,
   pipelineDefinitionsSetMapping,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesAdd, pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesAddMany,
   segmentInstancesRemove,
   segmentInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { workersAdd, workersAddMany, workersRemove } from "@mrc/server/store/slices/workersSlice";
import {
   connection,
   pipeline,
   pipeline_config,
   pipeline_def,
   pipeline_mappings,
   segments,
   worker,
   workers,
} from "@mrc/tests/defaultObjects";
import assert from "assert";

import { expectDefined } from "@mrc/tests/utils";
import { RootStore, setupStore } from "@mrc/server/store/store";

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

describe("From Config", () => {
   let created_def_id: string;

   beforeEach(() => {
      created_def_id = store.dispatch(pipelineDefinitionsCreateOrUpdate(pipeline_config)).id;
   });

   test("Select One", () => {
      const found = pipelineDefinitionsSelectById(store.getState(), created_def_id);

      expect(found).toBeDefined();

      expect(found).toHaveProperty("id", pipeline_def.id);
      expect(found?.config).toEqual(pipeline_config);
      expect(found?.instanceIds).toEqual([]);

      expect(found?.segments).toBeDefined();

      expect(Object.keys(found!.segments)).toEqual(Object.keys(pipeline_def.segments));

      Object.entries(found!.segments).forEach(([seg_name, seg_def]) => {
         expect(seg_def.id).toBe(pipeline_def.segments[seg_name].id);
         expect(seg_def.parentId).toBe(found?.id);
      });
   });
});

describe("Single", () => {
   beforeEach(() => {
      store.dispatch(pipelineDefinitionsAdd(pipeline_def));
   });

   test("Select All", () => {
      const found = pipelineDefinitionsSelectAll(store.getState());

      expect(found).toHaveLength(1);

      expect(found[0].id).toEqual(pipeline_def.id);
      expect(found[0].instanceIds).toEqual([]);

      expect(Object.keys(found[0].segments)).toEqual(Object.keys(pipeline_def.segments));

      Object.entries(found[0].segments).forEach(([seg_name, seg_def]) => {
         expect(seg_def.id).toBe(pipeline_def.segments[seg_name].id);
         expect(seg_def.parentId).toBe(found[0].id);
      });
   });

   test("Select One", () => {
      const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

      expect(found?.id).toEqual(pipeline_def.id);
      expect(found?.instanceIds).toEqual([]);

      expect(Object.keys(found!.segments)).toEqual(Object.keys(pipeline_config.segments));

      Object.entries(found!.segments).forEach(([seg_name, seg_def]) => {
         expect(seg_def.id).toBe(pipeline_def.segments[seg_name].id);
         expect(seg_def.parentId).toBe(found?.id);
      });
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
      assert.throws(() =>
         store.dispatch(
            pipelineDefinitionsRemove({
               ...pipeline_def,
               id: "9999",
            })
         )
      );
   });

   describe("With Mapping", () => {
      beforeEach(() => {
         // Add a connection and workers
         store.dispatch(connectionsAdd(connection));
         store.dispatch(workersAddMany(workers));

         // Now add a mapping
         store.dispatch(
            pipelineDefinitionsSetMapping({
               definition_id: pipeline_def.id,
               mapping: pipeline_mappings[workers[0].machineId],
            })
         );
      });

      test("Contains Mapping", () => {
         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expectDefined(found);

         expect(found?.mappings).toHaveProperty(workers[0].machineId);
      });

      test("Delete Worker", () => {
         store.dispatch(workersRemove(workers[0]));

         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expectDefined(found);

         expect(found?.mappings).not.toHaveProperty(workers[0].machineId);
      });
   });

   describe("With PipelineInstance", () => {
      beforeEach(() => {
         store.dispatch(connectionsAdd(connection));

         // Now add an instance
         store.dispatch(pipelineInstancesAdd(pipeline));
      });

      test("Contains Instance", () => {
         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expect(found?.instanceIds).toContain(pipeline.id);
      });

      test("Remove Instance", () => {
         store.dispatch(pipelineInstancesRemove(pipeline));

         const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

         expect(found?.instanceIds).not.toContain(pipeline.id);

         // Then remove the pipeline
         store.dispatch(pipelineDefinitionsRemove(pipeline_def));

         expect(pipelineDefinitionsSelectAll(store.getState())).toHaveLength(0);
      });

      test("Remove Pipeline Before Instance", () => {
         assert.throws(() => {
            // Remove the pipeline with running segments
            store.dispatch(pipelineDefinitionsRemove(pipeline_def));
         });
      });

      describe("With SegmentInstance", () => {
         beforeEach(() => {
            store.dispatch(workersAdd(worker));

            // Now add an instance
            store.dispatch(segmentInstancesAddMany(segments));
         });

         test("Contains Instance", () => {
            const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

            segments.forEach((s) => {
               expect(found?.segments).toHaveProperty(s.name);
               expect(found?.segments[s.name].instanceIds).toContain(s.id);
            });
         });

         test("Remove Instance", () => {
            segments.forEach((x) => {
               // Need to set the state first
               store.dispatch(
                  segmentInstancesUpdateResourceActualState({
                     resource: x,
                     status: ResourceActualStatus.Actual_Destroyed,
                  })
               );

               store.dispatch(segmentInstancesRemove(x));
            });

            const found = pipelineDefinitionsSelectById(store.getState(), pipeline_def.id);

            segments.forEach((s) => {
               expect(found?.segments).toHaveProperty(s.name);
               expect(found?.segments[s.name].instanceIds).not.toContain(s.id);
            });

            // Also remove the pipeline instance
            store.dispatch(pipelineInstancesRemove(pipeline));

            // Then remove the pipeline
            store.dispatch(pipelineDefinitionsRemove(pipeline_def));

            expect(pipelineDefinitionsSelectAll(store.getState())).toHaveLength(0);
         });

         test("Remove Pipeline Before Instance", () => {
            assert.throws(() => {
               // Remove the pipeline with running segments
               store.dispatch(pipelineDefinitionsRemove(pipeline_def));
            });
         });
      });
   });
});
