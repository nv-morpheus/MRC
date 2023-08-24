import { createSlice, PayloadAction } from "@reduxjs/toolkit";

import { PipelineConfiguration } from "@mrc/proto/mrc/protos/architect_state";
import { createWrappedEntityAdapter } from "@mrc/server/utils";
import { hashProtoMessage } from "@mrc/common/utils";
import { segmentInstancesAdd, segmentInstancesRemove } from "@mrc/server/store/slices/segmentInstancesSlice";
import { pipelineInstancesAdd, pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   IManifoldInstance,
   IPipelineConfiguration,
   IPipelineDefinition,
   IPipelineInstance,
   IPipelineMapping,
   ISegmentInstance,
} from "@mrc/common/entities";
import { manifoldInstancesAdd, manifoldInstancesRemove } from "@mrc/server/store/slices/manifoldInstancesSlice";
import { PipelineDefinitionWrapper } from "@mrc/common/pipelineDefinition";
import { workersRemove } from "@mrc/server/store/slices/workersSlice";
import { AppDispatch, AppGetState, RootState } from "@mrc/server/store/store";

const pipelineDefinitionsAdapter = createWrappedEntityAdapter<IPipelineDefinition>({
   selectId: (w) => w.id,
});

function pipelineInstanceAdded(
   state: PipelineDefinitionsStateType,
   instance: Pick<IPipelineInstance, "id" | "definitionId">
) {
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.definitionId);

   if (found) {
      found.instanceIds.push(instance.id);
   } else {
      throw new Error("Must add a PipelineDefinition before creating a PipelineInstance!");
   }
}

function segmentInstanceAdded(state: PipelineDefinitionsStateType, instance: ISegmentInstance) {
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.pipelineDefinitionId);

   if (found) {
      // Find the segment in the definition
      if (!Object.keys(found.segments).includes(instance.name)) {
         throw new Error(
            `Attempting to create SegmentInstance with Pipeline Definition with ID: ${instance.pipelineDefinitionId} but the segment name, ${instance.name} does not exist in the PipelineDefinition.`
         );
      }

      found.segments[instance.name].instanceIds.push(instance.id);
   } else {
      throw new Error("Must add a PipelineDefinition before creating a SegmentInstance!");
   }
}

function manifoldInstanceAdded(state: PipelineDefinitionsStateType, instance: IManifoldInstance) {
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.pipelineDefinitionId);

   if (found) {
      // Find the segment in the definition
      if (!Object.keys(found.manifolds).includes(instance.portName)) {
         throw new Error(
            `Attempting to create ManifoldInstance with Pipeline Definition with ID: ${instance.pipelineDefinitionId} but the manifold name, ${instance.portName} does not exist in the PipelineDefinition.`
         );
      }

      found.manifolds[instance.portName].instanceIds.push(instance.id);
   } else {
      throw new Error("Must add a PipelineDefinition before creating a ManifoldInstance!");
   }
}

export const pipelineDefinitionsSlice = createSlice({
   name: "pipelineDefinitions",
   initialState: pipelineDefinitionsAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<Omit<IPipelineDefinition, "mappings">>) => {
         if (pipelineDefinitionsAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} already exists`);
         }
         pipelineDefinitionsAdapter.addOne(state, {
            ...action.payload,
            mappings: {},
         });
      },
      remove: (state, action: PayloadAction<IPipelineDefinition>) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} not found`);
         }

         if (found.instanceIds.length > 0) {
            throw new Error(
               `Attempting to delete PipelineDefinition with ID: ${action.payload.id} while there are running instances. Stop PipelineInstances first!`
            );
         }

         if (Object.values(found.segments).reduce((accum: number, curr) => accum + curr.instanceIds.length, 0) > 0) {
            throw new Error(
               `Attempting to delete PipelineDefinition with ID: ${action.payload.id} with running segment instance. Remove segment instances first!`
            );
         }

         pipelineDefinitionsAdapter.removeOne(state, action.payload.id);
      },
      setMapping: (state, action: PayloadAction<{ definition_id: string; mapping: IPipelineMapping }>) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.definition_id);

         if (!found) {
            throw new Error(`Pipeline Definition with ID: ${action.payload.definition_id} not found`);
         }

         const mapping = action.payload.mapping;

         // Check to make sure we dont already have a matching mapping
         if (mapping.machineId in found.mappings) {
            throw new Error(
               `PipelineDefinition with ID: ${action.payload.definition_id}, already contains a mapping for machine ID: ${mapping.machineId}`
            );
         }

         found.mappings[action.payload.mapping.machineId] = action.payload.mapping;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(workersRemove, (state, action) => {
         const allDefinitions = pipelineDefinitionsAdapter.getAll(state);

         // Delete all mappings for this worker
         allDefinitions.forEach((d) => {
            if (action.payload.machineId in d.mappings) {
               delete d.mappings[action.payload.machineId];
            }
         });
      });

      builder.addCase(pipelineInstancesAdd, (state, action) => {
         pipelineInstanceAdded(state, action.payload);
      });
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.definitionId);

         if (found) {
            const index = found.instanceIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               found.instanceIds.splice(index, 1);
            }
         } else {
            throw new Error(
               `PipelineDefinition with ID: ${action.payload.definitionId}, not found. Must drop all Pipeline before removing a PipelineDefinition.`
            );
         }
      });

      builder.addCase(segmentInstancesAdd, (state, action) => {
         segmentInstanceAdded(state, action.payload);
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.pipelineDefinitionId);

         if (found) {
            // Find the segment in the definition
            if (!Object.keys(found.segments).includes(action.payload.name)) {
               throw new Error(
                  `Attempting to remove SegmentInstance with PipelineDefinition ID: ${action.payload.pipelineDefinitionId} but the segment name, ${action.payload.name} does not exist in the PipelineDefinition.`
               );
            }

            const foundSegment = found.segments[action.payload.name];

            const index = foundSegment.instanceIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               foundSegment.instanceIds.splice(index, 1);
            }
         } else {
            throw new Error(
               `PipelineDefinition with ID: ${action.payload.pipelineDefinitionId}, not found. Must drop all SegmentInstances before removing a PipelineDefinition`
            );
         }
      });

      builder.addCase(manifoldInstancesAdd, (state, action) => {
         manifoldInstanceAdded(state, action.payload);
      });
      builder.addCase(manifoldInstancesRemove, (state, action) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.pipelineDefinitionId);

         if (found) {
            // Find the segment in the definition
            if (!Object.keys(found.manifolds).includes(action.payload.portName)) {
               throw new Error(
                  `Attempting to remove ManifoldInstance with PipelineDefinition ID: ${action.payload.pipelineDefinitionId} but the manifold name, ${action.payload.portName} does not exist in the PipelineDefinition.`
               );
            }

            const foundManifold = found.manifolds[action.payload.portName];

            const index = foundManifold.instanceIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               foundManifold.instanceIds.splice(index, 1);
            }
         } else {
            throw new Error(
               `PipelineDefinition with ID: ${action.payload.pipelineDefinitionId}, not found. Must drop all ManifoldInstances before removing a PipelineDefinition`
            );
         }
      });
   },
});

export function pipelineDefinitionsCreateOrUpdate(pipeline_config: IPipelineConfiguration) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Compute the hash of the pipeline
      const pipeline_hash = hashProtoMessage(PipelineConfiguration.create(pipeline_config));

      // Check if this already exists
      let pipeline_def = pipelineDefinitionsSelectById(getState(), pipeline_hash);

      if (!pipeline_def) {
         // Generate a full pipeline definition with the ID as the hash
         pipeline_def = PipelineDefinitionWrapper.from(pipeline_config);

         dispatch(pipelineDefinitionsAdd(pipeline_def));
      }

      return pipeline_def;
   };
}

// export function pipelineDefinitionsCreateOrUpdate(
//    pipeline_config: IPipelineConfiguration,
//    pipeline_mapping: IPipelineMapping
// ) {
//    return (dispatch: AppDispatch, getState: AppGetState) => {
//       // Compute the hash of the pipeline
//       const pipeline_hash = hashProtoMessage(PipelineConfiguration.create(pipeline_config));

//       // Check if this already exists
//       let pipeline_def = pipelineDefinitionsSelectById(getState(), pipeline_hash);

//       if (!pipeline_def) {
//          // Generate a full pipeline definition with the ID as the hash
//          pipeline_def = PipelineDefinitionWrapper.from(pipeline_config);

//          dispatch(pipelineDefinitionsAdd(pipeline_def));
//       } else {
//          // Check to make sure we dont already have a matching mapping
//          if (pipeline_mapping.machineId in pipeline_def.mappings) {
//             throw new Error(
//                `PipelineDefinition with ID: ${pipeline_hash}, already contains a mapping for machine ID: ${pipeline_mapping.machineId}`
//             );
//          }
//       }

//       // Add the mapping to the pipeline config
//       dispatch(
//          pipelineDefinitionsSlice.actions.setMapping({
//             definition_id: pipeline_hash,
//             mapping: pipeline_mapping,
//          })
//       );

//       return {
//          pipeline: pipeline_def.id,
//       };
//    };
// }

type PipelineDefinitionsStateType = ReturnType<typeof pipelineDefinitionsSlice.getInitialState>;

export const {
   add: pipelineDefinitionsAdd,
   remove: pipelineDefinitionsRemove,
   setMapping: pipelineDefinitionsSetMapping,
} = pipelineDefinitionsSlice.actions;

export const {
   selectAll: pipelineDefinitionsSelectAll,
   selectById: pipelineDefinitionsSelectById,
   selectEntities: pipelineDefinitionsSelectEntities,
   selectIds: pipelineDefinitionsSelectIds,
   selectTotal: pipelineDefinitionsSelectTotal,
   selectByIds: pipelineDefinitionsSelectByIds,
} = pipelineDefinitionsAdapter.getSelectors((state: RootState) => state.pipelineDefinitions);

export function pipelineDefinitionsConfigureSlice() {
   return pipelineDefinitionsSlice.reducer;
}
