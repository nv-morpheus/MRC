/* eslint-disable @typescript-eslint/unbound-method */
import { IManifoldInstance, IPipelineInstance, ISegmentInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   ResourceRequestedStatus,
   SegmentMappingPolicies,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { connectionsRemove } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { RootState } from "@mrc/server/store/store";
import { createWrappedEntityAdapter } from "@mrc/server/utils";
import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { AppListenerAPI } from "@mrc/server/store/listener_middleware";
import { generateId, generateSegmentHash } from "@mrc/common/utils";
import { workersSelectByMachineId } from "@mrc/server/store/slices/workersSlice";
import {
   manifoldInstancesAdd,
   manifoldInstancesRemove,
   manifoldInstancesSelectById,
} from "@mrc/server/store/slices/manifoldInstancesSlice";
import { createWatcher } from "@mrc/server/store/resourceStateWatcher";
import {
   segmentInstancesAdd,
   segmentInstancesRemove,
   segmentInstancesSelectById,
} from "@mrc/server/store/slices/segmentInstancesSlice";

const pipelineInstancesAdapter = createWrappedEntityAdapter<IPipelineInstance>({
   selectId: (w) => w.id,
});

function segmentInstanceAdded(state: PipelineInstancesStateType, instance: ISegmentInstance) {
   // Handle synchronizing a new added instance
   const found = pipelineInstancesAdapter.getOne(state, instance.pipelineInstanceId);

   if (found) {
      found.segmentIds.push(instance.id);
   } else {
      throw new Error("Must add a PipelineInstance before a SegmentInstance!");
   }
}

export const pipelineInstancesSlice = createSlice({
   name: "pipelineInstances",
   initialState: pipelineInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<Pick<IPipelineInstance, "id" | "definitionId" | "machineId">>) => {
         if (pipelineInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Pipeline Instance with ID: ${action.payload.id} already exists`);
         }
         pipelineInstancesAdapter.addOne(state, {
            ...action.payload,
            segmentIds: [],
            manifoldIds: [],
            state: {
               requestedStatus: ResourceRequestedStatus.Requested_Created,
               actualStatus: ResourceActualStatus.Actual_Unknown,
               refCount: 0,
            },
         });
      },
      remove: (state, action: PayloadAction<IPipelineInstance>) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Pipeline Instance with ID: ${action.payload.id} not found`);
         }

         if (found.segmentIds.length > 0) {
            throw new Error(
               `Attempting to delete Pipeline Instance with ID: ${action.payload.id} with running segment instance. Remove segment instances first!`
            );
         }

         pipelineInstancesAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: IPipelineInstance; status: ResourceRequestedStatus }>
      ) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Pipeline Instance with ID: ${action.payload.resource.id} not found`);
         }

         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: IPipelineInstance; status: ResourceActualStatus }>
      ) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Pipeline Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceActualStatusToNumber(action.payload.status) >
            resourceRequestedStatusToNumber(found.state.requestedStatus) + 1
         ) {
            throw new Error(
               `Cannot update Pipeline Instance with ID: ${action.payload.resource.id} actual status to ${action.payload.status}. Requested status is ${found.state.requestedStatus}`
            );
         }

         found.state.actualStatus = action.payload.status;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any workers associated with that connection
         const connection_instances = selectByMachineId(state, action.payload.id);

         pipelineInstancesAdapter.removeMany(
            state,
            connection_instances.map((x) => x.id)
         );
      });
      builder.addCase(segmentInstancesAdd, (state, action) => {
         segmentInstanceAdded(state, action.payload);
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.pipelineInstanceId);

         if (found) {
            const index = found.segmentIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               found.segmentIds.splice(index, 1);
            }
         } else {
            throw new Error("Must drop all SegmentInstances before removing a PipelineInstance");
         }
      });
      builder.addCase(manifoldInstancesAdd, (state, action) => {
         // Handle synchronizing a new added instance
         const found = pipelineInstancesAdapter.getOne(state, action.payload.pipelineInstanceId);

         if (found) {
            found.manifoldIds.push(action.payload.id);
         } else {
            throw new Error("Must add a PipelineInstance before a ManifoldInstance!");
         }
      });
      builder.addCase(manifoldInstancesRemove, (state, action) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.pipelineInstanceId);

         if (found) {
            const index = found.manifoldIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               found.manifoldIds.splice(index, 1);
            }
         } else {
            throw new Error("Must drop all ManifoldInstances before removing a PipelineInstance");
         }
      });
   },
});

type PipelineInstancesStateType = ReturnType<typeof pipelineInstancesSlice.getInitialState>;

export const {
   add: pipelineInstancesAdd,
   remove: pipelineInstancesRemove,
   updateResourceRequestedState: pipelineInstancesUpdateResourceRequestedState,
   updateResourceActualState: pipelineInstancesUpdateResourceActualState,
} = pipelineInstancesSlice.actions;

export const {
   selectAll: pipelineInstancesSelectAll,
   selectById: pipelineInstancesSelectById,
   selectEntities: pipelineInstancesSelectEntities,
   selectIds: pipelineInstancesSelectIds,
   selectTotal: pipelineInstancesSelectTotal,
   selectByIds: pipelineInstancesSelectByIds,
} = pipelineInstancesAdapter.getSelectors((state: RootState) => state.pipelineInstances);

const selectByMachineId = createSelector(
   [pipelineInstancesAdapter.getAll, (state: PipelineInstancesStateType, machine_id: string) => machine_id],
   (pipelineInstances, machine_id) => pipelineInstances.filter((p) => p.machineId === machine_id)
);

export const pipelineInstancesSelectByMachineId = (state: RootState, machine_id: string) =>
   selectByMachineId(state.pipelineInstances, machine_id);

async function manifoldsFromInstance(
   listenerApi: AppListenerAPI,
   pipelineInstance: IPipelineInstance
): Promise<IManifoldInstance[]> {
   // Pipeline has been marked as ready. Update segment instances based on the pipeline config
   const pipeline_def = pipelineDefinitionsSelectById(listenerApi.getState(), pipelineInstance.definitionId);

   if (!pipeline_def) {
      throw new Error(
         `Could not find Pipeline Definition ID: ${pipelineInstance.definitionId}, for Pipeline Instance: ${pipelineInstance.id}`
      );
   }

   const manifolds = Object.entries(pipeline_def.manifolds).map(([manifold_name, manifold_def]) => {
      return {
         id: generateId(),
         actualInputSegments: {},
         actualOutputSegments: {},
         machineId: pipelineInstance.machineId,
         pipelineDefinitionId: pipeline_def.id,
         pipelineInstanceId: pipelineInstance.id,
         portName: manifold_name,
         requestedInputSegments: {},
         requestedOutputSegments: {},
         state: {
            refCount: 0,
            requestedStatus: ResourceRequestedStatus.Requested_Created,
            actualStatus: ResourceActualStatus.Actual_Unknown,
         },
      } as IManifoldInstance;
   });

   // For each one, make a fork to track progress
   const created_manifolds = await Promise.all(
      manifolds.map(async (m) => {
         const result = await listenerApi.fork(async () => {
            // Create the manifold
            listenerApi.dispatch(manifoldInstancesAdd(m));

            // Wait for it to be reported as created
            await listenerApi.condition((_, currentState) => {
               return (
                  manifoldInstancesSelectById(currentState, m.id)?.state.actualStatus ===
                  ResourceActualStatus.Actual_Created
               );
            });

            return manifoldInstancesSelectById(listenerApi.getState(), m.id);
         }).result;

         if (result.status === "ok") {
            return result.value;
         }
         return undefined;
      })
   );

   created_manifolds.forEach((result) => {
      if (result?.state.actualStatus !== ResourceActualStatus.Actual_Created) {
         throw new Error("Failed to create manifolds");
      }
   });

   return manifolds;
}

async function segmentsFromInstance(listenerApi: AppListenerAPI, pipelineInstance: IPipelineInstance) {
   const state = listenerApi.getState();

   // Pipeline has been marked as ready. Update segment instances based on the pipeline config
   const pipeline_def = pipelineDefinitionsSelectById(state, pipelineInstance.definitionId);

   if (!pipeline_def) {
      throw new Error(
         `Could not find Pipeline Definition ID: ${pipelineInstance.definitionId}, for Pipeline Instance: ${pipelineInstance.id}`
      );
   }

   // Get the mapping for this machine ID
   if (!(pipelineInstance.machineId in pipeline_def.mappings)) {
      throw new Error(
         `Could not find Mapping for Machine: ${pipelineInstance.machineId}, for Pipeline Definition: ${pipelineInstance.definitionId}`
      );
   }

   // Get the workers for this machine
   const workers = workersSelectByMachineId(state, pipelineInstance.machineId);

   const mapping = pipeline_def.mappings[pipelineInstance.machineId];

   // Now determine the segment instances that should be created
   const seg_to_workers = Object.fromEntries(
      Object.entries(mapping.segments).map(([seg_name, seg_map]) => {
         let workerIds: string[] = [];

         if (seg_map.byPolicy) {
            if (seg_map.byPolicy.value == SegmentMappingPolicies.OnePerWorker) {
               workerIds = workers.map((x) => x.id);
            } else {
               throw new Error(`Unsupported policy: ${seg_map.byPolicy.value}`);
            }
         } else if (seg_map.byWorker) {
            workerIds = seg_map.byWorker.workerIds;
         } else {
            throw new Error(`Invalid SegmentMap for ${seg_name}. No option set`);
         }

         return [seg_name, workerIds];
      })
   );

   // Now generate the segments that would need to be created
   const segments = Object.entries(seg_to_workers).flatMap(([seg_name, seg_assignment]) => {
      // For each assignment, create a segment instance
      return seg_assignment.map((wid) => {
         const address = generateSegmentHash(seg_name, wid);

         return {
            id: address.toString(),
            pipelineDefinitionId: pipeline_def.id,
            pipelineInstanceId: pipelineInstance.id,
            name: seg_name,
            address: address,
            workerId: wid,
            state: {
               refCount: 0,
               requestedStatus: ResourceRequestedStatus.Requested_Created,
               actualStatus: ResourceActualStatus.Actual_Unknown,
            },
         } as ISegmentInstance;
      });
   });

   // For each one, make a fork to track progress
   const created_segments = await Promise.all(
      segments.map(async (s) => {
         const result = await listenerApi.fork(async () => {
            // Create the manifold
            listenerApi.dispatch(segmentInstancesAdd(s));

            // Wait for it to be reported as created
            await listenerApi.condition((_, currentState) => {
               return (
                  segmentInstancesSelectById(currentState, s.id)?.state.actualStatus ===
                  ResourceActualStatus.Actual_Created
               );
            });

            return segmentInstancesSelectById(listenerApi.getState(), s.id);
         }).result;

         if (result.status === "ok") {
            return result.value;
         }
         return undefined;
      })
   );

   created_segments.forEach((result) => {
      if (result?.state.actualStatus !== ResourceActualStatus.Actual_Created) {
         throw new Error("Failed to create segments");
      }
   });

   return segments;
}

export function pipelineInstancesConfigureSlice() {
   createWatcher(
      "PipelineInstances",
      pipelineInstancesAdd,
      (state, id) => {
         return pipelineInstancesSelectById(state, id);
      },
      async (instance, listenerApi) => {
         // Create all of the manifold instances
         const manifolds = await manifoldsFromInstance(listenerApi, instance);

         // Before moving to RunUntilComplete, create the segment instances
         const segments = await segmentsFromInstance(listenerApi, instance);
      },
      undefined,
      undefined,
      undefined,
      async (instance, listenerApi) => {
         const is_ready = (p: IPipelineInstance | undefined) => {
            if (!p) {
               // Handled elsewhere
               return true;
            }

            return p.manifoldIds.length === 0 && p.segmentIds.length === 0;
         };

         // Only await here if we arent ready to delete right await
         if (!is_ready(instance)) {
            await listenerApi.condition((_, currentState) => {
               return is_ready(pipelineInstancesSelectById(currentState, instance.id));
            });
         }

         // Now we can actually just remove the object
         listenerApi.dispatch(pipelineInstancesRemove(instance));
      }
   );

   return pipelineInstancesSlice.reducer;
}
