/* eslint-disable @typescript-eslint/unbound-method */
import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { IManifoldInstance, ISegmentInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {
   segmentInstancesSelectById,
   segmentInstancesSelectByNameAndPipelineDef,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { startAppListening } from "@mrc/server/store/listener_middleware";
import { createWatcher } from "@mrc/server/store/resourceStateWatcher";
import { createWrappedEntityAdapter } from "@mrc/server/utils";
import { AppDispatch, RootState, AppGetState } from "@mrc/server/store/store";

const manifoldInstancesAdapter = createWrappedEntityAdapter<IManifoldInstance>({
   selectId: (w) => w.id,
});

export const manifoldInstancesSlice = createSlice({
   name: "manifoldInstances",
   initialState: manifoldInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IManifoldInstance>) => {
         if (manifoldInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Manifold Instance with ID: ${action.payload.id} already exists`);
         }
         manifoldInstancesAdapter.addOne(state, action.payload);
      },
      remove: (state, action: PayloadAction<IManifoldInstance>) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state.actualStatus != ResourceActualStatus.Actual_Destroyed) {
            throw new Error(
               `Attempting to delete Manifold Instance with ID: ${action.payload.id} while it has not finished. Stop ManifoldInstance first!`
            );
         }

         manifoldInstancesAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: IManifoldInstance; status: ResourceRequestedStatus }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceRequestedStatusToNumber(found.state.requestedStatus) >
            resourceRequestedStatusToNumber(action.payload.status)
         ) {
            throw new Error(
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state.requestedStatus} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: IManifoldInstance; status: ResourceActualStatus }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceActualStatusToNumber(found.state.actualStatus) > resourceActualStatusToNumber(action.payload.status)
         ) {
            throw new Error(
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state.actualStatus} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.actualStatus = action.payload.status;
      },
      attachRequestedSegment: (
         state,
         action: PayloadAction<{
            manifold: IManifoldInstance;
            is_input: boolean;
            segment: ISegmentInstance;
            is_local: boolean;
         }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.manifold.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.manifold.id} not found`);
         }

         // Check to make sure this hasnt been added already
         if (action.payload.is_input) {
            if (action.payload.segment.address in found.requestedInputSegments) {
               throw new Error("Segment already attached to manifold");
            }

            found.requestedInputSegments[action.payload.segment.address] = action.payload.is_local;
         } else {
            if (action.payload.segment.address in found.requestedOutputSegments) {
               throw new Error("Segment already attached to manifold");
            }

            found.requestedOutputSegments[action.payload.segment.address] = action.payload.is_local;
         }
      },
      detachRequestedSegment: (
         state,
         action: PayloadAction<{
            manifold: IManifoldInstance;
            is_input: boolean;
            segment: ISegmentInstance;
         }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.manifold.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.manifold.id} not found`);
         }

         // Check to make sure its already added
         if (action.payload.is_input) {
            if (!(action.payload.segment.address in found.requestedInputSegments)) {
               throw new Error("Segment not attached to manifold");
            }

            delete found.requestedInputSegments[action.payload.segment.address];
         } else {
            if (!(action.payload.segment.address in found.requestedOutputSegments)) {
               throw new Error("Segment not attached to manifold");
            }

            delete found.requestedOutputSegments[action.payload.segment.address];
         }
      },
   },
   extraReducers: (builder) => {
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // Need to delete any manifolds associated with the pipeline
         const instances = selectByPipelineId(state, action.payload.id);

         manifoldInstancesAdapter.removeMany(
            state,
            instances.map((x) => x.id)
         );
      });
   },
});

function syncSegmentNameForManifold(
   dispatch: AppDispatch,
   state: RootState,
   manifold: IManifoldInstance,
   segmentName: string,
   isInput: boolean
) {
   // Find all segments that match this name and definition pair
   const matchingSegments = segmentInstancesSelectByNameAndPipelineDef(
      state,
      segmentName,
      manifold.pipelineDefinitionId
   );

   // Find only the segment
   const activeSegmentIds = matchingSegments
      .filter(
         (s) =>
            resourceActualStatusToNumber(s.state.actualStatus) >=
               resourceActualStatusToNumber(ResourceActualStatus.Actual_Created) &&
            resourceActualStatusToNumber(s.state.actualStatus) <
               resourceActualStatusToNumber(ResourceActualStatus.Actual_Completed)
      )
      .map((s) => s.id);

   const currentSegmentIds = Object.keys(isInput ? manifold.requestedInputSegments : manifold.requestedOutputSegments);

   // Determine any that need to be added
   const toAdd = activeSegmentIds.filter((s) => !currentSegmentIds.includes(s));

   toAdd.forEach((segId) => {
      const seg = segmentInstancesSelectById(state, segId);

      if (!seg) {
         throw new Error(`Could not find segment with ID: ${segId}`);
      }

      // Figure out if this is local
      const is_local = manifold.pipelineInstanceId === seg.pipelineInstanceId;

      // Dispatch the attach action
      dispatch(
         manifoldInstancesSlice.actions.attachRequestedSegment({
            is_input: isInput,
            is_local: is_local,
            manifold: manifold,
            segment: seg,
         })
      );
   });

   // Determine any that need to be removed
   const toRemove = currentSegmentIds.filter((s) => !activeSegmentIds.includes(s));

   toRemove.forEach((segId) => {
      const seg = segmentInstancesSelectById(state, segId);

      if (!seg) {
         throw new Error(`Could not find segment with ID: ${segId}`);
      }

      // Dispatch the attach action
      dispatch(
         manifoldInstancesSlice.actions.detachRequestedSegment({
            is_input: isInput,
            manifold: manifold,
            segment: seg,
         })
      );
   });
}

export function manifoldInstancesSyncSegments(manifoldId: string) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();

      const found = manifoldInstancesSelectById(state, manifoldId);

      if (!found) {
         throw new Error(`Manifold Instance with ID: ${manifoldId} not found`);
      }

      // Find the pipeline definition we are attached to
      const pipeline_def = pipelineDefinitionsSelectById(state, found.pipelineDefinitionId);

      if (!pipeline_def) {
         throw new Error(`Could not find pipeline definition with ID: ${found.pipelineDefinitionId}`);
      }

      if (!(found.portName in pipeline_def.manifolds)) {
         throw new Error(
            `Could not find manifold ${found.portName} in definition with ID: ${found.pipelineDefinitionId}`
         );
      }

      const manifold_def = pipeline_def.manifolds[found.portName];

      // For each ingress, sync all segments
      Object.entries(manifold_def.outputSegmentIds).forEach(([segmentName]) => {
         syncSegmentNameForManifold(dispatch, state, found, segmentName, false);
      });

      // For each egress, sync all segments
      Object.entries(manifold_def.inputSegmentIds).forEach(([segmentName]) => {
         syncSegmentNameForManifold(dispatch, state, found, segmentName, true);
      });
   };
}

// export function manifoldInstancesAttachLocalSegment(manifold: IManifoldInstance, segment: ISegmentInstance) {
//    return (dispatch: AppDispatch, getState: AppGetState) => {
//       const state = getState();

//       const found = manifoldInstancesSelectById(state, manifold.id);

//       if (!found) {
//          throw new Error(`Manifold Instance with ID: ${manifold.id} not found`);
//       }

//       // Check to make sure they are actually local
//       if (manifold.machineId !== workersSelectById(state, segment.workerId)?.machineId) {
//          throw new Error("Invalid local manifold/segment pair. Manifold and segment are on different machines");
//       }

//       const pipeline_def = pipelineDefinitionsSelectById(state, manifold.pipelineDefinitionId);

//       if (!pipeline_def) {
//          throw new Error(`Could not find pipeline definition with ID: ${manifold.pipelineDefinitionId}`);
//       }

//       if (!(manifold.portName in pipeline_def.manifolds)) {
//          throw new Error(
//             `Could not find manifold ${manifold.portName} in definition with ID: ${manifold.pipelineDefinitionId}`
//          );
//       }

//       const manifold_def = pipeline_def.manifolds[manifold.portName];

//       if (!(segment.name in pipeline_def.segments)) {
//          throw new Error(
//             `Could not find segment ${segment.name} in definition with ID: ${manifold.pipelineDefinitionId}`
//          );
//       }

//       const segment_def = pipeline_def.segments[segment.name];

//       // Figure out if this is an egress or ingress (relative to the manifold. Opposite for segments)
//       let is_manifold_ingress = false;

//       if (manifold.portName in segment_def.egressPorts) {
//          is_manifold_ingress = true;
//       } else if (manifold.portName in segment_def.ingressPorts) {
//          is_manifold_ingress = false;
//       } else {
//          throw new Error("Manifold not found in segment definition ingress or egress ports");
//       }

//       // Now sync all other manifolds for this definition
//       manifold_def.instanceIds.forEach((manifold_id) => {
//          const manifold_instance = manifoldInstancesSelectById(state, manifold_id);

//          if (!manifold_instance) {
//             throw new Error("Could not find manifold by ID");
//          }

//          // Figure out if this is local
//          const is_local = manifold_instance.machineId === manifold.machineId;

//          // Dispatch the attach action
//          dispatch(
//             manifoldInstancesSlice.actions.attachRequestedSegment({
//                is_ingress: is_manifold_ingress,
//                is_local: is_local,
//                manifold: manifold_instance,
//                segment: segment,
//             })
//          );
//       });
//    };
// }

export function manifoldInstancesAddMany(instances: IManifoldInstance[]) {
   // To allow the watchers to work, we need to add all individually
   return (dispatch: AppDispatch) => {
      // Loop and dispatch each individually
      instances.forEach((m) => {
         dispatch(manifoldInstancesAdd(m));
      });
   };
}

type ManifoldInstancesStateType = ReturnType<typeof manifoldInstancesSlice.getInitialState>;

export const {
   add: manifoldInstancesAdd,
   remove: manifoldInstancesRemove,
   updateResourceRequestedState: manifoldInstancesUpdateResourceRequestedState,
   updateResourceActualState: manifoldInstancesUpdateResourceActualState,
} = manifoldInstancesSlice.actions;

export const {
   selectAll: manifoldInstancesSelectAll,
   selectById: manifoldInstancesSelectById,
   selectByIds: manifoldInstancesSelectByIds,
   selectEntities: manifoldInstancesSelectEntities,
   selectIds: manifoldInstancesSelectIds,
   selectTotal: manifoldInstancesSelectTotal,
} = manifoldInstancesAdapter.getSelectors((state: RootState) => state.manifoldInstances);

const selectByPipelineId = createSelector(
   [manifoldInstancesAdapter.getAll, (state: ManifoldInstancesStateType, pipeline_id: string) => pipeline_id],
   (manifoldInstances, pipeline_id) => manifoldInstances.filter((x) => x.pipelineInstanceId === pipeline_id)
);

export const manifoldInstancesSelectByPipelineId = (state: RootState, pipeline_id: string) =>
   selectByPipelineId(state.manifoldInstances, pipeline_id);

const selectByNameAndPipelineDef = createSelector(
   [
      manifoldInstancesAdapter.getAll,
      (state: ManifoldInstancesStateType, name: string) => name,
      (state: ManifoldInstancesStateType, name: string, pipelineDefinitionId: string) => pipelineDefinitionId,
   ],
   (manifoldInstances, name, pipelineDefinitionId) =>
      manifoldInstances.filter((x) => x.portName === name && x.pipelineDefinitionId === pipelineDefinitionId)
);

export const manifoldInstancesSelectByNameAndPipelineDef = (
   state: RootState,
   name: string,
   pipelineDefinitionId: string
) => selectByNameAndPipelineDef(state.manifoldInstances, name, pipelineDefinitionId);

// export function manifoldInstancesConfigureListeners() {
//    startAppListening({
//       actionCreator: manifoldInstancesAdd,
//       effect: async (action, listenerApi) => {
//          const id = action.payload.id;

//          const instance = manifoldInstancesSelectById(listenerApi.getState(), id);

//          if (!instance) {
//             throw new Error("Could not find segment instance");
//          }

//          // Now that the object has been created, set the requested status to Created
//          listenerApi.dispatch(
//             manifoldInstancesSlice.actions.updateResourceRequestedState({
//                resource: instance,
//                status: ResourceRequestedStatus.Requested_Created,
//             })
//          );

//          const monitor_instance = listenerApi.fork(async () => {
//             while (true) {
//                // Wait for the next update
//                const [, current_state] = await listenerApi.take((action) => {
//                   return manifoldInstancesUpdateResourceActualState.match(action) && action.payload.resource.id === id;
//                });

//                if (!current_state.system.requestRunning) {
//                   console.warn("Updating resource outside of a request will lead to undefined behavior!");
//                }

//                // Get the status of this instance
//                const instance = manifoldInstancesSelectById(listenerApi.getState(), id);

//                if (!instance) {
//                   throw new Error("Could not find instance");
//                }

//                if (instance.state.actualStatus === ResourceActualStatus.Actual_Created) {
//                   // Now that its created, sync our segments
//                   listenerApi.dispatch(manifoldInstancesSyncSegments(id));

//                   // Tell it to move running/completed
//                   listenerApi.dispatch(
//                      manifoldInstancesSlice.actions.updateResourceRequestedState({
//                         resource: instance,
//                         status: ResourceRequestedStatus.Requested_Completed,
//                      })
//                   );
//                } else if (instance.state.actualStatus === ResourceActualStatus.Actual_Completed) {
//                   // Before we can move to Stopped, all ref counts must be 0

//                   // Tell it to move to stopped
//                   listenerApi.dispatch(
//                      manifoldInstancesSlice.actions.updateResourceRequestedState({
//                         resource: instance,
//                         status: ResourceRequestedStatus.Requested_Stopped,
//                      })
//                   );
//                } else if (instance.state.actualStatus === ResourceActualStatus.Actual_Stopped) {
//                   // Tell it to move to stopped
//                   listenerApi.dispatch(
//                      manifoldInstancesSlice.actions.updateResourceRequestedState({
//                         resource: instance,
//                         status: ResourceRequestedStatus.Requested_Destroyed,
//                      })
//                   );
//                } else if (instance.state.actualStatus === ResourceActualStatus.Actual_Destroyed) {
//                   // Now we can actually just remove the object
//                   listenerApi.dispatch(manifoldInstancesRemove(instance));

//                   break;
//                } else {
//                   throw new Error("Unknow state type");
//                }
//             }
//          });

//          await listenerApi.condition((action) => {
//             return manifoldInstancesRemove.match(action) && action.payload.id === id;
//          });
//          monitor_instance.cancel();
//       },
//    });
// }

export function manifoldInstancesConfigureSlice() {
   createWatcher(
      "ManifoldInstances",
      manifoldInstancesAdd,
      manifoldInstancesSelectById,
      async (instance, listenerApi) => {
         // Now that its created, sync our segments
         listenerApi.dispatch(manifoldInstancesSyncSegments(instance.id));
      },
      undefined,
      undefined,
      undefined,
      async (instance, listenerApi) => {
         listenerApi.dispatch(manifoldInstancesRemove(instance));
      }
   );

   return manifoldInstancesSlice.reducer;
}
