/* eslint-disable @typescript-eslint/unbound-method */
import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { IManifoldDefinition, IManifoldInstance, ISegmentInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {
   segmentInstanceIncRefCount,
   segmentInstanceDecRefCount,
   segmentInstancesSelectById,
   segmentInstancesSelectByNameAndPipelineDef,
   segmentInstancesSelectBySegmentAddress,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { startAppListening } from "@mrc/server/store/listener_middleware";
import { createWatcher } from "@mrc/server/store/resourceStateWatcher";
import { createWrappedEntityAdapter } from "@mrc/server/utils";
import { AppDispatch, RootState, AppGetState } from "@mrc/server/store/store";
import { yield_immediate } from "@mrc/common/utils";

function filterMappingByLocality(mapping: { [key: number]: boolean }, isLocal: boolean): [string, boolean][] {
   // Ugh Object.entries & Object.keys both cast to string
   return Object.entries(mapping).filter(([_, segmentLocal]) => segmentLocal === isLocal);
}
function getNumLocal(mapping: { [key: string]: boolean }): number {
   // Counts the number of local segments in either an input or output mapping
   return filterMappingByLocality(mapping, true).length;
}

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

         const requestedMap: { [key: string]: boolean } = action.payload.is_input
            ? found.requestedInputSegments
            : found.requestedOutputSegments;

         // Check to make sure this hasnt been added already
         if (action.payload.segment.segmentAddress in requestedMap) {
            throw new Error("Segment already attached to manifold");
         }

         requestedMap[action.payload.segment.segmentAddress] = action.payload.is_local;
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

         const requestedMap: { [key: string]: boolean } = action.payload.is_input
            ? found.requestedInputSegments
            : found.requestedOutputSegments;

         // Check to make sure its already added
         if (!(action.payload.segment.segmentAddress in requestedMap)) {
            throw new Error("Segment not attached to manifold");
         }

         delete requestedMap[action.payload.segment.segmentAddress];
      },

      attachActualSegment: (
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

         const requestedMap: { [key: string]: boolean } = action.payload.is_input
            ? found.requestedInputSegments
            : found.requestedOutputSegments;
         const actualMap: { [key: string]: boolean } = action.payload.is_input
            ? found.actualInputSegments
            : found.actualOutputSegments;

         const isLocal = requestedMap[action.payload.segment.segmentAddress];
         if (isLocal === undefined) {
            throw new Error("Segment not attached to manifold");
         }

         actualMap[action.payload.segment.segmentAddress] = isLocal;
      },

      detachActualSegment: (
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

         const actualMap: { [key: string]: boolean } = action.payload.is_input
            ? found.actualInputSegments
            : found.actualOutputSegments;

         if (!(action.payload.segment.segmentAddress in actualMap)) {
            throw new Error("Segment not attached to manifold");
         }

         delete actualMap[action.payload.segment.segmentAddress];

         const numRequestedInputs: number = Object.keys(found.requestedInputSegments).length;
         const numActualInputs: number = Object.keys(found.actualInputSegments).length;
         const hasInputs: boolean = numActualInputs !== 0 || numRequestedInputs !== 0;

         // When actual inputs go to 0, and we have no requested inputs, we can request the outputs to be detached
         if (!hasInputs) {
            const numRequestedOutputs: number = Object.keys(found.requestedOutputSegments).length;

            if (action.payload.is_input) {
               if (numRequestedOutputs > 0) {
                  Object.keys(found.requestedOutputSegments).forEach((segmentAddress) => {
                     delete found.requestedOutputSegments[parseInt(segmentAddress)];
                     // TODO: we need decrement the segment's refcount
                  });
               } else {
                  // If we don't have any outputs. tell the manifold it is OK to shutdown
                  found.state.requestedStatus = ResourceRequestedStatus.Requested_Stopped;
               }
            } else if (Object.keys(actualMap).length === 0 && numRequestedOutputs === 0) {
               // If we're detaching an output, we don't have any new outputs being requested, and we don't have any inputs
               // then tell the manifold it is OK to shutdown
               found.state.requestedStatus = ResourceRequestedStatus.Requested_Stopped;
            }
         }
      },
   },
   extraReducers: (builder) => {
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // // Need to delete any manifolds associated with the pipeline
         // const instances = selectByPipelineId(state, action.payload.id);
         // manifoldInstancesAdapter.removeMany(
         //    state,
         //    instances.map((x) => x.id)
         // );
      });
   },
});

function determineManifoldSegmentMapping(state: RootState, manifold_def: IManifoldDefinition) {
   const active_input_segs = Object.entries(manifold_def.inputSegmentIds)
      .map(([segmentName]) => {
         // Find all segments that match this name and definition pair
         const matchingSegments = segmentInstancesSelectByNameAndPipelineDef(state, segmentName, manifold_def.parentId);

         // Find only the segment which are active
         return matchingSegments.filter(
            (s) =>
               resourceActualStatusToNumber(s.state.actualStatus) >=
                  resourceActualStatusToNumber(ResourceActualStatus.Actual_Created) &&
               resourceActualStatusToNumber(s.state.actualStatus) <
                  resourceActualStatusToNumber(ResourceActualStatus.Actual_Completed)
         );
      })
      .reduce((acc, val) => acc.concat(val), []);

   const active_output_segs = Object.entries(manifold_def.outputSegmentIds)
      .map(([segmentName]) => {
         // Find all segments that match this name and definition pair
         const matchingSegments = segmentInstancesSelectByNameAndPipelineDef(state, segmentName, manifold_def.parentId);

         // Find only the segment which are active
         return matchingSegments.filter(
            (s) =>
               resourceActualStatusToNumber(s.state.actualStatus) >=
                  resourceActualStatusToNumber(ResourceActualStatus.Actual_Created) &&
               resourceActualStatusToNumber(s.state.actualStatus) <
                  resourceActualStatusToNumber(ResourceActualStatus.Actual_Completed)
         );
      })
      .reduce((acc, val) => acc.concat(val), []);

   const mapping = {
      input: active_input_segs,
      output: active_output_segs,
   };

   // We must have at least one input and output to map any segments
   if (mapping.input.length === 0 || mapping.output.length === 0) {
      mapping.input = [];
      mapping.output = [];
   }

   return mapping;
}

function syncSegmentNameForManifold(
   dispatch: AppDispatch,
   state: RootState,
   manifold: IManifoldInstance,
   segmentMapping: { input: ISegmentInstance[]; output: ISegmentInstance[] },
   isInput: boolean
) {
   const currentSegmentIds = Object.keys(isInput ? manifold.requestedInputSegments : manifold.requestedOutputSegments);

   const activeSegments = isInput ? segmentMapping.input : segmentMapping.output;

   // Determine any that need to be added
   const toAdd = activeSegments.filter((s) => !currentSegmentIds.includes(s.segmentAddress));

   toAdd.forEach((seg) => {
      // Figure out if this is local
      const isLocal = manifold.pipelineInstanceId === seg.pipelineInstanceId;

      // Dispatch the attach action
      dispatch(
         manifoldInstancesSlice.actions.attachRequestedSegment({
            is_input: isInput,
            is_local: isLocal,
            manifold: manifold,
            segment: seg,
         })
      );
   });

   // Determine any that need to be removed
   const toRemove = currentSegmentIds.filter((s) => !activeSegments.map((s) => s.segmentAddress).includes(s));

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

function ensureOneLocal(dispatch: AppDispatch, state: RootState, manifold: IManifoldInstance, isInput: boolean) {
   // If we have 0 local inputs, then all of the remote outputs can be detached
   // If we have 0 local outputs, then all of our remote inputs can be detached
   let numDetached = 0;
   const requestedMap = isInput ? manifold.requestedInputSegments : manifold.requestedOutputSegments;
   if (getNumLocal(requestedMap) === 0) {
      const requestedInvMap = isInput ? manifold.requestedOutputSegments : manifold.requestedInputSegments;
      const requestedInvRemotes: [string, boolean][] = filterMappingByLocality(requestedInvMap, false);

      requestedInvRemotes.forEach(([segId, _]) => {
         const seg = segmentInstancesSelectById(state, segId);
         if (!seg) {
            throw new Error(`Could not find segment with ID: ${segId}`);
         }

         dispatch(
            manifoldInstancesSlice.actions.detachRequestedSegment({
               is_input: !isInput,
               manifold: manifold,
               segment: seg,
            })
         );

         numDetached++;
      });
   }

   return numDetached;
}

export function manifoldInstancesSyncSegments(manifoldId: string) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      let state = getState();

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

      // Determine the list of segments that should be attached to this manifold
      const segment_mapping = determineManifoldSegmentMapping(state, manifold_def);

      // Sync the input and output assignments with the mapping
      syncSegmentNameForManifold(dispatch, state, found, segment_mapping, true);
      syncSegmentNameForManifold(dispatch, state, found, segment_mapping, false);

      // Update the stage, and determine if we should detach any remote segments if we don't have a corresponding local
      // segment. Specifically if a manifold doesn't contain any local inputs, then all of the remote outputs should be
      // detached and if a manifold doesn't have any local outputs then all remote inputs should be detached.
      state = getState();
      // Ensure that we have at least one local input and one local output
      const manifold = manifoldInstancesSelectById(state, manifoldId);
      if (!manifold) {
         throw new Error(`Manifold Instance with ID: ${manifoldId} not found`);
      }

      let numDetached = ensureOneLocal(dispatch, state, manifold, false);
      numDetached += ensureOneLocal(dispatch, state, manifold, true);
      console.log(`Detached ${numDetached} segments from manifold ${manifoldId}`);
   };
}

function manifoldInstanceUpdateActualSegment(
   dispatch: AppDispatch,
   state: RootState,
   manifold: IManifoldInstance,
   isInput: boolean,
   segmentAddress: string,
   isLocal: boolean
) {
   const segment = segmentInstancesSelectBySegmentAddress(state, segmentAddress)[0];
   if (!segment) {
      throw new Error(`Could not find segment with segment address: ${segmentAddress}`);
   }

   const requestedMapping: { [key: string]: boolean } = isInput
      ? manifold.requestedInputSegments
      : manifold.requestedOutputSegments;
   const actualMapping: { [key: string]: boolean } = isInput
      ? manifold.actualInputSegments
      : manifold.actualOutputSegments;

   // figure out if we are adding or removing
   if (segment.segmentAddress in requestedMapping) {
      // segment exists in requested, and the client added it to the actual
      dispatch(
         manifoldInstancesSlice.actions.attachActualSegment({ manifold: manifold, is_input: isInput, segment: segment })
      );

      // Increment the ref count of the segment [{"type": "ManifoldInstance", "id": "id"}]
      dispatch(segmentInstanceIncRefCount({ segment: segment }));
   } else {
      // One of two cases:
      //   1) Server asked the client to remove the segment and they did in which, and now we need to remove it from the actual
      //   2) Client is sending us an invalid segmentId
      if (segment.segmentAddress in actualMapping) {
         dispatch(
            manifoldInstancesSlice.actions.detachActualSegment({
               manifold: manifold,
               is_input: isInput,
               segment: segment,
            })
         );
         dispatch(segmentInstanceDecRefCount({ segment: segment }));
      } else {
         throw new Error(`Actual segment ${segmentAddress} does not match an attached segment`);
      }
   }
}

export function manifoldInstancesUpdateActualSegments(
   manifoldInstanceId: string,
   actualInputSegments: { [key: string]: boolean },
   actualOutputSegments: { [key: string]: boolean }
) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();
      const manifold = manifoldInstancesSelectById(state, manifoldInstanceId);
      if (!manifold) {
         throw new Error(`Could not find manifold with ID: ${manifoldInstanceId}`);
      }

      const actualInputToAdd = Object.entries(actualInputSegments).filter(
         ([segmentAddress]) => !(segmentAddress in manifold.actualInputSegments)
      );
      const actualOutputToAdd = Object.entries(actualOutputSegments).filter(
         ([segmentAddress]) => !(segmentAddress in manifold.actualOutputSegments)
      );
      const actualInputToRemove = Object.entries(manifold.actualInputSegments).filter(
         ([segmentAddress]) => !(segmentAddress in actualInputSegments)
      );
      const actualOutputToRemove = Object.entries(manifold.actualOutputSegments).filter(
         ([segmentAddress]) => !(segmentAddress in actualOutputSegments)
      );

      // perform any adds
      actualInputToAdd.forEach(([segmentAddress, isLocal]) => {
         manifoldInstanceUpdateActualSegment(dispatch, state, manifold, true, segmentAddress, isLocal);
      });

      actualOutputToAdd.forEach(([segmentAddress, isLocal]) => {
         manifoldInstanceUpdateActualSegment(dispatch, state, manifold, false, segmentAddress, isLocal);
      });

      // perform any removes
      actualInputToRemove.forEach(([segmentAddress, isLocal]) => {
         manifoldInstanceUpdateActualSegment(dispatch, state, manifold, true, segmentAddress, isLocal);
      });

      actualOutputToRemove.forEach(([segmentAddress, isLocal]) => {
         manifoldInstanceUpdateActualSegment(dispatch, state, manifold, false, segmentAddress, isLocal);
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

export function manifoldInstancesDestroy(instance: IManifoldInstance) {
   // To allow the watchers to work, we need to set the requested and actual state
   return async (dispatch: AppDispatch, getState: AppGetState) => {
      const state_snapshot = getState();

      // For any found workers, set the requested and actual states to avoid errors
      const found = manifoldInstancesSelectById(getState(), instance.id);

      if (found) {
         // Set the requested to destroyed
         dispatch(
            manifoldInstancesUpdateResourceRequestedState({
               resource: instance,
               status: ResourceRequestedStatus.Requested_Destroyed,
            })
         );

         // Yield here to allow listeners to run
         await yield_immediate();

         // Set the actual to destroyed
         dispatch(
            manifoldInstancesUpdateResourceActualState({
               resource: instance,
               status: ResourceActualStatus.Actual_Destroyed,
            })
         );

         // Yield here to allow listeners to run
         await yield_immediate();

         // Finally, run the remove segment action just to be sure
         if (manifoldInstancesSelectById(getState(), instance.id)) {
            console.warn("ManifoldInstances watcher did not correctly destroy instance. Manually destroying.");
            dispatch(manifoldInstancesRemove(instance));
         }
      }
   };
}

type ManifoldInstancesStateType = ReturnType<typeof manifoldInstancesSlice.getInitialState>;

export const {
   add: manifoldInstancesAdd,
   detachRequestedSegment: manifoldInstancesDetachRequestedSegment,
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
