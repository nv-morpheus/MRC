import { IPipelineConfiguration, IPipelineInstance, IPipelineMapping, ISegmentInstance } from "@mrc/common/entities";
import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { connectionsRemove } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsCreateOrUpdate } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { AppDispatch, AppGetState, RootState } from "@mrc/server/store/store";
import { createWrappedEntityAdapter, generateId } from "@mrc/server/utils";
import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { segmentInstancesAdd, segmentInstancesAddMany, segmentInstancesRemove } from "./segmentInstancesSlice";
import { startAppListening } from "@mrc/server/store/listener_middleware";

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
      builder.addCase(segmentInstancesAddMany, (state, action) => {
         action.payload.forEach((segmentInstance) => {
            segmentInstanceAdded(state, segmentInstance);
         });
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
   },
});

export function pipelineInstancesAssign(payload: { pipeline: IPipelineConfiguration; mapping: IPipelineMapping }) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Dispatch the definition to get the definition IDs
      const definition_ids = dispatch(pipelineDefinitionsCreateOrUpdate(payload.pipeline, payload.mapping));

      const pipeline_id = generateId();

      // First dispatch the pipeline instance update
      dispatch(
         pipelineInstancesAdd({
            id: pipeline_id,
            definitionId: definition_ids.pipeline,
            machineId: payload.mapping.machineId,
         })
      );

      // // Get the workers for this machine
      // const workers = workersSelectByMachineId(getState(), payload.machineId);

      // if (payload.assignments.length == 0)
      // {
      //    // Default to auto assignment of one segment instance per worker per definition
      //    const assignment = Object.entries(payload.pipeline.segments).map(([seg_name, seg_config]) => {
      //       return {segmentName: seg_name, workerIds: workers.map((x) => x.id)} as ISegmentMapping;
      //    });
      // }

      // const segments = payload.assignments.flatMap((assign) => {  // For each worker, create a segment instance
      //    return assign.workerIds.map((wid) => {
      //       return {
      //          id: generateId(),
      //          pipelineDefinitionId: definition_ids.pipeline,
      //          pipelineInstanceId: pipeline_id,
      //          name: assign.segmentName,
      //          address: 0,
      //          workerId: wid,
      //          state: SegmentStates.Initialized,
      //       } as ISegmentInstance;
      //    });
      // });

      // // Then dispatch the segment instances update
      // dispatch(segmentInstancesAddMany(segments));

      return {
         pipelineDefinitionId: definition_ids.pipeline,
         pipelineInstanceId: pipeline_id,
      };
   };
}

type PipelineInstancesStateType = ReturnType<typeof pipelineInstancesSlice.getInitialState>;

export const {
   add: pipelineInstancesAdd,
   remove: pipelineInstancesRemove,
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

export function pipelineInstancesConfigureListeners() {
   startAppListening({
      actionCreator: pipelineInstancesAdd,
      effect: async (action, listenerApi) => {
         const pipeline_id = action.payload.id;

         let pipeline_instance = pipelineInstancesSelectById(listenerApi.getState(), pipeline_id);

         if (!pipeline_instance) {
            throw new Error("Could not find instance");
         }

         // Now that the object has been created, set the requested status to Created
         listenerApi.dispatch(
            pipelineInstancesSlice.actions.updateResourceRequestedState({
               resource: pipeline_instance,
               status: ResourceRequestedStatus.Requested_Created,
            })
         );

         while (true) {
            // Wait for the next update
            const [update_action, current_state] = await listenerApi.take((action) => {
               return (
                  pipelineInstancesUpdateResourceActualState.match(action) && action.payload.resource.id === pipeline_id
               );
            });

            // Get the status of this instance
            pipeline_instance = pipelineInstancesSelectById(listenerApi.getState(), pipeline_id);

            if (!pipeline_instance) {
               throw new Error("Could not find instance");
            }

            if (pipeline_instance.state.actualStatus === ResourceActualStatus.Actual_Created) {
               // Tell it to move running/completed
               listenerApi.dispatch(
                  pipelineInstancesSlice.actions.updateResourceRequestedState({
                     resource: pipeline_instance,
                     status: ResourceRequestedStatus.Requested_Created,
                  })
               );
            } else if (pipeline_instance.state.actualStatus === ResourceActualStatus.Actual_Completed) {
               // Tell it to move to stopped
               listenerApi.dispatch(
                  pipelineInstancesSlice.actions.updateResourceRequestedState({
                     resource: pipeline_instance,
                     status: ResourceRequestedStatus.Requested_Stopped,
                  })
               );
            } else if (pipeline_instance.state.actualStatus === ResourceActualStatus.Actual_Stopped) {
               // Tell it to move to stopped
               listenerApi.dispatch(
                  pipelineInstancesSlice.actions.updateResourceRequestedState({
                     resource: pipeline_instance,
                     status: ResourceRequestedStatus.Requested_Destroyed,
                  })
               );
            } else if (pipeline_instance.state.actualStatus === ResourceActualStatus.Actual_Destroyed) {
               // Now we can actually just remove the object
               listenerApi.dispatch(pipelineInstancesRemove(pipeline_instance));

               break;
            } else {
               throw new Error("Unknow state type");
            }
         }

         console.log("tests");
      },
   });
}

export default pipelineInstancesSlice.reducer;
