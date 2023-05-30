import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { createWrappedEntityAdapter } from "../../utils";

import type { RootState } from "../store";
import {
   pipelineInstancesRemove,
   pipelineInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import { IManifoldInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { startAppListening } from "@mrc/server/store/listener_middleware";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

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
      addMany: (state, action: PayloadAction<IManifoldInstance[]>) => {
         manifoldInstancesAdapter.addMany(state, action.payload);
      },
      remove: (state, action: PayloadAction<IManifoldInstance>) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state?.actualStatus != ResourceActualStatus.Actual_Destroyed) {
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
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state} is greater than requested state ${action.payload.status}`
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
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.actualStatus = action.payload.status;
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

type ManifoldInstancesStateType = ReturnType<typeof manifoldInstancesSlice.getInitialState>;

export const {
   add: manifoldInstancesAdd,
   addMany: manifoldInstancesAddMany,
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

export function manifoldInstancesConfigureListeners() {
   startAppListening({
      actionCreator: pipelineInstancesUpdateResourceActualState,
      effect: (action, listenerApi) => {
         if (action.payload.status == ResourceActualStatus.Actual_Ready) {
         }
      },
   });
}

export default manifoldInstancesSlice.reducer;
