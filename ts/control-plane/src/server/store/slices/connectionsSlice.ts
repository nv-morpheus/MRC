import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { createSlice, PayloadAction } from "@reduxjs/toolkit";

import { createWrappedEntityAdapter } from "../../utils.js";

import type { AppDispatch, AppGetState, RootState } from "../store.js";
import {
   pipelineInstancesAdd,
   pipelineInstancesRemove,
   pipelineInstancesSelectByIds,
} from "./pipelineInstancesSlice.js";
import { workersAdd, workersRemove, workersSelectByIds } from "./workersSlice.js";
import { segmentInstancesDestroy, segmentInstancesSelectByIds } from "@mrc/server/store/slices/segmentInstancesSlice";
import { systemStartRequest, systemStopRequest } from "@mrc/server/store/slices/systemSlice";
import { IConnection, IWorker } from "@mrc/common/entities";
import { createWatcher } from "@mrc/server/store/resourceStateWatcher";

const connectionsAdapter = createWrappedEntityAdapter<IConnection>({
   selectId: (x) => x.id,
});

function workerAdded(state: ConnectionsStateType, worker: IWorker) {
   // Handle synchronizing a new added worker
   const found_connection = connectionsAdapter.getOne(state, worker.machineId);

   if (found_connection) {
      found_connection.workerIds.push(worker.id);
   } else {
      throw new Error("Must add a connection before a worker!");
   }
}

export const connectionsSlice = createSlice({
   name: "connections",
   initialState: connectionsAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IConnection>) => {
         if (connectionsAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Connection with ID: ${action.payload.id} already exists`);
         }
         connectionsAdapter.addOne(state, {
            ...action.payload,
            workerIds: [],
            assignedPipelineIds: [],
            state: {
               actualStatus: ResourceActualStatus.Actual_Created,
               refCount: 0,
               requestedStatus: ResourceRequestedStatus.Requested_Completed,
            },
         });
      },
      remove: (state, action: PayloadAction<IConnection>) => {
         const found = connectionsAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Connection with ID: ${action.payload.id} not found`);
         }

         if (found.workerIds.length > 0) {
            throw new Error(
               `Attempting to delete Connection with ID: ${
                  action.payload.id
               } while it still has active workers. Active workers: ${JSON.stringify(
                  found.workerIds
               )}. Delete workers first`
            );
         }

         if (found.assignedPipelineIds.length > 0) {
            throw new Error(
               `Attempting to delete Connection with ID: ${
                  action.payload.id
               } while it still has active pipeline instances. Active PipelineInstances: ${JSON.stringify(
                  found.assignedPipelineIds
               )}. Delete PipelineInstances first`
            );
         }

         connectionsAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: IConnection; status: ResourceRequestedStatus }>
      ) => {
         const found = connectionsAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Connection with ID: ${action.payload.resource.id} not found`);
         }

         // Set value without checking since thats handled elsewhere
         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: IConnection; status: ResourceActualStatus }>
      ) => {
         const found = connectionsAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Connection with ID: ${action.payload.resource.id} not found`);
         }

         // Set value without checking since thats handled elsewhere
         found.state.actualStatus = action.payload.status;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(workersAdd, (state, action) => {
         workerAdded(state, action.payload);
      });
      builder.addCase(workersRemove, (state, action) => {
         // Handle removing a worker
         const foundConnection = connectionsAdapter.getOne(state, action.payload.machineId);

         if (foundConnection) {
            const index = foundConnection.workerIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               foundConnection.workerIds.splice(index, 1);
            }
         } else {
            throw new Error("Must drop all workers before removing a connection");
         }
      });
      builder.addCase(pipelineInstancesAdd, (state, action) => {
         // Handle removing a worker
         const foundConnection = connectionsAdapter.getOne(state, action.payload.machineId);

         if (foundConnection) {
            foundConnection.assignedPipelineIds.push(action.payload.id);
         } else {
            throw new Error("Cannot add a pipeline. Connection does not exist");
         }
      });
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // Handle removing a worker
         const foundConnection = connectionsAdapter.getOne(state, action.payload.machineId);

         if (foundConnection) {
            const index = foundConnection.assignedPipelineIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               foundConnection.assignedPipelineIds.splice(index, 1);
            }
         } else {
            throw new Error("Cannot remove pipeline instance, connection not found.");
         }
      });
   },
});

export function connectionsDropOne(payload: Pick<IConnection, "id">) {
   return async (dispatch: AppDispatch, getState: AppGetState) => {
      // Get the state once and use that to make sure we are consistent
      const state_snapshot = getState();

      // First, find the matching connection
      const connection = connectionsSelectById(state_snapshot, payload.id);

      if (!connection) {
         console.warn(`Connection '${payload.id}' lost, but it was not found in the state. Ignoring'`);
         return;
      }

      // Then, find any matching workers
      const workers = workersSelectByIds(state_snapshot, connection.workerIds);

      // Now find matching pipelines
      const pipelines = pipelineInstancesSelectByIds(state_snapshot, connection.assignedPipelineIds);

      // Finally, find matching segments
      const seg_ids = pipelines.reduce((sum_ids: string[], curr) => sum_ids.concat(curr.segmentIds), []);

      const segments = segmentInstancesSelectByIds(state_snapshot, seg_ids);

      // Remove them all in reverse order
      try {
         // Start a batch to avoid many notifications
         dispatch(systemStartRequest(`Dropping Connection: ${payload.id}`));

         for (const x of segments) {
            // Need to set the state first
            await dispatch(segmentInstancesDestroy(x));
         }

         pipelines.forEach((x) => dispatch(pipelineInstancesRemove(x)));

         workers.forEach((x) => dispatch(workersRemove(x)));

         dispatch(connectionsRemove(connection));
      } finally {
         dispatch(systemStopRequest(`Dropping Connection: ${payload.id}`));
      }
   };
}

type ConnectionsStateType = ReturnType<typeof connectionsSlice.getInitialState>;

export const {
   add: connectionsAdd,
   remove: connectionsRemove,
   updateResourceRequestedState: connectionsUpdateResourceRequestedState,
   updateResourceActualState: connectionsUpdateResourceActualState,
} = connectionsSlice.actions;

export const {
   selectAll: connectionsSelectAll,
   selectById: connectionsSelectById,
   selectByIds: connectionsSelectByIds,
   selectEntities: connectionsSelectEntities,
   selectIds: connectionsSelectIds,
   selectTotal: connectionsSelectTotal,
} = connectionsAdapter.getSelectors((state: RootState) => state.connections);

export function connectionsConfigureSlice() {
   createWatcher(
      "Connections",
      connectionsAdd,
      connectionsSelectById,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined
   );

   return connectionsSlice.reducer;
}
