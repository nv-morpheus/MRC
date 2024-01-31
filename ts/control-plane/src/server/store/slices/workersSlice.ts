import { createSelector, createSlice, current, PayloadAction } from "@reduxjs/toolkit";

import { IWorker } from "@mrc/common/entities";
import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { createWatcher } from "@mrc/server/store/resourceStateWatcher";
import { createWrappedEntityAdapter } from "@mrc/server/utils";
import { connectionsRemove } from "@mrc/server/store/slices/connectionsSlice";
import { segmentInstancesAdd, segmentInstancesRemove } from "@mrc/server/store/slices/segmentInstancesSlice";
import { AppDispatch, RootState } from "@mrc/server/store/store";

const workersAdapter = createWrappedEntityAdapter<IWorker>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (w) => w.id,
});

export const workersSlice = createSlice({
   name: "workers",
   initialState: workersAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IWorker>) => {
         if (workersAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Worker with ID: ${action.payload.id} already exists`);
         }
         workersAdapter.addOne(state, action.payload);
      },
      remove: (state, action: PayloadAction<IWorker>) => {
         const found = workersAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Worker with ID: ${action.payload.id} not found`);
         }

         if (found.assignedSegmentIds.length > 0) {
            throw new Error(
               `Attempting to delete Worker with ID: ${
                  action.payload.id
               } while it still has active SegmentInstances. Active SegmentInstances: ${JSON.stringify(
                  found.assignedSegmentIds
               )}. Delete SegmentInstances first`
            );
         }

         workersAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: IWorker; status: ResourceRequestedStatus }>
      ) => {
         const found = workersAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Worker with ID: ${action.payload.resource.id} not found`);
         }

         // Set value without checking since thats handled elsewhere
         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: IWorker; status: ResourceActualStatus }>
      ) => {
         const found = workersAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Worker with ID: ${action.payload.resource.id} not found`);
         }

         // Set value without checking since thats handled elsewhere
         found.state.actualStatus = action.payload.status;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any workers associated with that connection
         const connection_workers = selectByMachineId(state, action.payload.id);

         workersAdapter.removeMany(
            state,
            connection_workers.map((w) => w.id)
         );
      });
      builder.addCase(segmentInstancesAdd, (state, action) => {
         // For each, update the worker with the new running instance
         const found = workersAdapter.getOne(state, action.payload.workerId);

         if (!found) {
            throw new Error("No matching worker ID found");
         }

         found.assignedSegmentIds.push(action.payload.id);
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = workersAdapter.getOne(state, action.payload.workerId);

         if (found) {
            const index = found.assignedSegmentIds.findIndex((x) => x === action.payload.id);

            if (index !== -1) {
               found.assignedSegmentIds.splice(index, 1);
            }
         } else {
            throw new Error("Must drop all SegmentInstances before removing a Worker");
         }
      });
   },
});

export function workersAddMany(workers: IWorker[]) {
   // To allow the watchers to work, we need to add all segments individually
   return (dispatch: AppDispatch) => {
      // Loop and dispatch each segment individually
      workers.forEach((s) => {
         dispatch(workersAdd(s));
      });
   };
}

type WorkersStateType = ReturnType<typeof workersSlice.getInitialState>;

export const {
   add: workersAdd,
   remove: workersRemove,
   updateResourceRequestedState: workersUpdateResourceRequestedState,
   updateResourceActualState: workersUpdateResourceActualState,
} = workersSlice.actions;

export const {
   selectAll: workersSelectAll,
   selectById: workersSelectById,
   selectByIds: workersSelectByIds,
   selectEntities: workersSelectEntities,
   selectIds: workersSelectIds,
   selectTotal: workersSelectTotal,
} = workersAdapter.getSelectors((state: RootState) => state.workers);

const selectByMachineId = createSelector(
   [
      (state: WorkersStateType) => workersAdapter.getAll(state),
      (state: WorkersStateType, executorId: string) => executorId,
   ],
   (workers, executorId) => workers.filter((w) => w.executorId === executorId)
);

export const workersSelectByMachineId = (state: RootState, machine_id: string) =>
   selectByMachineId(state.workers, machine_id);

export function workersConfigureSlice() {
   createWatcher(
      "Workers",
      workersAdd,
      workersSelectById,
      async () => {},
      undefined,
      undefined,
      undefined,
      async (instance, listenerApi) => {
         listenerApi.dispatch(workersRemove(instance));
      }
   );

   return workersSlice.reducer;
}
