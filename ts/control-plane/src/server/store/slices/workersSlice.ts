import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {createWrappedEntityAdapter} from "../../utils";

import type {RootState} from "../store";
import {connectionsRemove} from "./connectionsSlice";
import {segmentInstancesAdd, segmentInstancesAddMany, segmentInstancesRemove} from "./segmentInstancesSlice";
import {IWorker} from "@mrc/common/entities";
import {ResourceStatus} from "@mrc/proto/mrc/protos/architect_state";

const workersAdapter = createWrappedEntityAdapter<IWorker>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (w) => w.id,
});

export const workersSlice = createSlice({
   name: "workers",
   initialState: workersAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IWorker>) => {
         if (workersAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Worker with ID: ${action.payload.id} already exists`);
         }
         workersAdapter.addOne(state, action.payload);
      },
      addMany: (state, action: PayloadAction<IWorker[]>) => {
         workersAdapter.addMany(state, action.payload);
      },
      remove: (state, action: PayloadAction<IWorker>) => {
         const found = workersAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Worker with ID: ${action.payload.id} not found`);
         }

         if (found.assignedSegmentIds.length > 0)
         {
            throw new Error(`Attempting to delete Worker with ID: ${
                action.payload.id} while it still has active SegmentInstances. Active SegmentInstances: ${
                found.assignedSegmentIds}. Delete SegmentInstances first`);
         }

         workersAdapter.removeOne(state, action.payload.id);
      },
      updateResourceState: (state, action: PayloadAction<{resources: IWorker[], status: ResourceStatus}>) => {
         // Check for incorrect IDs
         action.payload.resources.forEach((w) => {
            if (!workersAdapter.getOne(state, w.id))
            {
               throw new Error(`Worker with ID: ${w.id} not found`);
            }
         });

         workersAdapter.getMany(state, action.payload.resources.map((w) => w.id))
             .forEach((w) => w.state.status = action.payload.status);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any workers associated with that connection
         const connection_workers = selectByMachineId(state, action.payload.id);

         workersAdapter.removeMany(state, connection_workers.map((w) => w.id));
      });
      builder.addCase(segmentInstancesAdd, (state, action) => {
         // For each, update the worker with the new running instance
         const found = workersAdapter.getOne(state, action.payload.workerId);

         if (!found)
         {
            throw new Error("No matching worker ID found");
         }

         found.assignedSegmentIds.push(action.payload.id);
      });
      builder.addCase(segmentInstancesAddMany, (state, action) => {
         // For each, update the worker with the new running instance
         action.payload.forEach((instance) => {
            const foundWorker = workersAdapter.getOne(state, instance.workerId);

            if (!foundWorker)
            {
               throw new Error("No matching worker ID found");
            }

            foundWorker.assignedSegmentIds.push(instance.id);
         });
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = workersAdapter.getOne(state, action.payload.workerId);

         if (found)
         {
            const index = found.assignedSegmentIds.findIndex(x => x === action.payload.id);

            if (index !== -1)
            {
               found.assignedSegmentIds.splice(index, 1);
            }
         }
         else
         {
            throw new Error("Must drop all SegmentInstances before removing a Worker");
         }
      });
   },
});

type WorkersStateType = ReturnType<typeof workersSlice.getInitialState>;

export const {
   add: workersAdd,
   addMany: workersAddMany,
   remove: workersRemove,
   updateResourceState: workersUpdateResourceState,
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
    [workersAdapter.getAll, (state: WorkersStateType, machine_id: string) => machine_id],
    (workers, machine_id) => workers.filter((w) => w.machineId === machine_id));

export const workersSelectByMachineId = (state: RootState, machine_id: string) => selectByMachineId(state.workers,
                                                                                                    machine_id);

// // Other code such as selectors can use the imported `RootState` type
// export const findWorker = (state: RootState, id: number) => {

//    // if (id in state.workers) {
//    //    return state.workers[id];
//    // }

//    return null;
// };

// export const findWorkers = (state: RootState, machine_id: number) => {

//    const connection = findConnection(state, machine_id);

//    if (!connection) {
//       return null;
//    }

//    workersAdapter.;

//    return null;
// };

export default workersSlice.reducer;
