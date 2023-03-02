import {Worker, WorkerStates} from "@mrc/proto/mrc/protos/architect_state";
import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {createWrappedEntityAdapter} from "../../utils";

import type {RootState} from "../store";
import {removeConnection} from "./connectionsSlice";
import {addSegmentInstances} from "./segmentInstancesSlice";

export type IWorker = Omit<Worker, "$type">;

const workersAdapter = createWrappedEntityAdapter<IWorker>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (w) => w.id,
});

export const workersSlice = createSlice({
   name: "workers",
   initialState: workersAdapter.getInitialState(),
   reducers: {
      // addWorker,
      addWorker: (state, action: PayloadAction<IWorker>) => {
         if (workersAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Worker with ID: ${action.payload.id} already exists`);
         }
         workersAdapter.addOne(state, action.payload);
      },
      addWorkers: (state, action: PayloadAction<IWorker[]>) => {
         workersAdapter.addMany(state, action.payload);
      },
      removeWorker: (state, action: PayloadAction<IWorker>) => {
         if (!workersAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Worker with ID: ${action.payload.id} not found`);
         }
         workersAdapter.removeOne(state, action.payload.id);
      },
      activateWorkers: (state, action: PayloadAction<IWorker[]>) => {
         // Check for incorrect IDs
         action.payload.forEach((w) => {
            if (!workersAdapter.getOne(state, w.id))
            {
               throw new Error(`Worker with ID: ${w.id} not found`);
            }
         });

         workersAdapter.getMany(state, action.payload.map((w) => w.id)).forEach((w) => w.state = WorkerStates.Activated);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(removeConnection, (state, action) => {
         // Need to delete any workers associated with that connection
         const connection_workers = selectByMachineId(state, action.payload.id);

         workersAdapter.removeMany(state, connection_workers.map((w) => w.id));
      });
      builder.addCase(addSegmentInstances, (state, action) => {
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
   },
});

type WorkersStateType = ReturnType<typeof workersSlice.getInitialState>;

export const {addWorker, addWorkers, removeWorker, activateWorkers} = workersSlice.actions;

export const {
   selectAll: workersSelectAll,
   selectById: workersSelectById,
   selectByIds: workersInstancesSelectByIds,
   selectEntities: workersSelectEntities,
   selectIds: workersSelectIds,
   selectTotal: workersSelectTotal,
} = workersAdapter.getSelectors((state: RootState) => state.workers);

const selectByMachineId = createSelector(
    [workersAdapter.getAll, (state: WorkersStateType, machine_id: number) => machine_id],
    (workers, machine_id) => workers.filter((w) => w.machineId === machine_id));

export const workersSelectByMachineId = (state: RootState, machine_id: number) => selectByMachineId(state.workers,
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
