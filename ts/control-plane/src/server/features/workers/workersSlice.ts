import { CaseReducer, createEntityAdapter, createSelector, createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from "../../store";
import { findConnection } from "./connectionsSlice";
// import { nanoid } from '@reduxjs/toolkit';

export interface IWorker {
   // Generated worker ID
   id: number,
   // Serialized worker address
   worker_address: string,
   // Parent machine this worker belongs to
   parent_machine_id: number,
}

const workersAdapter = createEntityAdapter<IWorker>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (w) => w.id,
});

// interface WorkersState {
//    [worker_id: number]: Worker,
// }

// // Define the initial state using that type
// const initialState: WorkersState = {
// };

// const addWorker: CaseReducer<WorkersState, PayloadAction<Worker>> = (state, action) => {
//    // doesn't actually mutate the state because it uses the Immer library,
//    // which detects changes to a "draft state" and produces a brand new
//    // immutable state based off those changes
//    state.workers.push({
//       id: action.payload.id
//    });
// };

export const workersSlice = createSlice({
   name: 'workers',
   initialState: workersAdapter.getInitialState(),
   reducers: {
      // addWorker,
      addWorker: (state, action: PayloadAction<IWorker>) => {
         workersAdapter.addOne(state, action.payload);
      },
      addWorkers: (state, action: PayloadAction<IWorker[]>) => {
         workersAdapter.addMany(state, action.payload);
      },
      removeWorker: (state, action: PayloadAction<IWorker>) => {
         workersAdapter.removeOne(state, action.payload.id);
      },
      activateWorkers: (state, action: PayloadAction<IWorker>) => {
         workersAdapter.removeOne(state, action.payload.id);
      },
   },
});

export const { addWorker, addWorkers, removeWorker } = workersSlice.actions;

export const {
   selectAll: workersSelectAll,
   selectById: workersSelectById,
   selectEntities: workersSelectEntities,
   selectIds: workersSelectIds,
   selectTotal: workersSelectTotal,
} = workersAdapter.getSelectors((state: RootState) => state.workers);

export const workersSelectByMachineId = createSelector([workersSelectAll, (state: RootState, machine_id: number) => machine_id],
   (workers, machine_id) => workers.filter((w) => w.parent_machine_id === machine_id));

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
