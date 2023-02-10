import { createEntityAdapter, createSlice, current, EntityState, EntityStateAdapter, PayloadAction } from '@reduxjs/toolkit';
import { EntityAdapter, EntityId, PreventAny } from "@reduxjs/toolkit/dist/entities/models";
import { createWrappedEntityAdapter } from "../../utils";
import type { RootState } from "../store";
import { addWorker, addWorkers, IWorker, removeWorker } from "./workersSlice";

export interface IConnection {
   // id is the per machine assigned connection id
   id: number,
   peer_info: string,
   // List of worker IDs associated with this connection
   worker_ids: number[],
}

const connectionsAdapter = createWrappedEntityAdapter<IConnection>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (x) => x.id,
});

function workerAdded(state: ConnectionsStateType, worker: IWorker) {
   // Handle synchronizing a new added worker
   const found_connection = connectionsAdapter.getOne(state, worker.parent_machine_id);

   if (found_connection) {
      found_connection.worker_ids.push(worker.id);
   } else {
      throw new Error("Must add a connection before a worker!");
   }
}

export const connectionsSlice = createSlice({
   name: 'connections',
   initialState: connectionsAdapter.getInitialState(),
   reducers: {
      addConnection: (state, action: PayloadAction<IConnection>) => {
         connectionsAdapter.addOne(state, action.payload);
      },
      removeConnection: (state, action: PayloadAction<IConnection>) => {
         connectionsAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(addWorker, (state, action) => {
         workerAdded(state, action.payload);
      });
      builder.addCase(addWorkers, (state, action) => {

         // Handle synchronizing a new added worker
         action.payload.forEach((p) => {
            workerAdded(state, p);
         });
      });
      builder.addCase(removeWorker, (state, action) => {

         // Handle removing a worker
         const foundConnection = connectionsAdapter.getOne(state, action.payload.parent_machine_id);

         if (foundConnection) {
            const index = foundConnection.worker_ids.findIndex(x => x === action.payload.id);

            if (index !== -1) {
               foundConnection.worker_ids.splice(index, 1);
            }
         } else {
            throw new Error("Must drop all workers before removing a connection");
         }
      });
   }
});

type ConnectionsStateType = ReturnType<typeof connectionsSlice.getInitialState>;

export const { addConnection, removeConnection } = connectionsSlice.actions;

export const {
   selectAll: connectionsSelectAll,
   selectById: connectionsSelectById,
   selectEntities: connectionsSelectEntities,
   selectIds: connectionsSelectIds,
   selectTotal: connectionsSelectTotal,
} = connectionsAdapter.getSelectors((state: RootState) => state.connections);

export default connectionsSlice.reducer;
