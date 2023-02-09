import { createEntityAdapter, createSlice, current, EntityState, EntityStateAdapter, PayloadAction } from '@reduxjs/toolkit';
import { EntityAdapter, EntityId, PreventAny } from "@reduxjs/toolkit/dist/entities/models";
import { createWrappedEntityAdapter } from "../../utils";
import type { RootState } from "../store";
import { addWorker, addWorkers, removeWorker } from "./workersSlice";

interface IConnection {
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

const localSelectors = connectionsAdapter.getSelectors();

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
         // Handle synchronizing a new added worker
         connectionsAdapter.getOne(state, action.payload.parent_machine_id)?.worker_ids.push(action.payload.id);
      });
      builder.addCase(addWorkers, (state, action) => {

         // Handle synchronizing a new added worker
         action.payload.forEach((p) => {
            connectionsAdapter.getOne(state, p.parent_machine_id)?.worker_ids.push(p.id);
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
         }
      });
   }
});

export const { addConnection, removeConnection } = connectionsSlice.actions;

export const {
   selectAll: connectionsSelectAll,
   selectById: connectionsSelectById,
   selectEntities: connectionsSelectEntities,
   selectIds: connectionsSelectIds,
   selectTotal: connectionsSelectTotal,
} = connectionsAdapter.getSelectors((state: RootState) => state.connections);

export default connectionsSlice.reducer;
