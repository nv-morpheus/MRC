import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from "../../store";
import { addWorker, addWorkers } from "./workersSlice";

interface Connection {
   // id is the per machine assigned connection id
   id: number,
   peer_info: string,
   // List of worker IDs associated with this connection
   worker_ids: number[],
}

interface ConnectionsState {
   [connection_id: number]: Connection,
}

// Define the initial state using that type
const initialState: ConnectionsState = {
};

export const connectionsSlice = createSlice({
   name: 'connections',
   initialState,
   reducers: {
      addConnection: (state, action: PayloadAction<Connection>) => {
         state[action.payload.id] = action.payload;
      },
      removeConnection: (state, action: PayloadAction<Connection>) => {
         delete state[action.payload.id];
      },
   },
   extraReducers: (builder) => {
      builder.addCase(addWorker, (state, action) => {
         // Handle synchronizing a new added worker
         if (action.payload.parent_machine_id in state) {
            state[action.payload.parent_machine_id].worker_ids.push(action.payload.id);
         }
      });
      builder.addCase(addWorkers, (state, action) => {

         action.payload.forEach((p) => {
            // Handle synchronizing a new added worker
            if (p.parent_machine_id in state) {
               state[p.parent_machine_id].worker_ids.push(p.id);
            }
         });
      });
   }
});

export const { addConnection, removeConnection } = connectionsSlice.actions;

// Other code such as selectors can use the imported `RootState` type
export const findConnection = (state: RootState, machine_id: number) => {

   if (machine_id in state.connections) {
      return state.connections[machine_id];
   }

   return null;
};

export default connectionsSlice.reducer;
