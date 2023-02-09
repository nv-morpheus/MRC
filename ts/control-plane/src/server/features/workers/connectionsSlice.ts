import { createEntityAdapter, createSlice, current, EntityState, EntityStateAdapter, PayloadAction } from '@reduxjs/toolkit';
import { EntityAdapter, EntityId, PreventAny } from "@reduxjs/toolkit/dist/entities/models";
import type { RootState } from "../../store";
import { addWorker, addWorkers, removeWorker } from "./workersSlice";

interface IConnection {
   // id is the per machine assigned connection id
   id: number,
   peer_info: string,
   // List of worker IDs associated with this connection
   worker_ids: number[],
}

// interface ConnectionsState {
//    [connection_id: number]: IConnection,
// }

// // Define the initial state using that type
// const initialState: ConnectionsState = {
// };

type createEntityAdapterParameters<T> = Parameters<typeof createEntityAdapter<T>>;

interface WrappedEntityAdapter<T> extends EntityAdapter<T> {
   getOne<S extends EntityState<T>>(state: PreventAny<S, T>, id: EntityId): T | undefined;
   getMany<S extends EntityState<T>>(state: PreventAny<S, T>, ids: EntityId[]): T[];
}

function createWrappedEntityAdapter<T>(...args: createEntityAdapterParameters<T>): WrappedEntityAdapter<T> {

   const inner_adapter = createEntityAdapter<T>(...args);

   return {
      ...inner_adapter,
      getOne: (state, id) => {
         if (id in state.entities) {
            return state.entities[id];
         }

         return undefined;
      },
      getMany: (state, ids) => {

         const matched_entities = ids.map((id) => {
            return id in state.entities ? state.entities[id] : undefined;
         }).filter((x): x is T => x != null);

         return matched_entities;
      }
   };
}

// const connectionsAdapter: WrappedEntityStateAdapter<IConnection> = {
//    ...createEntityAdapter<IConnection>({
//       // sortComparer: (a, b) => b.id.localeCompare(a.date),
//       selectId: (x) => x.id,
//    }),
//    getOne: (state, key) => {
//       if (key in state.entities) {
//          return state.entities[key];
//       }

//       return undefined;
//    },
//    getMany: (state, key) => {
//       if (key in state.entities) {
//          return state.entities[key];
//       }

//       return undefined;
//    }
// };

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
         const foundConnection = localSelectors.selectById(state, action.payload.parent_machine_id);

         if (foundConnection) {
            connectionsAdapter.updateOne(state, {
               id: foundConnection.id,
               changes: {
                  worker_ids: [...foundConnection.worker_ids, action.payload.id],
               }
            });
         }
      });
      builder.addCase(addWorkers, (state, action) => {

         action.payload.forEach((p) => {
            // Handle synchronizing a new added worker
            const foundConnection = connectionsAdapter.getOne(state, p.parent_machine_id);

            if (foundConnection) {
               foundConnection.worker_ids.push(p.id);
               // connectionsAdapter.updateOne(state, {
               //    id: foundConnection.id,
               //    changes: {
               //       worker_ids: [...foundConnection.worker_ids, p.id],
               //    }
               // });
            }
         });
      });
      builder.addCase(removeWorker, (state, action) => {
         connectionsAdapter.updateOne;
         // Handle synchronizing a new added worker
         const foundConnection = localSelectors.selectById(state, action.payload.parent_machine_id);

         if (foundConnection) {
            const index = foundConnection.worker_ids.findIndex(x => x === action.payload.id);

            if (index !== -1) {
               connectionsAdapter.updateOne(state, {
                  id: foundConnection.id,
                  changes: {
                     worker_ids: foundConnection.worker_ids.splice(index, 1),
                  }
               });
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


// // Other code such as selectors can use the imported `RootState` type
// export const findConnection = (state: RootState, machine_id: number) => {

//    if (machine_id in state.connections) {
//       return state.connections[machine_id];
//    }

//    return null;
// };

export default connectionsSlice.reducer;
