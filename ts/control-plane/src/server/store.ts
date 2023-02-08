import { configureStore } from '@reduxjs/toolkit';
import connectionsReducer from "./features/workers/connectionsSlice";
import workersReducer from "./features/workers/workersSlice";

export const store = configureStore({
   reducer: {
      connections: connectionsReducer,
      workers: workersReducer,
   }
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
