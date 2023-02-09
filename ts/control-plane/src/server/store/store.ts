import { configureStore } from '@reduxjs/toolkit';
import connectionsReducer from "./slices/connectionsSlice";
import workersReducer from "./slices/workersSlice";

export const store = configureStore({
   reducer: {
      connections: connectionsReducer,
      workers: workersReducer,
   }
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
