import {devToolsEnhancer} from "@redux-devtools/remote";
import {
   AnyAction,
   combineReducers,
   configureStore,
   createAction,
   createReducer,
   isPlain,
   PreloadedState,
} from "@reduxjs/toolkit";

import connectionsReducer from "./slices/connectionsSlice";
import devToolsReducer from "./slices/devToolsSlice";
import pipelineInstancesReducer from "./slices/pipelineInstancesSlice";
import segmentInstancesReducer from "./slices/segmentInstancesSlice";
import workersReducer from "./slices/workersSlice";

// Create the root reducer separately so we can extract the RootState type
const slicesReducer = combineReducers({
   connections: connectionsReducer,
   workers: workersReducer,
   pipelineInstances: pipelineInstancesReducer,
   segmentInstances: segmentInstancesReducer,
});

export const startAction = createAction("start");
export const stopAction  = createAction("stop");

const rootReducer = createReducer({} as RootState, (builder) => {
   builder.addCase(startAction, (state, action) => {
      console.log("Starting");
      return state;
   });
   builder.addCase(stopAction, (state, action) => {
      console.log("Stopping");
      return state;
   });

   // builder.addMatcher((action: AnyAction): action is AnyAction => action.type.startsWith("devTools/"), (state,
   // action) => {
   //    return devToolsReducer(state, action);
   // });

   // Forward onto the slices
   builder.addDefaultCase((state, action) => {
      return slicesReducer(state, action);
   })
});

export const setupStore = (preloadedState?: PreloadedState<RootState>) => {
   return configureStore({
      reducer: rootReducer,
      preloadedState,
      middleware: (getDefaultMiddleware) => getDefaultMiddleware({
         serializableCheck: {
            isSerializable: (value: unknown) => {
               return isPlain(value) || (value instanceof Uint8Array);
            },
         },
      }),
      // Disable devtools and add it in manually
      devTools: false,
      enhancers:
          [devToolsEnhancer({realtime: true, sendOnError: 1, suppressConnectErrors: false, stopOn: stopAction.type})],
   });
};

const rootStore = setupStore();

export function getRootStore()
{
   return rootStore;
}

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootStore   = ReturnType<typeof setupStore>;
export type RootState   = ReturnType<typeof slicesReducer>;
export type AppDispatch = typeof rootStore.dispatch;
export type AppGetState = typeof rootStore.getState;
