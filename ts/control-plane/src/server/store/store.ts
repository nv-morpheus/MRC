import { listenerMiddleware } from "@mrc/server/store/listener_middleware";
import {
   combineReducers,
   configureStore,
   createAction,
   createReducer,
   isPlain,
   PreloadedState,
} from "@reduxjs/toolkit";

import { devToolsEnhancer } from "@mrc/server/devTools/index";
import { systemConfigureSlice, systemStartRequest, systemStopRequest } from "@mrc/server/store/slices/systemSlice";
import { customBatcherEnhancer } from "@mrc/server/store/custom_batcher_enhancer";
import { connectionsConfigureSlice } from "@mrc/server/store/slices/connectionsSlice";
import { manifoldInstancesConfigureSlice } from "@mrc/server/store/slices/manifoldInstancesSlice";
import { pipelineDefinitionsConfigureSlice } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesConfigureSlice } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { segmentInstancesConfigureSlice } from "@mrc/server/store/slices/segmentInstancesSlice";
import { workersConfigureSlice } from "@mrc/server/store/slices/workersSlice";

// Create the root reducer separately so we can extract the RootState type
const slicesReducer = combineReducers({
   system: systemConfigureSlice(),
   connections: connectionsConfigureSlice(),
   workers: workersConfigureSlice(),
   pipelineDefinitions: pipelineDefinitionsConfigureSlice(),
   pipelineInstances: pipelineInstancesConfigureSlice(),
   segmentInstances: segmentInstancesConfigureSlice(),
   manifoldInstances: manifoldInstancesConfigureSlice(),
});

export const startAction = createAction("start");
export const stopAction = createAction("stop");

export const startBatch = createAction("startBatch");
export const stopBatch = createAction("stopBatch");

const rootReducer = createReducer({} as RootState, (builder) => {
   builder.addCase(startAction, (state) => {
      console.log("Starting");
      return state;
   });
   builder.addCase(stopAction, (state) => {
      console.log("Stopping");
      return state;
   });

   builder.addCase(startBatch, (state) => {
      console.log("Starting batch");

      return state;
   });

   // builder.addMatcher((action: AnyAction): action is AnyAction => action.type.startsWith("devTools/"), (state,
   // action) => {
   //    return devToolsReducer(state, action);
   // });

   // Forward onto the slices
   builder.addDefaultCase((state, action) => {
      return slicesReducer(state, action);
   });
});

export interface CustomBatcherOptions {
   startBatchAction: any;
   stopBatchAction: any;
}

export const setupStore = (preloadedState?: PreloadedState<RootState>, addDevTools = false) => {
   let enhancers = [
      customBatcherEnhancer({
         startBatchAction: systemStartRequest.type,
         stopBatchAction: systemStopRequest.type,
      }),
   ];

   if (addDevTools) {
      enhancers = [
         ...enhancers,
         devToolsEnhancer({
            port: 9000,
            realtime: true,
            sendOnError: 1,
            suppressConnectErrors: false,
            stopOn: stopAction.type,
         }),
      ];
   }

   return configureStore({
      reducer: rootReducer,
      preloadedState,
      middleware: (getDefaultMiddleware) =>
         getDefaultMiddleware({
            serializableCheck: {
               isSerializable: (value: unknown) => {
                  return isPlain(value) || value instanceof Uint8Array;
               },
            },
         }).prepend(listenerMiddleware.middleware),
      // Disable devtools and add it in manually
      devTools: false,
      enhancers,
   });
};

const rootStore = setupStore();

export function getRootStore() {
   return rootStore;
}

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootStore = ReturnType<typeof setupStore>;
export type RootState = ReturnType<typeof slicesReducer>;
export type AppDispatch = typeof rootStore.dispatch;
export type AppGetState = typeof rootStore.getState;
