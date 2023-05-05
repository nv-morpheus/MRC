import {devToolsEnhancer} from "@redux-devtools/remote";
import {
   Action,
   autoBatchEnhancer,
   combineReducers,
   configureStore,
   createAction,
   createReducer,
   Dispatch,
   isPlain,
   PreloadedState,
   StoreEnhancer,
} from "@reduxjs/toolkit";

import connectionsReducer from "./slices/connectionsSlice";
import pipelineDefinitionsReducer from "./slices/pipelineDefinitionsSlice";
// import devToolsReducer from "./slices/devToolsSlice";
import pipelineInstancesReducer from "./slices/pipelineInstancesSlice";
import segmentDefinitionsReducer from "./slices/segmentDefinitionsSlice";
import segmentInstancesReducer from "./slices/segmentInstancesSlice";
import systemReducer, {systemStartRequest, systemStopRequest} from "./slices/systemSlice";
import workersReducer from "./slices/workersSlice";

// Create the root reducer separately so we can extract the RootState type
const slicesReducer = combineReducers({
   system: systemReducer,
   connections: connectionsReducer,
   workers: workersReducer,
   pipelineDefinitions: pipelineDefinitionsReducer,
   pipelineInstances: pipelineInstancesReducer,
   segmentDefinitions: segmentDefinitionsReducer,
   segmentInstances: segmentInstancesReducer,
});

export const startAction = createAction("start");
export const stopAction  = createAction("stop");

export const startBatch = createAction("startBatch");
export const stopBatch  = createAction("stopBatch");

const rootReducer = createReducer({} as RootState, (builder) => {
   builder.addCase(startAction, (state, action) => {
      console.log("Starting");
      return state;
   });
   builder.addCase(stopAction, (state, action) => {
      console.log("Stopping");
      return state;
   });

   builder.addCase(startBatch, (state, action) => {
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
   })
});

export interface CustomBatcherOptions {
   startBatchAction: any;
   stopBatchAction: any;
}

export function customBatcherEnhancer(options: CustomBatcherOptions): StoreEnhancer
{
   return (next) => (...args) => {
      const store = next(...args);

      let notifying               = true;
      let shouldNotifyAtEndOfTick = false;
      let notificationQueued      = false;

      const listeners = new Set<() => void>();

      // const queueCallback = options.type === 'tick'
      //    ? queueMicrotaskShim
      //    : options.type === 'raf'
      //       ? rAF
      //       : options.type === 'callback'
      //          ? options.queueNotification
      //          : createQueueWithTimer(options.timeout);

      const notifyListeners = () => {
         // We're running at the end of the event loop tick.
         // Run the real listener callbacks to actually update the UI.
         notificationQueued = false;
         if (shouldNotifyAtEndOfTick)
         {
            shouldNotifyAtEndOfTick = false;
            listeners.forEach((l) => l());
         }
      };

      return Object.assign({}, store, {
         // Override the base `store.subscribe` method to keep original listeners
         // from running if we're delaying notifications
         subscribe(listener: () => void) {
            // Each wrapped listener will only call the real listener if
            // the `notifying` flag is currently active when it's called.
            // This lets the base store work as normal, while the actual UI
            // update becomes controlled by this enhancer.
            const wrappedListener: typeof listener = () => {
               if (notifying)
               {
                  listener();
               }
            };
            const unsubscribe = store.subscribe(wrappedListener);
            listeners.add(listener);
            return () => {
               unsubscribe();
               listeners.delete(listener);
            };
         },
         // Override the base `store.dispatch` method so that we can check actions
         // for the `shouldAutoBatch` flag and determine if batching is active
         dispatch(action: any) {
            try
            {
               const action_type_name = action?.type;

               // Trigger notifying based on the action type
               if (action_type_name == options.startBatchAction)
               {
                  notifying = false;
               }
               else if (action_type_name == options.stopBatchAction)
               {
                  notifying = true;
               }

               // // If a `notifyListeners` microtask was queued, you can't cancel it.
               // // Instead, we set a flag so that it's a no-op when it does run
               // shouldNotifyAtEndOfTick = !notifying;
               // if (shouldNotifyAtEndOfTick)
               // {
               //    // We've seen at least 1 action with `SHOULD_AUTOBATCH`. Try to queue
               //    // a microtask to notify listeners at the end of the event loop tick.
               //    // Make sure we only enqueue this _once_ per tick.
               //    if (!notificationQueued)
               //    {
               //       notificationQueued = true;
               //       queueCallback(notifyListeners);
               //    }
               // }
               // Go ahead and process the action as usual, including reducers.
               // If normal notification behavior is enabled, the store will notify
               // all of its own listeners, and the wrapper callbacks above will
               // see `notifying` is true and pass on to the real listener callbacks.
               // If we're "batching" behavior, then the wrapped callbacks will
               // bail out, causing the base store notification behavior to be no-ops.
               return store.dispatch(action);
            } finally
            {
               // Assume we're back to normal behavior after each action
               notifying = true;
            }
         },
      });
   };
}

export const setupStore = (preloadedState?: PreloadedState<RootState>, addDevTools: boolean = false) => {
   let enhancers = undefined;

   if (addDevTools)
   {
      enhancers = [
         customBatcherEnhancer({
            startBatchAction: systemStartRequest.type,
            stopBatchAction: systemStopRequest.type,
         }),
         devToolsEnhancer(
             {port: 9000, realtime: true, sendOnError: 1, suppressConnectErrors: false, stopOn: stopAction.type}),
      ]
   }

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
      enhancers,
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
