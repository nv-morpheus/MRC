/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import { CustomBatcherOptions } from "@mrc/server/store/store";
import { StoreEnhancer } from "@reduxjs/toolkit";

export function customBatcherEnhancer(options: CustomBatcherOptions): StoreEnhancer {
   return (next) =>
      (...args) => {
         const store = next(...args);

         let notifying = true;
         let shouldNotifyAtEndOfTick = false;
         let notificationQueued = false;

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
            if (shouldNotifyAtEndOfTick) {
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
                  if (notifying) {
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
               const action_type_name = action?.type;

               // Trigger notifying based on the action type
               if (action_type_name == options.startBatchAction) {
                  notifying = false;
               } else if (action_type_name == options.stopBatchAction) {
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
            },
         });
      };
}
