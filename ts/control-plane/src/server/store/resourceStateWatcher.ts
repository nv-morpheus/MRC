import { IResourceInstance, ResourceStateTypeStrings } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { AppListenerAPI, startAppListening } from "@mrc/server/store/listener_middleware";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { ActionCreatorWithPayload } from "@reduxjs/toolkit";
import { resourceUpdateRequestedState } from "@mrc/server/store/slices/resourceActions";
import { RootState } from "@mrc/server/store/store";

type ResourceEvent<ResourceT> = (instance: ResourceT, listenerApi: AppListenerAPI) => Promise<void>;

export abstract class ResourceStateWatcher<ResourceT extends IResourceInstance, PayloadT extends { id: string }> {
   constructor(
      protected resourceType: ResourceStateTypeStrings,
      protected actionCreator: ActionCreatorWithPayload<PayloadT>
   ) {}

   public configureListener() {
      startAppListening({
         actionCreator: this.actionCreator,
         effect: async (action, listenerApi) => {
            const instanceId = action.payload.id;

            const instance = this._getResourceInstance(listenerApi.getState(), instanceId);

            if (!instance) {
               throw new Error("Could not find segment instance");
            }

            // // Now that the object has been created, set the requested status to Created
            // await listenerApi.dispatch(
            //    resourceUpdateRequestedState(this.resourceType, instanceId, ResourceRequestedStatus.Requested_Created)
            // );

            const monitor_instance = listenerApi.fork(async () => {
               while (true) {
                  // Wait for the next update
                  const [, current_state] = await listenerApi.take((action, currentState, originalState) => {
                     const currentInstance = this._getResourceInstance(currentState, instanceId);
                     const originalInstance = this._getResourceInstance(originalState, instanceId);

                     // If the object has been removed, dont exit. The entire fork will be cancelled
                     if (!currentInstance || !originalInstance) {
                        return false;
                     }

                     return (
                        resourceActualStatusToNumber(currentInstance.state.actualStatus) >
                        resourceActualStatusToNumber(originalInstance.state.actualStatus)
                     );
                  });

                  if (!current_state.system.requestRunning) {
                     console.warn("Updating resource outside of a request will lead to undefined behavior!");
                  }

                  // Get the status of this instance
                  const instance = this._getResourceInstance(listenerApi.getState(), instanceId);

                  if (!instance) {
                     throw new Error("Could not find instance");
                  }

                  switch (instance.state.actualStatus) {
                     case ResourceActualStatus.Actual_Unknown:
                        throw new Error("Should not get here");
                     case ResourceActualStatus.Actual_Initialized:
                        if (
                           resourceRequestedStatusToNumber(instance.state.requestedStatus) <
                           resourceRequestedStatusToNumber(ResourceRequestedStatus.Requested_Created)
                        ) {
                           // Tell it to move to created
                           await listenerApi.dispatch(
                              resourceUpdateRequestedState(
                                 this.resourceType,
                                 instanceId,
                                 ResourceRequestedStatus.Requested_Created
                              )
                           );
                        }
                        break;
                     case ResourceActualStatus.Actual_Creating:
                     case ResourceActualStatus.Actual_Created:
                        if (
                           resourceRequestedStatusToNumber(instance.state.requestedStatus) <
                           resourceRequestedStatusToNumber(ResourceRequestedStatus.Requested_Completed)
                        ) {
                           // Perform any processing
                           await this._onCreated(instance, listenerApi);

                           // Tell it to move running/completed
                           await listenerApi.dispatch(
                              resourceUpdateRequestedState(
                                 this.resourceType,
                                 instanceId,
                                 ResourceRequestedStatus.Requested_Completed
                              )
                           );
                        }
                        break;
                     case ResourceActualStatus.Actual_Running:
                        {
                           // Perform any processing
                           await this._onRunning(instance, listenerApi);
                        }
                        break;
                     case ResourceActualStatus.Actual_Completed:
                        if (
                           resourceRequestedStatusToNumber(instance.state.requestedStatus) <
                           resourceRequestedStatusToNumber(ResourceRequestedStatus.Requested_Stopped)
                        ) {
                           // TODO(MDD): Before we can move to Stopped, all ref counts must be 0

                           // Perform any processing
                           await this._onCompleted(instance, listenerApi);

                           // Tell it to move to stopped
                           await listenerApi.dispatch(
                              resourceUpdateRequestedState(
                                 this.resourceType,
                                 instanceId,
                                 ResourceRequestedStatus.Requested_Stopped
                              )
                           );
                        }
                        break;
                     case ResourceActualStatus.Actual_Stopping:
                     case ResourceActualStatus.Actual_Stopped:
                        if (
                           resourceRequestedStatusToNumber(instance.state.requestedStatus) <
                           resourceRequestedStatusToNumber(ResourceRequestedStatus.Requested_Destroyed)
                        ) {
                           // Perform any processing
                           await this._onStopped(instance, listenerApi);

                           // Tell it to move to stopped
                           await listenerApi.dispatch(
                              resourceUpdateRequestedState(
                                 this.resourceType,
                                 instanceId,
                                 ResourceRequestedStatus.Requested_Destroyed
                              )
                           );
                        }
                        break;
                     case ResourceActualStatus.Actual_Destroying:
                        // Do nothing
                        break;
                     case ResourceActualStatus.Actual_Destroyed:
                        // Now we can actually just remove the object
                        await this._onDestroyed(instance, listenerApi);

                        break;
                     default:
                        throw new Error("Unknow state type");
                  }
               }
            });

            await listenerApi.condition((action, currentState) => {
               // Exit if we cant find the object anymore
               return this._getResourceInstance(currentState, instanceId) === undefined;
            });
            monitor_instance.cancel();
         },
      });
   }

   protected abstract _getResourceInstance(state: RootState, id: string): ResourceT | undefined;

   protected abstract _onCreated(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void>;
   protected abstract _onRunning(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void>;
   protected abstract _onCompleted(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void>;
   protected abstract _onStopped(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void>;
   protected abstract _onDestroyed(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void>;
}

export class ResourceStateWatcherLambda<
   ResourceT extends IResourceInstance,
   PayloadT extends { id: string }
> extends ResourceStateWatcher<ResourceT, PayloadT> {
   constructor(
      resourceType: ResourceStateTypeStrings,
      actionCreator: ActionCreatorWithPayload<PayloadT>,
      private getResourceInstance: (state: RootState, id: string) => ResourceT | undefined,
      private onCreated?: ResourceEvent<ResourceT>,
      private onRunning?: ResourceEvent<ResourceT>,
      private onCompleted?: ResourceEvent<ResourceT>,
      private onStopped?: ResourceEvent<ResourceT>,
      private onDestroyed?: ResourceEvent<ResourceT>
   ) {
      super(resourceType, actionCreator);
   }

   protected _getResourceInstance(state: RootState, id: string): ResourceT | undefined {
      return this.getResourceInstance(state, id);
   }
   protected _onCreated(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void> {
      if (!this.onCreated) {
         return Promise.resolve();
      }

      return this.onCreated(instance, listenerApi);
   }

   protected _onRunning(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void> {
      if (!this.onRunning) {
         return Promise.resolve();
      }

      return this.onRunning(instance, listenerApi);
   }

   protected _onCompleted(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void> {
      if (!this.onCompleted) {
         return Promise.resolve();
      }

      return this.onCompleted(instance, listenerApi);
   }
   protected _onStopped(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void> {
      if (!this.onStopped) {
         return Promise.resolve();
      }

      return this.onStopped(instance, listenerApi);
   }
   protected _onDestroyed(instance: ResourceT, listenerApi: AppListenerAPI): Promise<void> {
      if (!this.onDestroyed) {
         return Promise.resolve();
      }

      return this.onDestroyed(instance, listenerApi);
   }
}

export function createWatcher<ResourceT extends IResourceInstance, PayloadT extends { id: string }>(
   resourceTypeString: ResourceStateTypeStrings,
   actionCreator: ActionCreatorWithPayload<PayloadT>,
   getResourceInstance: (state: RootState, id: string) => ResourceT | undefined,
   onCreated?: ResourceEvent<ResourceT>,
   onRunning?: ResourceEvent<ResourceT>,
   onCompleted?: ResourceEvent<ResourceT>,
   onStopped?: ResourceEvent<ResourceT>,
   onDestroyed?: ResourceEvent<ResourceT>
) {
   const watcher = new ResourceStateWatcherLambda<ResourceT, PayloadT>(
      resourceTypeString,
      actionCreator,
      getResourceInstance,
      onCreated,
      onRunning,
      onCompleted,
      onStopped,
      onDestroyed
   );

   watcher.configureListener();

   return watcher;
}
