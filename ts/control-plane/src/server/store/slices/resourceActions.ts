import {
   IManifoldInstance,
   IPipelineInstance,
   IResourceInstance,
   ISegmentInstance,
   IWorker,
} from "@mrc/common/entities";
import { sleep } from "@mrc/common/utils";
import {
   ResourceActualStatus,
   ResourceRequestedStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import {
   manifoldInstancesSelectById,
   manifoldInstancesUpdateResourceActualState,
   manifoldInstancesUpdateResourceRequestedState,
} from "@mrc/server/store/slices/manifoldInstancesSlice";
import {
   pipelineInstancesSelectById,
   pipelineInstancesUpdateResourceActualState,
   pipelineInstancesUpdateResourceRequestedState,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesSelectById,
   segmentInstancesUpdateResourceActualState,
   segmentInstancesUpdateResourceRequestedState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { systemStartRequest, systemStopRequest } from "@mrc/server/store/slices/systemSlice";
import {
   workersSelectById,
   workersUpdateResourceActualState,
   workersUpdateResourceRequestedState,
} from "@mrc/server/store/slices/workersSlice";
import { AppDispatch, AppGetState } from "@mrc/server/store/store";

export type ResourceStateTypes = IWorker | IPipelineInstance | ISegmentInstance | IManifoldInstance;
export type ResourceStateTypeStrings = "Workers" | "PipelineInstances" | "SegmentInstances" | "ManifoldInstances";

export function resourceUpdateRequestedState(
   resourceType: ResourceStateTypeStrings,
   resourceId: string,
   status: ResourceRequestedStatus
) {
   function errorPrefix() {
      return `Cannot update requested state of ${resourceType} with ID: ${resourceId}.`;
   }

   function checkUpdate<ResourceT extends IResourceInstance>(instance: ResourceT | undefined): ResourceT {
      if (!instance) {
         throw new Error(`${errorPrefix()} Object not found`);
      }

      const desiredRequestedStatusNumber = resourceRequestedStatusToNumber(status);

      const trueRequestedStatusNumber = resourceRequestedStatusToNumber(instance.state.requestedStatus);
      const trueActualStatusNumber = resourceActualStatusToNumber(instance.state.actualStatus);

      if (desiredRequestedStatusNumber < trueRequestedStatusNumber) {
         throw new Error(
            `${errorPrefix()} Current requested state ${
               instance.state.requestedStatus
            } is greater than desired requested state ${status}`
         );
      }

      if (desiredRequestedStatusNumber < trueActualStatusNumber) {
         throw new Error(
            `${errorPrefix()} Current actual state ${
               instance.state.actualStatus
            } is greater than desired requested state ${status}`
         );
      }

      return instance;
   }

   return async (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();

      let did_start_request = false;

      try {
         if (!state.system.requestRunning) {
            did_start_request = true;
            dispatch(systemStartRequest("resourceUpdateRequestedState"));
         }

         if (resourceType === "Workers") {
            const found = checkUpdate(workersSelectById(state, resourceId));

            dispatch(workersUpdateResourceRequestedState({ resource: found, status: status }));
         } else if (resourceType === "PipelineInstances") {
            const found = checkUpdate(pipelineInstancesSelectById(state, resourceId));

            dispatch(pipelineInstancesUpdateResourceRequestedState({ resource: found, status: status }));
         } else if (resourceType === "SegmentInstances") {
            const found = checkUpdate(segmentInstancesSelectById(state, resourceId));

            dispatch(segmentInstancesUpdateResourceRequestedState({ resource: found, status: status }));
         } else if (resourceType === "ManifoldInstances") {
            const found = checkUpdate(manifoldInstancesSelectById(state, resourceId));

            dispatch(manifoldInstancesUpdateResourceRequestedState({ resource: found, status: status }));
         } else {
            throw new Error("Unknow resource type");
         }
      } finally {
         if (did_start_request) {
            dispatch(systemStopRequest("resourceUpdateRequestedState"));
         }

         // Finally, we need to await here to allow any listeners to run
         await sleep(0);
      }
   };
}

export function resourceUpdateActualState(
   resourceType: ResourceStateTypeStrings,
   resourceId: string,
   status: ResourceActualStatus
) {
   function errorPrefix() {
      return `Cannot update actual state of ${resourceType} with ID: ${resourceId}.`;
   }

   function checkUpdate<ResourceT extends IResourceInstance>(instance: ResourceT | undefined): ResourceT {
      if (!instance) {
         throw new Error(`${errorPrefix()} Object not found`);
      }

      const desiredActualStatusNumber = resourceActualStatusToNumber(status);

      const trueRequestedStatusNumber = resourceRequestedStatusToNumber(instance.state.requestedStatus);
      const trueActualStatusNumber = resourceActualStatusToNumber(instance.state.actualStatus);

      if (desiredActualStatusNumber < trueActualStatusNumber) {
         throw new Error(
            `${errorPrefix()} Current actual state, ${
               instance.state.requestedStatus
            }, is greater than desired actual state, ${status}`
         );
      }

      if (desiredActualStatusNumber > trueRequestedStatusNumber + 1) {
         throw new Error(
            `${errorPrefix()} Desired actual state Current actual state, ${status}, beyond the allowed value for the current requested state, ${
               instance.state.requestedStatus
            }`
         );
      }

      return instance;
   }

   return async (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();

      let did_start_request = false;

      try {
         if (!state.system.requestRunning) {
            did_start_request = true;
            dispatch(systemStartRequest("resourceUpdateActualState"));
         }

         if (resourceType === "Workers") {
            const found = checkUpdate(workersSelectById(state, resourceId));

            dispatch(workersUpdateResourceActualState({ resource: found, status: status }));
         } else if (resourceType === "PipelineInstances") {
            const found = checkUpdate(pipelineInstancesSelectById(state, resourceId));

            dispatch(pipelineInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else if (resourceType === "SegmentInstances") {
            const found = checkUpdate(segmentInstancesSelectById(state, resourceId));

            dispatch(segmentInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else if (resourceType === "ManifoldInstances") {
            const found = checkUpdate(manifoldInstancesSelectById(state, resourceId));

            dispatch(manifoldInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else {
            throw new Error("Unknow resource type");
         }
      } finally {
         if (did_start_request) {
            dispatch(systemStopRequest("resourceUpdateActualState"));
         }

         // Finally, we need to await here to allow any listeners to run
         await sleep(0);
      }
   };
}
