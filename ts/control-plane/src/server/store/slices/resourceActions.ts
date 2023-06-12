import { sleep } from "@mrc/common/utils";
import { ResourceActualStatus } from "@mrc/proto/mrc/protos/architect_state";
import {
   manifoldInstancesSelectById,
   manifoldInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/manifoldInstancesSlice";
import {
   pipelineInstancesSelectById,
   pipelineInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesSelectById,
   segmentInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { systemStartRequest, systemStopRequest } from "@mrc/server/store/slices/systemSlice";
import { AppDispatch, AppGetState } from "@mrc/server/store/store";

export function updateResourceActualState(
   resourceType: "PipelineInstances" | "SegmentInstances" | "ManifoldInstances",
   resourceId: string,
   status: ResourceActualStatus
) {
   return async (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();

      let did_start_request = false;

      try {
         if (!state.system.requestRunning) {
            did_start_request = true;
            dispatch(systemStartRequest("updateResourceActualState"));
         }

         if (resourceType === "PipelineInstances") {
            // Find the object
            const found = pipelineInstancesSelectById(state, resourceId);

            if (!found) {
               throw new Error(`Could not find resource with ID: ${resourceId}`);
            }

            dispatch(pipelineInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else if (resourceType === "SegmentInstances") {
            // Find the object
            const found = segmentInstancesSelectById(state, resourceId);

            if (!found) {
               throw new Error(`Could not find resource with ID: ${resourceId}`);
            }

            dispatch(segmentInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else if (resourceType === "ManifoldInstances") {
            // Find the object
            const found = manifoldInstancesSelectById(state, resourceId);

            if (!found) {
               throw new Error(`Could not find resource with ID: ${resourceId}`);
            }

            dispatch(manifoldInstancesUpdateResourceActualState({ resource: found, status: status }));
         } else {
            throw new Error("Unknow resource type");
         }
      } finally {
         if (did_start_request) {
            dispatch(systemStopRequest("updateResourceActualState"));
         }

         // Finally, we need to await here to allow any listeners to run
         await sleep(0);
      }
   };
}
