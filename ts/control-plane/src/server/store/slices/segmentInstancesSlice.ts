import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { createWrappedEntityAdapter } from "../../utils";

import type { RootState } from "../store";
import { connectionsRemove } from "@mrc/server/store/slices/connectionsSlice";
import {
   pipelineInstancesRemove,
   pipelineInstancesUpdateResourceActualState,
} from "@mrc/server/store/slices/pipelineInstancesSlice";
import { workersRemove, workersSelectByMachineId } from "@mrc/server/store/slices/workersSlice";
import { ISegmentInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
   SegmentMappingPolicies,
} from "@mrc/proto/mrc/protos/architect_state";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { startAppListening } from "@mrc/server/store/listener_middleware";
import { generateSegmentHash } from "@mrc/common/utils";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

const segmentInstancesAdapter = createWrappedEntityAdapter<ISegmentInstance>({
   selectId: (w) => w.id,
});

export const segmentInstancesSlice = createSlice({
   name: "segmentInstances",
   initialState: segmentInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<ISegmentInstance>) => {
         if (segmentInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Segment Instance with ID: ${action.payload.id} already exists`);
         }
         segmentInstancesAdapter.addOne(state, action.payload);
      },
      addMany: (state, action: PayloadAction<ISegmentInstance[]>) => {
         segmentInstancesAdapter.addMany(state, action.payload);
      },
      remove: (state, action: PayloadAction<ISegmentInstance>) => {
         const found = segmentInstancesAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Segment Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state?.actualStatus != ResourceActualStatus.Actual_Destroyed) {
            throw new Error(
               `Attempting to delete Segment Instance with ID: ${action.payload.id} while it has not finished. Stop SegmentInstance first!`
            );
         }

         segmentInstancesAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: ISegmentInstance; status: ResourceRequestedStatus }>
      ) => {
         const found = segmentInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Segment Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceRequestedStatusToNumber(found.state.requestedStatus) >
            resourceRequestedStatusToNumber(action.payload.status)
         ) {
            throw new Error(
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: ISegmentInstance; status: ResourceActualStatus }>
      ) => {
         const found = segmentInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Segment Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceActualStatusToNumber(found.state.actualStatus) > resourceActualStatusToNumber(action.payload.status)
         ) {
            throw new Error(
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state.actualStatus} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.actualStatus = action.payload.status;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any segments associated with the worker
         const instances = selectByWorkerId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(
            state,
            instances.map((x) => x.id)
         );
      });
      builder.addCase(workersRemove, (state, action) => {
         // Need to delete any segments associated with the worker
         const instances = selectByWorkerId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(
            state,
            instances.map((x) => x.id)
         );
      });
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // Need to delete any segments associated with the pipeline
         const instances = selectByPipelineId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(
            state,
            instances.map((x) => x.id)
         );
      });
   },
});

type SegmentInstancesStateType = ReturnType<typeof segmentInstancesSlice.getInitialState>;

export const {
   add: segmentInstancesAdd,
   addMany: segmentInstancesAddMany,
   remove: segmentInstancesRemove,
   updateResourceRequestedState: segmentInstancesUpdateResourceRequestedState,
   updateResourceActualState: segmentInstancesUpdateResourceActualState,
} = segmentInstancesSlice.actions;

export const {
   selectAll: segmentInstancesSelectAll,
   selectById: segmentInstancesSelectById,
   selectByIds: segmentInstancesSelectByIds,
   selectEntities: segmentInstancesSelectEntities,
   selectIds: segmentInstancesSelectIds,
   selectTotal: segmentInstancesSelectTotal,
} = segmentInstancesAdapter.getSelectors((state: RootState) => state.segmentInstances);

const selectByWorkerId = createSelector(
   [segmentInstancesAdapter.getAll, (state: SegmentInstancesStateType, worker_id: string) => worker_id],
   (segmentInstances, worker_id) => segmentInstances.filter((x) => x.workerId === worker_id)
);

export const segmentInstancesSelectByWorkerId = (state: RootState, worker_id: string) =>
   selectByWorkerId(state.segmentInstances, worker_id);

const selectByPipelineId = createSelector(
   [segmentInstancesAdapter.getAll, (state: SegmentInstancesStateType, pipeline_id: string) => pipeline_id],
   (segmentInstances, pipeline_id) => segmentInstances.filter((x) => x.pipelineInstanceId === pipeline_id)
);

export const segmentInstancesSelectByPipelineId = (state: RootState, pipeline_id: string) =>
   selectByPipelineId(state.segmentInstances, pipeline_id);

export function segmentInstancesConfigureListeners() {
   startAppListening({
      actionCreator: pipelineInstancesUpdateResourceActualState,
      effect: (action, listenerApi) => {
         if (action.payload.status == ResourceActualStatus.Actual_Running) {
            // Pipeline has been marked as ready. Update segment instances based on the pipeline config
            const pipeline_def = pipelineDefinitionsSelectById(
               listenerApi.getState(),
               action.payload.resource.definitionId
            );

            if (!pipeline_def) {
               throw new Error(
                  `Could not find Pipeline Definition ID: ${action.payload.resource.definitionId}, for Pipeline Instance: ${action.payload.resource.id}`
               );
            }

            // Get the mapping for this machine ID
            if (!(action.payload.resource.machineId in pipeline_def.mappings)) {
               throw new Error(
                  `Could not find Mapping for Machine: ${action.payload.resource.machineId}, for Pipeline Definition: ${action.payload.resource.definitionId}`
               );
            }

            // Get the workers for this machine
            const workers = workersSelectByMachineId(listenerApi.getState(), action.payload.resource.machineId);

            const mapping = pipeline_def.mappings[action.payload.resource.machineId];

            // Now determine the segment instances that should be created
            const seg_to_workers = Object.fromEntries(
               Object.entries(mapping.segments).map(([seg_name, seg_map]) => {
                  let workerIds: string[] = [];

                  if (seg_map.byPolicy) {
                     if (seg_map.byPolicy.value == SegmentMappingPolicies.OnePerWorker) {
                        workerIds = workers.map((x) => x.id);
                     } else {
                        throw new Error(`Unsupported policy: ${seg_map.byPolicy.value}`);
                     }
                  } else if (seg_map.byWorker) {
                     workerIds = seg_map.byWorker.workerIds;
                  } else {
                     throw new Error(`Invalid SegmentMap for ${seg_name}. No option set`);
                  }

                  return [seg_name, workerIds];
               })
            );

            // Now generate the segments that would need to be created
            const segments = Object.entries(seg_to_workers).flatMap(([seg_name, seg_assignment]) => {
               // For each assignment, create a segment instance
               return seg_assignment.map((wid) => {
                  const address = generateSegmentHash(seg_name, wid);

                  return {
                     id: address.toString(),
                     pipelineDefinitionId: pipeline_def.id,
                     pipelineInstanceId: action.payload.resource.id,
                     name: seg_name,
                     address: address,
                     workerId: wid,
                     state: {
                        refCount: 0,
                        requestedStatus: ResourceRequestedStatus.Requested_Created,
                        actualStatus: ResourceActualStatus.Actual_Unknown,
                     },
                  } as ISegmentInstance;
               });
            });

            // Filter out any running ones

            // Then dispatch the segment instances update
            listenerApi.dispatch(segmentInstancesAddMany(segments));
         }
      },
   });
}

export default segmentInstancesSlice.reducer;
