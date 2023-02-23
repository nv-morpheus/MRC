import {createSlice, PayloadAction} from "@reduxjs/toolkit";

import {PipelineInstance} from "../../../proto/mrc/protos/architect_state";
import {createWrappedEntityAdapter, generateId} from "../../utils";

import type {AppDispatch, AppGetState, RootState} from "../store";
import {addSegmentInstances, ISegmentInstance} from "./segmentInstancesSlice";
import {workersSelectByMachineId} from "./workersSlice";

export type IPipelineInstance = Omit<PipelineInstance, "$type">;

const pipelineInstancesAdapter = createWrappedEntityAdapter<IPipelineInstance>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId: (w) => w.id,
});

export const pipelineInstancesSlice = createSlice({
   name: "pipelineInstances",
   initialState: pipelineInstancesAdapter.getInitialState(),
   reducers: {
      // addWorker,
      addPipelineInstance: (state, action: PayloadAction<IPipelineInstance>) => {
         if (pipelineInstancesAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Pipeline Instance with ID: ${action.payload.id} already exists`);
         }
         pipelineInstancesAdapter.addOne(state, action.payload);
      },
      removePipelineInstance: (state, action: PayloadAction<IPipelineInstance>) => {
         if (!pipelineInstancesAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Worker with ID: ${action.payload.id} not found`);
         }
         pipelineInstancesAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(addSegmentInstances, (state, action) => {
         action.payload.forEach((segmentInstance) => {
            // Find the matching pipeline instance for each segment instance
            const foundPipeline = pipelineInstancesAdapter.getOne(state, segmentInstance.pipelineId);

            if (!foundPipeline)
            {
               throw new Error("Did not find matching pipeline for segement");
            }

            foundPipeline.segmentIds.push(segmentInstance.id);
         });
      });
   },

});

export function assignPipelineInstance(
    payload: {machineId: number, pipelineId: number, segmentAssignments: {[id: number]: number;};})
{
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Get the pipeline definition

      // Get the matching segment definitions

      const pipeline_id = generateId();

      // First dispatch the pipeline instance update
      dispatch(addPipelineInstance({
         id: pipeline_id,
         definitionId: payload.pipelineId,
         machineId: payload.machineId,
         segmentIds: [],  // Empty for now
      }));

      // Get the workers for this machine
      const workers = workersSelectByMachineId(getState(), payload.machineId);

      // Then dispatch the segment instances update
      const segments:
          ISegmentInstance[] = Object.entries(payload.segmentAssignments).flatMap(([segmentId, workerId]) => {
             // For each worker, instantiate an copy of the segment
             return workers.map((wid) => {
                return {
                   id: generateId(),
                   definitionId: Number(segmentId),
                   pipelineId: pipeline_id,
                   address: 0,
                   workerId: wid.id,
                };
             });
          });

      dispatch(addSegmentInstances(segments));

      return {
         pipelineId: pipeline_id,
         segmentIds: segments.map((x) => x.id),
      };
   };
}

type PipelineInstancesStateType = ReturnType<typeof pipelineInstancesSlice.getInitialState>;

export const {addPipelineInstance, removePipelineInstance} = pipelineInstancesSlice.actions;

export const {
   selectAll: pipelineInstancesSelectAll,
   selectById: pipelineInstancesSelectById,
   selectEntities: pipelineInstancesSelectEntities,
   selectIds: pipelineInstancesSelectIds,
   selectTotal: pipelineInstancesSelectTotal,
   selectByIds: pipelineInstancesSelectByIds,
} = pipelineInstancesAdapter.getSelectors((state: RootState) => state.pipelineInstances);

export default pipelineInstancesSlice.reducer;
