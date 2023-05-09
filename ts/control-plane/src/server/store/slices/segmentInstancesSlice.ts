import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {createWrappedEntityAdapter} from "../../utils";

import type {RootState} from "../store";
import {connectionsRemove} from "@mrc/server/store/slices/connectionsSlice";
import {pipelineInstancesRemove} from "@mrc/server/store/slices/pipelineInstancesSlice";
import {workersRemove} from "@mrc/server/store/slices/workersSlice";
import {ISegmentInstance} from "@mrc/common/entities";
import {SegmentStates} from "@mrc/proto/mrc/protos/architect_state";

const segmentInstancesAdapter = createWrappedEntityAdapter<ISegmentInstance>({
   selectId: (w) => w.id,
});

export const segmentInstancesSlice = createSlice({
   name: "segmentInstances",
   initialState: segmentInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<ISegmentInstance>) => {
         if (segmentInstancesAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Segment Instance with ID: ${action.payload.id} already exists`);
         }
         segmentInstancesAdapter.addOne(state, action.payload);
      },
      addMany: (state, action: PayloadAction<ISegmentInstance[]>) => {
         segmentInstancesAdapter.addMany(state, action.payload);
      },
      remove: (state, action: PayloadAction<ISegmentInstance>) => {
         const found = segmentInstancesAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Segment Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state != SegmentStates.Completed)
         {
            throw new Error(`Attempting to delete Segment Instance with ID: ${
                action.payload.id} while it has not finished. Stop SegmentInstance first!`)
         }

         segmentInstancesAdapter.removeOne(state, action.payload.id);
      },
      updateState: (state, action: PayloadAction<Pick<ISegmentInstance, "id"|"state">>) => {
         const found = segmentInstancesAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Segment Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state > action.payload.state)
         {
            throw new Error(`Cannot update state of Instance with ID: ${action.payload.id}. Current state ${
                found.state} is greater than requested state ${action.payload.state}`);
         }

         found.state = action.payload.state;
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any segments associated with the worker
         const instances = selectByWorkerId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(state, instances.map((x) => x.id));
      });
      builder.addCase(workersRemove, (state, action) => {
         // Need to delete any segments associated with the worker
         const instances = selectByWorkerId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(state, instances.map((x) => x.id));
      });
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // Need to delete any segments associated with the pipeline
         const instances = selectByPipelineId(state, action.payload.id);

         segmentInstancesAdapter.removeMany(state, instances.map((x) => x.id));
      });
   },

});

type SegmentInstancesStateType = ReturnType<typeof segmentInstancesSlice.getInitialState>;

export const {
   add: segmentInstancesAdd,
   addMany: segmentInstancesAddMany,
   remove: segmentInstancesRemove,
   updateState: segmentInstancesUpdateState,
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
    (segmentInstances, worker_id) => segmentInstances.filter((x) => x.workerId === worker_id));

export const segmentInstancesSelectByWorkerId = (state: RootState, worker_id: string) => selectByWorkerId(
    state.segmentInstances,
    worker_id);

const selectByPipelineId = createSelector(
    [segmentInstancesAdapter.getAll, (state: SegmentInstancesStateType, pipeline_id: string) => pipeline_id],
    (segmentInstances, pipeline_id) => segmentInstances.filter((x) => x.pipelineInstanceId === pipeline_id));

export const segmentInstancesSelectByPipelineId = (state: RootState, pipeline_id: string) => selectByPipelineId(
    state.segmentInstances,
    pipeline_id);

export default segmentInstancesSlice.reducer;
