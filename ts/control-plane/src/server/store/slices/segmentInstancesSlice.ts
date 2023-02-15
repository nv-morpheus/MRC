import {createSlice, PayloadAction} from '@reduxjs/toolkit';

import {SegmentInstance} from "../../../proto/mrc/protos/architect_state";
import {createWrappedEntityAdapter} from "../../utils";

import type {RootState} from "../store";

export type ISegmentInstance = Omit<SegmentInstance, "$type">;

const segmentInstancesAdapter = createWrappedEntityAdapter<ISegmentInstance>({
   // sortComparer: (a, b) => b.id.localeCompare(a.date),
   selectId : (w) => w.id,
});

export const segmentInstancesSlice = createSlice({
   name : 'segmentInstances',
   initialState : segmentInstancesAdapter.getInitialState(),
   reducers : {
      // addWorker,
      addSegmentInstance : (state, action: PayloadAction<ISegmentInstance>) => {
         if (segmentInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Segment Instance with ID: ${action.payload.id} already exists`);
         }
         segmentInstancesAdapter.addOne(state, action.payload);
      },
      addSegmentInstances :
          (state,
           action: PayloadAction<ISegmentInstance[]>) => { segmentInstancesAdapter.addMany(state, action.payload);},
      removeSegmentInstance : (state, action: PayloadAction<ISegmentInstance>) => {
         if (!segmentInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Worker with ID: ${action.payload.id} not found`);
         }
         segmentInstancesAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers : (builder) => {
       // builder.addCase(addWorkers, (state, action) => {

       //    // Test, add a new segment instance
       //    segmentInstancesAdapter.addOne(state, {
       //       id: 0,
       //       definitionId: 0,
       //       segmentIds: [0],
       //    });
       // });
   },

});

type SegmentInstancesStateType = ReturnType<typeof segmentInstancesSlice.getInitialState>;

export const {addSegmentInstance, addSegmentInstances, removeSegmentInstance} = segmentInstancesSlice.actions;

export const {
   selectAll : segmentInstancesSelectAll,
   selectById : segmentInstancesSelectById,
   selectByIds : segmentInstancesSelectByIds,
   selectEntities : segmentInstancesSelectEntities,
   selectIds : segmentInstancesSelectIds,
   selectTotal : segmentInstancesSelectTotal,
} = segmentInstancesAdapter.getSelectors((state: RootState) => state.segmentInstances);

export default segmentInstancesSlice.reducer;
