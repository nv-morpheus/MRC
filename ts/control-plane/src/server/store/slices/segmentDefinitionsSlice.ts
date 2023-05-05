import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {createWrappedEntityAdapter} from "../../utils";

import type {RootState} from "../store";
import {pipelineDefinitionsRemove} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {PipelineRequestAssignmentRequest_SegmentMapping} from "@mrc/proto/mrc/protos/architect";
import {
   EgressPort,
   IngressPort,
   ScalingOptions,
   SegmentDefinition,
   SegmentOptions,
} from "@mrc/proto/mrc/protos/architect_state";

export type IIngressPort    = Omit<IngressPort, "$type">;
export type IEgressPort     = Omit<EgressPort, "$type">;
export type IScalingOptions = Omit<ScalingOptions, "$type">;
export type ISegmentOptions = Omit<SegmentOptions, "$type">&{
   scalingOptions?: IScalingOptions,
};

export type ISegmentDefinition = Omit<SegmentDefinition, "$type"|"ingressPorts"|"egressPorts"|"options">&{
   ingressPorts: IIngressPort[],
   egressPorts: IEgressPort[],
   options?: ISegmentOptions,
};

export type ISegmentMapping = Omit<PipelineRequestAssignmentRequest_SegmentMapping, "$type">;

const segmentDefinitionsAdapter = createWrappedEntityAdapter<ISegmentDefinition>({
   selectId: (w) => w.id,
});

export const segmentDefinitionsSlice = createSlice({
   name: "segmentDefinitions",
   initialState: segmentDefinitionsAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<ISegmentDefinition>) => {
         if (segmentDefinitionsAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Segment Definition with ID: ${action.payload.id} already exists`);
         }
         segmentDefinitionsAdapter.addOne(state, action.payload);
      },
      addMany: (state, action: PayloadAction<ISegmentDefinition[]>) => {
         segmentDefinitionsAdapter.addMany(state, action.payload);
      },
      remove: (state, action: PayloadAction<ISegmentDefinition>) => {
         const found = segmentDefinitionsAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Segment Definition with ID: ${action.payload.id} not found`);
         }

         if (found.instanceIds.length > 0)
         {
            throw new Error(`Attempting to delete Segment Definition with ID: ${
                action.payload.id} while there are running instances. Stop SegmentInstances first!`)
         }

         segmentDefinitionsAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(pipelineDefinitionsRemove, (state, action) => {
         // Need to delete any segments associated with the pipeline
         const found = selectByPipelineId(state, action.payload.id);

         segmentDefinitionsAdapter.removeMany(state, found.map((x) => x.id));
      });
   },

});

type SegmentDefinitionsStateType = ReturnType<typeof segmentDefinitionsSlice.getInitialState>;

export const {
   add: segmentDefinitionsAdd,
   addMany: segmentDefinitionsAddMany,
   remove: segmentDefinitionsRemove,
} = segmentDefinitionsSlice.actions;

export const {
   selectAll: segmentDefinitionsSelectAll,
   selectById: segmentDefinitionsSelectById,
   selectByIds: segmentDefinitionsSelectByIds,
   selectEntities: segmentDefinitionsSelectEntities,
   selectIds: segmentDefinitionsSelectIds,
   selectTotal: segmentDefinitionsSelectTotal,
} = segmentDefinitionsAdapter.getSelectors((state: RootState) => state.segmentDefinitions);

const selectByPipelineId = createSelector(
    [segmentDefinitionsAdapter.getAll, (state: SegmentDefinitionsStateType, id: number) => id],
    (segmentDefinitions, id) => segmentDefinitions.filter((x) => x.pipelineId === id));

export const segmentDefinitionsSelectByPipelineId = (state: RootState,
                                                     id: number) => selectByPipelineId(state.segmentDefinitions, id);

export default segmentDefinitionsSlice.reducer;
