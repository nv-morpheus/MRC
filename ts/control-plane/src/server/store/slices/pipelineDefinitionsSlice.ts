import {createSlice, PayloadAction} from "@reduxjs/toolkit";

import {PipelineDefinition} from "../../../proto/mrc/protos/architect_state";
import {createWrappedEntityAdapter, generateId} from "../../utils";

import type {AppDispatch, AppGetState, RootState} from "../store";
import {
   ISegmentDefinition,
   ISegmentMapping,
   segmentDefinitionsAdd,
   segmentDefinitionsAddMany,
   segmentDefinitionsRemove,
} from "@mrc/server/store/slices/segmentDefinitionsSlice";

export type IPipelineDefinition = Omit<PipelineDefinition, "$type">;

const pipelineDefinitionsAdapter = createWrappedEntityAdapter<IPipelineDefinition>({
   selectId: (w) => w.id,
});

function segmentDefinitionAdded(state: PipelineDefinitionsStateType, instance: ISegmentDefinition)
{
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.pipelineId);

   if (found)
   {
      found.segmentIds.push(instance.id);
   }
   else
   {
      throw new Error("Must add a PipelineDefinition before a SegmentDefinition!");
   }
}

export const pipelineDefinitionsSlice = createSlice({
   name: "pipelineDefinitions",
   initialState: pipelineDefinitionsAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<Pick<IPipelineDefinition, "id">>) => {
         if (pipelineDefinitionsAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} already exists`);
         }
         pipelineDefinitionsAdapter.addOne(state, {...action.payload, instanceIds: [], segmentIds: []});
      },
      remove: (state, action: PayloadAction<IPipelineDefinition>) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} not found`);
         }

         if (found.segmentIds.length > 0)
         {
            throw new Error(`Attempting to delete Pipeline Definition with ID: ${
                action.payload.id} with running segment instance. Remove segment instances first!`)
         }

         pipelineDefinitionsAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(segmentDefinitionsAdd, (state, action) => {
         segmentDefinitionAdded(state, action.payload);
      });
      builder.addCase(segmentDefinitionsAddMany, (state, action) => {
         action.payload.forEach((segmentDefinition) => {
            segmentDefinitionAdded(state, segmentDefinition);
         });
      });
      builder.addCase(segmentDefinitionsRemove, (state, action) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.pipelineId);

         if (found)
         {
            const index = found.segmentIds.findIndex(x => x === action.payload.id);

            if (index !== -1)
            {
               found.segmentIds.splice(index, 1);
            }
         }
         else
         {
            throw new Error("Must drop all SegmentDefinitions before removing a PipelineDefinition");
         }
      });
   },

});

export function pipelineDefinitionsCreate(payload: {pipeline: IPipelineDefinition, segments: ISegmentDefinition[]})
{
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Generate a new ID for this definition
      const pipeline = {
         ...payload.pipeline,
         id: generateId(),
      };

      const segments: ISegmentDefinition[] = payload.segments.map((def) => {
         // For each worker, instantiate an copy of the segment
         return {
            ...def,
            id: generateId(),
            pipelineId: pipeline.id,  // Update the pipeline ID since we just generated it
         };
      });

      dispatch(pipelineDefinitionsAdd(pipeline));

      // Then dispatch the segment definitions update
      dispatch(segmentDefinitionsAddMany(segments));

      return {
         pipeline: pipeline.id,
         segments: segments.map((x) => x.id),
      };
   };
}

type PipelineDefinitionsStateType = ReturnType<typeof pipelineDefinitionsSlice.getInitialState>;

export const {add: pipelineDefinitionsAdd, remove: pipelineDefinitionsRemove} = pipelineDefinitionsSlice.actions;

export const {
   selectAll: pipelineDefinitionsSelectAll,
   selectById: pipelineDefinitionsSelectById,
   selectEntities: pipelineDefinitionsSelectEntities,
   selectIds: pipelineDefinitionsSelectIds,
   selectTotal: pipelineDefinitionsSelectTotal,
   selectByIds: pipelineDefinitionsSelectByIds,
} = pipelineDefinitionsAdapter.getSelectors((state: RootState) => state.pipelineDefinitions);

export default pipelineDefinitionsSlice.reducer;
