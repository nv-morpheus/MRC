import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {PipelineInstance, SegmentStates} from "../../../proto/mrc/protos/architect_state";
import {createWrappedEntityAdapter, generateId} from "../../utils";

import type {AppDispatch, AppGetState, RootState} from "../store";
import {
   segmentInstancesAdd,
   segmentInstancesAddMany,
   ISegmentInstance,
   segmentInstancesRemove,
} from "./segmentInstancesSlice";
import {workersSelectByMachineId} from "./workersSlice";
import {connectionsRemove} from "@mrc/server/store/slices/connectionsSlice";
import {IPipelineDefinition, pipelineDefinitionsCreate} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {ISegmentDefinition, ISegmentMapping} from "@mrc/server/store/slices/segmentDefinitionsSlice";

export type IPipelineInstance = Omit<PipelineInstance, "$type">;

const pipelineInstancesAdapter = createWrappedEntityAdapter<IPipelineInstance>({
   selectId: (w) => w.id,
});

function segmentInstanceAdded(state: PipelineInstancesStateType, instance: ISegmentInstance)
{
   // Handle synchronizing a new added instance
   const found = pipelineInstancesAdapter.getOne(state, instance.pipelineId);

   if (found)
   {
      found.segmentIds.push(instance.id);
   }
   else
   {
      throw new Error("Must add a PipelineInstance before a SegmentInstance!");
   }
}

export const pipelineInstancesSlice = createSlice({
   name: "pipelineInstances",
   initialState: pipelineInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<Pick<IPipelineInstance, "id"|"definitionId"|"machineId">>) => {
         if (pipelineInstancesAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Pipeline Instance with ID: ${action.payload.id} already exists`);
         }
         pipelineInstancesAdapter.addOne(state, {...action.payload, segmentIds: []});
      },
      remove: (state, action: PayloadAction<IPipelineInstance>) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Pipeline Instance with ID: ${action.payload.id} not found`);
         }

         if (found.segmentIds.length > 0)
         {
            throw new Error(`Attempting to delete Pipeline Instance with ID: ${
                action.payload.id} with running segment instance. Remove segment instances first!`)
         }

         pipelineInstancesAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(connectionsRemove, (state, action) => {
         // Need to delete any workers associated with that connection
         const connection_instances = selectByMachineId(state, action.payload.id);

         pipelineInstancesAdapter.removeMany(state, connection_instances.map((x) => x.id));
      });
      builder.addCase(segmentInstancesAdd, (state, action) => {
         segmentInstanceAdded(state, action.payload);
      });
      builder.addCase(segmentInstancesAddMany, (state, action) => {
         action.payload.forEach((segmentInstance) => {
            segmentInstanceAdded(state, segmentInstance);
         });
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.pipelineId);

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
            throw new Error("Must drop all SegmentInstances before removing a PipelineInstance");
         }
      });
   },

});

export function pipelineInstancesAssign(payload: {
   machineId: number,
   pipeline: IPipelineDefinition,
   segments: ISegmentDefinition[],
   assignments: ISegmentMapping[],
})
{
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Dispatch the definition to get the definition IDs
      const definition_ids = dispatch(pipelineDefinitionsCreate({
         pipeline: payload.pipeline,
         segments: payload.segments,
      }));

      const pipeline_id = generateId();

      // First dispatch the pipeline instance update
      dispatch(pipelineInstancesAdd({
         id: pipeline_id,
         definitionId: definition_ids.pipeline,
         machineId: payload.machineId,
      }));

      // Get the workers for this machine
      const workers = workersSelectByMachineId(getState(), payload.machineId);

      if (payload.assignments.length == 0)
      {
         // Default to auto assignment of one segment instance per worker per definition
         payload.assignments = payload.segments.map((seg_def) => {
            return {segmentName: seg_def.name, workerIds: workers.map((x) => x.id)} as ISegmentMapping;
         });
      }

      const segments = payload.assignments.flatMap((assign) => {  // For each worker, create a segment instance
         // Find the segment definition ID from the name
         const def_idx = payload.segments.findIndex((x) => x.name == assign.segmentName);

         return assign.workerIds.map((wid) => {
            return {
               id: generateId(),
               definitionId: Number(definition_ids.segments[def_idx]),
               pipelineId: pipeline_id,
               address: 0,
               workerId: wid,
               state: SegmentStates.Initialized,
            } as ISegmentInstance;
         });
      });

      // Then dispatch the segment instances update
      dispatch(segmentInstancesAddMany(segments));

      return {
         pipelineDefinitionId: definition_ids.pipeline,
         segmentDefinitionIds: definition_ids.segments,
         pipelineInstanceId: pipeline_id,
         segmentInstanceIds: segments.map((x) => x.id),
      };
   };
}

type PipelineInstancesStateType = ReturnType<typeof pipelineInstancesSlice.getInitialState>;

export const {add: pipelineInstancesAdd, remove: pipelineInstancesRemove} = pipelineInstancesSlice.actions;

export const {
   selectAll: pipelineInstancesSelectAll,
   selectById: pipelineInstancesSelectById,
   selectEntities: pipelineInstancesSelectEntities,
   selectIds: pipelineInstancesSelectIds,
   selectTotal: pipelineInstancesSelectTotal,
   selectByIds: pipelineInstancesSelectByIds,
} = pipelineInstancesAdapter.getSelectors((state: RootState) => state.pipelineInstances);

const selectByMachineId = createSelector(
    [pipelineInstancesAdapter.getAll, (state: PipelineInstancesStateType, machine_id: number) => machine_id],
    (pipelineInstances, machine_id) => pipelineInstances.filter((p) => p.machineId === machine_id));

export const pipelineInstancesSelectByMachineId = (state: RootState, machine_id: number) => selectByMachineId(
    state.pipelineInstances,
    machine_id);

export default pipelineInstancesSlice.reducer;
