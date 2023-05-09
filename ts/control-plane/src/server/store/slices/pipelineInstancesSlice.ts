import {IPipelineConfiguration, IPipelineInstance, ISegmentInstance, ISegmentMapping} from "@mrc/common/entities";
import {ResourceStatus, SegmentStates} from "@mrc/proto/mrc/protos/architect_state";
import {connectionsRemove} from "@mrc/server/store/slices/connectionsSlice";
import {pipelineDefinitionsCreate} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {AppDispatch, AppGetState, RootState} from "@mrc/server/store/store";
import {createWrappedEntityAdapter, generateId} from "@mrc/server/utils";
import {createSelector, createSlice, PayloadAction} from "@reduxjs/toolkit";

import {
   segmentInstancesAdd,
   segmentInstancesAddMany,
   segmentInstancesRemove,
} from "./segmentInstancesSlice";
import {workersSelectByMachineId} from "./workersSlice";

const pipelineInstancesAdapter = createWrappedEntityAdapter<IPipelineInstance>({
   selectId: (w) => w.id,
});

function segmentInstanceAdded(state: PipelineInstancesStateType, instance: ISegmentInstance)
{
   // Handle synchronizing a new added instance
   const found = pipelineInstancesAdapter.getOne(state, instance.pipelineInstanceId);

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
         pipelineInstancesAdapter.addOne(state, {
            ...action.payload,
            segmentIds: [],
            state: {
               status: ResourceStatus.Registered,
               refCount: 0,
            },
         });
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
      updateResourceState: (state, action: PayloadAction<{resource: IPipelineInstance, status: ResourceStatus}>) => {
         const found = pipelineInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found)
         {
            throw new Error(`Pipeline Instance with ID: ${action.payload.resource.id} not found`);
         }

         found.state.status = action.payload.status;
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
         const found = pipelineInstancesAdapter.getOne(state, action.payload.pipelineInstanceId);

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
   machineId: string,
   pipeline: IPipelineConfiguration,
   assignments: ISegmentMapping[],
})
{
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Dispatch the definition to get the definition IDs
      const definition_ids = dispatch(pipelineDefinitionsCreate(payload.pipeline));

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
         payload.assignments = Object.entries(payload.pipeline.segments).map(([seg_name, seg_config]) => {
            return {segmentName: seg_name, workerIds: workers.map((x) => x.id)} as ISegmentMapping;
         });
      }

      const segments = payload.assignments.flatMap((assign) => {  // For each worker, create a segment instance
         return assign.workerIds.map((wid) => {
            return {
               id: generateId(),
               pipelineDefinitionId: definition_ids.pipeline,
               pipelineInstanceId: pipeline_id,
               name: assign.segmentName,
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
         pipelineInstanceId: pipeline_id,
         segmentInstanceIds: segments.map((x) => x.id),
      };
   };
}

type PipelineInstancesStateType = ReturnType<typeof pipelineInstancesSlice.getInitialState>;

export const {
   add: pipelineInstancesAdd,
   remove: pipelineInstancesRemove,
   updateResourceState: pipelineInstancesUpdateResourceState,
} = pipelineInstancesSlice.actions;

export const {
   selectAll: pipelineInstancesSelectAll,
   selectById: pipelineInstancesSelectById,
   selectEntities: pipelineInstancesSelectEntities,
   selectIds: pipelineInstancesSelectIds,
   selectTotal: pipelineInstancesSelectTotal,
   selectByIds: pipelineInstancesSelectByIds,
} = pipelineInstancesAdapter.getSelectors((state: RootState) => state.pipelineInstances);

const selectByMachineId = createSelector(
    [pipelineInstancesAdapter.getAll, (state: PipelineInstancesStateType, machine_id: string) => machine_id],
    (pipelineInstances, machine_id) => pipelineInstances.filter((p) => p.machineId === machine_id));

export const pipelineInstancesSelectByMachineId = (state: RootState, machine_id: string) => selectByMachineId(
    state.pipelineInstances,
    machine_id);

export default pipelineInstancesSlice.reducer;
