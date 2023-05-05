import {createSlice, PayloadAction} from "@reduxjs/toolkit";

import type {AppDispatch, AppGetState, RootState} from "../store";
import {
   EgressPort,
   IngressPort,
   PipelineConfiguration,
   PipelineConfiguration_SegmentConfiguration,
   PipelineDefinition,
   PipelineDefinition_SegmentDefinition,
   ScalingOptions,
   SegmentOptions,
} from "@mrc/proto/mrc/protos/architect_state";
import {createWrappedEntityAdapter} from "@mrc/server/utils";
import {hashProtoMessage} from "@mrc/common/utils";
import {
   ISegmentInstance,
   segmentInstancesAdd,
   segmentInstancesAddMany,
   segmentInstancesRemove,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import {
   IPipelineInstance,
   pipelineInstancesAdd,
   pipelineInstancesRemove,
} from "@mrc/server/store/slices/pipelineInstancesSlice";

export type IIngressPort    = Omit<IngressPort, "$type">;
export type IEgressPort     = Omit<EgressPort, "$type">;
export type IScalingOptions = Omit<ScalingOptions, "$type">;
export type ISegmentOptions = Omit<SegmentOptions, "$type">&{
   scalingOptions?: IScalingOptions,
};

export type ISegmentConfiguration =
    Omit<PipelineConfiguration_SegmentConfiguration, "$type"|"ingressPorts"|"egressPorts"|"options">&{
       ingressPorts: IIngressPort[],
       egressPorts: IEgressPort[],
       options?: ISegmentOptions,
    };

export type IPipelineConfiguration = Omit<PipelineConfiguration, "$type"|"segments">&{
   segments: {[key: string]: ISegmentConfiguration},
};

export type ISegmentDefinition = Omit<PipelineDefinition_SegmentDefinition, "$type">;

export type IPipelineDefinition = Omit<PipelineDefinition, "$type"|"config"|"segments">&{
   config: IPipelineConfiguration,
   segments: {[key: string]: ISegmentDefinition},
};

export type IPipelineDefinitionNested = {
   segments: Map<string, Omit<ISegmentDefinition, "id">>;
};

const pipelineDefinitionsAdapter = createWrappedEntityAdapter<IPipelineDefinition>({
   selectId: (w) => w.id,
});

function pipelineInstanceAdded(state: PipelineDefinitionsStateType,
                               instance: Pick<IPipelineInstance, "id"|"definitionId">)
{
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.definitionId);

   if (found)
   {
      found.instanceIds.push(instance.id);
   }
   else
   {
      throw new Error("Must add a PipelineDefinition before creating a PipelineInstance!");
   }
}

function segmentInstanceAdded(state: PipelineDefinitionsStateType, instance: ISegmentInstance)
{
   // Handle synchronizing a new added instance
   const found = pipelineDefinitionsAdapter.getOne(state, instance.pipelineDefinitionId);

   if (found)
   {
      // Find the segment in the definition
      if (!Object.keys(found.segments).includes(instance.name))
      {
         throw new Error(`Attempting to create SegmentInstance with Pipeline Definition with ID: ${
             instance.pipelineDefinitionId} but the segment name, ${
             instance.name} does not exist in the PipelineDefinition.`);
      }

      found.segments[instance.name].instanceIds.push(instance.id);
   }
   else
   {
      throw new Error("Must add a PipelineDefinition before creating a SegmentInstance!");
   }
}

export const pipelineDefinitionsSlice = createSlice({
   name: "pipelineDefinitions",
   initialState: pipelineDefinitionsAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IPipelineDefinition>) => {
         if (pipelineDefinitionsAdapter.getOne(state, action.payload.id))
         {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} already exists`);
         }
         pipelineDefinitionsAdapter.addOne(state, action.payload);
      },
      remove: (state, action: PayloadAction<IPipelineDefinition>) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.id);

         if (!found)
         {
            throw new Error(`Pipeline Definition with ID: ${action.payload.id} not found`);
         }

         if (found.instanceIds.length > 0)
         {
            throw new Error(`Attempting to delete PipelineDefinition with ID: ${
                action.payload.id} while there are running instances. Stop PipelineInstances first!`)
         }

         if (Object.values(found.segments).reduce((accum: number, curr) => accum + curr.instanceIds.length, 0) > 0)
         {
            throw new Error(`Attempting to delete PipelineDefinition with ID: ${
                action.payload.id} with running segment instance. Remove segment instances first!`)
         }

         pipelineDefinitionsAdapter.removeOne(state, action.payload.id);
      },
   },
   extraReducers: (builder) => {
      builder.addCase(pipelineInstancesAdd, (state, action) => {
         pipelineInstanceAdded(state, action.payload);
      });
      builder.addCase(pipelineInstancesRemove,
                      (state, action) => {
                         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.definitionId);

                         if (found)
                         {
                            const index = found.instanceIds.findIndex(x => x === action.payload.id);

                            if (index !== -1)
                            {
                               found.instanceIds.splice(index, 1);
                            }
                         }
                         else
                         {
                            throw new Error(`PipelineDefinition with ID: ${action.payload.definitionId}, not found. Must drop all Pipeline before removing a PipelineDefinition.`);
                         }
                      });

      builder.addCase(segmentInstancesAdd, (state, action) => {
         segmentInstanceAdded(state, action.payload);
      });
      builder.addCase(segmentInstancesAddMany, (state, action) => {
         action.payload.forEach((seg) => {
            segmentInstanceAdded(state, seg);
         });
      });
      builder.addCase(segmentInstancesRemove, (state, action) => {
         const found = pipelineDefinitionsAdapter.getOne(state, action.payload.pipelineDefinitionId);

         if (found)
         {
            // Find the segment in the definition
            if (!Object.keys(found.segments).includes(action.payload.name))
            {
               throw new Error(`Attempting to remove SegmentInstance with PipelineDefinition ID: ${
                   action.payload.pipelineDefinitionId} but the segment name, ${
                   action.payload.name} does not exist in the PipelineDefinition.`);
            }

            const foundSegment = found.segments[action.payload.name];

            const index = foundSegment.instanceIds.findIndex(x => x === action.payload.id);

            if (index !== -1)
            {
               foundSegment.instanceIds.splice(index, 1);
            }
         }
         else
         {
            throw new Error(`PipelineDefinition with ID: ${action.payload.pipelineDefinitionId}, not found. Must drop all SegmentInstances before removing a PipelineDefinition`);
         }
      });
   },
});

export function pipelineDefinitionsCreate(payload: IPipelineConfiguration)
{
   return (dispatch: AppDispatch, getState: AppGetState) => {
      // Compute the hash of the pipeline
      const pipeline_hash = hashProtoMessage(PipelineConfiguration.create(payload));

      // Generate a full pipeline definition with the ID as the hash
      const pipeline: IPipelineDefinition = {id: pipeline_hash, config: payload, instanceIds: [], segments: {}};

      const segs = Object.fromEntries(Object.entries(payload.segments).map(([seg_name, seg_config]) => {
         // Compute the hash of the segment
         const segment_hash = hashProtoMessage(PipelineConfiguration_SegmentConfiguration.create(seg_config));

         return [
            seg_name,
            {
               id: segment_hash,
               parentId: pipeline.id,
               name: seg_name,
               instanceIds: [],
            } as ISegmentDefinition,
         ]
      }));

      pipeline.segments = segs;

      dispatch(pipelineDefinitionsAdd(pipeline));

      return {
         pipeline: pipeline.id,
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
