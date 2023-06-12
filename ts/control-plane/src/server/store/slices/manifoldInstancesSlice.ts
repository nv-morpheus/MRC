/* eslint-disable @typescript-eslint/unbound-method */
import { createSelector, createSlice, PayloadAction } from "@reduxjs/toolkit";

import { createWrappedEntityAdapter } from "../../utils";

import type { AppDispatch, AppGetState, RootState } from "../store";
import { pipelineInstancesRemove } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { IManifoldInstance, ISegmentInstance } from "@mrc/common/entities";
import {
   ResourceActualStatus,
   resourceActualStatusToNumber,
   resourceRequestedStatusToNumber,
} from "@mrc/proto/mrc/protos/architect_state";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";
import { workersSelectById } from "@mrc/server/store/slices/workersSlice";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";

const manifoldInstancesAdapter = createWrappedEntityAdapter<IManifoldInstance>({
   selectId: (w) => w.id,
});

export const manifoldInstancesSlice = createSlice({
   name: "manifoldInstances",
   initialState: manifoldInstancesAdapter.getInitialState(),
   reducers: {
      add: (state, action: PayloadAction<IManifoldInstance>) => {
         if (manifoldInstancesAdapter.getOne(state, action.payload.id)) {
            throw new Error(`Manifold Instance with ID: ${action.payload.id} already exists`);
         }
         manifoldInstancesAdapter.addOne(state, action.payload);
      },
      remove: (state, action: PayloadAction<IManifoldInstance>) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.id} not found`);
         }

         if (found.state.actualStatus != ResourceActualStatus.Actual_Destroyed) {
            throw new Error(
               `Attempting to delete Manifold Instance with ID: ${action.payload.id} while it has not finished. Stop ManifoldInstance first!`
            );
         }

         manifoldInstancesAdapter.removeOne(state, action.payload.id);
      },
      updateResourceRequestedState: (
         state,
         action: PayloadAction<{ resource: IManifoldInstance; status: ResourceRequestedStatus }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.resource.id} not found`);
         }

         if (
            resourceRequestedStatusToNumber(found.state.requestedStatus) >
            resourceRequestedStatusToNumber(action.payload.status)
         ) {
            throw new Error(
               `Cannot update state of Instance with ID: ${action.payload.resource.id}. Current state ${found.state.requestedStatus} is greater than requested state ${action.payload.status}`
            );
         }

         found.state.requestedStatus = action.payload.status;
      },
      updateResourceActualState: (
         state,
         action: PayloadAction<{ resource: IManifoldInstance; status: ResourceActualStatus }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.resource.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.resource.id} not found`);
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
      attachRequestedSegment: (
         state,
         action: PayloadAction<{
            manifold: IManifoldInstance;
            is_ingress: boolean;
            segment: ISegmentInstance;
            is_local: boolean;
         }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.manifold.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.manifold.id} not found`);
         }

         // Check to make sure this hasnt been added already
         if (action.payload.is_ingress) {
            if (action.payload.segment.address in found.requestedIngressSegments) {
               throw new Error("Segment already attached to manifold");
            }

            found.requestedIngressSegments[action.payload.segment.address] = action.payload.is_local;
         } else {
            if (action.payload.segment.address in found.requestedEgressSegments) {
               throw new Error("Segment already attached to manifold");
            }

            found.requestedEgressSegments[action.payload.segment.address] = action.payload.is_local;
         }
      },
      detachRequestedSegment: (
         state,
         action: PayloadAction<{
            manifold: IManifoldInstance;
            is_ingress: boolean;
            segment: ISegmentInstance;
         }>
      ) => {
         const found = manifoldInstancesAdapter.getOne(state, action.payload.manifold.id);

         if (!found) {
            throw new Error(`Manifold Instance with ID: ${action.payload.manifold.id} not found`);
         }

         // Check to make sure its already added
         if (action.payload.is_ingress) {
            if (!(action.payload.segment.address in found.requestedIngressSegments)) {
               throw new Error("Segment not attached to manifold");
            }

            delete found.requestedIngressSegments[action.payload.segment.address];
         } else {
            if (!(action.payload.segment.address in found.requestedEgressSegments)) {
               throw new Error("Segment not attached to manifold");
            }

            delete found.requestedEgressSegments[action.payload.segment.address];
         }
      },
   },
   extraReducers: (builder) => {
      builder.addCase(pipelineInstancesRemove, (state, action) => {
         // Need to delete any manifolds associated with the pipeline
         const instances = selectByPipelineId(state, action.payload.id);

         manifoldInstancesAdapter.removeMany(
            state,
            instances.map((x) => x.id)
         );
      });
   },
});

export function manifoldInstancesAttachLocalSegment(manifold: IManifoldInstance, segment: ISegmentInstance) {
   return (dispatch: AppDispatch, getState: AppGetState) => {
      const state = getState();

      const found = manifoldInstancesSelectById(state, manifold.id);

      if (!found) {
         throw new Error(`Manifold Instance with ID: ${manifold.id} not found`);
      }

      // Check to make sure they are actually local
      if (manifold.machineId !== workersSelectById(state, segment.workerId)?.machineId) {
         throw new Error("Invalid local manifold/segment pair. Manifold and segment are on different machines");
      }

      const pipeline_def = pipelineDefinitionsSelectById(state, manifold.pipelineDefinitionId);

      if (!pipeline_def) {
         throw new Error(`Could not find pipeline definition with ID: ${manifold.pipelineDefinitionId}`);
      }

      if (!(manifold.portName in pipeline_def.manifolds)) {
         throw new Error(
            `Could not find manifold ${manifold.portName} in definition with ID: ${manifold.pipelineDefinitionId}`
         );
      }

      const manifold_def = pipeline_def.manifolds[manifold.portName];

      if (!(segment.name in pipeline_def.segments)) {
         throw new Error(
            `Could not find segment ${segment.name} in definition with ID: ${manifold.pipelineDefinitionId}`
         );
      }

      const segment_def = pipeline_def.segments[segment.name];

      // Figure out if this is an egress or ingress (relative to the manifold. Opposite for segments)
      let is_manifold_ingress = false;

      if (manifold.portName in segment_def.egressPorts) {
         is_manifold_ingress = true;
      } else if (manifold.portName in segment_def.ingressPorts) {
         is_manifold_ingress = false;
      } else {
         throw new Error("Manifold not found in segment definition ingress or egress ports");
      }

      // Now sync all other manifolds for this definition
      manifold_def.instanceIds.forEach((manifold_id) => {
         const manifold_instance = manifoldInstancesSelectById(state, manifold_id);

         if (!manifold_instance) {
            throw new Error("Could not find manifold by ID");
         }

         // Figure out if this is local
         const is_local = manifold_instance.machineId === manifold.machineId;

         // Dispatch the attach action
         dispatch(
            manifoldInstancesSlice.actions.attachRequestedSegment({
               is_ingress: is_manifold_ingress,
               is_local: is_local,
               manifold: manifold_instance,
               segment: segment,
            })
         );
      });
   };
}

export function manifoldInstancesAddMany(instances: IManifoldInstance[]) {
   // To allow the watchers to work, we need to add all individually
   return (dispatch: AppDispatch) => {
      // Loop and dispatch each individually
      instances.forEach((m) => {
         dispatch(manifoldInstancesAdd(m));
      });
   };
}

type ManifoldInstancesStateType = ReturnType<typeof manifoldInstancesSlice.getInitialState>;

export const {
   add: manifoldInstancesAdd,
   remove: manifoldInstancesRemove,
   updateResourceRequestedState: manifoldInstancesUpdateResourceRequestedState,
   updateResourceActualState: manifoldInstancesUpdateResourceActualState,
} = manifoldInstancesSlice.actions;

export const {
   selectAll: manifoldInstancesSelectAll,
   selectById: manifoldInstancesSelectById,
   selectByIds: manifoldInstancesSelectByIds,
   selectEntities: manifoldInstancesSelectEntities,
   selectIds: manifoldInstancesSelectIds,
   selectTotal: manifoldInstancesSelectTotal,
} = manifoldInstancesAdapter.getSelectors((state: RootState) => state.manifoldInstances);

const selectByPipelineId = createSelector(
   [manifoldInstancesAdapter.getAll, (state: ManifoldInstancesStateType, pipeline_id: string) => pipeline_id],
   (manifoldInstances, pipeline_id) => manifoldInstances.filter((x) => x.pipelineInstanceId === pipeline_id)
);

export const manifoldInstancesSelectByPipelineId = (state: RootState, pipeline_id: string) =>
   selectByPipelineId(state.manifoldInstances, pipeline_id);

export default manifoldInstancesSlice.reducer;
