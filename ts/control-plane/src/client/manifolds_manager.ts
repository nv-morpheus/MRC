import {
   ManifoldUpdateActualAssignmentsResponse,
   EventType,
   ManifoldUpdateActualAssignmentsRequest,
} from "@mrc/proto/mrc/protos/architect";
import { PipelineManager } from "./pipeline_manager";
import { ResourceActualStatus, ResourceRequestedStatus, resourceActualStatusToNumber, resourceRequestedStatusToNumber } from "@mrc/proto/mrc/protos/architect_state";

export class ManifoldClientInstance {
   constructor(public readonly manifoldId: string, private readonly manifoldsManager: ManifoldsManager) {}

   get connectionManager() {
      return this.manifoldsManager.connectionManager;
   }

   public async syncActualSegments() {
      const manifoldState = this.getState();

      const response =
         await this.connectionManager.send_request<ManifoldUpdateActualAssignmentsResponse>(
            EventType.ClientUnaryManifoldUpdateActualAssignments,
            ManifoldUpdateActualAssignmentsRequest.create({
               manifoldInstanceId: this.manifoldId,
               actualInputSegments: manifoldState.requestedInputSegments,
               actualOutputSegments: manifoldState.requestedOutputSegments,
            })
         );

      if (!response.ok) {
         throw new Error("Failed to sync actual segments");
      }
   }

   public async syncActualStatus() {
      const segmentState = this.getState();
      const reqStatus = segmentState.state!.requestedStatus;
      const currentActualStatus = segmentState.state!.actualStatus;
      if (resourceActualStatusToNumber(currentActualStatus) > resourceRequestedStatusToNumber(reqStatus)) {
         throw new Error(`Invalid state: actual status ${currentActualStatus} is greater than requested status ${reqStatus}`);
      }

      let actualStatus: ResourceActualStatus;
      switch (reqStatus) {
         case ResourceRequestedStatus.Requested_Completed:
            actualStatus = ResourceActualStatus.Actual_Running;
            break;
         case ResourceRequestedStatus.Requested_Stopped:
            actualStatus = ResourceActualStatus.Actual_Stopping;
            break;
         case ResourceRequestedStatus.Requested_Destroyed:
            actualStatus = ResourceActualStatus.Actual_Destroying;
            break;
         default:
            throw new Error(`Invalid state: requested status ${reqStatus} is not a valid requested status`);
      }

      return await this.updateActualStatus(actualStatus);
   }

   public async updateActualStatus(status: ResourceActualStatus) {
      return await this.connectionManager.update_resource_status(this.manifoldId, "ManifoldInstances", status);
   }

   public getState() {
      const state = this.manifoldsManager.connectionManager.getClientState();
      return state.manifoldInstances!.entities[this.manifoldId];
   }
}

export class ManifoldsManager {
   constructor(public readonly pipelineManager: PipelineManager) {}

   get manifoldIds() {
      const state = this.connectionManager.getClientState();
      const pipeline = state.pipelineInstances?.entities[this.pipelineManager.pipelineInstanceId];
      return pipeline?.manifoldIds ?? [];
   }

   get manifolds() {
      return this.manifoldIds.map((id) => new ManifoldClientInstance(id, this));
   }

   get connectionManager() {
      return this.pipelineManager.connectionManager;
   }
}
