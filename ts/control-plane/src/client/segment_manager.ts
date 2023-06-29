import { ResourceActualStatus, ResourceRequestedStatus, resourceActualStatusToNumber, resourceRequestedStatusToNumber } from "@mrc/proto/mrc/protos/architect_state";
import { WorkerClientInstance } from "./workers_manager";
import { ResourceStopResponse, EventType, ResourceStopRequest } from "@mrc/proto/mrc/protos/architect";


export class SegmentManager {
   constructor(public readonly worker: WorkerClientInstance, public readonly segmentId: string) { }

   get connectionManager() {
      return this.worker.connectionManager;
   }

   public getState() {
      const state = this.connectionManager.getClientState();
      return state.segmentInstances!.entities[this.segmentId];
   }

   public async updateActualStatus(status: ResourceActualStatus) {
      return await this.connectionManager.update_resource_status(this.segmentId, "SegmentInstances", status);
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

   public async sendSegmentRunning() {
      return await this.updateActualStatus(ResourceActualStatus.Actual_Running);
   }

   public async sendSegmentStopping() {
      return await this.updateActualStatus(ResourceActualStatus.Actual_Stopping);
   }

   public async sendSegmenStopped() {
      return await this.updateActualStatus(ResourceActualStatus.Actual_Stopped);
   }

   public async requestSegmentStop() {
      // Perform a graceful stop by first informing the server that we want to stop, allowing all of our upstream
      // segments to first detach from us. Once that is done we should receive a `Requested_Stopped` status from the
      // server, at which point we can stop our segment. 
      // This is different than completing, this would happen as the result of receiving a Cntrl-C, SIGTERM or a spot 
      // eviction from a cloud service provider.
      const response = await this.connectionManager.send_request<ResourceStopResponse>(
         EventType.ClientUnaryResourceStopRequest,
         ResourceStopRequest.create({
            resourceType: "SegmentInstances",
            resourceId: this.segmentId,
         })
      );

      if (!response.ok) {
         throw new Error(`Failed to request stop for segment ${this.segmentId}`);
      }
   }
}
