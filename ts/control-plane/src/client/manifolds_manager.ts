import {
   ManifoldUpdateActualAssignmentsResponse,
   EventType,
   ManifoldUpdateActualAssignmentsRequest,
} from "@mrc/proto/mrc/protos/architect";
import { PipelineManager } from "./pipeline_manager";

export class ManifoldClientInstance {
   constructor(public readonly manifoldId: string, private readonly manifoldsManager: ManifoldsManager) {}

   public async syncActualSegments() {
      const manifoldState = this.getState();

      const response =
         await this.manifoldsManager.connectionManager.send_request<ManifoldUpdateActualAssignmentsResponse>(
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
