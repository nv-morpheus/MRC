import { IExecutor } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generateExecutorAddress, generateResourceId } from "@mrc/common/utils";
import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

export class Executor implements IExecutor {
   private _instance: IExecutor;

   constructor(state: IExecutor) {
      this._instance = state;
   }

   public get id() {
      return this._instance.id;
   }

   public get executorAddress() {
      return this._instance.executorAddress;
   }

   public get peerInfo() {
      return this._instance.peerInfo;
   }

   public get ucxAddress() {
      return this._instance.ucxAddress;
   }

   public get assignedPipelineIds() {
      return this._instance.assignedPipelineIds;
   }

   public get mappedPipelineDefinitions() {
      return this._instance.mappedPipelineDefinitions;
   }

   public get assignedSegmentIds() {
      return this._instance.assignedSegmentIds;
   }

   public get state(): ResourceState {
      return new ResourceState(this._instance.state);
   }

   public get workerIds() {
      return this._instance.workerIds;
   }

   public get_interface(): IExecutor {
      return this._instance;
   }

   public static create(peerInfo: string) {
      const id = generateResourceId("Executor");

      return new Executor({
         id: id,
         executorAddress: generateExecutorAddress(Number(id)),
         peerInfo: peerInfo,
         ucxAddress: "",
         assignedPipelineIds: [],
         mappedPipelineDefinitions: [],
         assignedSegmentIds: [],
         state: {
            ...ResourceState.create().get_interface(),
            // Set the actual status to created and requested completed. Since any executor that connects will already
            // be running to completion
            actualStatus: ResourceActualStatus.Actual_Created,
            requestedStatus: ResourceRequestedStatus.Requested_Completed,
         },
         workerIds: [],
      });
   }
}
