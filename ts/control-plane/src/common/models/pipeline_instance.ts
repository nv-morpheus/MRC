import { IExecutor, IPipelineInstance } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generatePipelineAddress, generateResourceId } from "@mrc/common/utils";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

export class PipelineInstance implements IPipelineInstance {
   private _instance: IPipelineInstance;

   constructor(state: IPipelineInstance) {
      this._instance = state;
   }

   public get id(): string {
      return this._instance.id;
   }

   public get executorId(): string {
      return this._instance.executorId;
   }

   public get pipelineAddress(): number {
      return this._instance.pipelineAddress;
   }

   public get definitionId(): string {
      return this._instance.definitionId;
   }

   public get segmentIds(): string[] {
      return this._instance.segmentIds;
   }

   public get manifoldIds(): string[] {
      return this._instance.manifoldIds;
   }

   public get state(): ResourceState {
      return new ResourceState(this._instance.state);
   }

   public get_interface(): IPipelineInstance {
      return this._instance;
   }

   public static create(connectionInstance: IExecutor, definitionId: string) {
      const id = generateResourceId("PipelineInstance");

      return new PipelineInstance({
         id: id,
         executorId: connectionInstance.id,
         pipelineAddress: generatePipelineAddress(Number(connectionInstance.id), Number(id)),
         definitionId: definitionId,
         segmentIds: [],
         manifoldIds: [],
         state: {
            ...ResourceState.create().get_interface(),
            requestedStatus: ResourceRequestedStatus.Requested_Created,
         },
      });
   }
}
