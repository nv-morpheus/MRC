import { IPipelineInstance, ISegmentInstance } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generateResourceId, generateSegmentAddress } from "@mrc/common/utils";

export class SegmentInstance implements ISegmentInstance {
   private _instance: ISegmentInstance;

   constructor(state: ISegmentInstance) {
      this._instance = state;
   }

   public get id(): string {
      return this._instance.id;
   }

   public get executorId(): string {
      return this._instance.executorId;
   }

   public get pipelineInstanceId(): string {
      return this._instance.pipelineInstanceId;
   }

   public get segmentAddress(): string {
      return this._instance.segmentAddress;
   }

   public get pipelineDefinitionId(): string {
      return this._instance.pipelineDefinitionId;
   }

   public get name(): string {
      return this._instance.name;
   }

   public get egressManifoldInstanceIds(): string[] {
      return this._instance.egressManifoldInstanceIds;
   }

   public get ingressManifoldInstanceIds(): string[] {
      return this._instance.ingressManifoldInstanceIds;
   }

   public get workerId(): string {
      return this._instance.workerId;
   }

   public get state(): ResourceState {
      return new ResourceState(this._instance.state);
   }

   public get_interface(): ISegmentInstance {
      return this._instance;
   }

   public static create(pipelineInstance: IPipelineInstance, name: string, workerId: string) {
      const id = generateResourceId("SegmentInstance");

      return new SegmentInstance({
         id: id,
         executorId: pipelineInstance.executorId,
         pipelineInstanceId: pipelineInstance.id,
         segmentAddress: generateSegmentAddress(
            Number(pipelineInstance.executorId),
            Number(pipelineInstance.id),
            Number(id)
         ),
         pipelineDefinitionId: pipelineInstance.definitionId,
         name: name,
         egressManifoldInstanceIds: [],
         ingressManifoldInstanceIds: [],
         workerId: workerId,
         state: ResourceState.create().get_interface(),
      });
   }
}
