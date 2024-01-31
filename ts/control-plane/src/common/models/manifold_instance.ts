import { IManifoldInstance, IPipelineInstance, IResourceState } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generateManifoldAddress, generateResourceId } from "@mrc/common/utils";
import { ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

export class ManifoldInstance implements IManifoldInstance {
   private _interface: IManifoldInstance;

   constructor(state: IManifoldInstance) {
      this._interface = state;
   }

   public get id(): string {
      return this._interface.id;
   }

   public get executorId(): string {
      return this._interface.executorId;
   }

   public get pipelineInstanceId(): string {
      return this._interface.pipelineInstanceId;
   }

   public get manifoldAddress(): string {
      return this._interface.manifoldAddress;
   }

   public get pipelineDefinitionId(): string {
      return this._interface.pipelineDefinitionId;
   }

   public get portName(): string {
      return this._interface.portName;
   }

   public get requestedInputSegments(): { [key: number]: boolean } {
      return this._interface.requestedInputSegments;
   }

   public get requestedOutputSegments(): { [key: number]: boolean } {
      return this._interface.requestedOutputSegments;
   }

   public get actualInputSegments(): { [key: number]: boolean } {
      return this._interface.actualInputSegments;
   }

   public get actualOutputSegments(): { [key: number]: boolean } {
      return this._interface.actualOutputSegments;
   }

   public get state(): ResourceState {
      return new ResourceState(this._interface.state);
   }

   public get_interface(): IManifoldInstance {
      return this._interface;
   }

   public static create(pipelineInstance: IPipelineInstance, portName: string) {
      const id = generateResourceId("ManifoldInstance");

      return new ManifoldInstance({
         id: id,
         executorId: pipelineInstance.executorId,
         pipelineInstanceId: pipelineInstance.id,
         manifoldAddress: generateManifoldAddress(
            Number(pipelineInstance.executorId),
            Number(pipelineInstance.id),
            Number(id)
         ),
         pipelineDefinitionId: pipelineInstance.definitionId,
         portName: portName,
         requestedInputSegments: {},
         requestedOutputSegments: {},
         actualInputSegments: {},
         actualOutputSegments: {},
         state: {
            ...ResourceState.create().get_interface(),
            requestedStatus: ResourceRequestedStatus.Requested_Created,
         },
      });
   }
}
