import { IManifoldInstance, IPipelineInstance, IResourceState } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generateManifoldAddress, generateResourceId } from "@mrc/common/utils";

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface ManifoldInstanceState extends IManifoldInstance {}

export class ManifoldInstanceState implements IManifoldInstance {
   id: string = generateResourceId("ManifoldInstance");
   partitionId: string;
   pipelineInstanceId: string;
   manifoldAddress: string;
   pipelineDefinitionId: string;
   portName: string;
   requestedInputSegments: { [key: number]: boolean } = {};
   requestedOutputSegments: { [key: number]: boolean } = {};
   actualInputSegments: { [key: number]: boolean } = {};
   actualOutputSegments: { [key: number]: boolean } = {};
   connectionId: string;
   state: IResourceState = new ResourceState();

   constructor(pipelineInstance: IPipelineInstance, portName: string) {
      this.partitionId = pipelineInstance.partitionId;
      this.pipelineInstanceId = pipelineInstance.id;
      this.manifoldAddress = generateManifoldAddress(
         Number(this.partitionId),
         Number(this.pipelineInstanceId),
         Number(this.id)
      );
      this.pipelineDefinitionId = pipelineInstance.definitionId;
      this.portName = portName;
      this.connectionId = pipelineInstance.connectionId;
   }
}
