import { IConnection, IPipelineInstance, IResourceState, IWorker } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generatePipelineAddress, generateResourceId } from "@mrc/common/utils";

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface PipelineInstanceState extends IPipelineInstance {}

export class PipelineInstanceState {
   id: string = generateResourceId("PipelineInstance");
   partitionId: string;
   pipelineAddress: number;
   definitionId: string;
   segmentIds: string[] = [];
   manifoldIds: string[] = [];
   connectionId: string;
   state: IResourceState = new ResourceState();

   constructor(connectionInstance: IConnection, definitionId: string) {
      this.connectionId = connectionInstance.id;

      this.partitionId = this.connectionId; // TODO(MDD): Update this when removing partitions
      this.pipelineAddress = generatePipelineAddress(Number(this.partitionId), Number(this.id));
      this.definitionId = definitionId;
   }
}
