import { IPipelineInstance, IResourceState, ISegmentInstance } from "@mrc/common/entities";
import { ResourceState } from "@mrc/common/models/resource_state";
import { generateResourceId, generateSegmentAddress } from "@mrc/common/utils";

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface SegmentInstanceState extends ISegmentInstance {}

export class SegmentInstanceState {
   id: string = generateResourceId("SegmentInstance");
   partitionId: string;
   pipelineInstanceId: string;
   segmentAddress: string;
   pipelineDefinitionId: string;
   name: string;
   egressManifoldInstanceIds: string[] = [];
   ingressManifoldInstanceIds: string[] = [];
   connectionId: string;
   state: IResourceState = new ResourceState();

   constructor(pipelineInstance: IPipelineInstance, name: string) {
      this.partitionId = pipelineInstance.partitionId;
      this.pipelineInstanceId = pipelineInstance.id;
      this.segmentAddress = generateSegmentAddress(
         Number(this.partitionId),
         Number(this.pipelineInstanceId),
         Number(this.id)
      );
      this.pipelineDefinitionId = pipelineInstance.definitionId;
      this.name = name;
      this.connectionId = pipelineInstance.connectionId;
   }
}
