import {
   IExecutor,
   IManifoldInstance,
   IPipelineConfiguration,
   IPipelineDefinition,
   IPipelineInstance,
   IPipelineMapping,
   IResourceState,
   ISegmentInstance,
   ISegmentMapping,
   IWorker,
} from "@mrc/common/entities";
import { Executor } from "@mrc/common/models/executor";
import { ManifoldInstance } from "@mrc/common/models/manifold_instance";
import { PipelineInstance } from "@mrc/common/models/pipeline_instance";
import { SegmentInstance } from "@mrc/common/models/segment_instance";
import { PipelineDefinitionWrapper } from "@mrc/common/pipelineDefinition";
import {
   generateId,
   generatePartitionAddress,
   generatePipelineAddress,
   generateSegmentAddress,
   generateSegmentHash,
   hashProtoMessage,
   hashName16,
   stringToBytes,
} from "@mrc/common/utils";
import {
   ManifoldOptions_Policy,
   PipelineConfiguration,
   ResourceActualStatus,
   ResourceRequestedStatus,
} from "@mrc/proto/mrc/protos/architect_state";

const default_resource_state: IResourceState = {
   requestedStatus: ResourceRequestedStatus.Requested_Initialized,
   actualStatus: ResourceActualStatus.Actual_Unknown,
   refCount: 0,
};

export const executor: IExecutor = Executor.create("localhost:1234").get_interface();

const workerId = generateId();

export const worker: IWorker = {
   id: workerId,
   executorId: executor.id,
   partitionAddress: generatePartitionAddress(Number(workerId)),
   ucxAddress: "-----",
   state: {
      ...default_resource_state,
   },
   assignedSegmentIds: [],
};

export const workers: IWorker[] = [worker];

export const pipeline_config: IPipelineConfiguration = {
   segments: {
      seg1: {
         name: "seg1",
         nameHash: hashName16("seg1"),
         egressPorts: ["int_port"],
         ingressPorts: [],
      },
      seg2: {
         name: "seg2",
         nameHash: hashName16("seg2"),
         egressPorts: [],
         ingressPorts: ["int_port"],
      },
   },
   manifolds: {
      int_port: {
         portName: "int_port",
         portHash: hashName16("int_port"),
         typeId: 0,
         typeString: "int",
         options: {
            policy: ManifoldOptions_Policy.LoadBalance,
         },
      },
   },
};

const pipeline_config_hash = hashProtoMessage(PipelineConfiguration.create(pipeline_config));

export const pipeline_mappings = Object.fromEntries(
   [executor].map((c) => {
      return [
         c.id,
         {
            executorId: c.id,
            segments: Object.fromEntries(
               Object.entries(pipeline_config.segments).map(([seg_name]) => {
                  return [seg_name, { segmentName: seg_name, byWorker: { workerIds: [worker.id] } } as ISegmentMapping];
               })
            ),
         } as IPipelineMapping,
      ];
   })
);

export const pipeline_def: IPipelineDefinition = PipelineDefinitionWrapper.from(
   pipeline_config,
   Object.values(pipeline_mappings)
);

// {
//    id: pipeline_config_hash,
//    config: pipeline_config,
//    mappings: pipeline_mappings,
//    instanceIds: [],
//    segments: Object.fromEntries(
//       Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
//          return [
//             seg_name,
//             {
//                id: hashProtoMessage(PipelineConfiguration_SegmentConfiguration.create(seg_config)),
//                parentId: pipeline_config_hash,
//                name: seg_name,
//                instanceIds: [],
//                egressPorts: {},
//                ingressPorts: {},
//                egressManifoldIds: {},
//                ingressManifoldIds: {},
//             } as ISegmentDefinition,
//          ];
//       })
//    ),
//    manifolds: Object.fromEntries(
//       Object.entries(pipeline_config.manifolds).map(([man_name, man_config]) => {
//          return [
//             man_name,
//             {
//                id: hashProtoMessage(PipelineConfiguration_ManifoldConfiguration.create(man_config)),
//                parentId: pipeline_config_hash,
//                portName: man_name,
//                instanceIds: [],
//             } as IManifoldDefinition,
//          ];
//       })
//    ),
// };

export const pipeline: IPipelineInstance = PipelineInstance.create(executor, pipeline_def.id).get_interface();

export const segments: ISegmentInstance[] = Object.entries(pipeline_def.segments).map(([seg_name, seg_def]) => {
   return SegmentInstance.create(pipeline, seg_name, worker.id).get_interface();
});

export const segments_map = Object.fromEntries(segments.map((s) => [s.name, s]));

export const manifolds: IManifoldInstance[] = Object.entries(pipeline_def.manifolds).map(([man_name, man_def]) => {
   return ManifoldInstance.create(pipeline, man_name).get_interface();
});

export const manifolds_map = Object.fromEntries(manifolds.map((s) => [s.portName, s]));
