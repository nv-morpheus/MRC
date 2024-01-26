import {
   IConnection,
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
import { ManifoldInstanceState } from "@mrc/common/models/manifold_instance";
import { PipelineInstanceState } from "@mrc/common/models/pipeline_instance";
import { SegmentInstanceState } from "@mrc/common/models/segment_instance";
import { PipelineDefinitionWrapper } from "@mrc/common/pipelineDefinition";
import {
   generateId,
   generatePartitionAddress,
   generatePipelineAddress,
   generateSegmentAddress,
   generateSegmentHash,
   hashProtoMessage,
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

export const connection: IConnection = {
   id: generateId(),
   peerInfo: "localhost:1234",
   workerIds: [],
   assignedPipelineIds: [],
   mappedPipelineDefinitions: [],
   state: {
      actualStatus: ResourceActualStatus.Actual_Created,
      refCount: 0,
      requestedStatus: ResourceRequestedStatus.Requested_Completed,
   },
};

const workerId = generateId();

export const worker: IWorker = {
   id: workerId,
   connectionId: connection.id,
   partitionAddress: generatePartitionAddress(Number(workerId)),
   ucxAddress: stringToBytes("-----"),
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
         egressPorts: ["int_port"],
         ingressPorts: [],
      },
      seg2: {
         name: "seg2",
         egressPorts: [],
         ingressPorts: ["int_port"],
      },
   },
   manifolds: {
      int_port: {
         name: "int_port",
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
   [connection].map((c) => {
      return [
         c.id,
         {
            connectionId: c.id,
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

export const pipeline: IPipelineInstance = new PipelineInstanceState(connection, pipeline_def.id);

export const segments: ISegmentInstance[] = Object.entries(pipeline_def.segments).map(([seg_name, seg_def]) => {
   return new SegmentInstanceState(pipeline, seg_name);
});

export const segments_map = Object.fromEntries(segments.map((s) => [s.name, s]));

export const manifolds: IManifoldInstance[] = Object.entries(pipeline_def.manifolds).map(([man_name, man_def]) => {
   return new ManifoldInstanceState(pipeline, man_name);
});

export const manifolds_map = Object.fromEntries(manifolds.map((s) => [s.portName, s]));
