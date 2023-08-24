import {
   IManifoldDefinition,
   IPipelineConfiguration,
   IPipelineDefinition,
   IPipelineMapping,
   ISegmentDefinition,
} from "@mrc/common/entities";
import { hashProtoMessage } from "@mrc/common/utils";
import {
   PipelineConfiguration,
   PipelineConfiguration_ManifoldConfiguration,
   PipelineConfiguration_SegmentConfiguration,
} from "@mrc/proto/mrc/protos/architect_state";

export class PipelineDefinitionWrapper {
   public static from(
      config: IPipelineConfiguration,
      mapping?: IPipelineMapping | IPipelineMapping[]
   ): IPipelineDefinition {
      const pipeline_hash = hashProtoMessage(PipelineConfiguration.create(config));

      // Generate a full pipeline definition with the ID as the hash
      const pipeline_def: IPipelineDefinition = {
         id: pipeline_hash,
         config: config,
         instanceIds: [],
         segments: {},
         mappings: {},
         manifolds: {},
      };

      const manifolds = Object.fromEntries(
         Object.entries(config.manifolds).map(([man_name, man_config]) => {
            // Compute the hash of the segment
            const config_hash = hashProtoMessage(PipelineConfiguration_ManifoldConfiguration.create(man_config));

            return [
               man_name,
               {
                  id: config_hash,
                  parentId: pipeline_def?.id,
                  portName: man_name,
                  options: man_config.options,
                  instanceIds: [],
                  outputSegmentIds: {},
                  inputSegmentIds: {},
               } as IManifoldDefinition,
            ];
         })
      );

      const segs = Object.fromEntries(
         Object.entries(config.segments).map(([seg_name, seg_config]) => {
            // Compute the hash of the segment
            const config_hash = hashProtoMessage(PipelineConfiguration_SegmentConfiguration.create(seg_config));

            const egressManifolds = seg_config.egressPorts.map((p) => manifolds[p]);

            // Cross reference the egress ports (i.e. ingress on the manifold)
            egressManifolds.forEach((p) => {
               p.inputSegmentIds[seg_name] = config_hash;
            });

            const ingressManifolds = seg_config.ingressPorts.map((p) => manifolds[p]);

            // Cross reference the ingress ports (i.e. egress on the manifold)
            ingressManifolds.forEach((p) => {
               p.outputSegmentIds[seg_name] = config_hash;
            });

            return [
               seg_name,
               {
                  id: config_hash,
                  parentId: pipeline_def?.id,
                  name: seg_name,
                  options: seg_config.options,
                  instanceIds: [],
                  egressManifoldIds: Object.fromEntries(egressManifolds.map((p) => [p.portName, p.id])),
                  ingressManifoldIds: Object.fromEntries(ingressManifolds.map((p) => [p.portName, p.id])),
               } as ISegmentDefinition,
            ];
         })
      );

      pipeline_def.segments = segs;
      pipeline_def.manifolds = manifolds;

      if (mapping) {
         if (mapping instanceof Array<IPipelineMapping[]>) {
            pipeline_def.mappings = Object.fromEntries(
               mapping.map((m) => {
                  return [m.machineId, m];
               })
            );
         } else {
            pipeline_def.mappings[mapping.machineId] = mapping;
         }
      }

      return pipeline_def;
   }
}
