/* eslint-disable @typescript-eslint/no-non-null-assertion */
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import "ix/add/asynciterable-operators/last";

import { ResourceActualStatus, SegmentMappingPolicies } from "@mrc/proto/mrc/protos/architect_state";

import { throwExpression } from "@mrc/common/utils";

import { IPipelineConfiguration, IPipelineMapping, ISegmentMapping } from "@mrc/common/entities";
import { MrcTestClient } from "@mrc/client/client";
import { WorkersManager } from "@mrc/client/workers_manager";
import {
   Ack,
   EventType,
   ManifoldUpdateActualAssignmentsRequest,
   ManifoldUpdateActualAssignmentsResponse,
   PipelineAddMappingRequest,
   PipelineAddMappingResponse,
   PipelineRegisterConfigRequest,
   PipelineRegisterConfigResponse,
} from "@mrc/proto/mrc/protos/architect";
import { ConnectionManager } from "@mrc/client/connection_manager";
import { ManifoldsManager } from "./manifolds_manager";

export class PipelineManager {
   private _pipelineDefinitionId: string | undefined;
   private _pipelineInstanceId: string | undefined;

   private _isCreated = false;
   private _manifoldsManager: ManifoldsManager;

   constructor(public readonly workersManager: WorkersManager, public config: IPipelineConfiguration) {
      this._manifoldsManager = new ManifoldsManager(this);
   }

   public static create(
      config: IPipelineConfiguration,
      workerAddresses: string[],
      client?: MrcTestClient
   ): PipelineManager {
      if (!client) {
         client = new MrcTestClient();
      }

      const connectionManager = new ConnectionManager(client);
      const workersManager = new WorkersManager(connectionManager, workerAddresses);

      return new PipelineManager(workersManager, config);
   }

   get client() {
      return this.workersManager.client;
   }

   get connectionManager() {
      return this.workersManager.connectionManager;
   }

   get manifoldsManager() {
      return this._manifoldsManager;
   }

   get isRegistered() {
      return this._pipelineDefinitionId && this._pipelineInstanceId;
   }

   get isCreated() {
      return this._isCreated;
   }

   get pipelineDefinitionId() {
      return this._pipelineDefinitionId ?? throwExpression("Must register pipeline first");
   }

   get pipelineInstanceId() {
      return this._pipelineInstanceId ?? throwExpression("Must register pipeline first");
   }

   public async register() {
      if (this.isRegistered) {
         throw new Error("Already registered");
      }

      await this.workersManager.ensureRegistered();

      const mapping: IPipelineMapping = {
         connectionId: this.workersManager.connectionId,
         segments: Object.fromEntries(
            Object.entries(this.config.segments).map(([seg_name]) => {
               return [
                  seg_name,
                  {
                     segmentName: seg_name,
                     byPolicy: { value: SegmentMappingPolicies.OnePerWorker },
                  } as ISegmentMapping,
               ];
            })
         ),
      };

      // Now request to run a pipeline
      const configResponse = await PipelineManager.sendPipelineRegisterConfig(this.connectionManager, this.config);

      const mappingResponse = await PipelineManager.sendPipelineAddMapping(
         this.connectionManager,
         configResponse.pipelineDefinitionId,
         mapping
      );

      this._pipelineDefinitionId = configResponse.pipelineDefinitionId;
      this._pipelineInstanceId = mappingResponse.pipelineInstanceId;
   }

   public async ensureRegistered() {
      if (!this.isRegistered) {
         await this.register();
      }
   }

   public async unregister() {
      if (this.workersManager.isRegistered) {
         await this.workersManager.unregister();
      }

      this._pipelineDefinitionId = undefined;
      this._pipelineInstanceId = undefined;
   }

   public async createResources() {
      if (!this.isRegistered) {
         throw new Error("Must be registered first");
      }

      //  Update the PipelineInstance state to assign segment instances
      const pipeline_instance_state = await this.connectionManager.update_resource_status(
         this.pipelineInstanceId,
         "PipelineInstances",
         ResourceActualStatus.Actual_Created
      );

      const manifoldIds =
         this.connectionManager.getClientState().pipelineInstances!.entities[this.pipelineInstanceId].manifoldIds;

      // For each manifold, set it to created
      const manifolds = await Promise.all(
         manifoldIds.map(async (s) => {
            return await this.connectionManager.update_resource_status(
               s,
               "ManifoldInstances",
               ResourceActualStatus.Actual_Created
            )!;
         })
      );

      const segmentIds =
         this.connectionManager.getClientState().pipelineInstances!.entities[this.pipelineInstanceId].segmentIds;

      // For each segment, set it to created
      const segments = await Promise.all(
         segmentIds.map(async (s) => {
            return await this.connectionManager.update_resource_status(
               s,
               "SegmentInstances",
               ResourceActualStatus.Actual_Created
            )!;
         })
      );
   }

   public async ensureResourcesCreated() {
      await this.ensureRegistered();

      if (!this.isCreated) {
         await this.createResources();
      }
   }

   public static async sendPipelineRegisterConfig(
      connectionManager: ConnectionManager,
      config: IPipelineConfiguration
   ) {
      // Now request to run a pipeline
      const response = await connectionManager.send_request<PipelineRegisterConfigResponse>(
         EventType.ClientUnaryPipelineRegisterConfig,
         PipelineRegisterConfigRequest.create({
            config: config,
         })
      );

      return response;
   }

   public static async sendPipelineAddMapping(
      connectionManager: ConnectionManager,
      pipelineDefinitionId: string,
      mapping: IPipelineMapping
   ) {
      // Now request to run a pipeline
      const response = await connectionManager.send_request<PipelineAddMappingResponse>(
         EventType.ClientUnaryPipelineAddMapping,
         PipelineAddMappingRequest.create({
            definitionId: pipelineDefinitionId,
            mapping: mapping,
         })
      );

      return response;
   }
}
