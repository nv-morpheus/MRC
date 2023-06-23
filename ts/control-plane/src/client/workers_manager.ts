/* eslint-disable @typescript-eslint/no-non-null-assertion */
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import "ix/add/asynciterable-operators/last";

import { stringToBytes, throwExpression } from "@mrc/common/utils";
import { Ack, EventType, RegisterWorkersRequest, RegisterWorkersResponse } from "@mrc/proto/mrc/protos/architect";

import { MrcTestClient } from "@mrc/client/client";
import { ConnectionManager } from "@mrc/client/connection_manager";
import { ResourceActualStatus } from "@mrc/proto/mrc/protos/architect_state";

export class WorkersManager {
   private _registerResponse: RegisterWorkersResponse | undefined;
   private _isCreated = false;
   private _isRunning = false;

   constructor(public readonly connectionManager: ConnectionManager, public addresses: string[]) {}

   public static create(workerAddresses: string[], client?: MrcTestClient): WorkersManager {
      if (!client) {
         client = new MrcTestClient();
      }

      const connectionManager = new ConnectionManager(client);

      return new WorkersManager(connectionManager, workerAddresses);
   }

   get client() {
      return this.connectionManager.client;
   }

   get isRegistered() {
      return Boolean(this._registerResponse);
   }

   get isCreated() {
      return this._isCreated;
   }

   get isRunning() {
      return this._isRunning;
   }

   get machineId() {
      return this._registerResponse?.machineId ?? throwExpression("Must register first");
   }

   get workerIds() {
      return this._registerResponse?.instanceIds ?? throwExpression("Must register first");
   }

   public async register() {
      if (this.isRegistered) {
         throw new Error("Already registered");
      }

      await this.connectionManager.ensureResourcesCreated();

      // Now request to run a pipeline
      this._registerResponse = await WorkersManager.sendRegisterWorkersRequest(this.connectionManager, this.addresses);
   }

   public async ensureRegistered() {
      if (!this.isRegistered) {
         await this.register();
      }

      return this._registerResponse ?? throwExpression("Must register first");
   }

   public async unregister() {
      if (this.connectionManager.isRegistered) {
         await this.connectionManager.unregister();
      }

      this._registerResponse = undefined;
   }

   public async createResources() {
      if (this.isCreated) {
         throw new Error("Already created");
      }

      const registerResponse = await this.ensureRegistered();

      for (const workerId of registerResponse.instanceIds) {
         await WorkersManager.sendWorkerCreated(this.connectionManager, workerId);
      }

      this._isCreated = true;
   }

   public async ensureResourcesCreated() {
      await this.ensureRegistered();

      if (!this.isCreated) {
         await this.createResources();
      }
   }

   public async runResources() {
      if (this.isRunning) {
         throw new Error("Already running");
      }

      await this.ensureResourcesCreated();

      for (const workerId of this.workerIds) {
         await WorkersManager.sendWorkerRunning(this.connectionManager, workerId);
      }

      this._isCreated = true;
   }

   public async ensureResourcesRunning() {
      await this.ensureRegistered();

      if (!this.isCreated) {
         await this.createResources();
      }
   }

   public static async sendRegisterWorkersRequest(connectionManager: ConnectionManager, addresses: string[]) {
      const response = await connectionManager.send_request<RegisterWorkersResponse>(
         EventType.ClientUnaryRegisterWorkers,
         RegisterWorkersRequest.create({
            ucxWorkerAddresses: stringToBytes(addresses),
         })
      );

      return response;
   }

   public static async sendWorkerCreated(connectionManager: ConnectionManager, workerId: string) {
      await connectionManager.update_resource_status(workerId, "Workers", ResourceActualStatus.Actual_Created);

      return true;
   }

   public static async sendWorkerRunning(connectionManager: ConnectionManager, workerId: string) {
      await connectionManager.update_resource_status(workerId, "Workers", ResourceActualStatus.Actual_Running);

      return true;
   }
}
