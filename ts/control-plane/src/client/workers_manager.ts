/* eslint-disable @typescript-eslint/no-non-null-assertion */
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import "ix/add/asynciterable-operators/last";

import { stringToBytes, throwExpression } from "@mrc/common/utils";
import { Ack, EventType, RegisterWorkersRequest, RegisterWorkersResponse } from "@mrc/proto/mrc/protos/architect";

import { MrcTestClient } from "@mrc/client/client";
import { ConnectionManager } from "@mrc/client/connection_manager";

export class WorkersManager {
   private _client: MrcTestClient;
   private _registerResponse: RegisterWorkersResponse | undefined;
   private _isCreated = false;

   constructor(public readonly connectionManager: ConnectionManager, public addresses: string[]) {
      this._client = connectionManager.client;
   }

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

   public async createResources() {
      if (this.isCreated) {
         throw new Error("Already created");
      }

      const registerResponse = await this.ensureRegistered();

      await WorkersManager.sendActivateWorkersRequest(this.connectionManager, registerResponse);

      this._isCreated = true;
   }

   public async ensureResourcesCreated() {
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

   public static async sendActivateWorkersRequest(
      connectionManager: ConnectionManager,
      response: RegisterWorkersResponse
   ) {
      await connectionManager.send_request<Ack>(EventType.ClientUnaryActivateStream, response);

      return true;
   }

   public static async sendRegisterAndActivateWorkersRequest(
      connectionManager: ConnectionManager,
      addresses: string[]
   ) {
      const response = await WorkersManager.sendRegisterWorkersRequest(connectionManager, addresses);

      await WorkersManager.sendActivateWorkersRequest(connectionManager, response);

      return response;
   }
}
