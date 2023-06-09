/* eslint-disable @typescript-eslint/no-non-null-assertion */
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import "ix/add/asynciterable-operators/last";

import { url as inspectorUrl } from "node:inspector";
import { Channel, credentials } from "@grpc/grpc-js";
import { ConnectivityState } from "@grpc/grpc-js/build/src/connectivity-state";
import {
   ControlPlaneState,
   ManifoldInstance,
   PipelineInstance,
   ResourceActualStatus,
   SegmentInstance,
   SegmentMappingPolicies,
} from "@mrc/proto/mrc/protos/architect_state";
import { as, AsyncIterableX, AsyncSink } from "ix/asynciterable";
import { filter as filter_ix, share as share_ix, tap as tax_ix } from "ix/asynciterable/operators";
import { createChannel, createClient, waitForChannelReady } from "nice-grpc";

import { generateId, packEvent, sleep, stringToBytes, unpackEvent } from "@mrc/common/utils";
import {
   Ack,
   ArchitectClient,
   ArchitectDefinition,
   ClientConnectedResponse,
   Event,
   EventType,
   PingRequest,
   PingResponse,
   PipelineRequestAssignmentRequest,
   PipelineRequestAssignmentResponse,
   RegisterWorkersRequest,
   RegisterWorkersResponse,
   ResourceUpdateStatusRequest,
   ResourceUpdateStatusResponse,
} from "@mrc/proto/mrc/protos/architect";
import { ArchitectServer } from "@mrc/server/server";
import { RootStore, setupStore } from "@mrc/server/store/store";

import { unpack_first_event, unpack_unary_event } from "@mrc/client/utils";
import { UnknownMessage } from "@mrc/proto/typeRegistry";
import { Observable, filter, firstValueFrom, from, lastValueFrom, share, tap } from "rxjs";
import {
   IPipelineConfiguration,
   IPipelineInstance,
   IPipelineMapping,
   ISegmentInstance,
   ISegmentMapping,
} from "@mrc/common/entities";

export class MrcTestClient {
   public store: RootStore | null = null;
   public server: ArchitectServer | null = null;
   public client_channel: Channel | null = null;
   public client: ArchitectClient | null = null;
   private _abort_controller: AbortController = new AbortController();
   private _send_events: AsyncSink<Event> | null = null;
   private _receive_events$: Observable<Event> | null = null;
   public machineId: string | null = null;

   private _state_updates: Array<ControlPlaneState> = [];
   private _message_history: Array<Event> = [];
   // private _response_messages: Array<Event> = [];

   private _debugger_attached: boolean;
   private _response_stream$: Observable<Event> | null = null;
   private _receive_events_complete: Promise<Event> | null = null;

   constructor() {
      this._debugger_attached = inspectorUrl() !== undefined;

      if (this._debugger_attached) {
         console.log("Debugger attached. Creating dev tools connection");
      }

      // this._debugger_attached = false;
   }

   public async initializeClient() {
      const startTime = performance.now();

      this.store = setupStore(undefined, this._debugger_attached);

      // Use localhost:0 to bind to a random port to avoid collisions when testing
      this.server = new ArchitectServer(this.store, "localhost:0");

      const port = await this.server.start();

      this.client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

      // Now make the client
      this.client = createClient(ArchitectDefinition, this.client_channel);

      // Important to ensure the channel is ready before continuing
      await waitForChannelReady(this.client_channel, new Date(Date.now() + 1000));

      // console.log("Sleeping");
      // await sleep(1000);
      // console.log("Sleeping done");

      // console.log(`beforeEach took ${performance.now() - startTime} milliseconds`);
   }

   public async finalizeClient() {
      const startTime = performance.now();

      if (this._debugger_attached) {
         // We have a debugger attached. Sleep to keep the state open
         console.log("Sleeping for 60s to inspect state");
         await sleep(60000);
         console.log("Sleeping done");
      }

      // if (this.client) {
      //    await this.client.shutdown({});
      //    this.client = null;
      // }

      // if (this.store) {
      //    // Send the stop message to the state
      //    this.store?.dispatch(stopAction());
      // }

      if (this.client_channel) {
         this.client_channel.close();
         this.client_channel = null;
      }

      if (this.server) {
         await this.server.stop();
         await this.server.join();
         this.server = null;
      }

      // console.log(`afterEach took ${performance.now() - startTime} milliseconds`);
   }

   public async initializeEventStream() {
      if (!this.client) {
         throw new Error("Must initialize client before stream");
      }

      this._abort_controller = new AbortController();
      this._send_events = new AsyncSink<Event>();

      const receive_events = as(
         this.client.eventStream(this._send_events, {
            signal: this._abort_controller.signal,
         })
      );

      this._receive_events$ = from(receive_events).pipe(
         tap((event) => {
            // Save a history of the messages to help with debugging
            this._message_history.push(event);
         }),
         share()
      );

      this._receive_events_complete = lastValueFrom(this._receive_events$, {
         defaultValue: Event.create({}),
      });

      // Subscribe permenantly to keep the stream hot
      this._receive_events$
         .pipe(
            filter((value) => {
               return value.event === EventType.ServerStateUpdate;
            })
         )
         .forEach((value: Event) => {
            // Save all of the server state updates
            this._state_updates.push(unpackEvent<ControlPlaneState>(value));
         });

      // Wait for the connected response before filtering off the state update
      const connected_response = await unpack_first_event<ClientConnectedResponse>(
         this._receive_events$,
         (event) => event.event === EventType.ClientEventStreamConnected
      );

      // this._response_stream = this._recieve_events.pipe(
      //    filter((value) => {
      //       return value.event !== EventType.ServerStateUpdate;
      //    })
      // );
      this._response_stream$ = this._receive_events$;

      // this._recieve_events
      //    .pipe(
      //       filter((value) => {
      //          return value.event !== EventType.ServerStateUpdate;
      //       })
      //    )
      //    .forEach((value: Event, index, signal) => {
      //       // Save all of the server state updates
      //       this._response_messages.push(value);
      //    });

      this.machineId = connected_response.machineId;
   }

   public async finalizeEventStream() {
      if (this._send_events) {
         this._send_events.end();
         this._send_events = null;
      }

      // Need to await for all events to flush through
      if (this._receive_events_complete) {
         await this._receive_events_complete;
         this._receive_events_complete = null;
      }

      // if (this._recieve_events) {
      //    // This can fail so unset the variable before the for loop
      //    const recieve_events = this._recieve_events;
      //    this._recieve_events = null;

      //    for await (const item of recieve_events) {
      //       console.log(`Excess messages left in recieve queue. Msg: ${item}`);
      //    }
      // }
   }

   public isChannelConnected() {
      return this.client_channel?.getConnectivityState(true) == ConnectivityState.READY;
   }

   public async abortConnection(reason = "Abort requested") {
      if (!this.client) {
         throw new Error("Client is not connected");
      }

      this._abort_controller.abort(reason);

      // Call finalize to close the input stream and pull off any messages before exiting
      await this.finalizeEventStream();
   }

   public getServerState() {
      if (!this.store) {
         throw new Error("Client is not connected");
      }

      return this.store.getState();
   }

   public getClientState() {
      if (this._state_updates.length === 0) {
         throw new Error("Cant get client state. No state updates have been received");
      }

      return this._state_updates[this._state_updates.length - 1];
   }

   public async ping(request: PingRequest): Promise<PingResponse> {
      if (!this.client) {
         throw new Error("Client is not connected");
      }

      return await this.client.ping(request);
   }

   public async send_request<ResponseT extends UnknownMessage>(event_type: EventType, request: UnknownMessage) {
      if (!this._response_stream$ || !this._send_events) {
         throw new Error("Client is not connected");
      }

      // Pack message with random tag
      const message = packEvent(event_type, generateId().toString(), request);

      return await unpack_unary_event<ResponseT>(this._response_stream$, this._send_events, message);
   }

   // public async unary_event<MessageT extends UnknownMessage>(message: Event) {
   //    if (!this._response_stream$ || !this._send_events) {
   //       throw new Error("Client is not connected");
   //    }

   //    return await unpack_unary_event<MessageT>(this._response_stream$, this._send_events, message);
   // }

   public async register_workers(addresses: string[]) {
      const response = await this.send_request<RegisterWorkersResponse>(
         EventType.ClientUnaryRegisterWorkers,
         RegisterWorkersRequest.create({
            ucxWorkerAddresses: stringToBytes(addresses),
         })
      );

      return response;
   }

   public async activate_workers(response: RegisterWorkersResponse) {
      await this.send_request<Ack>(EventType.ClientUnaryActivateStream, response);

      return true;
   }

   public async register_and_activate_workers(addresses: string[]) {
      const response = await this.register_workers(addresses);

      await this.activate_workers(response);

      return response;
   }

   public async register_pipeline_config(config: IPipelineConfiguration) {
      const mapping: IPipelineMapping = {
         machineId: this.machineId!,
         segments: Object.fromEntries(
            Object.entries(config.segments).map(([seg_name]) => {
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
      const response = await this.send_request<PipelineRequestAssignmentResponse>(
         EventType.ClientUnaryRequestPipelineAssignment,

         PipelineRequestAssignmentRequest.create({
            pipeline: config,
            mapping: mapping,
         })
      );

      return response;
   }

   public async update_resource_status(
      id: string,
      resource_type: "PipelineInstances",
      status: ResourceActualStatus
   ): Promise<PipelineInstance | null>;
   public async update_resource_status(
      id: string,
      resource_type: "SegmentInstances",
      status: ResourceActualStatus
   ): Promise<SegmentInstance | null>;
   public async update_resource_status(
      id: string,
      resource_type: "ManifoldInstances",
      status: ResourceActualStatus
   ): Promise<ManifoldInstance | null>;
   public async update_resource_status(
      id: string,
      resource_type: "PipelineInstances" | "SegmentInstances" | "ManifoldInstances",
      status: ResourceActualStatus
   ) {
      const response = await this.send_request<ResourceUpdateStatusResponse>(
         EventType.ClientUnaryResourceUpdateStatus,
         ResourceUpdateStatusRequest.create({
            resourceId: id,
            resourceType: resource_type,
            status: status,
         })
      );

      // Now return the correct instance from the updated state
      if (resource_type === "PipelineInstances") {
         const entities = this.getClientState().pipelineInstances!.entities;

         if (!(id in entities)) {
            return null;
         }

         return entities[id];
      } else if (resource_type === "SegmentInstances") {
         const entities = this.getClientState().segmentInstances!.entities;

         if (!(id in entities)) {
            return null;
         }

         return entities[id];
      } else if (resource_type === "ManifoldInstances") {
         const entities = this.getClientState().manifoldInstances!.entities;

         if (!(id in entities)) {
            return null;
         }

         return entities[id];
      } else {
         throw new Error("Unknow resource type");
      }
   }
}
