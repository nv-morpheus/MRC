import { Channel, credentials, ServerCredentials } from "@grpc/grpc-js";
import { ConnectivityState } from "@grpc/grpc-js/build/src/connectivity-state";
import assert from "assert";
import { createChannel, createClient, createServer, Server, waitForChannelReady } from "nice-grpc";
import { ArchitectClient, ArchitectDefinition, PingRequest, Event, EventType, ClientConnectedResponse, RegisterWorkersRequest, RegisterWorkersResponse, Ack } from "../proto/mrc/protos/architect";

import { Architect } from "../server/architect";
import { ArchitectServer } from "../server/server";
import { connectionsSelectAll, connectionsSelectById, IConnection } from "../server/store/slices/connectionsSlice";
import { RootState, RootStore, setupStore } from "../server/store/store";
import { as, AsyncIterableX, AsyncSink, from, zip } from 'ix/asynciterable';
import { Observable, Subject } from 'rxjs';
import { pack, packEvent, unpackEvent } from "../common/utils";
import { pluck, share } from 'ix/asynciterable/operators';
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import { unary_event, unpack_first_event, unpack_unary_event } from "./utils";
import { workersSelectById } from "../server/store/slices/workersSlice";

describe("Client", () => {

   let store: RootStore;
   let server: ArchitectServer;
   let client_channel: Channel;
   let client: ArchitectClient;

   beforeEach(async () => {
      store = setupStore();

      server = new ArchitectServer(store);

      const port = await server.start();

      client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

      // Now make the client
      client = createClient(ArchitectDefinition, client_channel);

      // Important to ensure the channel is ready before continuing
      await waitForChannelReady(client_channel, new Date(Date.now() + 1000));
   });

   afterEach(async () => {
      client_channel.close();
      await server.stop();
      await server.join();
   });

   describe("connection", () => {
      it("is ready", () => {
         expect(client_channel.getConnectivityState(true)).toBe(ConnectivityState.READY);
      });

      it("ping", async () => {
         const req = PingRequest.create({
            tag: 1234,
         });

         const resp = await client.ping(req);

         expect(resp.tag).toBe(req.tag);
      });

      it("no connections on start", async () => {
         expect(connectionsSelectAll(store.getState())).toHaveLength(0);
      });

      describe("eventStream", () => {

         let abort_controller: AbortController;
         let send_events: AsyncSink<Event>;
         let recieve_events: AsyncIterableX<Event>;
         let connected_response: ClientConnectedResponse;

         beforeEach(async () => {
            abort_controller = new AbortController();
            send_events = new AsyncSink<Event>();

            recieve_events = as(client.eventStream(send_events, {
               signal: abort_controller.signal,
            })).pipe(share());

            connected_response = await unpack_first_event<ClientConnectedResponse>(recieve_events, {
               predicate: (event) => event.event === EventType.ClientEventStreamConnected
            });
         });

         afterEach(async () => {
            // Make sure to close down the send stream
            send_events.end();
         });

         it("connect to event stream", async () => {

            let connections: IConnection[];

            // Verify the number of connections is 1
            const connection = connectionsSelectById(store.getState(), connected_response.machineId);

            expect(connection).toBeDefined();
            expect(connection?.id).toBe(connected_response.machineId);

            // Cause a disconnect
            send_events.end();

            recieve_events.finalize(() => {
               connections = connectionsSelectAll(store.getState());

               expect(connections).toHaveLength(0);
            });

         });

         it("abort after connection", async () => {

            try {
               for await (const req of recieve_events) {
                  abort_controller.abort();

                  send_events.end();
               }
            } catch (error: any) {
               expect(error.message).toMatch("The operation has been aborted");
            }
         });

         it("add one worker", async () => {

            const registered_response = await unpack_unary_event<RegisterWorkersResponse>(recieve_events, send_events, packEvent(EventType.ClientUnaryRegisterWorkers, 9876, RegisterWorkersRequest.create({
               ucxWorkerAddresses: [new TextEncoder().encode("test data")],
            })));

            expect(registered_response.machineId).toBe(connected_response.machineId);

            // Need to do deeper checking here
         });

         it("activate stream", async () => {

            const registered_response = await unpack_unary_event<RegisterWorkersResponse>(recieve_events, send_events, packEvent(EventType.ClientUnaryRegisterWorkers, 9876, RegisterWorkersRequest.create({
               ucxWorkerAddresses: [new TextEncoder().encode("test data")],
            })));

            const activated_response = await unpack_unary_event<Ack>(recieve_events, send_events, packEvent(EventType.ClientUnaryActivateStream, 2, registered_response));

            // Check to make sure its activated
            const found_worker = workersSelectById(store.getState(), registered_response.instanceIds[0]);

            expect(found_worker?.activated).toBe(true);
         });
      });
   });
});
