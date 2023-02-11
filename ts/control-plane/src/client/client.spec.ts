import { Channel, credentials, ServerCredentials } from "@grpc/grpc-js";
import { ConnectivityState } from "@grpc/grpc-js/build/src/connectivity-state";
import assert from "assert";
import { createChannel, createClient, createServer, Server, waitForChannelReady } from "nice-grpc";
import { ArchitectClient, ArchitectDefinition, PingRequest, Event, EventType } from "../proto/mrc/protos/architect";

import { Architect } from "../server/architect";
import { ArchitectServer } from "../server/server";
import { connectionsSelectAll, IConnection } from "../server/store/slices/connectionsSlice";
import { RootState, RootStore, setupStore } from "../server/store/store";
import { AsyncSink, from, zip } from 'ix/asynciterable';
import { Observable, Subject } from 'rxjs';

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

      it("connect to event stream", async () => {

         let connections: IConnection[];
         const send_events = new AsyncSink<Event>();

         const state_update_sink = new AsyncSink<RootState>();

         // Subscribe to the stores next update
         const store_unsub = store.subscribe(() => {
            const state = store.getState();
            state_update_sink.write(state);
         });

         const resp_stream = client.eventStream(send_events);

         // Wait for the next state update
         // const first_update = await state_update_sink.next() as IteratorYieldResult<RootState>;

         // let connections = connectionsSelectAll(first_update.value);

         // expect(connections).toHaveLength(1);

         // // Send a start message
         // send_events.write(Event.create({
         //    event: EventType.ClientEventRequestStateUpdate
         // }));

         for await (const req of resp_stream) {
            fail("Should not have recieved any responses");
         }

         store_unsub();
         state_update_sink.end();

         connections = connectionsSelectAll(store.getState());

         expect(connections).toHaveLength(0);
      });

      it("abort after connection", async () => {

         const abort_controller = new AbortController();

         const event_generator = async function* (events: Event[]) {
            yield* events;
         };

         const resp_stream = client.eventStream(event_generator([]), {
            signal: abort_controller.signal,
         });

         abort_controller.abort();

         try {
            for await (const req of resp_stream) {
               fail("Should not have recieved any responses");
            }
         } catch (error: any) {
            expect(error.message).toMatch("The operation has been aborted");
         }
      });
   });
});
