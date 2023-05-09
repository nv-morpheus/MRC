import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";

import {Channel, credentials} from "@grpc/grpc-js";
import {ConnectivityState} from "@grpc/grpc-js/build/src/connectivity-state";
import {IConnection, IPipelineConfiguration, IPipelineMapping, ISegmentMapping} from "@mrc/common/entities";
import {ResourceStatus} from "@mrc/proto/mrc/protos/architect_state";
import {
   pipelineDefinitionsSelectById,
} from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import {pipeline_mappings} from "@mrc/tests/defaultObjects";
import {as, AsyncIterableX, AsyncSink} from "ix/asynciterable";
import {share} from "ix/asynciterable/operators";
import {createChannel, createClient, waitForChannelReady} from "nice-grpc";

import {packEvent, stringToBytes} from "../common/utils";
import {
   Ack,
   ArchitectClient,
   ArchitectDefinition,
   ClientConnectedResponse,
   Event,
   EventType,
   PingRequest,
   PipelineRequestAssignmentRequest,
   PipelineRequestAssignmentResponse,
   RegisterWorkersRequest,
   RegisterWorkersResponse,
   ResourceUpdateStatusRequest,
   ResourceUpdateStatusResponse,
} from "../proto/mrc/protos/architect";
import {ArchitectServer} from "../server/server";
import {connectionsSelectAll, connectionsSelectById} from "../server/store/slices/connectionsSlice";
import {pipelineInstancesSelectById} from "../server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesSelectByIds,
   segmentInstancesSelectByPipelineId,
   segmentInstancesSelectByWorkerId,
} from "../server/store/slices/segmentInstancesSlice";
import {workersSelectById} from "../server/store/slices/workersSlice";
import {RootStore, setupStore} from "../server/store/store";

import {unpack_first_event, unpack_unary_event} from "./utils";

describe("Client", () => {
   let store: RootStore;
   let server: ArchitectServer;
   let client_channel: Channel;
   let client: ArchitectClient;

   beforeEach(async () => {
      const startTime = performance.now();

      store = setupStore();

      // Use localhost:0 to bind to a random port to avoid collisions when testing
      server = new ArchitectServer(store, "localhost:0");

      const port = await server.start();

      client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

      // Now make the client
      client = createClient(ArchitectDefinition, client_channel);

      // Important to ensure the channel is ready before continuing
      await waitForChannelReady(client_channel, new Date(Date.now() + 1000));

      console.log(`beforeEach took ${performance.now() - startTime} milliseconds`);
   });

   afterEach(async () => {
      const startTime = performance.now();

      client_channel.close();
      await server.stop();
      await server.join();

      console.log(`afterEach took ${performance.now() - startTime} milliseconds`);
   });

   describe("connection", () => {
      it("is ready", () => {
         expect(client_channel.getConnectivityState(true)).toBe(ConnectivityState.READY);
      });

      it("ping", async () => {
         const req = PingRequest.create({
            tag: "1234",
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
            send_events      = new AsyncSink<Event>();

            recieve_events = as (client.eventStream(send_events, {
                                   signal: abort_controller.signal,
                                })).pipe(share());

            connected_response = await unpack_first_event<ClientConnectedResponse>(
                recieve_events,
                {predicate: (event) => event.event === EventType.ClientEventStreamConnected});
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
            try
            {
               for await (const req of recieve_events)
               {
                  abort_controller.abort();

                  send_events.end();
               }
            } catch (error: any)
            {
               expect(error.message).toMatch("The operation has been aborted");
            }
         });

         it("add one worker", async () => {
            const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
                recieve_events,
                send_events,
                packEvent(EventType.ClientUnaryRegisterWorkers, "9876", RegisterWorkersRequest.create({
                   ucxWorkerAddresses: stringToBytes(["test data"]),
                })));

            expect(registered_response.machineId).toBe(connected_response.machineId);

            // Need to do deeper checking here
         });

         it("activate stream", async () => {
            const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
                recieve_events,
                send_events,
                packEvent(EventType.ClientUnaryRegisterWorkers, "9876", RegisterWorkersRequest.create({
                   ucxWorkerAddresses: stringToBytes(["test data"]),
                })));

            const activated_response = await unpack_unary_event<Ack>(
                recieve_events,
                send_events,
                packEvent(EventType.ClientUnaryActivateStream, "2", registered_response));

            // Check to make sure its activated
            const found_worker = workersSelectById(store.getState(), registered_response.instanceIds[0]);

            expect(found_worker?.state.status).toBe(ResourceStatus.Activated);
         });

         describe("pipeline", () => {
            test("request pipeline", async () => {
               const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
                   recieve_events,
                   send_events,
                   packEvent(EventType.ClientUnaryRegisterWorkers, "9876", RegisterWorkersRequest.create({
                      ucxWorkerAddresses: stringToBytes(["test data", "test data 2"]),
                   })));

               const activated_response = await unpack_unary_event<Ack>(
                   recieve_events,
                   send_events,
                   packEvent(EventType.ClientUnaryActivateStream, "2", registered_response));

               const pipeline_config: IPipelineConfiguration = {
                  segments: {
                     my_seg1: {
                        egressPorts: [],
                        ingressPorts: [],
                        name: "my_seg",
                     },
                     my_seg2: {
                        egressPorts: [],
                        ingressPorts: [],
                        name: "my_seg2",
                     },
                  },
               };

               const pipeline_mapping: IPipelineMapping = {
                  machineId: connected_response.machineId,
                  segments:
                      Object.fromEntries(Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
                         return [
                            seg_name,
                            {segmentName: seg_name, byWorker: {workerIds: registered_response.instanceIds}} as
                                ISegmentMapping,
                         ];
                      })),
               };

               // Now request to run a pipeline
               const request_pipeline_response = await unpack_unary_event<PipelineRequestAssignmentResponse>(
                   recieve_events,
                   send_events,
                   packEvent(EventType.ClientUnaryRequestPipelineAssignment,
                             "12345",
                             PipelineRequestAssignmentRequest.create({
                                pipeline: pipeline_config,
                                mapping: pipeline_mapping,
                             })));

               // Check the pipeline definition
               const foundPipelineDefinition = pipelineDefinitionsSelectById(
                   store.getState(),
                   request_pipeline_response.pipelineDefinitionId);

               expect(foundPipelineDefinition?.id).toBe(request_pipeline_response.pipelineDefinitionId);

               // Check pipeline instances
               const foundPipelineInstance = pipelineInstancesSelectById(store.getState(),
                                                                         request_pipeline_response.pipelineInstanceId);

               expect(foundPipelineInstance?.machineId).toBe(connected_response.machineId);
               expect(foundPipelineInstance?.segmentIds).toEqual(request_pipeline_response.segmentInstanceIds);

               // Check segments exist in state
               let foundSegmentInstances = segmentInstancesSelectByIds(store.getState(),
                                                                       request_pipeline_response.segmentInstanceIds);

               expect(foundSegmentInstances).toHaveLength(0);

               //  Update the PipelineInstance state to assign segment instances
               const update_pipeline_status_response = await unpack_unary_event<ResourceUpdateStatusResponse>(
                   recieve_events,
                   send_events,
                   packEvent(EventType.ClientUnaryResourceUpdateStatus, "12345", ResourceUpdateStatusRequest.create({
                      resourceId: foundPipelineInstance?.id,
                      resourceType: "PipelineInstances",
                      status: ResourceStatus.Ready,
                   })));

               expect(update_pipeline_status_response.ok).toBeTruthy();

               foundSegmentInstances = segmentInstancesSelectByPipelineId(store.getState(), foundPipelineInstance?.id!);

               expect(foundSegmentInstances)
                   .toHaveLength(Object.keys(pipeline_mapping.segments).length *
                                 registered_response.instanceIds.length);
            });
         });
      });
   });
});

// Disabling this for now since it causes jest to crash when importing a ESM module

// describe("ClientWithDevTools", () => {
//    let store: RootStore;
//    let server: ArchitectServer;
//    let client_channel: Channel;
//    let client: ArchitectClient;

//    beforeEach(async () => {
//       // First create the dev server
//       await launchDevtoolsCli();

//       // Start the store with dev tools enabled
//       store = setupStore(undefined, true);

//       server = new ArchitectServer(store);

//       const port = await server.start();

//       client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

//       // Now make the client
//       client = createClient(ArchitectDefinition, client_channel);

//       // Important to ensure the channel is ready before continuing
//       await waitForChannelReady(client_channel, new Date(Date.now() + 1000));
//    });

//    it("ping", async () => {
//       const req = PingRequest.create({
//          tag: 1234,
//       });

//       const resp = await client.ping(req);

//       expect(resp.tag).toBe(req.tag);

//       await new Promise(r => setTimeout(r, 2000));
//    });

//    afterEach(async () => {
//       client_channel.close();
//       await server.stop();
//       await server.join();
//    });
// });
