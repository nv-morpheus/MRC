/* eslint-disable @typescript-eslint/no-non-null-asserted-optional-chain */
/* eslint-disable @typescript-eslint/no-non-null-assertion */
import "ix/add/asynciterable-operators/first";
import "ix/add/asynciterable-operators/finalize";
import "ix/add/asynciterable-operators/last";

import { Channel, credentials } from "@grpc/grpc-js";
import { ConnectivityState } from "@grpc/grpc-js/build/src/connectivity-state";
import { IConnection, IPipelineConfiguration, IPipelineMapping, ISegmentMapping } from "@mrc/common/entities";
import { ResourceActualStatus, ResourceStatus } from "@mrc/proto/mrc/protos/architect_state";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipeline_mappings } from "@mrc/tests/defaultObjects";
import { as, AsyncIterableX, AsyncSink, from } from "ix/asynciterable";
import { filter, finalize, map, share } from "ix/asynciterable/operators";
import { createChannel, createClient, waitForChannelReady } from "nice-grpc";

import { packEvent, stringToBytes } from "../common/utils";
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
} from "../proto/mrc/protos/architect";
import { ArchitectServer } from "../server/server";
import { connectionsSelectAll, connectionsSelectById } from "../server/store/slices/connectionsSlice";
import { pipelineInstancesSelectById } from "../server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesSelectByIds,
   segmentInstancesSelectByPipelineId,
} from "../server/store/slices/segmentInstancesSlice";
import { workersSelectById } from "../server/store/slices/workersSlice";
import { RootStore, setupStore } from "../server/store/store";

import { unpack_first_event, unpack_unary_event } from "./utils";
import { UnknownMessage } from "@mrc/proto/typeRegistry";

// class AsyncFlag<T> {
//    private _promise: Promise<T>;
//    private _resolve: ((value: T) => void) | undefined;
//    private _reject: ((reason?: any) => void) | undefined;

//    constructor() {
//       this._promise = new Promise((resolve, reject) => {
//          this._resolve = resolve;
//          this._reject = reject;
//       });
//    }

//    public reset() {
//       this._promise = new Promise((resolve, reject) => {
//          this._resolve = resolve;
//          this._reject = reject;
//       });
//    }

//    public set(data: T) {
//       if (!this._resolve) {
//          throw new Error("Something bad happened");
//       }

//       this._resolve(data);
//    }

//    async get(timeout = 0) {
//       if (timeout <= 0) {
//          return this._promise;
//       }

//       return new Promise((resolve, reject) => {
//          const timer = setTimeout(() => {
//             const msg = "AsyncFlag timeout";
//             reject(msg);
//          }, timeout);

//          this._promise
//             .then(resolve)
//             .catch(reject)
//             .then(() => clearTimeout(timer));
//       });
//    }

//    public error(err: any) {
//       if (!this._reject) {
//          throw new Error("Something bad happened");
//       }

//       this._reject(err);
//    }
// }

// async function clientBeforeEach() {
//    const startTime = performance.now();

//    const store = setupStore();

//    // Use localhost:0 to bind to a random port to avoid collisions when testing
//    const server = new ArchitectServer(store, "localhost:0");

//    const port = await server.start();

//    const client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

//    // Now make the client
//    const client = createClient(ArchitectDefinition, client_channel);

//    // Important to ensure the channel is ready before continuing
//    await waitForChannelReady(client_channel, new Date(Date.now() + 1000));

//    // console.log(`beforeEach took ${performance.now() - startTime} milliseconds`);
//    return {
//       store,
//       server,
//       client_channel,
//       client,
//    };
// }

// async function clientAfterEach(server: ArchitectServer, client_channel: Channel) {
//    const startTime = performance.now();

//    client_channel.close();
//    await server.stop();
//    await server.join();

//    // console.log(`afterEach took ${performance.now() - startTime} milliseconds`);
// }

class MrcTestClient {
   public store: RootStore | null = null;
   public server: ArchitectServer | null = null;
   public client_channel: Channel | null = null;
   public client: ArchitectClient | null = null;
   private _abort_controller: AbortController = new AbortController();
   private _send_events: AsyncSink<Event> | null = null;
   private _recieve_events: AsyncIterableX<Event> | null = null;
   public machineId: string | null = null;

   constructor() {}

   public async initializeClient() {
      const startTime = performance.now();

      this.store = setupStore();

      // Use localhost:0 to bind to a random port to avoid collisions when testing
      this.server = new ArchitectServer(this.store, "localhost:0");

      const port = await this.server.start();

      this.client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

      // Now make the client
      this.client = createClient(ArchitectDefinition, this.client_channel);

      // Important to ensure the channel is ready before continuing
      await waitForChannelReady(this.client_channel, new Date(Date.now() + 1000));

      // console.log(`beforeEach took ${performance.now() - startTime} milliseconds`);
   }

   public async finalizeClient() {
      const startTime = performance.now();

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

      this._recieve_events = as(
         this.client.eventStream(this._send_events, {
            signal: this._abort_controller.signal,
         })
      ).pipe(share());

      const connected_response = await unpack_first_event<ClientConnectedResponse>(this._recieve_events, {
         predicate: (event) => event.event === EventType.ClientEventStreamConnected,
      });

      this.machineId = connected_response.machineId;
   }

   public async finalizeEventStream() {
      if (this._send_events) {
         this._send_events.end();
         this._send_events = null;
      }

      if (this._recieve_events) {
         // This can fail so unset the variable before the for loop
         const recieve_events = this._recieve_events;
         this._recieve_events = null;

         for await (const item of recieve_events) {
            console.log(`Excess messages left in recieve queue. Msg: ${item}`);
         }
      }
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

   public getState() {
      if (!this.store) {
         throw new Error("Client is not connected");
      }

      return this.store.getState();
   }

   public async ping(request: PingRequest): Promise<PingResponse> {
      if (!this.client) {
         throw new Error("Client is not connected");
      }

      return await this.client.ping(request);
   }

   public async unary_event<MessageT extends UnknownMessage>(message: Event) {
      if (!this._recieve_events || !this._send_events) {
         throw new Error("Client is not connected");
      }

      return unpack_unary_event<MessageT>(this._recieve_events, this._send_events, message);
   }
}

// describe("Client", () => {
//    let store: RootStore;
//    let server: ArchitectServer;
//    let client_channel: Channel;
//    let client: ArchitectClient;

//    beforeEach(async () => {
//       // ({ store, server, client_channel, client } = await clientBeforeEach());

//       const startTime = performance.now();

//       store = setupStore();

//       // Use localhost:0 to bind to a random port to avoid collisions when testing
//       server = new ArchitectServer(store, "localhost:0");

//       const port = await server.start();

//       client_channel = createChannel(`localhost:${port}`, credentials.createInsecure());

//       // Now make the client
//       client = createClient(ArchitectDefinition, client_channel);

//       // Important to ensure the channel is ready before continuing
//       await waitForChannelReady(client_channel, new Date(Date.now() + 1000));

//       // console.log(`beforeEach took ${performance.now() - startTime} milliseconds`);
//    });

//    afterEach(async () => {
//       // await clientAfterEach(server, client_channel);

//       const startTime = performance.now();

//       client_channel.close();
//       await server.stop();
//       await server.join();

//       // console.log(`afterEach took ${performance.now() - startTime} milliseconds`);
//    });

//    describe("connection", () => {
//       it("is ready", () => {
//          expect(client_channel.getConnectivityState(true)).toBe(ConnectivityState.READY);
//       });

//       it("ping", async () => {
//          const req = PingRequest.create({
//             tag: "1234",
//          });

//          const resp = await client.ping(req);

//          expect(resp.tag).toBe(req.tag);
//       });

//       it("no connections on start", async () => {
//          expect(connectionsSelectAll(store.getState())).toHaveLength(0);
//       });

//       describe("eventStream", () => {
//          let abort_controller: AbortController;
//          let send_events: AsyncSink<Event>;
//          let recieve_events: AsyncIterableX<Event>;
//          let connected_response: ClientConnectedResponse;

//          beforeEach(async () => {
//             abort_controller = new AbortController();
//             send_events = new AsyncSink<Event>();

//             recieve_events = as(
//                client.eventStream(send_events, {
//                   signal: abort_controller.signal,
//                })
//             ).pipe(share());

//             connected_response = await unpack_first_event<ClientConnectedResponse>(recieve_events, {
//                predicate: (event) => event.event === EventType.ClientEventStreamConnected,
//             });
//          });

//          afterEach(async () => {
//             // Make sure to close down the send stream
//             send_events.end();
//          });

//          it("connect to event stream", async () => {
//             let connections: IConnection[];

//             // Verify the number of connections is 1
//             const connection = connectionsSelectById(store.getState(), connected_response.machineId);

//             expect(connection).toBeDefined();
//             expect(connection?.id).toBe(connected_response.machineId);

//             recieve_events.finalize(() => {
//                connections = connectionsSelectAll(store.getState());

//                expect(connections).toHaveLength(0);
//             });

//             // Cause a disconnect
//             send_events.end();
//          });

//          it("abort after connection", async () => {
//             try {
//                for await (const req of recieve_events) {
//                   abort_controller.abort();

//                   send_events.end();
//                }
//             } catch (error: any) {
//                expect(error.message).toMatch("The operation has been aborted");
//             }
//          });

//          it("add one worker", async () => {
//             const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
//                recieve_events,
//                send_events,
//                packEvent(
//                   EventType.ClientUnaryRegisterWorkers,
//                   "9876",
//                   RegisterWorkersRequest.create({
//                      ucxWorkerAddresses: stringToBytes(["test data"]),
//                   })
//                )
//             );

//             expect(registered_response.machineId).toBe(connected_response.machineId);

//             // Need to do deeper checking here
//          });

//          it("activate stream", async () => {
//             const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
//                recieve_events,
//                send_events,
//                packEvent(
//                   EventType.ClientUnaryRegisterWorkers,
//                   "9876",
//                   RegisterWorkersRequest.create({
//                      ucxWorkerAddresses: stringToBytes(["test data"]),
//                   })
//                )
//             );

//             const activated_response = await unpack_unary_event<Ack>(
//                recieve_events,
//                send_events,
//                packEvent(EventType.ClientUnaryActivateStream, "2", registered_response)
//             );

//             // Check to make sure its activated
//             const found_worker = workersSelectById(store.getState(), registered_response.instanceIds[0]);

//             expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Ready);
//          });

//          describe("pipeline", () => {
//             test("request pipeline", async () => {
//                const registered_response = await unpack_unary_event<RegisterWorkersResponse>(
//                   recieve_events,
//                   send_events,
//                   packEvent(
//                      EventType.ClientUnaryRegisterWorkers,
//                      "9876",
//                      RegisterWorkersRequest.create({
//                         ucxWorkerAddresses: stringToBytes(["test data", "test data 2"]),
//                      })
//                   )
//                );

//                const activated_response = await unpack_unary_event<Ack>(
//                   recieve_events,
//                   send_events,
//                   packEvent(EventType.ClientUnaryActivateStream, "2", registered_response)
//                );

//                const pipeline_config: IPipelineConfiguration = {
//                   segments: {
//                      my_seg1: {
//                         egressPorts: {},
//                         ingressPorts: {},
//                         name: "my_seg",
//                      },
//                      my_seg2: {
//                         egressPorts: {},
//                         ingressPorts: {},
//                         name: "my_seg2",
//                      },
//                   },
//                   manifolds: {},
//                };

//                const pipeline_mapping: IPipelineMapping = {
//                   machineId: connected_response.machineId,
//                   segments: Object.fromEntries(
//                      Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
//                         return [
//                            seg_name,
//                            {
//                               segmentName: seg_name,
//                               byWorker: { workerIds: registered_response.instanceIds },
//                            } as ISegmentMapping,
//                         ];
//                      })
//                   ),
//                };

//                // Now request to run a pipeline
//                const request_pipeline_response = await unpack_unary_event<PipelineRequestAssignmentResponse>(
//                   recieve_events,
//                   send_events,
//                   packEvent(
//                      EventType.ClientUnaryRequestPipelineAssignment,
//                      "12345",
//                      PipelineRequestAssignmentRequest.create({
//                         pipeline: pipeline_config,
//                         mapping: pipeline_mapping,
//                      })
//                   )
//                );

//                // Check the pipeline definition
//                const foundPipelineDefinition = pipelineDefinitionsSelectById(
//                   store.getState(),
//                   request_pipeline_response.pipelineDefinitionId
//                );

//                expect(foundPipelineDefinition?.id).toBe(request_pipeline_response.pipelineDefinitionId);

//                // Check pipeline instances
//                const foundPipelineInstance = pipelineInstancesSelectById(
//                   store.getState(),
//                   request_pipeline_response.pipelineInstanceId
//                );

//                expect(foundPipelineInstance?.machineId).toBe(connected_response.machineId);
//                expect(foundPipelineInstance?.segmentIds).toEqual(request_pipeline_response.segmentInstanceIds);

//                // Check segments exist in state
//                let foundSegmentInstances = segmentInstancesSelectByIds(
//                   store.getState(),
//                   request_pipeline_response.segmentInstanceIds
//                );

//                expect(foundSegmentInstances).toHaveLength(0);

//                //  Update the PipelineInstance state to assign segment instances
//                const update_pipeline_status_response = await unpack_unary_event<ResourceUpdateStatusResponse>(
//                   recieve_events,
//                   send_events,
//                   packEvent(
//                      EventType.ClientUnaryResourceUpdateStatus,
//                      "12345",
//                      ResourceUpdateStatusRequest.create({
//                         resourceId: foundPipelineInstance?.id,
//                         resourceType: "PipelineInstances",
//                         status: ResourceActualStatus.Actual_Ready,
//                      })
//                   )
//                );

//                expect(update_pipeline_status_response.ok).toBeTruthy();

//                foundSegmentInstances = segmentInstancesSelectByPipelineId(store.getState(), foundPipelineInstance?.id!);

//                expect(foundSegmentInstances).toHaveLength(
//                   Object.keys(pipeline_mapping.segments).length * registered_response.instanceIds.length
//                );
//             });
//          });
//       });
//    });
// });

describe("Connection", () => {
   const client: MrcTestClient = new MrcTestClient();

   beforeEach(async () => {
      await client.initializeClient();
   });

   afterEach(async () => {
      await client.finalizeClient();
   });

   test("Is Ready", () => {
      expect(client.isChannelConnected()).toBeTruthy();
   });

   test("Ping", async () => {
      const req = PingRequest.create({
         tag: "1234",
      });

      const resp = await client.ping(req);

      expect(resp.tag).toBe(req.tag);
   });

   test("No Connections On Start", async () => {
      expect(connectionsSelectAll(client.getState())).toHaveLength(0);
   });

   test("No Connections After Disconnect", async () => {
      // Connect then disconnect
      await client.initializeEventStream();
      await client.finalizeEventStream();

      // Should have 0 connections in the state
      expect(connectionsSelectAll(client.getState())).toHaveLength(0);
   });

   describe("With EventStream", () => {
      beforeEach(async () => {
         await client.initializeEventStream();
      });

      afterEach(async () => {
         await client.finalizeEventStream();
      });

      test("Found Connection", () => {
         // Verify the number of connections is 1
         const connection = connectionsSelectById(client.getState(), client.machineId!);

         expect(connection).toBeDefined();
         expect(connection?.id).toEqual(client.machineId!);
      });

      test("Abort", async () => {
         expect(client.abortConnection()).rejects.toThrow("The operation has been aborted");
      });
   });
});

describe("Worker", () => {
   const client: MrcTestClient = new MrcTestClient();

   beforeEach(async () => {
      await client.initializeClient();
      await client.initializeEventStream();
   });

   afterEach(async () => {
      await client.finalizeEventStream();
      await client.finalizeClient();
   });

   test("Add One", async () => {
      const registered_response = await client.unary_event<RegisterWorkersResponse>(
         packEvent(
            EventType.ClientUnaryRegisterWorkers,
            "9876",
            RegisterWorkersRequest.create({
               ucxWorkerAddresses: stringToBytes(["test data"]),
            })
         )
      );

      expect(registered_response.machineId).toBe(client.machineId);

      // Need to do deeper checking here
   });

   it("Activate", async () => {
      const registered_response = await client.unary_event<RegisterWorkersResponse>(
         packEvent(
            EventType.ClientUnaryRegisterWorkers,
            "9876",
            RegisterWorkersRequest.create({
               ucxWorkerAddresses: stringToBytes(["test data"]),
            })
         )
      );

      const activated_response = await client.unary_event<Ack>(
         packEvent(EventType.ClientUnaryActivateStream, "2", registered_response)
      );

      // Check to make sure its activated
      const found_worker = workersSelectById(client.getState(), registered_response.instanceIds[0]);

      expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Ready);
   });
});

describe("Pipeline", () => {
   const client: MrcTestClient = new MrcTestClient();

   beforeEach(async () => {
      await client.initializeClient();
      await client.initializeEventStream();
   });

   afterEach(async () => {
      await client.finalizeEventStream();
      await client.finalizeClient();
   });

   test("Request Assignment", async () => {
      const registered_response = await client.unary_event<RegisterWorkersResponse>(
         packEvent(
            EventType.ClientUnaryRegisterWorkers,
            "9876",
            RegisterWorkersRequest.create({
               ucxWorkerAddresses: stringToBytes(["test data", "test data 2"]),
            })
         )
      );

      const activated_response = await client.unary_event<Ack>(
         packEvent(EventType.ClientUnaryActivateStream, "2", registered_response)
      );

      const pipeline_config: IPipelineConfiguration = {
         segments: {
            my_seg1: {
               egressPorts: {},
               ingressPorts: {},
               name: "my_seg",
            },
            my_seg2: {
               egressPorts: {},
               ingressPorts: {},
               name: "my_seg2",
            },
         },
         manifolds: {},
      };

      const pipeline_mapping: IPipelineMapping = {
         machineId: client.machineId!,
         segments: Object.fromEntries(
            Object.entries(pipeline_config.segments).map(([seg_name, seg_config]) => {
               return [
                  seg_name,
                  {
                     segmentName: seg_name,
                     byWorker: { workerIds: registered_response.instanceIds },
                  } as ISegmentMapping,
               ];
            })
         ),
      };

      // Now request to run a pipeline
      const request_pipeline_response = await client.unary_event<PipelineRequestAssignmentResponse>(
         packEvent(
            EventType.ClientUnaryRequestPipelineAssignment,
            "12345",
            PipelineRequestAssignmentRequest.create({
               pipeline: pipeline_config,
               mapping: pipeline_mapping,
            })
         )
      );

      // Check the pipeline definition
      const foundPipelineDefinition = pipelineDefinitionsSelectById(
         client.getState(),
         request_pipeline_response.pipelineDefinitionId
      );

      expect(foundPipelineDefinition?.id).toBe(request_pipeline_response.pipelineDefinitionId);

      // Check pipeline instances
      const foundPipelineInstance = pipelineInstancesSelectById(
         client.getState(),
         request_pipeline_response.pipelineInstanceId
      );

      expect(foundPipelineInstance).toBeDefined();

      expect(foundPipelineInstance?.machineId).toEqual(client.machineId);
      expect(foundPipelineInstance?.segmentIds).toEqual(request_pipeline_response.segmentInstanceIds);

      // Check segments exist in state
      let foundSegmentInstances = segmentInstancesSelectByIds(
         client.getState(),
         request_pipeline_response.segmentInstanceIds
      );

      expect(foundSegmentInstances).toHaveLength(0);

      //  Update the PipelineInstance state to assign segment instances
      const update_pipeline_status_response = await client.unary_event<ResourceUpdateStatusResponse>(
         packEvent(
            EventType.ClientUnaryResourceUpdateStatus,
            "12345",
            ResourceUpdateStatusRequest.create({
               resourceId: foundPipelineInstance?.id,
               resourceType: "PipelineInstances",
               status: ResourceActualStatus.Actual_Ready,
            })
         )
      );

      expect(update_pipeline_status_response.ok).toBeTruthy();

      foundSegmentInstances = segmentInstancesSelectByPipelineId(client.getState(), foundPipelineInstance?.id!);

      expect(foundSegmentInstances).toHaveLength(
         Object.keys(pipeline_mapping.segments).length * registered_response.instanceIds.length
      );
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
