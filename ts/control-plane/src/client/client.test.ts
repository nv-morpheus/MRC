/* eslint-disable @typescript-eslint/no-non-null-asserted-optional-chain */
/* eslint-disable @typescript-eslint/no-non-null-assertion */

import { MrcTestClient } from "@mrc/client/client";
import { IPipelineConfiguration, IPipelineMapping, ISegmentMapping } from "@mrc/common/entities";
import { packEvent, stringToBytes } from "@mrc/common/utils";
import {
   Ack,
   EventType,
   PingRequest,
   PipelineRequestAssignmentRequest,
   PipelineRequestAssignmentResponse,
   RegisterWorkersRequest,
   RegisterWorkersResponse,
   ResourceUpdateStatusRequest,
   ResourceUpdateStatusResponse,
} from "@mrc/proto/mrc/protos/architect";
import {
   ManifoldOptions_Policy,
   PipelineInstance,
   PipelineMapping_SegmentMapping_ByPolicy,
   ResourceActualStatus,
   ResourceRequestedStatus,
   SegmentInstance,
   SegmentMappingPolicies,
} from "@mrc/proto/mrc/protos/architect_state";
import { connectionsSelectAll, connectionsSelectById } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesSelectById } from "@mrc/server/store/slices/pipelineInstancesSlice";
import {
   segmentInstancesSelectByIds,
   segmentInstancesSelectByPipelineId,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import { workersSelectById } from "@mrc/server/store/slices/workersSlice";

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

//             expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Running);
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
//                         status: ResourceActualStatus.Actual_Running,
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
      expect(connectionsSelectAll(client.getServerState())).toHaveLength(0);
   });

   test("No Connections After Disconnect", async () => {
      // Connect then disconnect
      await client.initializeEventStream();
      await client.finalizeEventStream();

      // Should have 0 connections in the state
      expect(connectionsSelectAll(client.getServerState())).toHaveLength(0);
   });

   describe("With EventStream", () => {
      beforeEach(async () => {
         await client.initializeEventStream();
      });

      afterEach(async () => {
         await client.finalizeEventStream();
      });

      test("Found Connection", async () => {
         // Verify the number of connections is 1
         const connection = connectionsSelectById(client.getServerState(), client.machineId!);

         expect(connection).toBeDefined();
         expect(connection?.id).toEqual(client.machineId!);
      });

      // test("Abort", async () => {
      //    expect(client.abortConnection()).rejects.toThrow("The operation has been aborted");
      // });
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
      const response = await client.register_workers(["test data"]);

      expect(response.machineId).toBe(client.machineId);

      // Need to do deeper checking here
   });

   test("Activate", async () => {
      const registered_response = await client.register_and_activate_workers(["test data"]);

      // Check to make sure its activated
      const found_worker = workersSelectById(client.getServerState(), registered_response.instanceIds[0]);

      expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Running);
   });
});

describe("Pipeline", () => {
   const client: MrcTestClient = new MrcTestClient();
   let workerIds: string[] = [];

   beforeEach(async () => {
      await client.initializeClient();
      await client.initializeEventStream();

      // Also register 2 workers before starting any pipeline activities
      workerIds = (await client.register_and_activate_workers(["test data", "test data 2"])).instanceIds;
   });

   afterEach(async () => {
      await client.finalizeEventStream();
      await client.finalizeClient();
   });

   test("Request Assignment", async () => {
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

      // Now request to run a pipeline
      const request_pipeline_response = await client.register_pipeline_config(pipeline_config);

      // Check the pipeline definition
      const foundPipelineDefinition = pipelineDefinitionsSelectById(
         client.getServerState(),
         request_pipeline_response.pipelineDefinitionId
      );

      expect(foundPipelineDefinition?.id).toBe(request_pipeline_response.pipelineDefinitionId);

      // Check pipeline instances
      const foundPipelineInstance = pipelineInstancesSelectById(
         client.getServerState(),
         request_pipeline_response.pipelineInstanceId
      );

      expect(foundPipelineInstance).toBeDefined();

      expect(foundPipelineInstance?.machineId).toEqual(client.machineId);
      expect(foundPipelineInstance?.segmentIds).toEqual(request_pipeline_response.segmentInstanceIds);

      // Check segments exist in state
      const foundSegmentInstances = segmentInstancesSelectByIds(
         client.getServerState(),
         request_pipeline_response.segmentInstanceIds
      );

      expect(foundSegmentInstances).toHaveLength(0);
   });

   describe("Config", () => {
      let pipelineDefinitionId = "";
      let pipelineInstanceId = "";

      beforeEach(async () => {
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

         // Now request to run a pipeline
         const response = await client.register_pipeline_config(pipeline_config);

         pipelineDefinitionId = response.pipelineDefinitionId;
         pipelineInstanceId = response.pipelineInstanceId;
      });

      test("Resource States", async () => {
         let pipeline_instance_state: PipelineInstance | null =
            client.getClientState().pipelineInstances!.entities[pipelineInstanceId];

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Created);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await client.update_resource_status(
            pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Created
         );

         // For each segment, set it to created
         const segments = await Promise.all(
            pipeline_instance_state!.segmentIds.map(async (s) => {
               return await client.update_resource_status(s, "SegmentInstances", ResourceActualStatus.Actual_Created)!;
            })
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await client.update_resource_status(
            pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Completed
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Stopped);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await client.update_resource_status(
            pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Stopped
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Destroyed);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await client.update_resource_status(
            pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Destroyed
         );

         // The pipeline instance should be gone
         expect(pipeline_instance_state).toBeNull();
      });

      test("Resource State Handle Errors", async () => {
         const pipeline_instance_state: PipelineInstance | null =
            client.getClientState().pipelineInstances!.entities[pipelineInstanceId];

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Created);

         // Move to completed before marking as created
         expect(
            client.update_resource_status(
               pipelineInstanceId,
               "PipelineInstances",
               ResourceActualStatus.Actual_Completed
            )
         ).rejects.toThrow();
      });
   });
});

describe("Manifold", () => {
   const client: MrcTestClient = new MrcTestClient();
   let workerIds: string[] = [];
   let pipelineDefinitionId = "";
   let pipelineInstanceId = "";

   beforeEach(async () => {
      await client.initializeClient();
      await client.initializeEventStream();

      // Also register 2 workers before starting any pipeline activities
      workerIds = (await client.register_and_activate_workers(["test data", "test data 2"])).instanceIds;

      const pipeline_config: IPipelineConfiguration = {
         segments: {
            my_seg1: {
               ingressPorts: {},
               egressPorts: {
                  port1: {
                     portName: "port1",
                     typeId: 1234,
                     typeString: "int",
                  },
               },
               name: "my_seg1",
            },
            my_seg2: {
               ingressPorts: {
                  port1: {
                     portName: "port1",
                     typeId: 1234,
                     typeString: "int",
                  },
               },
               egressPorts: {},
               name: "my_seg2",
            },
         },
         manifolds: {
            port1: {
               name: "port1",
               options: {
                  policy: ManifoldOptions_Policy.LoadBalance,
               },
            },
         },
      };

      // Now request to run a pipeline
      const response = await client.register_pipeline_config(pipeline_config);

      pipelineDefinitionId = response.pipelineDefinitionId;
      pipelineInstanceId = response.pipelineInstanceId;

      // Finally, indicate the pipeline has been created
      await client.update_resource_status(pipelineInstanceId, "PipelineInstances", ResourceActualStatus.Actual_Created);
   });

   afterEach(async () => {
      await client.finalizeEventStream();
      await client.finalizeClient();
   });

   test("Resource States", async () => {
      //  Update the PipelineInstance state to assign segment instances
      const pipeline_instance_state = await client.update_resource_status(
         pipelineInstanceId,
         "PipelineInstances",
         ResourceActualStatus.Actual_Created
      );

      // For each manifold, set it to created
      const manifolds = await Promise.all(
         pipeline_instance_state!.manifoldIds.map(async (s) => {
            return await client.update_resource_status(s, "ManifoldInstances", ResourceActualStatus.Actual_Created)!;
         })
      );

      // For each segment, set it to created
      const segments = await Promise.all(
         client.getClientState().pipelineInstances!.entities[pipelineInstanceId].segmentIds.map(async (s) => {
            return await client.update_resource_status(s, "SegmentInstances", ResourceActualStatus.Actual_Created)!;
         })
      );

      // Now update the attached manifolds for each segment
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
