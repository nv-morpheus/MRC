/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/require-await */
/* eslint-disable @typescript-eslint/no-non-null-asserted-optional-chain */
/* eslint-disable @typescript-eslint/no-non-null-assertion */

import { MrcTestClient } from "@mrc/client/client";
import { ConnectionManager } from "@mrc/client/connection_manager";
import { PipelineManager } from "@mrc/client/pipeline_manager";
import { WorkersManager } from "@mrc/client/workers_manager";
import { IPipelineConfiguration } from "@mrc/common/entities";
import { PingRequest } from "@mrc/proto/mrc/protos/architect";
import {
   ManifoldOptions_Policy,
   PipelineInstance,
   ResourceActualStatus,
   ResourceRequestedStatus,
} from "@mrc/proto/mrc/protos/architect_state";
import { connectionsSelectAll, connectionsSelectById } from "@mrc/server/store/slices/connectionsSlice";
import { pipelineDefinitionsSelectById } from "@mrc/server/store/slices/pipelineDefinitionsSlice";
import { pipelineInstancesSelectById } from "@mrc/server/store/slices/pipelineInstancesSlice";
import { workersSelectById } from "@mrc/server/store/slices/workersSlice";
import exp from "constants";

/*
// Uncomment this block to disable Jest's rather verbose logging
const jestConsole = console;

beforeEach(() => {
  global.console = require('console');
});

afterEach(() => {
  global.console = jestConsole;
});
*/

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
      const manager = ConnectionManager.create(client);
      await manager.register();
      await manager.unregister();

      // Should have 0 connections in the state
      expect(connectionsSelectAll(client.getServerState())).toHaveLength(0);
   });

   describe("With EventStream", () => {
      const connectionManager = ConnectionManager.create(client);

      beforeEach(async () => {
         await connectionManager.ensureResourcesCreated();
      });

      afterEach(async () => {
         await connectionManager.unregister();
      });

      test("Found Connection", async () => {
         // Verify the number of connections is 1
         const connection = connectionsSelectById(client.getServerState(), connectionManager.machineId!);

         expect(connection).toBeDefined();
         expect(connection?.id).toEqual(connectionManager.machineId!);
      });

      // test("Abort", async () => {
      //    expect(client.abortConnection()).rejects.toThrow("The operation has been aborted");
      // });
   });
});

describe("Worker", () => {
   const client: MrcTestClient = new MrcTestClient();
   const connectionManager = ConnectionManager.create(client);

   beforeEach(async () => {
      await connectionManager.ensureResourcesCreated();
   });

   afterEach(async () => {
      await connectionManager.unregister();
      await client.finalizeClient();
   });

   test("Add One", async () => {
      const manager = new WorkersManager(connectionManager, ["test data"]);

      await manager.register();

      expect(manager.machineId).toBe(connectionManager.machineId);

      // Need to do deeper checking here
   });

   test("Activate", async () => {
      const manager = new WorkersManager(connectionManager, ["test data"]);

      await manager.createResources();

      // Check to make sure its activated
      let found_worker = workersSelectById(client.getServerState(), manager.workerIds[0]);

      expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Created);

      // Now set it as running
      await manager.runResources();

      // Ensure its running
      found_worker = workersSelectById(client.getServerState(), manager.workerIds[0]);

      expect(found_worker?.state.actualStatus).toBe(ResourceActualStatus.Actual_Running);
   });
});

describe("Pipeline", () => {
   const client: MrcTestClient = new MrcTestClient();
   const workersManager: WorkersManager = WorkersManager.create(["test data", "test data 2"], client);

   beforeEach(async () => {
      // Ensure everything up to the workers is ready to go
      await workersManager.ensureResourcesCreated();
   });

   afterEach(async () => {
      await workersManager.unregister();
      await client.finalizeClient();
   });

   test("Request Assignment", async () => {
      const pipeline_config: IPipelineConfiguration = {
         segments: {
            my_seg1: {
               name: "my_seg",
               egressPorts: [],
               ingressPorts: [],
            },
            my_seg2: {
               name: "my_seg2",
               egressPorts: [],
               ingressPorts: [],
            },
         },
         manifolds: {},
      };

      const manager = new PipelineManager(workersManager, pipeline_config);

      // Now request to run a pipeline
      await manager.register();

      // Check the pipeline definition
      const foundPipelineDefinition = pipelineDefinitionsSelectById(
         client.getServerState(),
         manager.pipelineDefinitionId
      );

      expect(foundPipelineDefinition?.id).toBe(manager.pipelineDefinitionId);

      // Check pipeline instances
      const foundPipelineInstance = pipelineInstancesSelectById(client.getServerState(), manager.pipelineInstanceId);

      expect(foundPipelineInstance).toBeDefined();

      expect(foundPipelineInstance?.machineId).toEqual(workersManager.connectionManager.machineId);

      // Should be no segments to start
      expect(foundPipelineInstance?.segmentIds).toHaveLength(0);
   });

   describe("Config", () => {
      const pipelineManager = new PipelineManager(workersManager, {
         segments: {
            my_seg1: {
               egressPorts: ["port1"],
               ingressPorts: [],
               name: "my_seg",
            },
            my_seg2: {
               egressPorts: [],
               ingressPorts: ["port1"],
               name: "my_seg2",
            },
         },
         manifolds: {
            port1: {
               name: "port1",
               typeId: 0,
               typeString: "int",
            },
         },
      });

      beforeEach(async () => {
         await pipelineManager.ensureRegistered();
      });

      afterEach(async () => {
         await pipelineManager.unregister();
      });

      test("Resource States", async () => {
         let pipeline_instance_state: PipelineInstance | null =
            workersManager.connectionManager.getClientState().pipelineInstances!.entities[
            pipelineManager.pipelineInstanceId
            ];

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Created);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await workersManager.connectionManager.update_resource_status(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Created
         );

         // For each manifold, set it to created
         const manifolds = await Promise.all(
            pipeline_instance_state!.manifoldIds.map(async (m) => {
               return await workersManager.connectionManager.update_resource_status(
                  m,
                  "ManifoldInstances",
                  ResourceActualStatus.Actual_Created
               )!;
            })
         );

         // Update the resource to get the assigned segments
         pipeline_instance_state = workersManager.connectionManager.getResource(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances"
         );

         // For each segment, set it to created
         const segments = await Promise.all(
            pipeline_instance_state!.segmentIds.map(async (s) => {
               return await workersManager.connectionManager.update_resource_status(
                  s,
                  "SegmentInstances",
                  ResourceActualStatus.Actual_Created
               )!;
            })
         );

         // Update the resource to get the assigned segments
         pipeline_instance_state = workersManager.connectionManager.getResource(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances"
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);

         // Set both the segments and the manifolds to stopped to allow the pipeline to shutdown
         await Promise.all(
            segments.map(async (s) => {
               if (!s) {
                  throw new Error("Segment should not be undefined");
               }

               return await workersManager.connectionManager.update_resource_status(
                  s.id,
                  "SegmentInstances",
                  ResourceActualStatus.Actual_Destroyed
               )!;
            })
         );

         await Promise.all(
            manifolds.map(async (m) => {
               if (!m) {
                  throw new Error("Manifold should not be undefined");
               }

               return await workersManager.connectionManager.update_resource_status(
                  m.id,
                  "ManifoldInstances",
                  ResourceActualStatus.Actual_Destroyed
               )!;
            })
         );

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await workersManager.connectionManager.update_resource_status(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Completed
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Stopped);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await workersManager.connectionManager.update_resource_status(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Stopped
         );

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Destroyed);

         //  Update the PipelineInstance state to assign segment instances
         pipeline_instance_state = await workersManager.connectionManager.update_resource_status(
            pipelineManager.pipelineInstanceId,
            "PipelineInstances",
            ResourceActualStatus.Actual_Destroyed
         );

         // The pipeline instance should be gone
         expect(pipeline_instance_state).toBeNull();
      });

      test("Resource State Handle Errors", async () => {
         const pipeline_instance_state: PipelineInstance | null =
            workersManager.connectionManager.getClientState().pipelineInstances!.entities[
            pipelineManager.pipelineInstanceId
            ];

         expect(pipeline_instance_state?.state?.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Created);

         // Move to completed before marking as created
         void expect(
            workersManager.connectionManager.update_resource_status(
               pipelineManager.pipelineInstanceId,
               "PipelineInstances",
               ResourceActualStatus.Actual_Completed
            )
         ).rejects.toThrow();
      });
   });
});

describe("Manifold", () => {
   const pipeline_config: IPipelineConfiguration = {
      segments: {
         my_seg1: {
            ingressPorts: [],
            egressPorts: ["port1"],
            name: "my_seg1",
         },
         my_seg2: {
            ingressPorts: ["port1"],
            egressPorts: [],
            name: "my_seg2",
         },
      },
      manifolds: {
         port1: {
            name: "port1",
            typeId: 1234,
            typeString: "int",
            options: {
               policy: ManifoldOptions_Policy.LoadBalance,
            },
         },
      },
   };

   const client: MrcTestClient = new MrcTestClient();
   const pipelineManager = PipelineManager.create(pipeline_config, ["test data"], client);

   beforeEach(async () => {
      await pipelineManager.ensureResourcesCreated();
   });

   afterEach(async () => {
      await pipelineManager.unregister();
      await client.finalizeClient();
   });

   test("Resource States", async () => {
      //  Update the PipelineInstance state to assign segment instances
      const pipeline_instance_state = await pipelineManager.connectionManager.update_resource_status(
         pipelineManager.pipelineInstanceId,
         "PipelineInstances",
         ResourceActualStatus.Actual_Created
      );

      // For each manifold, set it to created
      const manifolds = await Promise.all(
         pipeline_instance_state!.manifoldIds.map(async (s) => {
            return await pipelineManager.connectionManager.update_resource_status(
               s,
               "ManifoldInstances",
               ResourceActualStatus.Actual_Created
            )!;
         })
      );

      // For each segment, set it to created
      const segments = await Promise.all(
         pipelineManager.connectionManager
            .getClientState()
            .pipelineInstances!.entities[pipelineManager.pipelineInstanceId].segmentIds.map(async (s) => {
               return await pipelineManager.connectionManager.update_resource_status(
                  s,
                  "SegmentInstances",
                  ResourceActualStatus.Actual_Created
               )!;
            })
      );

      // Now update the attached manifolds for each segment
   });

   describe("Second Connection", () => {
      let pipelineManager2: PipelineManager;

      afterEach(async () => {
         if (pipelineManager2 !== undefined && pipelineManager2.isRegistered) {
            await pipelineManager2.unregister();
         }
      });

      test("Ref Counting", async () => {
         let state = pipelineManager.connectionManager.getClientState();
         expect(state.manifoldInstances!.ids).toHaveLength(1);
         expect(pipelineManager.manifoldsManager.manifoldIds).toEqual(state.manifoldInstances!.ids);

         const manifold1 = pipelineManager.manifoldsManager.manifolds[0];
         let manifold1State = manifold1.getState();

         // Verify that with a single connection, the `my_seg1` segment has a single local connection to `my_seg2`
         expect(state.segmentInstances!.ids).toHaveLength(2);
         const pipe1seg1Id: number = parseInt(state.segmentInstances!.ids[0]);
         const pipe1seg2Id: number = parseInt(state.segmentInstances!.ids[1]);

         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(manifold1State.requestedInputSegments[pipe1seg1Id]).toBe(true);

         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(manifold1State.requestedOutputSegments[pipe1seg2Id]).toBe(true);

         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(0);

         // update requested to actual
         await manifold1.syncActualSegments();

         // requested segments should be actual segments now
         state = pipelineManager.connectionManager.getClientState();
         manifold1State = manifold1.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(manifold1State.requestedInputSegments[pipe1seg1Id]).toBe(true);
         expect(manifold1State.actualInputSegments).toEqual(manifold1State.requestedInputSegments);

         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(manifold1State.requestedOutputSegments[pipe1seg2Id]).toBe(true);
         expect(manifold1State.actualOutputSegments).toEqual(manifold1State.requestedOutputSegments);

         // Might need to change this to a search if the order stops being deterministic
         let pipe1seg1 = state.segmentInstances!.entities[pipe1seg1Id!];
         expect(pipe1seg1.name).toEqual("my_seg1");
         expect(pipe1seg1.state!.refCount).toEqual(1);

         let pipe1seg2 = state.segmentInstances!.entities[pipe1seg2Id!];
         expect(pipe1seg2.name).toEqual("my_seg2");
         expect(pipe1seg2.state!.refCount).toEqual(1);

         // Now create a second connection
         pipelineManager2 = PipelineManager.create(pipeline_config, ["test data2"], client);
         await pipelineManager2.ensureResourcesCreated();

         // Now see what the state is, we should have 2 manifolds, 2 actual segments and 2 requested segments
         state = pipelineManager2.connectionManager.getClientState();
         expect(state.manifoldInstances!.ids).toHaveLength(2);
         expect(state.segmentInstances!.ids).toHaveLength(4);

         manifold1State = manifold1.getState();
         const manifold2 = pipelineManager2.manifoldsManager.manifolds[0];
         let manifold2State = manifold2.getState();

         const pipe2seg1Id: number = parseInt(state.segmentInstances!.ids[2]);
         const pipe2seg2Id: number = parseInt(state.segmentInstances!.ids[3]);
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(1);
         expect(manifold1State.requestedInputSegments[pipe1seg1Id]).toBe(true);
         expect(manifold1State.requestedInputSegments[pipe2seg1Id]).toBe(false);
         expect(manifold1State.requestedOutputSegments[pipe1seg2Id]).toBe(true);
         expect(manifold1State.requestedOutputSegments[pipe2seg2Id]).toBe(false);


         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(0);
         expect(manifold2State.requestedInputSegments[pipe1seg1Id]).toBe(false);
         expect(manifold2State.requestedInputSegments[pipe2seg1Id]).toBe(true);
         expect(manifold2State.requestedOutputSegments[pipe1seg2Id]).toBe(false);
         expect(manifold2State.requestedOutputSegments[pipe2seg2Id]).toBe(true);

         // Both manifolds need to update their requested/actual segments
         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();

         state = pipelineManager2.connectionManager.getClientState();

         // fetch an updated version of the first manifold
         manifold1State = manifold1.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(2);
         expect(manifold1State.actualInputSegments).toEqual(manifold1State.requestedInputSegments);

         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(2);
         expect(manifold1State.actualOutputSegments).toEqual(manifold1State.requestedOutputSegments);

         manifold2State = manifold2.getState();
         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(2);
         expect(manifold2State.actualInputSegments).toEqual(manifold2State.requestedInputSegments);

         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(2);
         expect(manifold2State.actualOutputSegments).toEqual(manifold2State.requestedOutputSegments);

         pipe1seg1 = state.segmentInstances!.entities[pipe1seg1Id!];
         expect(pipe1seg1.state!.refCount).toEqual(2);

         pipe1seg2 = state.segmentInstances!.entities[pipe1seg2Id!];
         expect(pipe1seg2.state!.refCount).toEqual(2);

         const pipe2seg1 = state.segmentInstances!.entities[pipe2seg1Id!];
         expect(pipe2seg1.name).toEqual("my_seg1");
         expect(pipe2seg1.state!.refCount).toEqual(2);

         const pipe2seg2 = state.segmentInstances!.entities[pipe2seg2Id!];
         expect(pipe2seg2.name).toEqual("my_seg2");
         expect(pipe2seg2.state!.refCount).toEqual(2);

         // Simulate a graceful shutdown of the second pipeline.
         for (const worker of pipelineManager2.workersManager.workers) {
            for (const seg of worker.segments) {
               await seg.requestSegmentStop();
               let segmentState = seg.getState();
               expect(segmentState.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopping);
            }
         }

         // Both manifolds should have some of their reqiested segments removed
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(2);

         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(2);

         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();

         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(1);

         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(0);

         // veirfy the refcount went down
         state = pipelineManager2.connectionManager.getClientState();
         pipe1seg1 = state.segmentInstances!.entities[pipe1seg1Id!];
         expect(pipe1seg1.name).toEqual("my_seg1");
         expect(pipe1seg1.state!.refCount).toEqual(1);

         pipe1seg2 = state.segmentInstances!.entities[pipe1seg2Id!];
         expect(pipe1seg2.name).toEqual("my_seg2");
         expect(pipe1seg2.state!.refCount).toEqual(1);

         for (const worker of pipelineManager2.workersManager.workers) {
            for (const seg of worker.segments) {
               await seg.sendSegmenStopped();
               let segmentState = seg.getState();
               expect(segmentState.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Destroyed);
               expect(segmentState.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopped);
            }
         }

         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();

         // Manifold2 should have been asked to shut down, manifold1 should still be running
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(1);

         // Still shouldn't be connected to anything
         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(0);

         expect(manifold1State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);
         expect(manifold2State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Stopped);

         await manifold2.syncActualStatus();
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(manifold2State.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopping);

         await manifold2.updateActualStatus(ResourceActualStatus.Actual_Stopped);
         manifold2State = manifold2.getState();
         expect(manifold2State.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopped);

         await pipelineManager2.unregister();

         state = pipelineManager.connectionManager.getClientState();
         expect(state.manifoldInstances!.ids).toHaveLength(1);

         // make sure we didn't mess up the first manifold somehow
         manifold1State = manifold1.getState();
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(manifold1State.actualInputSegments[pipe1seg1Id]).toBe(true);

         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(1);
         expect(manifold1State.actualOutputSegments[pipe1seg2Id]).toBe(true);

      });

      test("Shutdown second input", async () => {
         let state = pipelineManager.connectionManager.getClientState();
         const manifold1 = pipelineManager.manifoldsManager.manifolds[0];

         const pipe1seg1Id: number = parseInt(state.segmentInstances!.ids[0]);
         const pipe1seg2Id: number = parseInt(state.segmentInstances!.ids[1]);

         await manifold1.syncActualSegments();
         let manifold1State = manifold1.getState();

         // Now create a second connection
         pipelineManager2 = PipelineManager.create(pipeline_config, ["test data2"], client);
         await pipelineManager2.ensureResourcesCreated();

         // Now see what the state is, we should have 2 manifolds, 2 actual segments and 2 requested segments
         state = pipelineManager2.connectionManager.getClientState();
         expect(state.manifoldInstances!.ids).toHaveLength(2);
         expect(state.segmentInstances!.ids).toHaveLength(4);

         manifold1State = manifold1.getState();

         const manifold2 = pipelineManager2.manifoldsManager.manifolds[0];
         let manifold2State = manifold2.getState();

         const pipe2seg1Id: number = parseInt(state.segmentInstances!.ids[2]);
         const pipe2seg2Id: number = parseInt(state.segmentInstances!.ids[3]);

         // Both manifolds need to update their requested/actual segments
         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();

         state = pipelineManager2.connectionManager.getClientState();

         // fetch an updated version of the first manifold
         manifold1State = manifold1.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(2);
         expect(manifold1State.actualInputSegments).toEqual(manifold1State.requestedInputSegments);

         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(2);
         expect(manifold1State.actualOutputSegments).toEqual(manifold1State.requestedOutputSegments);

         manifold2State = manifold2.getState();
         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(2);
         expect(manifold2State.actualInputSegments).toEqual(manifold2State.requestedInputSegments);

         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(2);
         expect(manifold2State.actualOutputSegments).toEqual(manifold2State.requestedOutputSegments);

         // Now we need to stop the first segment in pipe2
         let foundPipe2Seg1: boolean = false;
         for (const worker of pipelineManager2.workersManager.workers) {
            for (const seg of worker.segments) {
               if (parseInt(seg.segmentId) === pipe2seg1Id) {
                  foundPipe2Seg1 = true;
                  await seg.requestSegmentStop();
                  let segmentState = seg.getState();
                  expect(segmentState.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopping);
               }
            }
         }

         expect(foundPipe2Seg1).toBe(true);

         // Both manifolds should have some of their reqiested segments removed
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(2);

         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(2);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(2);

         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();

         expect(manifold1State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);
         expect(manifold2State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(2);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(2);

         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(1);

         // pipe2seg2 should still be running just fine consuming input from pipe1seg1
         let foundPipe2Seg2: boolean = false;
         for (const worker of pipelineManager2.workersManager.workers) {
            for (const seg of worker.segments) {
               if (parseInt(seg.segmentId) === pipe2seg2Id) {
                  foundPipe2Seg2 = true;
                  let segmentState = seg.getState();
                  expect(segmentState.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);
               }
            }
         }

         expect(foundPipe2Seg2).toBe(true);


         // veirfy the refcount went down
         state = pipelineManager2.connectionManager.getClientState();
         let pipe1seg1 = state.segmentInstances!.entities[pipe1seg1Id!];
         expect(pipe1seg1.name).toEqual("my_seg1");
         expect(pipe1seg1.state!.refCount).toEqual(2);

         let pipe1seg2 = state.segmentInstances!.entities[pipe1seg2Id!];
         expect(pipe1seg2.name).toEqual("my_seg2");
         expect(pipe1seg2.state!.refCount).toEqual(1);


         // now stop pipe2seg2
         foundPipe2Seg2 = false;
         for (const worker of pipelineManager2.workersManager.workers) {
            for (const seg of worker.segments) {
               if (parseInt(seg.segmentId) === pipe2seg2Id) {
                  foundPipe2Seg2 = true;
                  await seg.requestSegmentStop();
               }
            }
         }

         await manifold1.syncActualSegments();
         await manifold2.syncActualSegments();

         // veirfy the refcount went down
         state = pipelineManager2.connectionManager.getClientState();
         pipe1seg1 = state.segmentInstances!.entities[pipe1seg1Id!];
         expect(pipe1seg1.name).toEqual("my_seg1");
         expect(pipe1seg1.state!.refCount).toEqual(1);

         pipe1seg2 = state.segmentInstances!.entities[pipe1seg2Id!];
         expect(pipe1seg2.name).toEqual("my_seg2");
         expect(pipe1seg2.state!.refCount).toEqual(1);


         // Manifold2 should have been asked to shut down, manifold1 should still be running
         manifold1State = manifold1.getState();
         manifold2State = manifold2.getState();
         expect(Object.keys(manifold1State.requestedInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualInputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.requestedOutputSegments)).toHaveLength(1);
         expect(Object.keys(manifold1State.actualOutputSegments)).toHaveLength(1);

         // Still shouldn't be connected to anything
         expect(Object.keys(manifold2State.requestedInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualInputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.requestedOutputSegments)).toHaveLength(0);
         expect(Object.keys(manifold2State.actualOutputSegments)).toHaveLength(0);

         expect(manifold1State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Completed);
         expect(manifold2State.state!.requestedStatus).toEqual(ResourceRequestedStatus.Requested_Stopped);

         await manifold2.updateActualStatus(ResourceActualStatus.Actual_Stopped);
         manifold2State = manifold2.getState();
         expect(manifold2State.state!.actualStatus).toEqual(ResourceActualStatus.Actual_Stopped);
      });
   });

});
