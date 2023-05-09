

import {ServerDuplexStream} from "@grpc/grpc-js";
import {IConnection, IWorker} from "@mrc/common/entities";
import {
   segmentInstancesSelectById,
   segmentInstancesUpdateResourceState,
} from "@mrc/server/store/slices/segmentInstancesSlice";
import {systemStartRequest, systemStopRequest} from "@mrc/server/store/slices/systemSlice";
import {as, AsyncSink, merge} from "ix/asynciterable";
import {withAbort} from "ix/asynciterable/operators";
import {CallContext} from "nice-grpc";
import {firstValueFrom, Subject} from "rxjs";

import {pack, packEvent, unpackEvent} from "../common/utils";
import {Any} from "../proto/google/protobuf/any";
import {
   Ack,
   ArchitectServiceImplementation,
   ClientConnectedResponse,
   ErrorCode,
   Event,
   EventType,
   eventTypeToJSON,
   PingRequest,
   PingResponse,
   PipelineRequestAssignmentRequest,
   PipelineRequestAssignmentResponse,
   RegisterWorkersRequest,
   RegisterWorkersResponse,
   ResourceUpdateStatusRequest,
   ResourceUpdateStatusResponse,
   ServerStreamingMethodResult,
   ShutdownRequest,
   ShutdownResponse,
   StateUpdate,
   TaggedInstance,
} from "../proto/mrc/protos/architect";
import {ControlPlaneState, ResourceStatus} from "../proto/mrc/protos/architect_state";
import {DeepPartial, messageTypeRegistry} from "../proto/typeRegistry";

import {connectionsAdd, connectionsDropOne} from "./store/slices/connectionsSlice";
import {
   pipelineInstancesAssign,
   pipelineInstancesSelectById,
   pipelineInstancesUpdateResourceState,
} from "./store/slices/pipelineInstancesSlice";
import {
   workersAddMany,
   workersRemove,
   workersSelectById,
   workersSelectByMachineId,
   workersUpdateResourceState,
} from "./store/slices/workersSlice";
import {getRootStore, RootStore, stopAction} from "./store/store";
import {generateId} from "./utils";

interface IncomingData
{
   msg: Event, stream?: ServerDuplexStream<Event, Event>, machineId: string,
}

function unaryResponse<MessageDataT>(event: IncomingData|undefined, message_class: any, data: MessageDataT): Event
{
   // Lookup message type
   const type_registry = messageTypeRegistry.get(message_class.$type);

   const response = message_class.create(data);

   // const writer = new BufferWriter();

   // RegisterWorkersResponse.encode(response, writer);

   const any_msg = Any.create({
      typeUrl: `type.googleapis.com/${message_class.$type}`,
      value: message_class.encode(response).finish(),
   });

   return Event.create({
      event: EventType.Response,
      tag: event?.msg.tag ?? "0",
      message: any_msg,
   });
}

// function pack<MessageDataT extends UnknownMessage>(data: MessageDataT): Any {

//    // Load the type from the registry
//    const message_type = messageTypeRegistry.get(data.$type);

//    if (!message_type) {
//       throw new Error("Unknown type in type registry");
//    }

//    const any_msg = Any.create({
//       typeUrl: `type.googleapis.com/${message_type.$type}`,
//       value: message_type.encode(data).finish(),
//    });

//    return any_msg;
// }

// function unpack<MessageT extends UnknownMessage>(event: IncomingData) {
//    const message_type_str = event.msg.message?.typeUrl.split('/').pop();

//    // Load the type from the registry
//    const message_type = messageTypeRegistry.get(message_type_str ?? "");

//    if (!message_type) {
//       throw new Error(`Could not unpack message with type: ${event.msg.message?.typeUrl}`);
//    }

//    const message = message_type.decode(event.msg.message?.value as Uint8Array) as MessageT;

//    return message;
// }

// function unaryResponse<MessageDataT extends Message>(event: IncomingData, data: MessageDataT): void {

//    const any_msg = new Any();
//    any_msg.pack(data.serializeBinary(), typeUrlFromMessageClass(data) as string);

//    const message = new Event();
//    message.setEvent(EventType.RESPONSE);
//    message.setTag(event.msg.getTag());
//    message.setMessage(any_msg);

//    event.stream.write(message);
// }

class Architect implements ArchitectServiceImplementation
{
   public service: ArchitectServiceImplementation;

   private _store: RootStore;

   private shutdown_subject: Subject<void> = new Subject<void>();
   private _stop_controller: AbortController;

   constructor(store?: RootStore)
   {
      // Use the default store if not supplied
      if (!store)
      {
         store = getRootStore();
      }

      this._store = store;

      this._stop_controller = new AbortController();

      // Have to do this. Look at
      // https://github.com/paymog/grpc_tools_node_protoc_ts/blob/master/doc/server_impl_signature.md to see about
      // getting around this restriction
      this.service = {
         eventStream: (request, context) => {
            return this.do_eventStream(request, context);
         },
         ping: async(request, context): Promise<DeepPartial<PingResponse>> => {
            return await this.do_ping(request, context);
         },
         shutdown: async(request, context): Promise<DeepPartial<ShutdownResponse>> => {
            return await this.do_shutdown(request, context);
         },
      };
   }

   public stop()
   {
      // Trigger a stop cancellation for any connected streams
      this._stop_controller.abort("Stop signaled");

      this._store.dispatch(stopAction());
   }

   public eventStream(request: AsyncIterable<Event>, context: CallContext): ServerStreamingMethodResult<{
      event?: EventType | undefined;
      tag?: string | undefined;
      message?: {value?: Uint8Array | undefined; typeUrl?: string | undefined;} | undefined;
      error?: {message?: string | undefined; code?: ErrorCode | undefined;} | undefined;
   }>
   {
      return this.do_eventStream(request, context);
   }
   public ping(request: PingRequest, context: CallContext): Promise<{tag?: string | undefined;}>
   {
      return this.do_ping(request, context);
   }
   public shutdown(request: ShutdownRequest, context: CallContext): Promise<{tag?: string | undefined;}>
   {
      return this.do_shutdown(request, context);
   }

   public onShutdownSignaled()
   {
      return firstValueFrom(this.shutdown_subject);
   }

   private async * do_eventStream(stream: AsyncIterable<Event>, context: CallContext): AsyncIterable<DeepPartial<Event>>
   {
      console.log(`Event stream created for ${context.peer}`);

      const connection: IConnection = {
         id: generateId(),
         peerInfo: context.peer,
         workerIds: [],
         assignedPipelineIds: [],
      };

      context.metadata.set("mrc-machine-id", connection.id.toString());

      const store_update_sink = new AsyncSink<Event>();

      // Subscribe to the stores next update
      const store_unsub = this._store.subscribe(() => {
         const state = this._store.getState();

         // Remove the system object from the state
         const {system: _, ...out_state} = {...state, system: {extra: true}};

         console.log("Pushing state update");

         // Push out the state update
         store_update_sink.write(
             packEvent<ControlPlaneState>(EventType.ServerStateUpdate,
                                          "0",
                                          ControlPlaneState.create(out_state as ControlPlaneState)));
      });

      // Create a new connection
      this._store.dispatch(connectionsAdd(connection));

      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const self = this;

      // Yield a connected even
      yield Event.create({
         event: EventType.ClientEventStreamConnected,
         message: pack(ClientConnectedResponse.create({
            machineId: connection.id,
         })),
      });

      const event_stream = async function*() {
         try
         {
            for await (const req of stream)
            {
               try
               {
                  console.log(
                      `Event stream start for ${connection.peerInfo} with message: ${eventTypeToJSON(req.event)}`);

                  self._store.dispatch(systemStartRequest());

                  yield* self.do_handle_event({
                     msg: req,
                     machineId: connection.id,
                  },
                                              context);
               } finally
               {
                  self._store.dispatch(systemStopRequest());

                  console.log(
                      `Event stream end for ${connection.peerInfo} with message: ${eventTypeToJSON(req.event)}`);
               }
            }
         } catch (error)
         {
            console.log(`Error occurred in stream. Error: ${error}`);
         } finally
         {
            console.log(`Event stream closed for ${connection.peerInfo}.`);

            // Input stream has completed so stop pushing events
            store_unsub();

            store_update_sink.end();
         }
      };

      try
      {
         // Make the combined async iterable
         const combined_iterable =
             merge(as<Event>(event_stream()).pipe(withAbort(this._stop_controller.signal)), as<Event>(store_update_sink));

         for await (const out_event of combined_iterable)
         {
            yield out_event;
         }

      } catch (error)
      {
         console.log(`Error occurred in stream. Error: ${error}`);
      } finally
      {
         console.log(`All streams closed for ${connection.peerInfo}. Deleting connection.`);

         // Ensure the other streams are cleaned up
         store_unsub();

         store_update_sink.end();

         // Use the Lost Connection action to force all child objects to be removed too
         this._store.dispatch(connectionsDropOne(connection));
      }
   }

   private async * do_handle_event(event: IncomingData, context: CallContext)
   {
      try
      {
         switch (event.msg.event)
         {
         case EventType.ClientEventPing:
            const payload = unpackEvent<PingRequest>(event.msg);

            console.log(`Ping from ${context.peer}. Tag: ${payload.tag}.`);

            yield unaryResponse(event, PingResponse, PingResponse.create({
               tag: payload.tag,
            }));

            break;

         case EventType.ClientEventRequestStateUpdate:

            yield unaryResponse(event,
                                StateUpdate,
                                StateUpdate.create({

                                }));

            break;
         case EventType.ClientUnaryRegisterWorkers: {
            const payload = unpackEvent<RegisterWorkersRequest>(event.msg);

            const workers: IWorker[] = payload.ucxWorkerAddresses.map((value): IWorker => {
               return {
                  id: generateId(),
                  machineId: event.machineId,
                  workerAddress: value,
                  state: {
                     status: ResourceStatus.Registered,
                     refCount: 0,
                  },
                  assignedSegmentIds: [],
               };
            });

            // Add the workers
            this._store.dispatch(workersAddMany(workers));

            const resp = RegisterWorkersResponse.create({
               machineId: event.machineId,
               instanceIds: workersSelectByMachineId(this._store.getState(), event.machineId).map((worker) => worker.id)
            });

            yield unaryResponse(event, RegisterWorkersResponse, resp);

            break;
         }
         case EventType.ClientUnaryActivateStream: {
            const payload = unpackEvent<RegisterWorkersResponse>(event.msg);

            const workers = payload.instanceIds.map((id) => {
               const w = workersSelectById(this._store.getState(), id);

               if (!w)
               {
                  throw new Error(`Cannot activate Worker ${id}. ID does not exist`);
               }

               return w;
            });

            this._store.dispatch(workersUpdateResourceState({resources: workers, status: ResourceStatus.Activated}));

            yield unaryResponse(event, Ack, {});

            break;
         }
         case EventType.ClientUnaryDropWorker: {
            const payload = unpackEvent<TaggedInstance>(event.msg);

            const found_worker = workersSelectById(this._store.getState(), payload.instanceId);

            if (found_worker)
            {
               this._store.dispatch(workersRemove(found_worker));
            }

            yield unaryResponse(event, Ack, {});

            break;
         }
         case EventType.ClientUnaryRequestPipelineAssignment: {
            const payload = unpackEvent<PipelineRequestAssignmentRequest>(event.msg);

            // Check to make sure its not null
            if (!payload.pipeline)
            {
               throw new Error("`pipeline` cannot be undefined");
               // Use default values for now since the pipeline def is empty
               // payload.pipeline = PipelineDefinition.create({
               //    id: 0,
               //    instanceIds: [],
               //    segmentIds: [],
               // });
            }

            if (!payload.mapping)
            {
               throw new Error("`mapping` cannot be undefined");
            }

            if (payload.mapping.machineId == "0")
            {
               payload.mapping.machineId = event.machineId;
            }
            else if (payload.mapping.machineId != event.machineId)
            {
               throw new Error("Incorrect machineId");
            }

            // Add a pipeline assignment to the machine
            const addedInstances = this._store.dispatch(pipelineInstancesAssign({
               pipeline: payload.pipeline,
               mapping: payload.mapping,
            }));

            yield unaryResponse(event,
                                PipelineRequestAssignmentResponse,
                                PipelineRequestAssignmentResponse.create(addedInstances));

            break;
         }
         case EventType.ClientUnaryResourceUpdateStatus: {
            const payload = unpackEvent<ResourceUpdateStatusRequest>(event.msg);

            // Check to make sure its not null
            switch (payload.resourceType)
            {
            case "PipelineInstances": {
               const found = pipelineInstancesSelectById(this._store.getState(), payload.resourceId);

               if (!found)
               {
                  throw new Error(`Could not find PipelineInstance for ID: ${payload.resourceId}`);
               }

               this._store.dispatch(pipelineInstancesUpdateResourceState({
                  resource: found,
                  status: payload.status,
               }));

               break;
            }
            case "SegmentInstances": {
               const found = segmentInstancesSelectById(this._store.getState(), payload.resourceId);

               if (!found)
               {
                  throw new Error(`Could not find SegmentInstance for ID: ${payload.resourceId}`);
               }

               this._store.dispatch(segmentInstancesUpdateResourceState({
                  resource: found,
                  status: payload.status,
               }));

               break;
            }
            default:
               throw new Error("Unsupported resource type");
            }

            yield unaryResponse(event, ResourceUpdateStatusResponse, ResourceUpdateStatusResponse.create({ok: true}));

            break;
         }
         default:
            break;
         }
      } catch (error)
      {
         console.log(`Error occurred handing event. Error: ${error}`);
      }
   }

   private async do_shutdown(req: ShutdownRequest, context: CallContext): Promise<DeepPartial<ShutdownResponse>>
   {
      console.log(`Issuing shutdown promise from ${context.peer}`);

      // Signal that shutdown was requested
      this.shutdown_subject.next();

      return ShutdownResponse.create();
   }

   private async do_ping(req: PingRequest, context: CallContext): Promise<DeepPartial<PingResponse>>
   {
      console.log(`Ping from ${context.peer}`);

      return PingResponse.create({
         tag: req.tag,
      });
   }
}

// class Architect implements ArchitectServer{
//    [name: string]: UntypedHandleCall;
//    eventStream: handleBidiStreamingCall<Event, Event>;
//    shutdown: handleUnaryCall<ShutdownRequest, ShutdownResponse>;

// }

// class Architect implements ArchitectServer {
//    [name: string]: UntypedHandleCall;

//    constructor() {
//       console.log("Created");

//       this.a = 5;
//    }

//    public eventStream(call: ServerDuplexStream<Event, Event>): void {
//       console.log(`Event stream created for ${call.getPeer()}`);

//       call.on("data", (req: Event) => {
//          console.log(`Event stream data for ${call.getPeer()} with message: ${req.event.toString()}`);
//          // this.do_handle_event(req, call);
//       });

//       call.on("error", (err: Error) => {
//          console.log(`Event stream errored for ${call.getPeer()} with message: ${err.message}`);
//       });

//       call.on("end", () => {
//          console.log(`Event stream closed for ${call.getPeer()}`);
//       });
//    }

//    private do_handle_event(event: Event, call: ServerDuplexStream<Event, Event>): void{

//       try {
//          switch (event.event) {
//             case EventType.ClientEventRequestStateUpdate:

//                break;

//             default:
//                break;
//          }
//       } catch (error) {

//       }

//    }

// }

export {
   Architect,
};
