

import {
   ServerDuplexStream, ServerUnaryCall
} from '@grpc/grpc-js';
import { firstValueFrom, Observable, Subject } from "rxjs";

import { getRootStore, RootState, RootStore } from "./store/store";
import { addWorkers, IWorker, removeWorker, workersSelectById, workersSelectByMachineId } from "./store/slices/workersSlice";
import { addConnection, removeConnection } from "./store/slices/connectionsSlice";
import { RegisterWorkersRequest, RegisterWorkersResponse, Event, Ack, EventType, PingRequest, PingResponse, ShutdownRequest, ShutdownResponse, TaggedInstance, ArchitectServiceImplementation, ServerStreamingMethodResult, StateUpdate, ErrorCode, ClientConnectedResponse } from "../proto/mrc/protos/architect";
import { Any } from "../proto/google/protobuf/any";
import { DeepPartial, MessageType, messageTypeRegistry, UnknownMessage } from "../proto/typeRegistry";
import { CallContext } from "nice-grpc";
import { as, AsyncSink, concat, from, merge, zip } from 'ix/asynciterable';

interface IncomingData {
   msg: Event,
   stream?: ServerDuplexStream<Event, Event>,
}

// type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

// type DeepPartial<T> = T extends Builtin ? T
//    : T extends Array<infer U> ? Array<DeepPartial<U>> : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
//    : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
//    : Partial<T>;

// type KeysOfUnion<T> = T extends T ? keyof T : never;
// type Exact<P, I extends P> = P extends Builtin ? P
//    : P & { [K in keyof P]: Exact<P[K], I[K]> } & { [K in Exclude<keyof I, KeysOfUnion<P>>]: never };

// interface ProtoMessageBase<MessageDataT> {
//    create<I extends Exact<DeepPartial<MessageDataT>, I>>(base?: I): ProtoMessageBase<MessageDataT>;
//    // create(base?: MessageDataT): ProtoMessageBase<MessageDataT>;
//    // encode(message: ProtoMessageBase<MessageDataT>, writer?: proto_min.Writer): proto_min.Writer;
// }

// function testResponse<MessageClassT>(message_class: MessageClassT): void {

// }

// function typeUrlFromMessageClass(message_class: any) {

//    const prefix = "mrc.protos";

//    if (message_class instanceof RegisterWorkersRequest) {
//       return `${prefix}.RegisterWorkersRequest`;
//    } else if (message_class instanceof RegisterWorkersResponse) {
//       return `${prefix}.RegisterWorkersResponse`;
//    } else if (message_class instanceof Ack) {
//       return `${prefix}.Ack`;
//    } else {
//       throw new Error(`Unknown message type: ${typeof message_class}`);
//    }
// }

// function unaryResponse<MessageT extends ProtoMessageBase, I extends Exact<DeepPartial<MessageT>, I>>(event: IncomingData, message_class: MessageT, data: I): MessageT {
// function unaryResponse<MessageDataT extends Exact<DeepPartial<ProtoMessageBase<MessageDataT>>, MessageDataT>, MessageClassT extends ProtoMessageBase<MessageDataT>>(event: IncomingData, message_class: MessageClassT, data: MessageDataT): void {
function unaryResponse2<MessageDataT extends UnknownMessage, MessageClass extends MessageType<MessageDataT>>(event: IncomingData, message_class: MessageClass, data: MessageDataT): void {

   const registered_type = messageTypeRegistry.get(message_class.$type);

}
function unaryResponse<MessageDataT>(event: IncomingData | undefined, message_class: any, data: MessageDataT): Event {

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
      tag: event?.msg.tag ?? 0,
      message: any_msg,
   });
}

function pack<MessageDataT extends UnknownMessage>(data: MessageDataT): Any {

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(data.$type);

   if (!message_type) {
      throw new Error("Unknown type in type registry");
   }

   const any_msg = Any.create({
      typeUrl: `type.googleapis.com/${message_type.$type}`,
      value: message_type.encode(data).finish(),
   });

   return any_msg;
}

function unpack<MessageT extends UnknownMessage>(event: IncomingData) {
   const message_type_str = event.msg.message?.typeUrl.split('/').pop();

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(message_type_str ?? "");

   if (!message_type) {
      throw new Error(`Could not unpack message with type: ${event.msg.message?.typeUrl}`);
   }

   const message = message_type.decode(event.msg.message?.value as Uint8Array) as MessageT;

   return message;
}

// function unaryResponse<MessageDataT extends Message>(event: IncomingData, data: MessageDataT): void {

//    const any_msg = new Any();
//    any_msg.pack(data.serializeBinary(), typeUrlFromMessageClass(data) as string);

//    const message = new Event();
//    message.setEvent(EventType.RESPONSE);
//    message.setTag(event.msg.getTag());
//    message.setMessage(any_msg);

//    event.stream.write(message);
// }

class Architect implements ArchitectServiceImplementation {
   public service: ArchitectServiceImplementation;

   private _store: RootStore;

   private shutdown_subject: Subject<void> = new Subject<void>();

   constructor(store?: RootStore) {

      // Use the default store if not supplied
      if (!store) {
         store = getRootStore();
      }

      this._store = store;

      // Have to do this. Look at https://github.com/paymog/grpc_tools_node_protoc_ts/blob/master/doc/server_impl_signature.md to see about getting around this restriction
      this.service = {
         eventStream: (request, context) => {
            return this.do_eventStream(request, context);
         },
         ping: async (request, context): Promise<DeepPartial<PingResponse>> => {
            return await this.do_ping(request, context);
         },
         shutdown: async (request, context): Promise<DeepPartial<ShutdownResponse>> => {
            return await this.do_shutdown(request, context);
         }
      };
   }
   eventStream(request: AsyncIterable<Event>, context: CallContext): ServerStreamingMethodResult<{ error?: { message?: string | undefined; code?: ErrorCode | undefined; } | undefined; event?: EventType | undefined; tag?: number | undefined; message?: { typeUrl?: string | undefined; value?: Uint8Array | undefined; } | undefined; }> {
      return this.do_eventStream(request, context);
   }
   ping(request: PingRequest, context: CallContext): Promise<{ tag?: number | undefined; }> {
      return this.do_ping(request, context);
   }
   shutdown(request: ShutdownRequest, context: CallContext): Promise<{ tag?: number | undefined; }> {
      return this.do_shutdown(request, context);
   }

   public onShutdownSignaled() {
      return firstValueFrom(this.shutdown_subject);
   }

   private async *do_eventStream(stream: AsyncIterable<Event>, context: CallContext): AsyncIterable<DeepPartial<Event>> {
      console.log(`Event stream created for ${context.peer}`);

      const connection = {
         id: 1111,
         peer_info: context.peer,
         worker_ids: [],
      };

      context.metadata.set("mrc-machine-id", connection.id.toString());



      // const state_updates$ = new Observable<Event>((subscriber) => {


      //    async function* pull_messages(){
      //       for await (const req of stream) {
      //          console.log(`Event stream data for ${connection.peer_info} with message: ${req.event.toString()}`);

      //          yield* this.do_handle_event({
      //             msg: req,
      //          }, context);
      //       }
      //    }

      //    store_unsub
      // });

      // const send_events = new Subject<Event>();

      // merge();

      const store_update_sink = new AsyncSink<Event>();

      // Subscribe to the stores next update
      const store_unsub = this._store.subscribe(() => {
         const state = this._store.getState();

         // // Convert to an event
         // subscriber.next(Event.create({
         //    event: EventType.ServerStateUpdate,
         // }));

         store_update_sink.write(Event.create({
            event: EventType.ServerStateUpdate,
         }));
      });

      // Create a new connection
      this._store.dispatch(addConnection(connection));

      const self = this;

      const event_stream = async function* () {
         for await (const req of stream) {
            console.log(`Event stream data for ${connection.peer_info} with message: ${req.event.toString()}`);

            yield* self.do_handle_event({
               msg: req,
            }, context);
         }

         // Input stream has completed so stop pushing events
         store_unsub();

         store_update_sink.end();
      };

      // Yield a connected even
      yield Event.create({
         event: EventType.ClientEventStreamConnected,
         message: pack(ClientConnectedResponse.create({
            machineId: connection.id,
         }))
      });

      for await (const out_event of merge(as(event_stream()), as<Event>(store_update_sink))) {
         yield out_event;
      }


      // for await (const req of stream) {
      //    console.log(`Event stream data for ${connection.peer_info} with message: ${req.event.toString()}`);

      //    yield* this.do_handle_event({
      //       msg: req,
      //    }, context);
      // }

      console.log(`Event stream closed for ${connection.peer_info}. Deleting connection.`);

      // Create a new connection
      this._store.dispatch(removeConnection(connection));

      // call.on("data", (req: Event) => {
      //    console.log(`Event stream data for ${connection.peer_info} with message: ${req.getEvent().toString()}`);

      //    this.do_handle_event({
      //       msg: req,
      //       stream: call
      //    });
      // });

      // call.on("error", (err: Error) => {
      //    console.log(`Event stream errored for ${connection.peer_info} with message: ${err.message}`);
      // });

      // call.on("end", () => {
      //    console.log(`Event stream closed for ${connection.peer_info}. Deleting connection.`);

      //    // Create a new connection
      //    store.dispatch(removeConnection(connection));
      // });
   }

   private async *do_handle_event(event: IncomingData, context: CallContext) {
      try {
         switch (event.msg.event) {
            case EventType.ClientEventRequestStateUpdate:

               yield unaryResponse(event, StateUpdate, StateUpdate.create({

               }));

               break;
            case EventType.ClientUnaryRegisterWorkers:
               {
                  const payload = unpack<RegisterWorkersRequest>(event);

                  const machine_id = Number.parseInt(context.metadata.get("mrc-machine-id") as string);

                  const workers: IWorker[] = payload.ucxWorkerAddresses.map((value) => {
                     return {
                        id: 1234,
                        parent_machine_id: machine_id,
                        worker_address: value.toString(),
                        activated: false,
                     };
                  });

                  // Add the workers
                  this._store.dispatch(addWorkers(workers));

                  const resp = RegisterWorkersResponse.create({
                     machineId: machine_id,
                     instanceIds: workersSelectByMachineId(this._store.getState(), machine_id).map((worker) => worker.id)
                  });

                  yield unaryResponse(event, RegisterWorkersResponse, resp);

                  break;
               }
            case EventType.ClientUnaryActivateStream:
               {
                  const payload = unpack<RegisterWorkersResponse>(event);

                  yield unaryResponse(event, Ack, {});

                  break;
               }
            case EventType.ClientUnaryDropWorker:
               {
                  const payload = unpack<TaggedInstance>(event);

                  const found_worker = workersSelectById(this._store.getState(), payload.instanceId);

                  if (found_worker) {
                     this._store.dispatch(removeWorker(found_worker));
                  }

                  yield unaryResponse(event, Ack, {});

                  break;
               }

            default:
               break;
         }
      } catch (error) {
         console.log(`Error occurred handing event. Error: ${error}`);
      }
   }

   private async do_shutdown(req: ShutdownRequest, context: CallContext): Promise<DeepPartial<ShutdownResponse>> {

      console.log(`Issuing shutdown promise from ${context.peer}`);

      // Signal that shutdown was requested
      this.shutdown_subject.next();

      return ShutdownResponse.create();
   }

   private async do_ping(req: PingRequest, context: CallContext): Promise<DeepPartial<PingResponse>> {
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
