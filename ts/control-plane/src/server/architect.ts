

import {
   handleBidiStreamingCall, handleUnaryCall, sendUnaryData, ServerDuplexStream, ServerReadableStream, ServerUnaryCall, ServerWritableStream,
   status, UntypedHandleCall
} from '@grpc/grpc-js';
import { firstValueFrom, Subject } from "rxjs";

// import { Ack, ArchitectServer, ArchitectService, Event, EventType, PingRequest, PingResponse, RegisterWorkersRequest, RegisterWorkersResponse, ShutdownRequest, ShutdownResponse } from "../proto/mrc/protos/architect";
import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { store } from "./store";
import { addWorker, addWorkers, IWorker, removeWorker, workersSelectById, workersSelectByMachineId } from "./features/workers/workersSlice";
import proto_min from "protobufjs/minimal";
import { addConnection, removeConnection } from "./features/workers/connectionsSlice";
import { IArchitectServer } from "../proto/mrc/protos/architect_grpc_pb";
import { Ack, Event, EventType, PingRequest, PingResponse, RegisterWorkersRequest, RegisterWorkersResponse, ShutdownRequest, ShutdownResponse, TaggedInstance } from "../proto/mrc/protos/architect_pb";
import { Any } from "google-protobuf/google/protobuf/any_pb";
import { Message } from "google-protobuf";

interface IncomingData {
   msg: Event,
   stream: ServerDuplexStream<Event, Event>,
}

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

type DeepPartial<T> = T extends Builtin ? T
   : T extends Array<infer U> ? Array<DeepPartial<U>> : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
   : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
   : Partial<T>;

type KeysOfUnion<T> = T extends T ? keyof T : never;
type Exact<P, I extends P> = P extends Builtin ? P
   : P & { [K in keyof P]: Exact<P[K], I[K]> } & { [K in Exclude<keyof I, KeysOfUnion<P>>]: never };

interface ProtoMessageBase<MessageDataT> {
   create<I extends Exact<DeepPartial<MessageDataT>, I>>(base?: I): ProtoMessageBase<MessageDataT>;
   // create(base?: MessageDataT): ProtoMessageBase<MessageDataT>;
   // encode(message: ProtoMessageBase<MessageDataT>, writer?: proto_min.Writer): proto_min.Writer;
}

function testResponse<MessageClassT>(message_class: MessageClassT): void {

}

function typeUrlFromMessageClass(message_class: any) {

   const prefix = "mrc.protos";

   if (message_class instanceof RegisterWorkersRequest) {
      return `${prefix}.RegisterWorkersRequest`;
   } else if (message_class instanceof RegisterWorkersResponse) {
      return `${prefix}.RegisterWorkersResponse`;
   } else if (message_class instanceof Ack) {
      return `${prefix}.Ack`;
   } else {
      throw new Error(`Unknown message type: ${typeof message_class}`);
   }
}

// function unaryResponse<MessageT extends ProtoMessageBase, I extends Exact<DeepPartial<MessageT>, I>>(event: IncomingData, message_class: MessageT, data: I): MessageT {
// function unaryResponse<MessageDataT extends Exact<DeepPartial<ProtoMessageBase<MessageDataT>>, MessageDataT>, MessageClassT extends ProtoMessageBase<MessageDataT>>(event: IncomingData, message_class: MessageClassT, data: MessageDataT): void {
// function unaryResponse<MessageDataT>(event: IncomingData, message_class: any, data: MessageDataT): void {

//    const response = message_class.create(data);

//    // const writer = new BufferWriter();

//    // RegisterWorkersResponse.encode(response, writer);

//    const any_msg = Any. .create({
//       typeUrl: typeUrlFromMessageClass(message_class),
//       value: message_class.encode(response).finish(),
//    });

//    event.stream.write(Event.create({
//       event: EventType.RESPONSE,
//       tag: event.msg.getTag(),
//       message: any_msg,
//    }));
// }

function unaryResponse<MessageDataT extends Message>(event: IncomingData, data: MessageDataT): void {

   const any_msg = new Any();
   any_msg.pack(data.serializeBinary(), typeUrlFromMessageClass(data) as string);

   const message = new Event();
   message.setEvent(EventType.RESPONSE);
   message.setTag(event.msg.getTag());
   message.setMessage(any_msg);

   event.stream.write(message);
}

class Architect {
   public service: IArchitectServer;

   private shutdown_subject: Subject<void> = new Subject<void>();

   constructor() {
      // Have to do this. Look at https://github.com/paymog/grpc_tools_node_protoc_ts/blob/master/doc/server_impl_signature.md to see about getting around this restriction
      this.service = {
         eventStream: (call: ServerDuplexStream<Event, Event>): void => {
            this.do_eventStream(call);
         },
         ping: (call: ServerUnaryCall<PingRequest, PingResponse>, callback: sendUnaryData<PingResponse>): void => {
            this.do_ping(call, callback);
         },
         shutdown: (call: ServerUnaryCall<ShutdownRequest, ShutdownResponse>, callback: sendUnaryData<ShutdownResponse>): void => {
            this.do_shutdown(call, callback);
         },
      };
   }

   public getShutdownPromise() {
      return firstValueFrom(this.shutdown_subject);
   }

   public async shutdown() {
      await firstValueFrom(this.shutdown_subject);
   }

   private do_eventStream(call: ServerDuplexStream<Event, Event>): void {
      console.log(`Event stream created for ${call.getPeer()}`);

      const connection = {
         id: 1111,
         peer_info: call.getPeer(),
         worker_ids: [],
      };

      call.metadata.add("mrc-machine-id", connection.id.toString());

      // Create a new connection
      store.dispatch(addConnection(connection));

      call.on("data", (req: Event) => {
         console.log(`Event stream data for ${connection.peer_info} with message: ${req.getEvent().toString()}`);

         this.do_handle_event({
            msg: req,
            stream: call
         });
      });

      call.on("error", (err: Error) => {
         console.log(`Event stream errored for ${connection.peer_info} with message: ${err.message}`);
      });

      call.on("end", () => {
         console.log(`Event stream closed for ${connection.peer_info}. Deleting connection.`);

         // Create a new connection
         store.dispatch(removeConnection(connection));
      });
   }

   private do_handle_event(event: IncomingData): void {
      try {
         switch (event.msg.getEvent()) {
            case EventType.CLIENTEVENTREQUESTSTATEUPDATE:

               break;
            case EventType.CLIENTUNARYREGISTERWORKERS:
               {

                  const payload = RegisterWorkersRequest.deserializeBinary(event.msg.getMessage()?.getValue_asU8() as Uint8Array);

                  const machine_id = Number.parseInt(event.stream.metadata.get("mrc-machine-id")[0] as string);

                  const workers: IWorker[] = payload.getUcxWorkerAddressesList_asB64().map((value) => {
                     return {
                        id: 1234,
                        parent_machine_id: machine_id,
                        worker_address: value.toString(),
                        activated: false,
                     };
                  });

                  // Add the workers
                  store.dispatch(addWorkers(workers));

                  // const response = RegisterWorkersResponse.create({
                  //    machineId: machine_id,
                  //    instanceIds: workersSelectByMachineId(store.getState(), machine_id).map((worker) => worker.id),
                  // });

                  // // const writer = new BufferWriter();

                  // // RegisterWorkersResponse.encode(response, writer);

                  // const any_msg = Any.create({
                  //    typeUrl: `type.googleapis.com/mrc.protos.RegisterWorkersResponse`,
                  //    value: RegisterWorkersResponse.encode(response).finish(),
                  // });

                  // event.stream.write(Event.create({
                  //    event: EventType.Response,
                  //    tag: event.msg.tag,
                  //    message: any_msg,
                  // }));

                  const resp = new RegisterWorkersResponse();
                  resp.setMachineId(machine_id);
                  resp.setInstanceIdsList(workersSelectByMachineId(store.getState(), machine_id).map((worker) => worker.id));

                  unaryResponse(event, resp);

                  break;
               }
            case EventType.CLIENTUNARYACTIVATESTREAM:
               {
                  const payload = RegisterWorkersResponse.deserializeBinary(event.msg.getMessage()?.getValue_asU8() as Uint8Array);

                  unaryResponse(event, new Ack());

                  break;
               }
            case EventType.CLIENTUNARYDROPWORKER:
               {
                  const payload = TaggedInstance.deserializeBinary(event.msg.getMessage()?.getValue_asU8() as Uint8Array);

                  const found_worker = workersSelectById(store.getState(), payload.getInstanceId());

                  if (found_worker) {
                     store.dispatch(removeWorker(found_worker));
                  }

                  unaryResponse(event, new Ack());

                  break;
               }

            default:
               break;
         }
      } catch (error) {
         console.log(`Error occurred handing event. Error: ${error}`);
      }

   }

   private do_shutdown(call: ServerUnaryCall<ShutdownRequest, ShutdownResponse>, callback: sendUnaryData<ShutdownResponse>) {

      console.log(`Issuing shutdown promise from ${call.getPeer()}`);

      this.shutdown_subject.next();

      callback(null, new ShutdownResponse());
   }

   private do_ping(call: ServerUnaryCall<PingRequest, PingResponse>, callback: sendUnaryData<PingResponse>) {

      console.log(`Ping from ${call.getPeer()}`);

      callback(null, new PingResponse());
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
