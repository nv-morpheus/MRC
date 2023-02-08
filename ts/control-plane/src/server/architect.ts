

import {
   handleBidiStreamingCall, handleUnaryCall, sendUnaryData, ServerDuplexStream, ServerReadableStream, ServerUnaryCall, ServerWritableStream,
   status, UntypedHandleCall
} from '@grpc/grpc-js';
import { firstValueFrom, Subject } from "rxjs";

import { Ack, ArchitectServer, ArchitectService, Event, EventType, PingRequest, PingResponse, RegisterWorkersRequest, RegisterWorkersResponse, ShutdownRequest, ShutdownResponse } from "../proto/mrc/protos/architect";
import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { store } from "./store";
import { addWorker, addWorkers, IWorker, workersSelectByMachineId } from "./features/workers/workersSlice";
import { Any } from "../proto/google/protobuf/any";
import proto_min from "protobufjs/minimal";
import { addConnection, removeConnection } from "./features/workers/connectionsSlice";

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

   const prefix = "type.googleapis.com/mrc.protos";

   if (message_class === RegisterWorkersRequest) {
      return `${prefix}.RegisterWorkersRequest`;
   } else if (message_class === RegisterWorkersResponse) {
      return `${prefix}.RegisterWorkersResponse`;
   }

   return undefined;
}

// function unaryResponse<MessageT extends ProtoMessageBase, I extends Exact<DeepPartial<MessageT>, I>>(event: IncomingData, message_class: MessageT, data: I): MessageT {
// function unaryResponse<MessageDataT extends Exact<DeepPartial<ProtoMessageBase<MessageDataT>>, MessageDataT>, MessageClassT extends ProtoMessageBase<MessageDataT>>(event: IncomingData, message_class: MessageClassT, data: MessageDataT): void {
function unaryResponse<MessageDataT>(event: IncomingData, message_class: any, data: MessageDataT): void {

   const response = message_class.create(data);

   // const writer = new BufferWriter();

   // RegisterWorkersResponse.encode(response, writer);

   const any_msg = Any.create({
      typeUrl: typeUrlFromMessageClass(message_class),
      value: message_class.encode(response).finish(),
   });

   event.stream.write(Event.create({
      event: EventType.Response,
      tag: event.msg.tag,
      message: any_msg,
   }));
}

class Architect {
   public service: ArchitectServer;

   private shutdown_subject: Subject<void> = new Subject<void>();

   constructor() {
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
         console.log(`Event stream data for ${connection.peer_info} with message: ${req.event.toString()}`);

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
         switch (event.msg.event) {
            case EventType.ClientEventRequestStateUpdate:

               break;
            case EventType.ClientUnaryRegisterWorkers:
               {

                  const payload = RegisterWorkersRequest.decode(event.msg.message?.value as Uint8Array);

                  const machine_id = Number.parseInt(event.stream.metadata.get("mrc-machine-id")[0] as string);

                  const workers: IWorker[] = payload.ucxWorkerAddresses.map((value) => {
                     return {
                        id: 1234,
                        parent_machine_id: machine_id,
                        worker_address: value.toString(),
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

                  unaryResponse(event, RegisterWorkersResponse, {
                     machineId: machine_id,
                     instanceIds: workersSelectByMachineId(store.getState(), machine_id).map((worker) => worker.id),
                  });

                  break;
               }
            case EventType.ClientUnaryActivateStream:
               {
                  const payload = RegisterWorkersResponse.decode(event.msg.message?.value as Uint8Array);

                  unaryResponse(event, Ack, {});

                  break;
               }

            default:
               break;
         }
      } catch (error) {

      }

   }

   private do_shutdown(call: ServerUnaryCall<ShutdownRequest, ShutdownResponse>, callback: sendUnaryData<ShutdownResponse>) {

      console.log(`Issuing shutdown promise from ${call.getPeer()}`);

      this.shutdown_subject.next();

      callback(null, ShutdownResponse.create());
   }

   private do_ping(call: ServerUnaryCall<PingRequest, PingResponse>, callback: sendUnaryData<PingResponse>) {

      console.log(`Ping from ${call.getPeer()}`);

      callback(null, PingResponse.create());
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
   ArchitectService,
};
