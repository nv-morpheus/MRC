

import {
   handleBidiStreamingCall, handleUnaryCall, sendUnaryData, ServerDuplexStream, ServerReadableStream, ServerUnaryCall, ServerWritableStream,
   status, UntypedHandleCall
} from '@grpc/grpc-js';
import { firstValueFrom, Subject } from "rxjs";

import { ArchitectServer, ArchitectService, Event, EventType, PingRequest, PingResponse, ShutdownRequest, ShutdownResponse } from "../proto/mrc/protos/architect";

class Architect {
   public service: ArchitectServer;

   private shutdown_subject: Subject<void> = new Subject<void>();

   constructor(){
      this.service = {
         eventStream: (call: ServerDuplexStream<Event, Event>): void => {
            this.do_eventStream(call);
         },
         ping: (call: ServerUnaryCall<PingRequest, PingResponse>, callback: sendUnaryData<PingResponse>): void =>{
            this.do_ping(call, callback);
         },
         shutdown: (call: ServerUnaryCall<ShutdownRequest, ShutdownResponse>, callback: sendUnaryData<ShutdownResponse>): void =>{
            this.do_shutdown(call, callback);
         },
      };
   }

   public getShutdownPromise(){
      return firstValueFrom(this.shutdown_subject);
   }

   public async shutdown(){
      await firstValueFrom(this.shutdown_subject);
   }

   private do_eventStream(call: ServerDuplexStream<Event, Event>): void {
      console.log(`Event stream created for ${call.getPeer()}`);

      call.on("data", (req: Event) => {
         console.log(`Event stream data for ${call.getPeer()} with message: ${req.event.toString()}`);
         this.do_handle_event(req, call);
      });

      call.on("error", (err: Error) => {
         console.log(`Event stream errored for ${call.getPeer()} with message: ${err.message}`);
      });

      call.on("end", () => {
         console.log(`Event stream closed for ${call.getPeer()}`);
      });
   }

   private do_handle_event(event: Event, call: ServerDuplexStream<Event, Event>): void{

      try {
         switch (event.event) {
            case EventType.ClientEventRequestStateUpdate:

               break;

            default:
               break;
         }
      } catch (error) {

      }

   }

   private do_shutdown(call: ServerUnaryCall<ShutdownRequest, ShutdownResponse>, callback: sendUnaryData<ShutdownResponse>){

      console.log(`Issuing shutdown promise from ${call.getPeer()}`);

      this.shutdown_subject.next();

      callback(null, ShutdownResponse.create());
   }

   private do_ping(call: ServerUnaryCall<PingRequest, PingResponse>, callback: sendUnaryData<PingResponse>){

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
