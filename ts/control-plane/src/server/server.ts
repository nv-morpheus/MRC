import {
   ServerCredentials,
} from '@grpc/grpc-js';
import { createServer } from "nice-grpc";
import { ArrowFunction } from "typescript";
import { ArchitectDefinition } from "../proto/mrc/protos/architect";
import { Architect } from "./architect";
import { RootStore } from "./store/store";

// async function main(): Promise<void> {
//    const server = createServer();

//    const architect = new Architect();

//    server.add(ArchitectDefinition, architect.service);

//    const port = await server.listen('0.0.0.0:13337', ServerCredentials.createInsecure());

//    console.log(`server is running on 0.0.0.0:${port}`);

//    // Wait for the architect to shutdown
//    await architect.shutdown();

//    // Try to shutdown first
//    console.log("Server shutting down...");

//    try {
//       await server.shutdown();

//       console.log("Server shutdown");

//    } catch (error) {
//       console.log("Server shutdown failed. Forcing shutdown");
//       server.forceShutdown();
//    }

//    console.log("Exiting script");
// }

// main();


export class ArchitectServer {

   private _server = createServer();
   private _architect: Architect;

   private _completed_signal?: (value: void | PromiseLike<void>) => void = undefined;
   private _completed_promise: Promise<void> | undefined;

   constructor(store?: RootStore) {
      // Create the architect
      this._architect = new Architect(store);

      this._server.add(ArchitectDefinition, this._architect.service);
   }

   public async start() {
      const port = await this._server.listen('0.0.0.0:13338', ServerCredentials.createInsecure());

      console.log(`server is running on 0.0.0.0:${port}`);

      this._completed_promise = new Promise<void>((resolve, reject) => {
         // Save the resolve function to signal this outside of this function
         this._completed_signal = resolve;
      });

      this._architect.onShutdownSignaled().then(() => {
         // Call stop
         this.stop();
      });

      return port;
   }

   public async stop() {
      // Try to shutdown first
      console.log("Server shutting down...");

      try {
         await this._server.shutdown();

         console.log("Server shutdown");

      } catch (error) {

         console.log("Server shutdown failed. Forcing shutdown");
         this._server.forceShutdown();

      } finally {

         if (this._completed_signal) {
            this._completed_signal();
         }
      }
   }

   public async join() {
      if (!this._completed_promise) {
         throw new Error("Must start server before joining");
      }

      await this._completed_promise;
   }
}
