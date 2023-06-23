import { ServerCredentials } from "@grpc/grpc-js";
import { ArchitectDefinition } from "@mrc/proto/mrc/protos/architect";
import { Architect } from "@mrc/server/architect";
import { RootStore } from "@mrc/server/store/store";
import { createServer } from "nice-grpc";

export class ArchitectServer {
   private _server = createServer();
   private _architect: Architect;

   private _completed_signal?: (value: void | PromiseLike<void>) => void = undefined;
   private _completed_promise: Promise<void> | undefined;

   constructor(store?: RootStore, private hostname: string = "0.0.0.0:13337") {
      // Create the architect
      this._architect = new Architect(store);

      this._server.add(ArchitectDefinition, this._architect);
   }

   public async start() {
      const port = await this._server.listen(this.hostname, ServerCredentials.createInsecure());

      console.log(`server is running on ${this.hostname}`);

      this._completed_promise = new Promise<void>((resolve, reject) => {
         // Save the resolve function to signal this outside of this function
         this._completed_signal = resolve;
      });

      void this._architect.onShutdownSignaled().then(async () => {
         // Call stop
         await this.stop();
      });

      return port;
   }

   public async stop() {
      // Try to shutdown first
      console.log("Server shutting down...");

      // Tell the architect we are stopping
      await this._architect.stop();

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
