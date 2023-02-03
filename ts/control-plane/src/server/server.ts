import {
   Server,
   ServerCredentials,
} from '@grpc/grpc-js';
import { Architect, ArchitectService } from "./Architect";

async function main(): Promise<void> {
   const server = new Server();

   const architect = new Architect();

   server.addService(ArchitectService, architect.service);

   server.bindAsync('0.0.0.0:13337', ServerCredentials.createInsecure(), (error, port) => {
      server.start();

      console.log(`server is running on 0.0.0.0:${port}`);
   });

   // Wait for the architect to shutdown
   await architect.shutdown();

   // Try to shutdown first
   console.log("Server shutting down...");

   server.tryShutdown((error) => {

      if (error) {
         console.log("Server shutdown failed. Forcing shutdown");
         server.forceShutdown();
      }
      else {
         console.log("Server shutdown");
      }
   });

   console.log("Exiting script");
}

main();
