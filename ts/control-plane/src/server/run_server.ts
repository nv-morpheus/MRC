
import {ArchitectServer} from "./server";

// Needed because import{ArchitectServer} from "./server.js" does not work in .mts files
async function main()
{
   const reduxDevToolsServer = await import("@redux-devtools/cli");

   if (process.env.NODE_ENV !== "production")
   {
      const hostname = "localhost";
      const port     = "8000";

      // If not in production, start the redux-devtools service
      const devToolsServer = await reduxDevToolsServer.default({
         "hostname": hostname,
         "port": port,
      });

      // Wait for it to be ready
      await devToolsServer.ready;

      console.log(`Started Redux DevTools server at http://${hostname}:${
          port}. Open the URL in a browser to view the Redux state`);
   }

   const server = new ArchitectServer();

   await server.start();

   // process.on("SIGINT", async () => {

   // });

   await server.join();

   console.log("Exiting script");
}

main();
