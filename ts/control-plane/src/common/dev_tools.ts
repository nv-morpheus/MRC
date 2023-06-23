export async function launchDevtoolsCli(hostname: string = "localhost", port: string = "9000") {
   const reduxDevToolsServer = await import(/*webpackIgnore: true*/ "@redux-devtools/cli");

   const devToolsServer = await reduxDevToolsServer.default({
      hostname: hostname,
      port: port,
   });

   if (devToolsServer.portAlreadyUsed) {
      console.error("Could not start Redux DevTools. Port already in use.");

      return;
   }

   // Wait for it to be ready
   await devToolsServer.ready;

   // Were good to go
   console.log(
      `Started Redux DevTools server at http://${hostname}:${port}. Open the URL in a browser to view the Redux state`
   );
}
