
export async function launchDevtoolsCli(hostname: string = "localhost", port: string = "8000")
{
   const reduxDevToolsServer = await import("@redux-devtools/cli");

   const devToolsServer = await reduxDevToolsServer.default({
      "hostname": hostname,
      "port": port,
   });

   // Wait for it to be ready
   await devToolsServer.ready;

   // Were good to go
   console.log(`Started Redux DevTools server at http://${hostname}:${
       port}. Open the URL in a browser to view the Redux state`);
}
