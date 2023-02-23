import reduxDevToolsServer from "@redux-devtools/cli";

// Needed because import{ArchitectServer} from "./server.js" does not work in .mts files
import server_pkg from './server.js';
const {ArchitectServer} = server_pkg;

if (process.env.NODE_ENV !== "production")
{
    const hostname = "localhost";
    const port     = "8000";

    // If not in production, start the redux-devtools service
    const devToolsServer = await reduxDevToolsServer({
        "hostname" : hostname,
        "port" : port,
    });

    // Wait for it to be ready
    await devToolsServer.ready;

    console.log(`Started Redux DevTools server at https://${hostname}:${port}. Open the URL in a browser to view the Redux state`);
}

const server = new ArchitectServer();

await server.start();

// process.on("SIGINT", async () => {

// });

await server.join();

console.log("Exiting script");

