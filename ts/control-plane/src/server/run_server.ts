import { ArchitectServer } from "./server";


async function main(): Promise<void> {

   const server = new ArchitectServer();

   await server.start();

   process.on("SIGINT", async () => {

   });

   await server.join();

   console.log("Exiting script");
}

main();
