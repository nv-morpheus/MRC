import { launchDevtoolsCli } from "@mrc/common/dev_tools";
import { sleep } from "@mrc/common/utils";
import { ArchitectServer } from "@mrc/server/server";
import { setupStore } from "@mrc/server/store/store";

async function main() {
   console.log("Sleeping...");
   await sleep(4000);
   console.log("Sleeping... Done");

   const addDevTools = process.env.NODE_ENV !== "production";

   console.log(`addDevTools: ${addDevTools}`);

   if (addDevTools) {
      // If not in production, start the redux-devtools service
      await launchDevtoolsCli("localhost", "9000");
   }

   const server = new ArchitectServer(setupStore(undefined, addDevTools));

   await server.start();

   await server.join();

   console.log("Exiting script");
}

main();
