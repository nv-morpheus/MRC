import { credentials, Deadline, Server, ServerCredentials } from "@grpc/grpc-js";
import { ServiceClient } from "@grpc/grpc-js/build/src/make-client";
import assert from "assert";

import { ArchitectClient, ArchitectService } from "../proto/mrc/protos/architect_grpc_pb";
import { Architect } from "../server/architect";
import { Event, PingRequest } from "../proto/mrc/protos/architect_pb";
import { PromisifiedClient, promisifyClient } from "./utils";

describe("Client", () => {

   let server: Server;
   let client: PromisifiedClient<ArchitectClient>;

   beforeEach(done => {
      server = new Server();
      const architect = new Architect();

      server.addService(ArchitectService, architect.service);

      server.bindAsync('0.0.0.0:13337', ServerCredentials.createInsecure(), (error, port) => {
         assert.ifError(error);

         // Now make the client
         client = promisifyClient(new ArchitectClient(`localhost:${port}`, credentials.createInsecure()));

         server.start();

         done();
      });
   });

   afterEach(done => {
      client.$.close();
      server.tryShutdown(done);
   });

   describe("connection", () => {
      it("is ready", () => {
         client.$.waitForReady(10, (error) => {
            assert.ifError(error);
         });
      });

      it("ping", async () => {
         // client.$.waitForReady(10, (error) => {
         //    assert.ifError(error);
         // });

         const req = new PingRequest();
         req.setTag(1234);

         const resp = await client.ping(req);

         expect(resp.getTag()).toBe(req.getTag());
      });

      it("open bidi event stream", () => {

         const stream = client.$.eventStream();

         const event = new Event();
         // event.setEvent();

         // stream.write();
      });
   });
});
