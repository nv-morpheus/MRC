import assert from "assert";
import { RootStore, setupStore } from "../store";
import { activateWorkers, addWorker, IWorker, removeWorker, workersSelectAll, workersSelectById } from "./workersSlice";
import { addConnection, IConnection } from "./connectionsSlice";
import { expect, jest, test } from '@jest/globals';

describe("Workers", () => {
   let store: RootStore;

   const connection: IConnection = {
      id: 1111,
      peerInfo: "localhost:1234",
      workerIds: [],
      assignedPipelineIds: [],
   };

   const worker: IWorker = {
      id: 1234,
      activated: false,
      machineId: 1111,
      workerAddress: "-----",
      assignedSegmentIds: [],
   };

   // Get a clean store each time
   beforeEach((done) => {
      store = setupStore();
      done();
   });

   describe("add", () => {
      it("worker before connection", () => {
         assert.throws(() => store.dispatch(addWorker(worker)));
      });

      it("adds a worker", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         expect(workersSelectById(store.getState(), worker.id)).toBe(worker);
      });

      it("adds a worker with duplicate ID", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         expect(workersSelectById(store.getState(), worker.id)).toBe(worker);

         assert.throws(() => store.dispatch(addWorker(worker)));
      });
   });

   describe("remove", () => {

      it("worker before connection", () => {
         assert.throws(() => store.dispatch(removeWorker(worker)));
      });

      it("remove a worker", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         expect(workersSelectById(store.getState(), worker.id)).toBe(worker);

         store.dispatch(removeWorker(worker));

         expect(workersSelectAll(store.getState())).toHaveLength(0);
      });

      it("remove a non existant worker", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         expect(workersSelectById(store.getState(), worker.id)).toBe(worker);

         assert.throws(() => {
            store.dispatch(removeWorker({
               ...worker,
               id: 2222,
            }));
         });
      });
   });

   describe("activate", () => {
      it("default value", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         expect(workersSelectById(store.getState(), worker.id)).toHaveProperty("activated", false);
      });

      it("activate a worker", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         store.dispatch(activateWorkers([worker]));

         expect(workersSelectById(store.getState(), worker.id)).toHaveProperty("activated", true);
      });

      it("activate a worker again", () => {
         store.dispatch(addConnection(connection));

         store.dispatch(addWorker(worker));

         store.dispatch(activateWorkers([worker]));
         store.dispatch(activateWorkers([worker]));

         expect(workersSelectById(store.getState(), worker.id)).toHaveProperty("activated", true);
      });
   });
});
