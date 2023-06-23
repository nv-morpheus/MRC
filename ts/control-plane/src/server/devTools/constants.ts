import { ClientOptions } from "socketcluster-client/lib/clientsocket.js";

export const defaultSocketOptions: ClientOptions = {
   secure: false,
   hostname: "localhost",
   port: 8000,
   autoReconnect: true,
   autoReconnectOptions: {
      randomness: 30000,
      maxDelay: 600000,
   },
   pingTimeout: 600000,
   pingTimeoutDisabled: true,
   ackTimeout: 600000,
};
