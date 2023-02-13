import type { Client, ServiceError, CallOptions, ClientUnaryCall, Metadata } from '@grpc/grpc-js';
import { AsyncIterableX, AsyncSink } from "ix/asynciterable";
import { UnknownMessage } from "../proto/typeRegistry";
import { Event, EventType } from "../proto/mrc/protos/architect";
import "ix/add/asynciterable-operators/first";
import { OptionalFindOptions } from "ix/asynciterable/findoptions";
import { unpackEvent } from "../common/utils";

type OriginalCall<T, U> = (
   request: T,
   metadata: Metadata,
   options: Partial<CallOptions>,
   callback: (err: ServiceError | null, res?: U) => void,
) => ClientUnaryCall;

type PromisifiedCall<T, U> = (
   request: T,
   metadata?: Metadata,
   options?: Partial<CallOptions>,
) => Promise<U>;

export type PromisifiedClient<C> = { $: C; } & {
   [prop in Exclude<keyof C, keyof Client>]: C[prop] extends OriginalCall<infer T, infer U>
   ? PromisifiedCall<T, U>
   : never
};

export function promisifyClient<C extends Client>(client: C) {
   return new Proxy(client, {
      get: (target, descriptor) => {
         const key = descriptor as keyof PromisifiedClient<C>;

         if (key === '$') return target;

         const func = target[key];
         if (typeof func === 'function')
            return (...args: unknown[]) =>
               new Promise((resolve, reject) =>
                  func.call(
                     target,
                     ...[...args, (err: unknown, res: unknown) => (err ? reject(err) : resolve(res))],
                  ),
               );
      },
   }) as unknown as PromisifiedClient<C>;
}

export async function get_first_event(incoming_events: AsyncIterableX<Event>, options?: OptionalFindOptions<Event>) {

   const first_event = await incoming_events.first(options);

   if (!first_event) {
      throw new Error("Did not get response");
   }

   return first_event;
}

export async function unpack_first_event<MessageT extends UnknownMessage>(incoming_events: AsyncIterableX<Event>, options?: OptionalFindOptions<Event>) {

   const first_event = await get_first_event(incoming_events, options);

   return unpackEvent<MessageT>(first_event);
}

export async function unary_event(incoming_events: AsyncIterableX<Event>, outgoing_events: AsyncSink<Event>, message: Event) {

   const response_promise = get_first_event(incoming_events, {
      predicate: (event) => {
         return event.event === EventType.Response && event.tag === message.tag;
      }
   });

   // Now send the message
   outgoing_events.write(message);

   // Await the response
   return await response_promise;
}

export async function unpack_unary_event<MessageT extends UnknownMessage>(incoming_events: AsyncIterableX<Event>, outgoing_events: AsyncSink<Event>, message: Event) {

   return unpackEvent<MessageT>(await unary_event(incoming_events, outgoing_events, message));
}
