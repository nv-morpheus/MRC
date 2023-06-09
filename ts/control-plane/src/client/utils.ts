import { AsyncIterableX, AsyncSink } from "ix/asynciterable";
import "ix/add/asynciterable-operators/first";
import { Observable, filter, firstValueFrom } from "rxjs";
import { UnknownMessage } from "@mrc/proto/typeRegistry";
import { unpackEvent } from "@mrc/common/utils";
import { Event, EventType } from "@mrc/proto/mrc/protos/architect";

export async function get_first_event(
   incoming_events: AsyncIterableX<Event> | Observable<Event>,
   predicate: (value: Event, index: number) => boolean
) {
   let first_event: Event | undefined;

   if (incoming_events instanceof AsyncIterableX<Event>) {
      first_event = await incoming_events.first({
         predicate,
      });
   } else {
      first_event = await firstValueFrom(incoming_events.pipe(filter(predicate)));
   }

   if (!first_event) {
      throw new Error("Did not get response");
   }

   return first_event;
}

export async function unpack_first_event<MessageT extends UnknownMessage>(
   incoming_events: AsyncIterableX<Event> | Observable<Event>,
   predicate: (value: Event, index: number) => boolean
) {
   const first_event = await get_first_event(incoming_events, predicate);

   return unpackEvent<MessageT>(first_event);
}

export async function unary_event(
   incoming_events: AsyncIterableX<Event> | Observable<Event>,
   outgoing_events: AsyncSink<Event>,
   message: Event
) {
   const response_promise = get_first_event(incoming_events, (event) => {
      return event.event === EventType.Response && event.tag === message.tag;
   });

   // Now send the message
   outgoing_events.write(message);

   // Await the response
   const response_event = await response_promise;

   if (response_event.error) {
      throw new Error(response_event.error.message);
   }

   return response_event;
}

export async function unpack_unary_event<MessageT extends UnknownMessage>(
   incoming_events: AsyncIterableX<Event> | Observable<Event>,
   outgoing_events: AsyncSink<Event>,
   message: Event
) {
   return unpackEvent<MessageT>(await unary_event(incoming_events, outgoing_events, message));
}
