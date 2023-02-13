import { Any } from "../proto/google/protobuf/any";
import { Event, EventType } from "../proto/mrc/protos/architect";
import { messageTypeRegistry, UnknownMessage } from "../proto/typeRegistry";

export function pack<MessageDataT extends UnknownMessage>(data: MessageDataT): Any {

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(data.$type);

   if (!message_type) {
      throw new Error("Unknown type in type registry");
   }

   const any_msg = Any.create({
      typeUrl: `type.googleapis.com/${message_type.$type}`,
      value: message_type.encode(data).finish(),
   });

   return any_msg;
}

export function unpack<MessageT extends UnknownMessage>(message: Any) {
   const message_type_str = message.typeUrl.split('/').pop();

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(message_type_str ?? "");

   if (!message_type) {
      throw new Error(`Could not unpack message with type: ${message.typeUrl}`);
   }

   const decoded = message_type.decode(message.value as Uint8Array) as MessageT;

   return decoded;
}

export function packEvent<MessageDataT extends UnknownMessage>(data: MessageDataT, incoming_event?: Event, event_type?: EventType): Event {

   const any_msg = pack<MessageDataT>(data);

   return Event.create({
      event: event_type ?? EventType.Response,
      tag: incoming_event?.tag,
      message: any_msg,
   });
}

export function unpackEvent<MessageT extends UnknownMessage>(message: Event): MessageT {

   if (!message.message) {
      throw new Error("Message body for event was undefined. Cannot unpack");
   }

   return unpack<MessageT>(message.message);
}
