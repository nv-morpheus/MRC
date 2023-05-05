import {BinaryLike, createHash} from "node:crypto";
import {isBigUint64Array} from "node:util/types";
import {BufferWriter} from "protobufjs";

import {Any} from "../proto/google/protobuf/any";
import {Event, EventType} from "../proto/mrc/protos/architect";
import {messageTypeRegistry, UnknownMessage} from "../proto/typeRegistry";

export function stringToBytes(value: string[]): Uint8Array[];
export function stringToBytes(value: string): Uint8Array;
export function stringToBytes(value: string|string[])
{
   if (value instanceof Array<string>)
   {
      return value.map((s) => new TextEncoder().encode(s));
   }

   return new TextEncoder().encode(value);
}

export function bytesToString(value: Uint8Array[]): string[];
export function bytesToString(value: Uint8Array): string;
export function bytesToString(value: Uint8Array|Uint8Array[])
{
   if (value instanceof Array<Uint8Array>)
   {
      return value.map((s) => new TextDecoder().decode(s));
   }

   return new TextDecoder().decode(value);
}

export function pack<MessageDataT extends UnknownMessage>(data: MessageDataT): Any
{
   // Load the type from the registry
   const message_type = messageTypeRegistry.get(data.$type);

   if (!message_type)
   {
      throw new Error("Unknown type in type registry");
   }

   const any_msg = Any.create({
      typeUrl: `type.googleapis.com/${message_type.$type}`,
      value: message_type.encode(data).finish(),
   });

   return any_msg;
}

export function unpack<MessageT extends UnknownMessage>(message: Any)
{
   const message_type_str = message.typeUrl.split("/").pop();

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(message_type_str ?? "");

   if (!message_type)
   {
      throw new Error(`Could not unpack message with type: ${message.typeUrl}`);
   }

   const decoded = message_type.decode(message.value as Uint8Array) as MessageT;

   return decoded;
}

export function packEvent<MessageDataT extends UnknownMessage>(event_type: EventType,
                                                               event_tag: string,
                                                               data: MessageDataT): Event
{
   const any_msg = pack<MessageDataT>(data);

   return Event.create({
      event: event_type,
      tag: event_tag,
      message: any_msg,
   });
}

export function unpackEvent<MessageT extends UnknownMessage>(message: Event): MessageT
{
   if (!message.message)
   {
      throw new Error("Message body for event was undefined. Cannot unpack");
   }

   return unpack<MessageT>(message.message);
}

export function packEventResponse<MessageDataT extends UnknownMessage>(incoming_event: Event, data: MessageDataT): Event
{
   const any_msg = pack<MessageDataT>(data);

   return Event.create({
      event: EventType.Response,
      tag: incoming_event.tag,
      message: any_msg,
   });
}

// Generats a hash for a serialized object in string or buffer form
export function hashObject(data: BinaryLike): string
{
   const hash = createHash("md5");

   // Get the hash of the object encoded in base64
   const hash_str = hash.update(data).digest();

   // Only take the first 64 bits so it fits into uint64
   const hash_uint = hash_str.readBigUInt64LE(0);

   // Write it out as a string
   return hash_uint.toString();
}

export function hashProtoMessage<MessageDataT extends UnknownMessage>(data: MessageDataT): string
{
   // Load the type from the registry
   const message_type = messageTypeRegistry.get(data.$type);

   if (!message_type)
   {
      throw new Error("Unknown type in type registry");
   }

   const buffer = new BufferWriter();

   message_type.encode(data, buffer);

   return hashObject(buffer.finish());
}
