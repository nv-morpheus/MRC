import { BinaryLike, createHash } from "node:crypto";
import { BufferWriter } from "protobufjs";

import { Any } from "../proto/google/protobuf/any";
import { Event, EventType } from "../proto/mrc/protos/architect";
import { messageTypeRegistry, UnknownMessage } from "../proto/typeRegistry";

export function generateId(max = 4294967295): string {
   return Math.floor(Math.random() * max).toString();
}

export function sleep(ms: number) {
   return new Promise((r) => setTimeout(r, ms));
}

export function stringToBytes(value: string[]): Uint8Array[];
export function stringToBytes(value: string): Uint8Array;
export function stringToBytes(value: string | string[]) {
   if (value instanceof Array<string>) {
      return value.map((s) => new TextEncoder().encode(s));
   }

   return new TextEncoder().encode(value);
}

export function bytesToString(value: Uint8Array[]): string[];
export function bytesToString(value: Uint8Array): string;
export function bytesToString(value: Uint8Array | Uint8Array[]) {
   if (value instanceof Array<Uint8Array>) {
      return value.map((s) => new TextDecoder().decode(s));
   }

   return new TextDecoder().decode(value);
}

export function pack(data: UnknownMessage): Any {
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
   const message_type_str = message.typeUrl.split("/").pop();

   // Load the type from the registry
   const message_type = messageTypeRegistry.get(message_type_str ?? "");

   if (!message_type) {
      throw new Error(`Could not unpack message with type: ${message.typeUrl}`);
   }

   const decoded = message_type.decode(message.value as Uint8Array) as MessageT;

   return decoded;
}

export function packEvent(event_type: EventType, event_tag: string, data: UnknownMessage): Event {
   const any_msg = pack(data);

   return Event.create({
      event: event_type,
      tag: event_tag,
      message: any_msg,
   });
}

export function unpackEvent<MessageT extends UnknownMessage>(message: Event): MessageT {
   if (!message.message) {
      throw new Error("Message body for event was undefined. Cannot unpack");
   }

   return unpack<MessageT>(message.message);
}

export function packEventResponse(incoming_event: Event, data: UnknownMessage): Event {
   const any_msg = pack(data);

   return Event.create({
      event: EventType.Response,
      tag: incoming_event.tag,
      message: any_msg,
   });
}

function hashName16(name: string): bigint {
   // Implement the fnvla algorighm: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
   const hash_32_offset = 2_166_136_261n; // 0x811C9DC5
   const hash_32_prime = 16_777_619n; // 0x01000193

   let hash_u32 = hash_32_offset;

   for (let index = 0; index < name.length; index++) {
      const element = name.charCodeAt(index);

      hash_u32 ^= BigInt(element);
      hash_u32 = BigInt.asUintN(32, hash_u32 * hash_32_prime);
   }

   // Only take the last 16 bits
   return BigInt.asUintN(16, hash_u32);
}

export function generateSegmentHash(seg_name: string, worker_id: string): number {
   const name_hash = hashName16(seg_name);
   const worker_hash = hashName16(worker_id);

   // Shift the name over 16
   return Number((name_hash << 16n) | worker_hash);
}

// Generats a hash for a serialized object in string or buffer form
export function hashObject(data: BinaryLike): string {
   const hash = createHash("md5");

   // Get the hash of the object encoded in base64
   const hash_str = hash.update(data).digest();

   // Only take the first 64 bits so it fits into uint64
   const hash_uint = hash_str.readBigUInt64LE(0);

   // Write it out as a string
   return hash_uint.toString();
}

export function hashProtoMessage<MessageDataT extends UnknownMessage>(data: MessageDataT): string {
   // Load the type from the registry
   const message_type = messageTypeRegistry.get(data.$type);

   if (!message_type) {
      throw new Error("Unknown type in type registry");
   }

   const buffer = new BufferWriter();

   message_type.encode(data, buffer);

   return hashObject(buffer.finish());
}

export function ensureError(value: unknown): Error {
   if (value instanceof Error) {
      return value;
   }

   let stringified = "[Unable to stringify the thrown value]";
   try {
      stringified = JSON.stringify(value);
   } catch {
      /* empty */
   }

   const error = new Error(`This value was thrown as is, not through an Error: ${stringified}`);
   return error;
}
