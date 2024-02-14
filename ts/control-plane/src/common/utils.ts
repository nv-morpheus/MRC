import { BinaryLike, createHash } from "node:crypto";
import { BufferWriter } from "protobufjs";

import { Any } from "@mrc/proto/google/protobuf/any";
import { Event, EventType } from "@mrc/proto/mrc/protos/architect";
import { messageTypeRegistry, UnknownMessage } from "@mrc/proto/typeRegistry";

export function generateId(max = 4294967295): string {
   return Math.floor(Math.random() * max).toString();
}

const resourceIdCounts = new Map<string, number>();

export function generateResourceId(resourceType: string): string {
   const id = resourceIdCounts.get(resourceType) ?? 0;

   resourceIdCounts.set(resourceType, id + 1);

   return id.toString();
}

export function yield_(name = "") {
   // console.log(`Yield start. Name: ${name}`);

   // Yield uses setImmediate to allow other promises to finish before setTimeout is run
   // return new Promise((r) => setImmediate(r));
   return Promise.resolve().then(() => {
      // console.log(`Yield done. Name: ${name}`);
   });
}

export function yield_immediate(name = "") {
   // console.log(`Yield_immediate start. Name: ${name}`);

   // Yield uses setImmediate to allow other promises to finish before setTimeout is run
   // return new Promise((r) => setImmediate(r));
   return new Promise((r) => setImmediate(r)).then(() => {
      // console.log(`Yield_immediate done. Name: ${name}`);
   });
}

export function yield_timeout(name = "") {
   // console.log(`yield_timeout start. Name: ${name}`);

   // Yield uses setImmediate to allow other promises to finish before setTimeout is run
   // return new Promise((r) => setImmediate(r));
   return new Promise((r) => setTimeout(r)).then(() => {
      // console.log(`yield_timeout done. Name: ${name}`);
   });
}

export function sleep(ms: number, name = "") {
   // console.log(`Sleep start. Name: ${name}`);

   // Stronger than yield_. Forces full loop
   return new Promise((r) => setTimeout(r, ms)).then(() => {
      // console.log(`Sleep done. Name: ${name}`);
   });
}

export function stringToBytes(value: string[]): Uint8Array[];
export function stringToBytes(value: string): Uint8Array;
export function stringToBytes(value: string | string[]) {
   if (value instanceof Array) {
      return value.map((s) => new TextEncoder().encode(s));
   }

   return new TextEncoder().encode(value);
}

export function bytesToString(value: Uint8Array[]): string[];
export function bytesToString(value: Uint8Array): string;
export function bytesToString(value: Uint8Array | Uint8Array[]) {
   if (value instanceof Array) {
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

   const decoded = message_type.decode(message.value) as MessageT;

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

export function hashName16(name: string) {
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
   return Number(BigInt.asUintN(16, hash_u32));
}

// Generates a 64 bit address from 4 16 bit parts. The parts are shifted into place. Part 1 is the most significant and
// part 4 is the least significant.
export function generateAddress64(part1: number, part2: number, part3: number, part4: number): string {
   const address = (BigInt(part1) << 48n) | (BigInt(part2) << 32n) | (BigInt(part3) << 16n) | BigInt(part4);

   return address.toString();
}

// Generates a 64 bit address from 2 16 bit parts. The parts are shifted into place. Part 1 is the most significant and
// part 2 is the least significant.
export function generateAddress32(part1: number, part2: number): number {
   const address = (part1 << 16) | part2;

   return address;
}

export function generateExecutorAddress(executorId: number): number {
   return generateAddress32(0, executorId);
}

export function generatePartitionAddress(partitionId: number): number {
   return generateAddress32(0, partitionId);
}

export function generatePipelineAddress(executorId: number, pipelineId: number): number {
   return generateAddress32(executorId, pipelineId);
}

export function generateSegmentAddress(executorId: number, pipelineId: number, segmentHash: number, segmentId: number) {
   return generateAddress64(executorId, pipelineId, segmentHash, segmentId);
}

export function generateManifoldAddress(executorId: number, pipelineId: number, portHash: number, manifoldId: number) {
   return generateAddress64(executorId, pipelineId, portHash, manifoldId);
}

export function generatePortAddress(executorId: number, pipelineId: number, segmentId: number, portId: number) {
   return generateAddress64(executorId, pipelineId, segmentId, portId);
}

export function generateSegmentHash(seg_name: string, worker_id: string): number {
   const name_hash = BigInt(hashName16(seg_name));
   const worker_hash = BigInt(hashName16(worker_id));

   // Shift the name over 16
   return Number((name_hash << 16n) | worker_hash);
}

export function sortObjectByKeys(x: any): any {
   if (typeof x !== "object" || !x) {
      return x;
   }
   if (Array.isArray(x)) {
      return x.map(sortObjectByKeys);
   }
   return Object.keys(x)
      .sort()
      .reduce((o, k) => ({ ...o, [k]: sortObjectByKeys(x[k]) }), {});
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

   const sorted_data = sortObjectByKeys(data);

   message_type.encode(sorted_data, buffer);

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

/**
 * Utility function for throwing errors in null coalescing operators. i.e. `some_value ?? throwExpression("Error
 * message")`
 * @date 6/9/2023 - 5:16:28 PM
 *
 * @export
 * @param {string} errorMessage The message to pass into the Error constructor
 * @returns {never}
 */
export function throwExpression(errorMessage: string): never {
   throw new Error(errorMessage);
}

export function compareDifference<T>(currSet: Array<T>, newSet: Array<T>): [Array<T>, Array<T>] {
   const toRemove = currSet.filter((i) => !newSet.includes(i));

   const toAdd = newSet.filter((i) => !currSet.includes(i));

   return [toAdd, toRemove];
}
