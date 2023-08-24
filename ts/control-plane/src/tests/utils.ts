export function expectDefined<T>(obj: T | undefined): obj is T {
   expect(obj).toBeDefined();

   return obj !== undefined;
}
