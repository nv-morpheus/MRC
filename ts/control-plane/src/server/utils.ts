import { createEntityAdapter, EntityState } from "@reduxjs/toolkit";
import { EntityId, PreventAny, EntityAdapter } from "@reduxjs/toolkit/dist/entities/models";


type createEntityAdapterParameters<T> = Parameters<typeof createEntityAdapter<T>>;

export interface WrappedEntityAdapter<T> extends EntityAdapter<T> {
   getAll<S extends EntityState<T>>(state: PreventAny<S, T>): T[];
   getOne<S extends EntityState<T>>(state: PreventAny<S, T>, id: EntityId): T | undefined;
   getMany<S extends EntityState<T>>(state: PreventAny<S, T>, ids: EntityId[]): T[];
}

export function createWrappedEntityAdapter<T>(...args: createEntityAdapterParameters<T>): WrappedEntityAdapter<T> {

   const inner_adapter = createEntityAdapter<T>(...args);

   return {
      ...inner_adapter,
      getAll(state) {
         return state.ids.map((id) => state.entities[id]!);
      },
      getOne: (state, id) => {
         if (id in state.entities) {
            return state.entities[id];
         }

         return undefined;
      },
      getMany: (state, ids) => {

         const matched_entities = ids.map((id) => {
            return id in state.entities ? state.entities[id] : undefined;
         }).filter((x): x is T => x != null);

         return matched_entities;
      }
   };
}
