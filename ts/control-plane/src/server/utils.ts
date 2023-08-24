import { createDraftSafeSelector, createEntityAdapter, EntityState, Selector } from "@reduxjs/toolkit";
import {
   Dictionary,
   EntityAdapter,
   EntityId,
   EntitySelectors,
   PreventAny,
} from "@reduxjs/toolkit/dist/entities/models";

type createEntityAdapterParameters<T> = Parameters<typeof createEntityAdapter<T>>;

export interface WrappedEntitySelectors<T, V> extends EntitySelectors<T, V> {
   selectByIds: (state: V, id: EntityId[]) => T[];
}

export interface WrappedEntityAdapter<T> extends EntityAdapter<T> {
   getAll<S extends EntityState<T>>(state: PreventAny<S, T>): T[];
   getOne<S extends EntityState<T>>(state: PreventAny<S, T>, id: EntityId): T | undefined;
   getMany<S extends EntityState<T>>(state: PreventAny<S, T>, ids: EntityId[]): T[];
   getSelectors(): WrappedEntitySelectors<T, EntityState<T>>;
   getSelectors<V>(selectState: (state: V) => EntityState<T>): WrappedEntitySelectors<T, V>;
}

export function createWrappedEntityAdapter<T>(...args: createEntityAdapterParameters<T>): WrappedEntityAdapter<T> {
   const inner_adapter = createEntityAdapter<T>(...args);

   function getSelectors(): WrappedEntitySelectors<T, EntityState<T>>;
   function getSelectors<V>(selectState: (state: V) => EntityState<T>): WrappedEntitySelectors<T, V>;
   function getSelectors<V>(selectState?: (state: V) => EntityState<T>): WrappedEntitySelectors<T, any> {
      const selectEntities = (state: EntityState<T>) => state.entities;

      const selectIds = (_: unknown, ids: EntityId[]) => ids;

      const selectByIds = (entities: Dictionary<T>, ids: EntityId[]) => ids.map((id) => entities[id]!);

      if (!selectState) {
         return {
            ...inner_adapter.getSelectors(),
            selectByIds: createDraftSafeSelector(selectEntities, selectIds, selectByIds),
         };
      }

      const selectGlobalizedEntities = createDraftSafeSelector(
         selectState as Selector<V, EntityState<T>>,
         selectEntities
      );

      return {
         ...inner_adapter.getSelectors(selectState),
         selectByIds: createDraftSafeSelector(selectGlobalizedEntities, selectIds, selectByIds),
      };
   }

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
         const matched_entities = ids
            .map((id) => {
               return id in state.entities ? state.entities[id] : undefined;
            })
            .filter((x): x is T => x != null);

         return matched_entities;
      },
      getSelectors,
   };
}
