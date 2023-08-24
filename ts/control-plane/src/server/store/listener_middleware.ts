import { AppDispatch, RootState } from "@mrc/server/store/store";
import { ListenerEffectAPI, addListener, createListenerMiddleware } from "@reduxjs/toolkit";

import type { TypedStartListening, TypedAddListener } from "@reduxjs/toolkit";

export const listenerMiddleware = createListenerMiddleware<RootState>();

export type AppStartListening = TypedStartListening<RootState, AppDispatch>;
export type AppListenerAPI = ListenerEffectAPI<RootState, AppDispatch>;

export const startAppListening = listenerMiddleware.startListening as AppStartListening;

export const addAppListener = addListener as TypedAddListener<RootState, AppDispatch>;
