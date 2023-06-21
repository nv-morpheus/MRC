import { createSlice, PayloadAction, prepareAutoBatched } from "@reduxjs/toolkit";

export interface ISystem {
   requestRunning: boolean;
   requestRunningName: string | null;
   requestRunningNonce: number;
}

export const systemSlice = createSlice({
   name: "system",
   initialState: { requestRunning: false, requestRunningName: null, requestRunningNonce: 0 } as ISystem,
   reducers: {
      startRequest: (state, action: PayloadAction<string>) => {
         if (state.requestRunning) {
            console.warn("Request is already running. Did you forget to call endRequest?");
         }

         state.requestRunning = true;
         state.requestRunningName = action.payload;
         state.requestRunningNonce++;
      },
      stopRequest: (state, action: PayloadAction<string>) => {
         if (!state.requestRunning) {
            console.warn("No request is running. Did you forget to call startRequest?");
         }

         if (state.requestRunningName !== action.payload) {
            console.warn("Call to startRequest did not match with call to stopRequest");
         }

         state.requestRunning = false;
         state.requestRunningName = null;
      },
   },
});

export const { startRequest: systemStartRequest, stopRequest: systemStopRequest } = systemSlice.actions;

export function systemConfigureSlice() {
   return systemSlice.reducer;
}
