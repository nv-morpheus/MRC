import {createSlice, PayloadAction, prepareAutoBatched} from "@reduxjs/toolkit";

export interface ISystem {
   requestRunning: boolean;
}

export const systemSlice = createSlice({
   name: "system",
   initialState: {requestRunning: false} as ISystem,
   reducers: {
      startRequest: {
         reducer: (state, action: PayloadAction<void>) => {
            if (state.requestRunning)
            {
               console.warn("Request is already running. Did you forget to call endRequest?");
            }

            state.requestRunning = true;
         },
         prepare: prepareAutoBatched<void>(),
      },
      stopRequest: {
         reducer: (state, action: PayloadAction<void>) => {
            if (!state.requestRunning)
            {
               console.warn("No request is running. Did you forget to call startRequest?");
            }

            state.requestRunning = false;
         },
         prepare: prepareAutoBatched<void>(),
      },
   },
});

export const {startRequest: systemStartRequest, stopRequest: systemStopRequest} = systemSlice.actions;

export default systemSlice.reducer;
