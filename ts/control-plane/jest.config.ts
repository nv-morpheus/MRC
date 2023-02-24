import type {JestConfigWithTsJest} from "ts-jest";
import {pathsToModuleNameMapper} from "ts-jest";
import * as inspector from "inspector";
// import requireJSON5 from "require-json5";
import {compilerOptions} from "./test.json";

// const tsconfig = requireJSON5("./tsconfig.json");

function isDebuggerAttached()
{
   return inspector.url() !== undefined;
}

const timeout = isDebuggerAttached() ? 600000  // 10 minutes to debug
                                     : 5000;   // the default 5s for jest

if (isDebuggerAttached())
{
   console.log(`Detected attached debugger. Setting Jest timeout to ${timeout}`);
}

const jestConfig: JestConfigWithTsJest = {
   // [...]
   // Replace `ts-jest` with the preset you want to use
   // from the above list
   preset: "ts-jest",
   testEnvironment: "node",
   testTimeout: timeout,
   modulePaths: [compilerOptions.baseUrl],  // <-- This will be set to 'baseUrl' value
   // moduleNameMapper: pathsToModuleNameMapper(compilerOptions.paths /*, { prefix: '<rootDir>/' } */),
   modulePathIgnorePatterns: ["<rootDir>/dist/"],
   // reporters: [
   //    "default",
   //    "<rootDir>/test-reporter/index.js",
   //    [
   //       "@jest-performance-reporter/core",
   //       {
   //          "errorAfterMs": 1000,
   //          "warnAfterMs": 500,
   //          "logLevel": "warn",
   //          "maxItems": 5,
   //          "jsonReportPath": "performance-report.json",
   //          "csvReportPath": "performance-report.csv",
   //       },
   //    ],
   // ],
   // transform: {"^.+\\.(t|j)sx?$": "@swc/jest"},
}

export default jestConfig
