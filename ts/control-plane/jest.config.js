const inspector = require('inspector');
const { pathsToModuleNameMapper } = require('ts-jest');
const requireJSON5 = require("require-json5");

const { compilerOptions } = requireJSON5("./tsconfig");

function isDebuggerAttached() {
  return inspector.url() !== undefined;
}

const timeout = isDebuggerAttached()
  ? 600000 // 10 minutes to debug
  : 5000; // the default 5s for jest

if (isDebuggerAttached()) {
  console.log(`Detected attached debugger. Setting Jest timeout to ${timeout}`);
}

/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testTimeout: timeout,
  modulePathIgnorePatterns: ["<rootDir>/dist/"],
  testPathIgnorePatterns: ["<rootDir>/dist/"],
  roots: ['<rootDir>'],
  modulePaths: [compilerOptions.baseUrl],
  moduleNameMapper: pathsToModuleNameMapper(compilerOptions.paths, { prefix: `<rootDir>/${compilerOptions.baseUrl}` }),
  transform: {"^.+\\.tsx?$": "@swc/jest"},
};
