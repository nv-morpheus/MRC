/** @type {import('ts-jest').JestConfigWithTsJest} */
const inspector = require('inspector');

function isDebuggerAttached() {
  return inspector.url() !== undefined;
}

const timeout = isDebuggerAttached()
  ? 600000 // 10 minutes to debug
  : 5000; // the default 5s for jest

if (isDebuggerAttached()) {
  console.log(`Detected attached debugger. Setting Jest timeout to ${timeout}`);
}

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testTimeout: timeout,
  modulePathIgnorePatterns: ["<rootDir>/dist/"]
};
