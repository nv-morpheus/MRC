const tsNode = require("ts-node");
const requireJSON5 = require("require-json5")

tsNode.register({
  transpileOnly: true,
  compilerOptions: requireJSON5("/work/ts/control-plane/tsconfig.json").compilerOptions,
});

module.exports = require("./test-reporter");
