const path = require("path");

// Webpack plugins
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const ForkTsCheckerPlugin = require("fork-ts-checker-webpack-plugin");
const NodemonPlugin = require("nodemon-webpack-plugin");
const TsconfigPathsPlugin = require("tsconfig-paths-webpack-plugin");

const { NODE_ENV = "production" } = process.env;

module.exports = {
   entry: {
      index: "./src/server/run_server.ts",
      // run_server: './src/server/run_server.ts',
   },
   mode: NODE_ENV,
   target: "node",
   devtool: "inline-source-map",
   output: {
      path: path.resolve(__dirname, "build"),
      filename: "[name].bundle.js",
   },
   resolve: {
      extensions: [".ts", ".js"],
      extensionAlias: {
         ".js": [".js", ".ts"],
         ".cjs": [".cjs", ".cts"],
         ".mjs": [".mjs", ".mts"],
      },
      plugins: [new TsconfigPathsPlugin()],
   },
   externals: [
      {
         "utf-8-validate": "commonjs utf-8-validate",
         "bufferutil": "commonjs bufferutil",
         "sqlite3": "commonjs sqlite3",
         "knex": "commonjs knex",
         "@redux-devtools/cli": "import @redux-devtools/cli",
      },
   ],
   module: getLoaders(),
   plugins: getPlugins(),
   //    optimization: {
   //       // chunkIds: "named",
   //       moduleIds: "deterministic",
   //       splitChunks: {
   //           chunks: "all",
   //       },
   //   }
};

function getLoaders() {
   return {
      rules: [
         {
            test: /\.([cm]?ts|tsx)$/,
            // test: /\.[cm]?ts$/,
            // exclude: [/node_modules/],
            loader: "ts-loader",
         },
         // {
         //    test: /\.ts$/,
         //    exclude: [/node_modules/],
         //    use: [
         //       'ts-loader',
         //    ]
         // }
      ],
   };
}

function getPlugins() {
   return [
      // Clean the output directory
      new CleanWebpackPlugin(),
      new ForkTsCheckerPlugin(),
      new NodemonPlugin({
         nodeArgs: ["--inspect"],
      }),
   ];
}
