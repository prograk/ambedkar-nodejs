const path = require('path');
const webpack = require('webpack');

module.exports = {
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: (modulePath) => {
          return /node_modules/.test(modulePath) &&
                 !/node_modules\/(@xenova|@huggingface|@langchain|@qdrant)/.test(modulePath);
        },
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
            plugins: [
              '@babel/plugin-proposal-optional-chaining',
              '@babel/plugin-proposal-nullish-coalescing-operator',
            ],
          },
        },
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.mjs'],
    fallback: {
      "crypto": require.resolve("crypto-browserify"),
      "stream": require.resolve("stream-browserify"),
      "buffer": require.resolve("buffer"),
      "util": require.resolve("util"),
      "path": require.resolve("path-browserify"),
      "fs": false,
      "os": require.resolve("os-browserify/browser"),
    },
    alias: {
      "node:crypto": "crypto",
      "node:stream": "stream",
      "node:buffer": "buffer",
      "node:util": "util",
      "node:path": "path",
      "node:os": "os",
      "node:fs": false,
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
      process: 'process/browser',
    }),
  ],
};