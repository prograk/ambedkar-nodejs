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
      "crypto": "crypto-browserify",
      "stream": "stream-browserify",
      "buffer": "buffer",
      "util": "util",
      "path": "path-browserify",
      "fs": false,
      "os": "os-browserify",
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