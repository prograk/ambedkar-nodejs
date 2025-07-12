const path = require('path');

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
  },
};
