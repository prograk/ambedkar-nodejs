const path = require('path');

module.exports = {
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.m?js$/,
        // Only transpile @xenova and @huggingface modules (modern ESM)
        include: [
          path.resolve(__dirname, 'functions'),
          path.resolve(__dirname, 'node_modules/@xenova'),
          path.resolve(__dirname, 'node_modules/@huggingface'),
        ],
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', {
                targets: {
                  node: '18' // Match your Netlify runtime or even 16
                },
                useBuiltIns: 'usage',
                corejs: 3,
              }]
            ],
            plugins: ['@babel/plugin-proposal-optional-chaining', '@babel/plugin-proposal-nullish-coalescing-operator'],
          },
        },
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.mjs'],
  },
};
