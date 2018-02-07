# node-nn
A simple neural network to run with Node.js.

[![Codeship  Status for BenShelton/node-nn](https://app.codeship.com/projects/de87bd70-ee4b-0135-80c9-76500b92b2bd/status?branch=master)](https://app.codeship.com/projects/270896)

## Why does this exist

In order to learn some of basic concepts of Artificial Neural Networks I decided to build my own one from scratch.

This isn't meant to be a highly efficient replacement for other neural networks, rather a demonstration of the core principles involved.

## What features does it have

- Setup Multi-layer Networks
- Testing via Feed Forward
- Training via Back Propagation
- Built in Matrix math
- NeuroEvolution (genetic algorithm to evaluate & breed groups of networks)
- Zero dependency

## How do I use it

Simple! Just clone the repo:

``` bash
git clone https://github.com/BenShelton/node-nn.git
```

There are some examples setup for you to try using `npm run` or `yarn`:

``` bash
yarn xor
yarn mnist
yarn evolution
```

If you want to create your own, then you can import the necessary files from the library:

``` javascript
const { NeuralNetwork, NeuroEvolution } = require('./lib')
```

## I see room for improvement

Feel free to raise an issue or create a pull request. Contributions are always welcome!

## TODO

- [ ] Create documentation for creating your own
- [ ] Update examples
- [ ] Go through & comment on each stage of code
- [ ] Add testing for NeuralNetwork, Layer & NeuroEvolution
