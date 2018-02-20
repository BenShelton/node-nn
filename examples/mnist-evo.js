const fs = require('fs')
const csv = require('csv-parse/lib/sync')

const { Matrix, NeuralNetwork, NeuroEvolution } = require('../lib')

const template = new NeuralNetwork({
  layerSizes: [784, 20, 10],
  // batchSize: 1,
  learningRate: 1,
  testFn: (outputs, targets) => targets.highest() === outputs.highest()
})

const fitnessFn = bot => {
  trainingData.sort(() => Math.random() - 1)
  for (const line of trainingData) {
    bot.train(line.inputs, line.targets)
  }
  let error = 0
  for (const line of testingData) {
    error += bot.test(line.inputs, line.targets) ? 0 : 1
  }
  return (testingData.length - error) * (100 / testingData.length)
}

const botnet = new NeuroEvolution({
  size: 10,
  killRate: 0.5,
  mutationRate: 0.1,
  mutationPower: 0.05,
  template,
  fitnessFn,
  logStats: true
})

console.log('Loading Data')
const trainingData = csv(fs.readFileSync('examples/mnist-data/mnist_train_1k.csv')).map(parseLine)
const testingData = csv(fs.readFileSync('examples/mnist-data/mnist_test_100.csv')).map(parseLine)
console.log('Data Loaded')

botnet.runGenerations(10)
  .then(() => botnet.best())
  .then(best => {
    console.log('Exporting bot to examples/trained/mnist-evo-bot.json')
    fs.writeFileSync('examples/trained/mnist-evo-bot.json', best.toJSON())
    console.log('Complete!')
  })

function parseLine (line) {
  const answer = new Array(10).fill(0)
  answer[+line[0]] = 1
  return { inputs: Matrix.fromArray(line.slice(1)), targets: Matrix.fromArray(answer) }
}
