const fs = require('fs')
const csv = require('csv-parse/lib/sync')

const { Matrix, NeuralNetwork } = require('../lib')

const bot = new NeuralNetwork({
  layerSizes: [784, 64, 10],
  batchSize: 1,
  learningRate: 0.1,
  testFn: (outputs, targets) => outputs.highest() === targets.highest(),
  trackTrainingSuccess: true
})

console.log('Loading Data')
const trainingData = csv(fs.readFileSync('examples/mnist-data/mnist_train_1k.csv')).map(parseLine)
const testingData = csv(fs.readFileSync('examples/mnist-data/mnist_test_100.csv')).map(parseLine)

for (let epoch = 0; epoch < 30; epoch++) {
  console.log('Starting Training - Epoch ' + epoch)
  train()
  const endingMessage = bot.trackTrainingSuccess
    ? `Training Complete, Accuracy: ${bot.successRate()}%`
    : 'Training Complete'
  console.log(endingMessage)
  console.log('Starting Testing')
  test()
  console.log(`Testing Complete, Accuracy: ${bot.successRate()}%`)
  console.log('-----------------')
}
console.log('Exporting bot to examples/trained/mnist-bot.json')
fs.writeFileSync('examples/trained/mnist-bot.json', bot.toJSON())
console.log('Complete!')

function parseLine (line) {
  const answer = new Array(10).fill(0)
  answer[+line[0]] = 1
  return { inputs: Matrix.fromArray(line.slice(1)), targets: Matrix.fromArray(answer) }
}

function train () {
  let lines = 0
  for (const line of trainingData) {
    bot.train(line.inputs, line.targets)
    if (++lines % 1000 === 0) console.log(`${lines} lines trained`)
  }
}

function test () {
  for (const line of testingData) {
    bot.test(line.inputs, line.targets)
  }
}
