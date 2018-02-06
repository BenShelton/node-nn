const fs = require('fs')
const csv = require('csv-parse/lib/sync')

const { NeuralNetwork } = require('../lib')

const bot = new NeuralNetwork({
  layerSizes: [784, 20, 10],
  batchSize: 1,
  learningRate: 0.2,
  testFn: (outputs, targets) => targets.highest() === outputs.highest(),
  logTests: false,
  trackTrainingSuccess: true
})
// const bot = NeuralNetwork.fromJSON(fs.readFileSync('examples/trained/mnist-bot.json'))

console.log('Loading Data')
const trainingData = csv(fs.readFileSync('examples/mnist-data/mnist_train_1k.csv'))
const testingData = csv(fs.readFileSync('examples/mnist-data/mnist_test_1k.csv'))

for (let epoch = 0; epoch < 100; epoch++) {
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
console.log('Exporting bot to examples/mnistbot.json')
fs.writeFileSync('examples/mnistbot.json', bot.toJSON())
console.log('Complete!')

function train () {
  let lines = 0
  trainingData.sort(() => Math.random() - 1)
  for (const line of trainingData) {
    const answer = new Array(10).fill(0)
    answer[+line[0]] = 1
    bot.train(line.slice(1), answer)
    lines++
    if (lines % 1000 === 0) console.log(`${lines} lines trained`)
  }
}

function test () {
  for (const line of testingData) {
    const answer = new Array(10).fill(0)
    answer[+line[0]] = 1
    bot.test(line.slice(1), answer)
  }
}
