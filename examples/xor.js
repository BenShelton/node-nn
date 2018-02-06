const { NeuralNetwork } = require('../lib')

const bot = new NeuralNetwork({
  layerSizes: [2, 2, 1],
  learningRate: 0.2,
  logTests: true
})
// const bot = NeuralNetwork.fromJSON(require('fs').readFileSync('examples/trained/xor-bot.json'))

const testData = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] }
]

for (let epochs = 0; epochs < 10; epochs++) {
  for (let i = 0; i < 10000; i++) {
    const index = Math.floor(Math.random() * testData.length)
    const data = testData[index]
    bot.train(data.inputs, data.targets)
  }
  for (const data of testData) {
    bot.test(data.inputs, data.targets)
  }
}

console.log(`Success Rate: ${bot.successRate()}%`)
