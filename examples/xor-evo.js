const fs = require('fs')
const { Matrix, NeuralNetwork, NeuroEvolution } = require('../lib')

const template = new NeuralNetwork({
  layerSizes: [2, 2, 1],
  learningRate: 0.2,
  testFn: (outputs, targets) => Matrix.difference(outputs, targets).map(Math.abs).data[0][0]
})

const fitnessFn = bot => {
  for (let i = 0; i < 100; i++) {
    const data = testData[Math.floor(Math.random() * testData.length)]
    bot.train(data.inputs, data.targets)
  }
  let error = 0
  for (const data of testData) {
    error += bot.test(data.inputs, data.targets)
  }
  return (4 - error) * 25
}

const botnet = new NeuroEvolution({
  size: 250,
  killRate: 0.8,
  mutationRate: 0.05,
  mutationPower: 0.02,
  template,
  fitnessFn,
  logStats: true
})

const testData = [
  { inputs: Matrix.fromArray([0, 0]), targets: Matrix.fromArray([0]) },
  { inputs: Matrix.fromArray([0, 1]), targets: Matrix.fromArray([1]) },
  { inputs: Matrix.fromArray([1, 0]), targets: Matrix.fromArray([1]) },
  { inputs: Matrix.fromArray([1, 1]), targets: Matrix.fromArray([0]) }
]

for (let generation = 0; generation < 100; generation++) {
  botnet.runGeneration()
}

fs.writeFileSync('examples/trained/xor-evo-bot.json', botnet.best().toJSON())
