const Layer = require('./Layer')
const Matrix = require('./Matrix')

module.exports = class NeuralNetwork {
  constructor ({
    layerSizes = [],
    layers = null,
    learningRate = 0.1,
    batchSize = 1,
    activation = 'sigmoid',
    testFn = (outputs, targets) => Matrix.perceptron(outputs).compareMatrix(targets),
    logTests = false,
    trackTrainingSuccess = false
  }) {
    if (layers && layers.some(layer => !(layer instanceof Layer))) throw new Error('NeuralNetwork Constructor - Layers should be passed as Layer objects, did you mean layerSizes?')
    if (layerSizes.length < 2) throw new Error('NeuralNetwork Constructor - At least 2 layerSizes are required')
    this.layerSizes = layerSizes
    this.layers = layers || layerSizes.reduce((acc, nodes, index) => index ? [...acc, new Layer(nodes, layerSizes[index - 1])] : acc, [])
    this.learningRate = learningRate
    this.batchSize = batchSize
    switch (activation) {
      case 'sigmoid':
      default:
        this.activation = x => 1 / (1 + Math.exp(-x))
        this.derivative = x => x * (1 - x)
    }
    this.testFn = testFn
    this.logTests = logTests
    this.trackTrainingSuccess = trackTrainingSuccess

    this._batchCounter = 0
    this._tests = 0
    this._corrects = 0
  }

  static fromJSON (json) {
    const settings = JSON.parse(json)
    settings.layers.forEach((v, i, a) => { a[i] = Layer.fromJSON(v) })
    return new NeuralNetwork(settings)
  }

  static fromTemplate (template) {
    if (!(template instanceof NeuralNetwork)) throw new Error('NeuralNetwork FromTemplate - Template must be a NeuralNetwork')
    return new NeuralNetwork({
      layerSizes: template.layerSizes,
      learningRate: template.learningRate,
      batchSize: template.batchSize,
      activation: template.activation,
      testFn: template.testFn,
      logTests: template.logTests,
      trackTrainingSuccess: template.trackTrainingSuccess
    })
  }

  static crossover (a, b) {
    if (!(a instanceof NeuralNetwork) || !(b instanceof NeuralNetwork)) throw new Error('NeuralNetwork Crossover - 2 NeuralNetworks must be provided')
    const layers = []
    a.layers.forEach((layer, i) => {
      layers.push(Layer.fromMatrices(
        // Matrix.copy(layer.weights).map((v, wi) => Math.random() < 0.5 ? v : b.layers[i].weights.data[wi][0]),
        // Matrix.copy(layer.bias)
        Matrix.average([layer.weights, b.layers[i].weights]),
        Matrix.copy(layer.bias)
        // Matrix.average([layer.bias, b.layers[i].bias])
      ))
    })
    return new NeuralNetwork({
      layers,
      layerSizes: a.layerSizes,
      learningRate: a.learningRate,
      batchSize: a.batchSize,
      activation: a.activation,
      testFn: a.testFn,
      logTests: a.logTests,
      trackTrainingSuccess: a.trackTrainingSuccess
    })
  }

  toJSON () {
    return JSON.stringify({
      layers: this.layers.map(({ weights, bias }) => ({ weights, bias })),
      learningRate: this.learningRate,
      batchSize: this.batchSize,
      activation: this.activation,
      testFn: this.testFn,
      logTests: this.logTests,
      trackTrainingSuccess: this.trackTrainingSuccess
    })
  }

  train (inputs, targets) {
    if (!(inputs instanceof Matrix)) inputs = Matrix.fromArray(inputs)
    if (!(targets instanceof Matrix)) targets = Matrix.fromArray(targets)
    let activation = inputs
    for (const layer of this.layers) {
      layer.activation = activation
      activation = Matrix
        .dot(layer.weights, activation)
        .addMatrix(layer.bias)
        .map(this.activation)
    }

    if (this.trackTrainingSuccess) {
      const correct = this.testFn(activation, targets)
      this._tests++
      if (correct) this._corrects++
    }

    let outputs = activation
    let errors = Matrix.difference(targets, outputs)
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const gradients = outputs
        .map(this.derivative)
        .multiplyMatrix(errors)
        .multiply(this.learningRate)
      const deltas = Matrix.dot(gradients, Matrix.transpose(layer.activation))
      layer.weights.addMatrix(deltas)
      layer.bias.addMatrix(gradients)
      errors = Matrix.dot(Matrix.transpose(layer.weights), errors)
      outputs = layer.activation
    }
  }

  feedForward (inputs) {
    if (!(inputs instanceof Matrix)) inputs = Matrix.fromArray(inputs)
    let activation = inputs
    for (const layer of this.layers) {
      activation = Matrix
        .dot(layer.weights, activation)
        .addMatrix(layer.bias)
        .map(this.activation)
    }
    return activation
  }

  test (inputs, targets) {
    if (!(inputs instanceof Matrix)) inputs = Matrix.fromArray(inputs)
    if (!(targets instanceof Matrix)) targets = Matrix.fromArray(targets)
    const outputs = this.feedForward(inputs)
    const correct = this.testFn(outputs, targets)
    if (this.logTests) {
      console.log(correct ? 'Correct' : 'Wrong')
      console.log('Outputs: ', outputs.data)
      console.log('Targets: ', targets.data)
      console.log('-----------------')
    }
    this._tests++
    if (correct) this._corrects++
    return correct
  }

  successRate (persist = false) {
    const rate = (this._corrects / this._tests) * 100
    if (!persist) {
      this._tests = 0
      this._corrects = 0
    }
    return rate
  }
}
