const Matrix = require('./Matrix')

module.exports = class Layer {
  constructor (nodes, prevNodes) {
    this.weights = new Matrix(nodes, prevNodes)
    this.bias = new Matrix(nodes, 1)
    this.activation = null
  }

  static fromJSON (json) {
    const layer = new Layer(json.weights.rows, json.weights.cols, json.weights, json.bias)
    layer.weights.data = json.weights.data
    layer.bias.data = json.bias.data
    return layer
  }

  static fromMatrices (weights, bias) {
    const layer = new Layer(1, 1)
    layer.weights = weights
    layer.bias = bias
    return layer
  }

  mutate (power) {
    this.weights.map(v => Math.random() < power ? Matrix._random() : v)
    this.bias.map(v => Math.random() < power ? Matrix._random() : v)
  }
}
