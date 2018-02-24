const { describe, beforeEach, it } = require('mocha')
const assert = require('assert')

const { Matrix, NeuralNetwork } = require('../lib')

describe('NeuralNetwork', () => {
  describe('Training', () => {
    let nn
    beforeEach(() => {
      nn = new NeuralNetwork({
        layerSizes: [5, 3, 2]
      })
    })
    it('backpropagates correctly', () => {
      const nn = NeuralNetwork.fromJSON(JSON.stringify({
        layers: [
          {
            weights: { rows: 2, cols: 2, data: [[0.1, 0.2], [0.3, 0.4]] },
            bias: { rows: 2, cols: 1, data: [[0.5], [0.6]] }
          },
          {
            weights: { rows: 1, cols: 2, data: [[0.7, 0.8]] },
            bias: { rows: 1, cols: 1, data: [[0.9]] }
          }
        ],
        layerSizes: [2, 2, 1],
        activation: 'sigmoid',
        learningRate: 0.1,
        batchSize: 1
      }))
      nn.train([0, 0], [0])
      nn.train([0, 1], [1])
      nn.train([1, 0], [1])
      nn.train([1, 1], [0])
      assert.deepEqual(nn.layers[0].weights.data, [[0.08906864769787945, 0.1889549321126967], [0.29040985295725386, 0.3902676566761416]])
      assert.deepEqual(nn.layers[0].bias.data, [[0.4769405828828057], [0.5767147790962412]])
      assert.deepEqual(nn.layers[1].weights.data, [[0.6891883851420223, 0.7882486417061167]])
      assert.deepEqual(nn.layers[1].bias.data, [[0.8834728616674726]])
    })
    it('does not mutate passed in Matrices', () => {
      const inputs = Matrix.fromArray([1, 2, 3, 4, 5])
      const targets = Matrix.fromArray([6, 7])
      nn.train(inputs, targets)
      assert.deepEqual(inputs.data, [[1], [2], [3], [4], [5]])
      assert.deepEqual(targets.data, [[6], [7]])
    })
  })
})
