const { describe, beforeEach, it } = require('mocha')
const assert = require('assert')

const { NeuralNetwork, NeuroEvolution } = require('../lib')

describe('NeuroEvolution', () => {
  const template = new NeuralNetwork({
    layerSizes: [2, 4, 1]
  })
  const fitnessFn = () => 1
  let NE
  beforeEach(() => {
    NE = new NeuroEvolution({
      size: 10,
      killRate: 0.6,
      mutationRate: 0.1,
      mutationPower: 0.1,
      template,
      fitnessFn
    })
  })
  describe('Initialization', () => {
    it('requires a template & fitness function', () => {
      assert.throws(() => new NeuroEvolution({ fitnessFn, template: undefined }), /required/)
      assert.throws(() => new NeuroEvolution({ template, fitnessFn: undefined }), /required/)
    })
  })
  describe('Generation', () => {
    it('allows async fitness functions', done => {
      NE.fitnessFn = () => new Promise(resolve => setTimeout(resolve, 1, 1))
      NE.runGenerations(5)
        .then(() => done())
        .catch(done)
    })
    it('can return the best child of a generation', done => {
      NE.best()
        .then(best => {
          assert(best.toJSON())
          done()
        })
        .catch(done)
    })
  })
})
