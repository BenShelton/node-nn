const { describe, beforeEach, it } = require('mocha')
const assert = require('assert')

const { Matrix } = require('../lib')

describe('Matrix', () => {
  let m
  beforeEach(done => {
    m = new Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    done()
  })
  describe('Initialization', () => {
    it('creates a matrix of correct dimensions', done => {
      assert.equal(m.rows, 2)
      assert.equal(m.cols, 3)
      assert.equal(m.data.length, 2)
      assert.equal(m.data[0].length, 3)
      done()
    })
    it('creates a random matrix by default', done => {
      const randomM = new Matrix(2, 2)
      assert.notDeepEqual(randomM.data, [[0, 0], [0, 0]])
      done()
    })
    it('creates a matrix from passed in data', done => {
      assert.deepEqual(m.data, [[1, 2, 3], [4, 5, 6]])
      done()
    })
    it('throws an error when creating a matrix from passed in data mismatches row/cols', done => {
      assert.throws(() => new Matrix(5, 5, m.data), /mismatch/)
      done()
    })
  })
  describe('Methods', () => {
    it('can apply a function to each element', done => {
      m.map(val => 9)
      assert.deepEqual(m.data, [[9, 9, 9], [9, 9, 9]])
      done()
    })
    it('can subtract a number from each element', done => {
      m.subtract(2)
      assert.deepEqual(m.data, [[-1, 0, 1], [2, 3, 4]])
      done()
    })
    it('can multiply each element by a number', done => {
      m.multiply(2)
      assert.deepEqual(m.data, [[2, 4, 6], [8, 10, 12]])
      done()
    })
    it('can add another matrix', done => {
      const otherM = new Matrix(2, 3, [[2, 2, 2], [4, 4, 4]])
      m.addMatrix(otherM)
      assert.deepEqual(m.data, [[3, 4, 5], [8, 9, 10]])
      done()
    })
    it('throws an error when adding a matrix of mismatched size', done => {
      const otherM = new Matrix(2, 4, [[2, 2, 2, 2], [4, 4, 4, 4]])
      assert.throws(() => m.addMatrix(otherM), /mismatch/)
      done()
    })
  })
  describe('Static Methods', () => {
    it('can copy a matrix', done => {
      const copyM = Matrix.copy(m)
      assert(copyM instanceof Matrix)
      assert.notEqual(m, copyM)
      assert.deepEqual(copyM.data, [[1, 2, 3], [4, 5, 6]])
      done()
    })
    it('can transpose a matrix', done => {
      const transposeM = Matrix.transpose(m)
      assert(transposeM instanceof Matrix)
      assert.notEqual(m, transposeM)
      assert.deepEqual(transposeM.data, [[1, 4], [2, 5], [3, 6]])
      done()
    })
    it('can calculate the dot product of 2 matrices', done => {
      const otherM = new Matrix(3, 2, [[7, 8], [9, 10], [11, 12]])
      const dotM = Matrix.dot(m, otherM)
      assert(dotM instanceof Matrix)
      assert.notEqual(m, dotM)
      assert.notEqual(otherM, dotM)
      assert.deepEqual(dotM.data, [[58, 64], [139, 154]])
      done()
    })
    it('throws an error when dot product matrices have mismatched sizes', done => {
      const otherM = new Matrix(1, 1)
      assert.throws(() => Matrix.dot(m, otherM), /mismatch/)
      done()
    })
    it('can calculate the difference between 2 matrices', done => {
      const firstM = new Matrix(3, 1, [[1], [3], [5]])
      const otherM = new Matrix(3, 1, [[0], [1], [2]])
      const differenceM = Matrix.difference(firstM, otherM)
      assert(differenceM instanceof Matrix)
      assert.notEqual(firstM, differenceM)
      assert.notEqual(otherM, differenceM)
      assert.deepEqual(differenceM.data, [[1], [2], [3]])
      done()
    })
    it('can create a 2D matrix from an array', done => {
      const arr = [1, 2, 3]
      const arrayM = Matrix.fromArray(arr)
      assert(arrayM instanceof Matrix)
      assert.equal(arrayM.rows, 3)
      assert.equal(arrayM.cols, 1)
      assert.equal(arrayM.data.length, 3)
      assert.equal(arrayM.data[0].length, 1)
      assert.deepEqual(arrayM.data, [[1], [2], [3]])
      done()
    })
    it('can find the index of the highest number', done => {
      const highest = Matrix.fromArray([0, 0.1, 0.999, 0.3, 0.5, 0.01]).highest()
      assert.equal(highest, 2)
      done()
    })
  })
})
