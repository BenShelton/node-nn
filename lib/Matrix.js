module.exports = class Matrix {
  constructor (rows = 1, cols = 1, data) {
    if (data && (data.length !== rows || data[0].length !== cols)) {
      throw new Error('Matrix Constructor - Data size mismatch')
    }
    this.rows = rows
    this.cols = cols
    this.data = data ||
      [...new Array(rows)].map(() => [...new Array(cols)].map(Matrix._random))
  }

  static fromArray (arr) {
    return new Matrix(arr.length, 1, arr.map(v => [+v]))
  }

  static copy (m) {
    return new Matrix(m.rows, m.cols, [...m.data.map(v => [...v])])
  }

  static dot (a, b) {
    if (a.cols !== b.rows) throw new Error('Matrix Dot - Size mismatch')
    const data = [...new Array(a.rows)]
      .map((av, ai) => [...new Array(b.cols)]
        .map((bv, bi) => a.data[ai].reduce((sum, cv, ci) => sum + cv * b.data[ci][bi], 0))
      )
    return new Matrix(data.length, data[0].length, data)
  }

  static difference (a, b) {
    if (a.cols > 1 || b.cols > 1) throw new Error('Matrix Difference - Only 2D Matrices can be used')
    if (a.rows !== b.rows) throw new Error('Matrix Difference - Size mismatch')
    return Matrix.copy(a).map((val, row, col) => val - b.data[row][col])
  }

  static average (matrices) {
    const sum = Matrix.copy(matrices.shift())
    for (const matrix of matrices) {
      sum.addMatrix(matrix)
    }
    return sum.divide(matrices.length + 1)
  }

  static transpose (m) {
    const data = [...new Array(m.cols)]
      .map((rv, ri) => [...new Array(m.rows)]
        .map((cv, ci) => m.data[ci][ri])
      )
    return new Matrix(m.cols, m.rows, data)
  }

  static perceptron (m) {
    return Matrix.copy(m).map(Math.round)
  }

  static _random () {
    return Math.random() * 2 - 1
  }

  map (fn) {
    this.data.forEach((row, rowIndex) => {
      row.forEach((col, colIndex) => {
        row[colIndex] = fn(row[colIndex], rowIndex, colIndex)
      })
    })
    return this
  }

  compareMatrix (m) {
    this._checkSizes(m, 'CompareMatrix')
    let same = true
    this.data.forEach((row, rowIndex) => {
      row.forEach((col, colIndex) => {
        if (same && row[colIndex] !== m.data[rowIndex][colIndex]) same = false
      })
    })
    return same
  }

  addMatrix (m) {
    this._checkSizes(m, 'AddMatrix')
    return this.map((val, row, col) => val + m.data[row][col])
  }

  subtractMatrix (m) {
    this._checkSizes(m, 'SubtractMatrix')
    return this.map((val, row, col) => val - m.data[row][col])
  }

  multiplyMatrix (m) {
    this._checkSizes(m, 'MultiplyMatrix')
    return this.map((val, row, col) => val * m.data[row][col])
  }

  subtract (num) {
    return this.map(val => val - num)
  }

  multiply (num) {
    return this.map(val => val * num)
  }

  divide (num) {
    if (num === 0) throw new Error('Matrix Divide - Cannot divide by 0')
    return this.map(val => num !== 0 ? val / num : num)
  }

  highest () {
    if (this.cols > 1) throw new Error('Matrix Highest - Only 2D Matrices can be used')
    return this.data.reduce((max, v, i, a) => v[0] > a[max][0] ? i : max, 0)
  }

  _checkSizes (m, method) {
    if (this.rows !== m.rows || this.cols !== m.cols) {
      // console.log(`Rows: ${this.rows} - ${m.rows}`)
      // console.log(`Cols: ${this.cols} - ${m.cols}`)
      throw new Error(`Matrix ${method} - Size mismatch`)
    }
  }
}
