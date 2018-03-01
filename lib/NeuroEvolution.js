const NeuralNetwork = require('./NeuralNetwork')

module.exports = class NeuroEvolution {
  constructor ({
    size = 100,
    killRate = 0.5,
    mutationRate = 0.5,
    mutationPower = 0.1,
    template,
    fitnessFn,
    genFn,
    logStats = false
  }) {
    if (!template || !(template instanceof NeuralNetwork)) throw new Error('NeuroEvolution Constructor - Template NeuralNetwork is required')
    if (!fitnessFn || !(fitnessFn instanceof Function)) throw new Error('NeuroEvolution Constructor - FitnessFn Function is required')
    this.size = size
    this.killRate = killRate
    this.mutationRate = mutationRate
    this.mutationPower = mutationPower
    this.generation = [...Array(size)]
      .map(() => ({
        fitness: 0,
        network: NeuralNetwork.fromTemplate(template)
      }))
    this.fitnessFn = fitnessFn
    this.genFn = genFn
    this.logStats = logStats
    this._generationNum = 0
    this._fitnessRange = 0
    this._fitnessAverage = 0
    this._fitnessMin = 0
    this._fitnessMax = 0
    this._best = null
  }

  async runGenerations (num = 5) {
    await this.calculateFitness()
    this.kill()
    this.crossover()
    if (this.logStats) {
      console.log('Generation: ' + this._generationNum)
      console.log('Average: ' + this._fitnessAverage)
      console.log('Min: ' + this._fitnessMin)
      console.log('Max: ' + this._fitnessMax)
      console.log('----------------')
    }
    if (this.genFn) {
      this.genFn({
        best: this._best,
        generation: this._generationNum,
        average: this._fitnessAverage,
        min: this._fitnessMin,
        max: this._fitnessMax
      })
    }
    return --num > 0 ? this.runGenerations(num) : true
  }

  async calculateFitness () {
    if (this._fitnessRange) return
    this._fitnessRange = 0
    this._fitnessMin = Infinity
    this._fitnessMax = -Infinity
    await Promise.all(this.generation.map(async child => {
      const fitness = await this.fitnessFn(child.network)
      child.fitness = fitness
      return fitness
    })).then(fitnesses => {
      fitnesses.forEach((fitness, i) => {
        this._fitnessRange += fitness
        this._fitnessMin = Math.min(this._fitnessMin, fitness)
        this._fitnessMax = Math.max(this._fitnessMax, fitness)
        if (fitness === this._fitnessMax) this._best = this.generation[i].network
      })
      this._fitnessAverage = this._fitnessRange / this.size
    })
  }

  kill () {
    if (this._fitnessRange === 0) throw new Error('NeuroEvolution Kill - Fitnesses not calculated, await calculateFitness() first')
    this.generation.sort((a, b) => a.fitness - b.fitness)
    this.generation.splice(0, Math.floor((this.killRate / 1) * this.size))
    this._fitnessRange = this.generation.reduce((a, v) => a + v.fitness, 0)
  }

  crossover () {
    if (this._fitnessRange === 0) throw new Error('NeuroEvolution Crossover - Fitnesses not calculated, await calculateFitness() first')
    const newGeneration = []
    for (let i = this.generation.length; i < this.size; i++) {
      const [a, b] = this.chooseParents()
      const child = { fitness: 0, network: NeuralNetwork.crossover(a.network, b.network) }
      if (Math.random() < this.mutationRate) child.network.layers.forEach(l => l.mutate(this.mutationPower))
      newGeneration.push(child)
    }
    this.generation = newGeneration.concat(this.generation)
    this._fitnessRange = 0
    this._generationNum++
  }

  chooseParents (attempts = 0) {
    if (attempts > 1000) throw new Error('NeuroEvolution ChooseParent - Too many attempts to find parents')
    if (this.generation.length < 2) throw new Error('NeuroEvolution ChooseParent - Not enough parents left in generation')
    const choiceA = Math.random() * this._fitnessRange
    const choiceB = Math.random() * this._fitnessRange
    let a, b
    let sum = 0
    for (const parent of this.generation) {
      sum += parent.fitness
      if (!a && sum >= choiceA) a = parent
      if (!b && sum >= choiceB) b = parent
      if (a === b) return this.chooseParents(attempts++)
      if (a && b) return [a, b]
    }
    throw new Error('NeuroEvolution ChooseParent - Parents could not be selected from fitness range')
  }

  async best () {
    if (!this._best) {
      await this.calculateFitness()
      this._best = this.generation.find(b => b.fitness === this._fitnessMax).network
    }
    return this._best
  }
}
