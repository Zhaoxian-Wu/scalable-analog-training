// Synthetic, zero-mean population models with unit population variance.
// A fixed seed preserves the vector when rail or precision changes; changing
// dimension extends/truncates that same sample rather than resampling it.
export function sampleVector(distribution, size, seed = 1337) {
  let state = seed >>> 0
  const uniform = () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0
    return (state + 0.5) / 4294967296
  }
  return Array.from({ length: size }, () => {
    const u = uniform(), v = uniform()
    return distribution === 'gaussian'
      ? Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
      : -Math.sign(u - 0.5) * Math.log(1 - 2 * Math.abs(u - 0.5)) / Math.sqrt(2)
  })
}

export const quantizationIntervals = bits => 2 ** bits - 1

export function quantize(values, rail, bits) {
  const intervals = quantizationIntervals(bits)
  const delta = 2 * rail / intervals
  const roundEven = value => {
    const lower = Math.floor(value)
    return value - lower === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(value)
  }
  const reconstructed = values.map(x =>
    -rail + delta * Math.max(0, Math.min(intervals, roundEven((x + rail) / delta))))
  const mse = values.reduce((sum, x, i) => sum + (x - reconstructed[i]) ** 2, 0) / values.length
  const power = values.reduce((sum, x) => sum + x * x, 0) / values.length
  return { reconstructed, clipped: values.filter(x => Math.abs(x) > rail).length, mse, nmse: power ? mse / power : 0, delta }
}

// Global optimum for this vector and the symmetric endpoint quantizer, not the
// distribution-averaged norm–rail optimum analyzed in the manuscript. Between
// assignment boundaries the squared error is P - 2BC + AC². Sweep all boundaries
// and minimize that quadratic on each interval, including the unbounded tail.
export function optimalRail(values, bits) {
  const intervals = quantizationIntervals(bits)
  const magnitudes = values.map(Math.abs)
  if (magnitudes.every(x => x === 0)) return 1
  const events = []
  let a = 0, b = 0, power = 0
  for (const value of magnitudes) {
    a += value ? 1 : 1 / intervals ** 2
    b += value
    power += value ** 2
    if (!value) continue
    for (let r = 1; r <= (intervals - 1) / 2; r++) {
      events.push({ rail: value * intervals / (2 * r), value, r })
    }
  }
  events.sort((left, right) => left.rail - right.rail)
  let lower = 0, bestError = Infinity, bestRail = Math.max(...magnitudes)
  const consider = upper => {
    const rail = Math.max(lower, Math.min(upper, b / a))
    const error = power - 2 * b * rail + a * rail ** 2
    if (error < bestError) { bestError = error; bestRail = rail }
  }
  for (const event of events) {
    consider(event.rail)
    const before = (2 * event.r + 1) / intervals
    const after = (2 * event.r - 1) / intervals
    a += after ** 2 - before ** 2
    b += event.value * (after - before)
    lower = event.rail
  }
  consider(Infinity)
  return bestRail
}

// ACIQ population-error approximation, adapted to the endpoint quantizer above.
// https://github.com/submission2019/cnn-quantization/blob/master/mse_analysis.py
// Its K=2^bits-1 intervals give Delta=2C/K and rounding MSE Delta^2/12.
// Gaussian: (σ²+C²)erfc(C/(√2σ)) - √(2/π)Cσ exp(-C²/(2σ²))
//           + C²/(3K²)
// Laplace: 2b² exp(-C/b) + C²/(3K²).
// Solve their derivatives for the dimensionless multiplier by bisection.
function erfc(value) {
  const t = 1 / (1 + 0.5 * value)
  return t * Math.exp(-value * value - 1.26551223 + t * (1.00002368 + t *
    (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t *
    (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t *
    (-0.82215223 + t * 0.17087277)))))))))
}
export function aciqMultiplier(distribution, bits) {
  const denominator = 3 * quantizationIntervals(bits) ** 2
  const derivative = distribution === 'gaussian'
    ? t => t * erfc(t / Math.sqrt(2)) - Math.sqrt(2 / Math.PI) * Math.exp(-t * t / 2) + t / denominator
    : t => t / denominator - Math.exp(-t)
  let lower = 0, upper = 30
  for (let i = 0; i < 60; i++) {
    const midpoint = (lower + upper) / 2
    if (derivative(midpoint) > 0) upper = midpoint
    else lower = midpoint
  }
  return (lower + upper) / 2
}
export function aciqEstimate(values, bits, distribution) {
  // Fit a centered population model without changing the vector being quantized.
  const scale = distribution === 'gaussian'
    ? Math.sqrt(values.reduce((sum, x) => sum + x * x, 0) / values.length)
    : values.reduce((sum, x) => sum + Math.abs(x), 0) / values.length
  const multiplier = aciqMultiplier(distribution, bits)
  return { scale, multiplier, intervals: quantizationIntervals(bits), rail: Math.max(Number.EPSILON, multiplier * scale) }
}

export function aciqRail(values, bits, distribution) {
  return aciqEstimate(values, bits, distribution).rail
}
