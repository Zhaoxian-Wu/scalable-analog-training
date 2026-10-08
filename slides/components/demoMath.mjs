// Deterministic teaching examples, separate from manuscript experiment data.
export function residualTrace(increment, quantum, signed = false, steps = 40) {
  const ideal = [0], direct = [0], programmed = [0], residuals = [0], counts = [0]
  let residual = 0
  for (let t = 0; t < steps; t++) {
    const update = signed
      ? increment * (t % 12 < 7 ? 1 : -0.8)
      : increment * (1 + 0.2 * Math.sin(t * 0.6))
    ideal.push(ideal[t] + update)
    direct.push(direct[t] + Math.trunc(update / quantum) * quantum)
    residual += update
    const pulses = Math.trunc(residual / quantum)
    programmed.push(programmed[t] + pulses * quantum)
    residual -= pulses * quantum
    residuals.push(residual)
    counts.push(pulses)
  }
  return { ideal, direct, programmed, residuals, counts }
}

export const converterSignal = Array.from({ length: 56 }, (_, i) =>
  0.52 * Math.sin(i * 0.71) + 0.23 * Math.cos(i * 1.31) +
  (i === 17 ? 1.45 : i === 39 ? -1.3 : 0))

export function quantize(values, rail, bits) {
  const intervals = 2 ** bits - 1
  const delta = 2 * rail / intervals
  const roundEven = value => {
    const lower = Math.floor(value)
    return value - lower === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(value)
  }
  const reconstructed = values.map(x =>
    -rail + delta * Math.max(0, Math.min(intervals, roundEven((x + rail) / delta))))
  const mse = values.reduce((sum, x, i) => sum + (x - reconstructed[i]) ** 2, 0) / values.length
  const power = values.reduce((sum, x) => sum + x * x, 0) / values.length
  return { reconstructed, clipped: values.filter(x => Math.abs(x) > rail).length, nmse: mse / power, delta }
}

export function mappingStats(width, policy) {
  const target = 1 / Math.sqrt(width)
  const physical = policy === 'native' ? 1 / Math.sqrt(2 * Math.log(width * width)) : 0.25
  const logical = policy === 'fixed' ? physical : target
  return { physical, logical, scale: logical / physical, target }
}
