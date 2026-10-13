// Linear response with a balance point spanning the full response-zero range.
// balancePosition is rho in [0,1]: w^diamond = (2*rho-1)/beta for beta>0.
// Store physical weights directly. The response zeros define the device bounds.
export function responseFactors(weight, balancePosition, beta) {
  return {
    up: 2 * balancePosition * (1 - beta * weight),
    down: 2 * (1 - balancePosition) * (1 + beta * weight),
  }
}

export function pulseResponse(weight, direction, states, balancePosition, beta) {
  const factors = responseFactors(weight, balancePosition, beta)
  const response = direction > 0 ? factors.up : factors.down
  return weight + direction * (2 / states) * response
}

export function responseBounds(beta) {
  return beta > 0
    ? { min: -1 / beta, max: 1 / beta }
    : { min: -Infinity, max: Infinity }
}

export function responseBalance(balancePosition, beta) {
  return beta > 0 ? (2 * balancePosition - 1) / beta : null
}

export function initialPulseWeight(beta) {
  if (beta === 0) return -0.8 // No finite bounds in the constant-response case.
  const { min, max } = responseBounds(beta)
  return min + 0.1 * (max - min)
}
