<script setup>
import { computed, ref, watch } from 'vue'
import data from '../assets/scaling-results-data.json'
import MathTex from './MathTex.vue'

const selected = ref(0), modelId = ref('all')
const inspected = ref(null), pinned = ref(false)
const panels = [
  { label: 'Model size', kind: 'params', symbol: 'N', axis: 'Model parameters', takeaway: 'Analog loss improves with model size at a rate close to the matched digital baseline, with a persistent loss gap.' },
  { label: 'Token trajectory', kind: 'tokens', symbol: 'D', axis: 'Processed tokens', takeaway: 'Raw evaluations show learning over each token schedule. The overlaid fits describe the five model endpoints under the Chinchilla allocation rule.' },
  { label: 'Compute trajectory', kind: 'compute', symbol: 'C', axis: 'Algorithmic training compute (PF-days)', takeaway: 'Loss decreases with the shared full-model compute proxy in both paths. The overlaid fits describe the five model endpoints.' },
]
const panel = computed(() => panels[selected.value])
const fitFormula = profile => String.raw`L\propto ${panel.value.symbol}^{-${data.fits[profile][panel.value.kind].exponent.toFixed(3)}}`
const trajectory = computed(() => panel.value.kind !== 'params')
const models = computed(() => trajectory.value && modelId.value !== 'all' ? data.models.filter(m => m.id === modelId.value) : data.models)
const profiles = [{ id: 'digital', label: 'Digital AdamW', color: '#1764aa' }, { id: 'analog', label: 'MP Analog Adam', color: '#b31b1b' }]
const colors = { tiny: '#7853a2', small: '#31688e', medium: '#26865e', large: '#a97b00', xlarge: '#c55415' }
const format = value => value >= 1e9 ? `${+(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${+(value / 1e6).toFixed(2)}M` : value >= 1e3 ? `${+(value / 1e3).toFixed(1)}k` : +value.toPrecision(3)
const coordinate = (model, run, index) => panel.value.kind === 'params' ? model.params
  : Math.max(run.steps[index], 1) * data.tokensPerStep * (panel.value.kind === 'compute' ? model.computePerToken : 1)
const series = computed(() => models.value.flatMap(model => profiles.map(profile => {
  const run = model.runs[profile.id]
  const indexes = trajectory.value ? run.steps.map((_, i) => i) : [run.steps.length - 1]
  return { model, profile, color: trajectory.value ? colors[model.id] : profile.color,
    points: indexes.map(i => ({ model, profile, step: run.steps[i], loss: run.losses[i],
      tokens: run.steps[i] * data.tokensPerStep, x: coordinate(model, run, i), endpoint: i === run.steps.length - 1 })) }
})))
// Fits always describe all five model endpoints, even when a trajectory is filtered.
const fitBounds = computed(() => {
  const values = data.models.flatMap(model => profiles.map(profile => {
    const run = model.runs[profile.id]
    return coordinate(model, run, run.steps.length - 1)
  }))
  return { lo: Math.min(...values), hi: Math.max(...values) }
})
const bounds = computed(() => {
  const starts = data.models.flatMap(model => profiles.map(profile => {
    const run = model.runs[profile.id]
    return coordinate(model, run, trajectory.value ? 0 : run.steps.length - 1)
  }))
  return { lo: Math.min(...starts) / 1.15, hi: fitBounds.value.hi * 10 * 1.15 }
})
const x = value => 67 + (Math.log10(value) - Math.log10(bounds.value.lo)) / (Math.log10(bounds.value.hi) - Math.log10(bounds.value.lo)) * 613
const fitLoss = (profile, value) => {
  const fit = data.fits[profile][panel.value.kind]
  return fit.coefficient * value ** -fit.exponent
}
// Include the fitted range, upper extrapolation, and every recorded loss without changing axes on filtering.
const lossBounds = computed(() => {
  const values = data.models.flatMap(model => profiles.flatMap(profile => {
    const losses = model.runs[profile.id].losses
    return trajectory.value ? losses : [losses[losses.length - 1]]
  }))
  for (const profile of profiles) values.push(fitLoss(profile.id, fitBounds.value.lo), fitLoss(profile.id, fitBounds.value.hi * 10))
  return { lo: Math.floor(Math.min(...values)), hi: Math.ceil(Math.max(...values) * 1.05) }
})
const y = value => 306 - (value - lossBounds.value.lo) / (lossBounds.value.hi - lossBounds.value.lo) * 274
const yTicks = computed(() => {
  const { lo, hi } = lossBounds.value
  const step = [1, 2, 5, 10].find(value => value >= (hi - lo) / 6) ?? 10
  return Array.from({ length: Math.floor(hi / step) - Math.ceil(lo / step) + 1 }, (_, i) => (Math.ceil(lo / step) + i) * step)
})
const xTicks = computed(() => {
  const ticks = []
  for (let power = Math.floor(Math.log10(bounds.value.lo)); power <= Math.ceil(Math.log10(bounds.value.hi)); power++) {
    for (const factor of [1, 2, 5]) {
      const value = factor * 10 ** power
      if (value >= bounds.value.lo && value <= bounds.value.hi) ticks.push(value)
    }
  }
  // Preserve readable labels for the wide trajectory ranges.
  return ticks.filter((_, i) => ticks.length <= 7 || i % 2 === 0)
})
const path = values => values.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.loss)}`).join(' ')
function fitPath(profile) {
  const lo = fitBounds.value.lo, hi = fitBounds.value.hi * 10
  return path(Array.from({ length: 160 }, (_, i) => {
    const value = lo * (hi / lo) ** (i / 159)
    return { x: value, loss: fitLoss(profile, value) }
  }))
}
function inspect(point, pin = false) {
  if (!pinned.value || pin) inspected.value = point
  if (pin) pinned.value = true
}
function clear() { inspected.value = null; pinned.value = false }
watch([selected, modelId], clear)
function navigate(event) {
  let next = selected.value
  if (event.key === 'ArrowRight') next = (next + 1) % panels.length
  else if (event.key === 'ArrowLeft') next = (next + panels.length - 1) % panels.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = panels.length - 1
  else return
  event.preventDefault()
  selected.value = next
  event.currentTarget.querySelectorAll('[role="tab"]')[next].focus()
}
</script>

<template>
  <div class="scaling-results" @keydown.esc="clear">
    <div class="evidence-tabs scaling-tabs" role="tablist" aria-label="Scaling figure panels" @keydown="navigate">
      <button v-for="(item, i) in panels" :id="`results-tab-${i}`" :key="item.label" type="button" role="tab" :aria-selected="selected === i" :tabindex="selected === i ? 0 : -1" aria-controls="results-panel" @click="selected = i"><span>{{ String.fromCharCode(97 + i) }}</span>{{ item.label }}</button>
    </div>
    <div id="results-panel" role="tabpanel" :aria-labelledby="`results-tab-${selected}`">
      <div class="results-controls">
        <div class="plot-legend"><span v-for="profile in profiles" :key="profile.id"><i :style="{ background: profile.color, color: profile.color }" :class="{ 'dotted-key': profile.id === 'analog' }"></i>{{ profile.label }} · {{ profile.id === 'analog' ? 'dotted' : 'solid' }}</span></div>
        <label v-if="trajectory">Model <select v-model="modelId" aria-label="Trajectory model"><option value="all">All five scales</option><option v-for="model in data.models" :key="model.id" :value="model.id">{{ model.label }}</option></select></label>
      </div>
      <div class="results-plot-scroll">
        <svg class="results-plot" viewBox="0 0 710 370" role="group" :aria-label="`${panel.label}: recorded analog and digital validation losses; logarithmic horizontal axis`">
          <defs><clipPath id="results-plot-clip"><rect x="67" y="25" width="613" height="281" /></clipPath></defs>
          <rect class="fit-extrapolation-region" :x="x(fitBounds.hi)" y="25" :width="680 - x(fitBounds.hi)" height="281" fill="var(--soft)"/>
          <g v-for="tick in yTicks" :key="`y-${tick}`"><line x1="67" :y1="y(tick)" x2="680" :y2="y(tick)" class="results-grid"/><text x="55" :y="y(tick) + 4" text-anchor="end">{{ tick }}</text></g>
          <g v-for="tick in xTicks" :key="`x-${tick}`"><line :x1="x(tick)" y1="25" :x2="x(tick)" y2="306" class="results-grid"/><text :x="x(tick)" y="329" text-anchor="middle">{{ format(tick) }}</text></g>
          <path d="M67 25 V306 H680" fill="none" stroke="var(--muted)" />
          <text x="374" y="361" text-anchor="middle">{{ panel.axis }} · log scale</text>
          <text x="18" y="166" transform="rotate(-90 18 166)" text-anchor="middle">Validation loss ↓</text>
          <g clip-path="url(#results-plot-clip)">
            <template v-if="trajectory"><path v-for="line in series" :key="`${line.model.id}-${line.profile.id}`" :d="path(line.points)" fill="none" :stroke="line.color" stroke-width="1.8" stroke-linecap="round" :stroke-dasharray="line.profile.id === 'analog' ? '1 4' : undefined" /></template>
            <g v-for="profile in profiles" :key="profile.id" class="endpoint-fit">
              <path :d="fitPath(profile.id)" fill="none" stroke="white" stroke-width="6"/>
              <path class="fit-line" :d="fitPath(profile.id)" fill="none" :stroke="profile.color" stroke-width="3" stroke-linecap="round" :stroke-dasharray="profile.id === 'analog' ? '1 6' : undefined"/>
            </g>
            <g v-for="line in series" :key="`${line.model.id}-${line.profile.id}`">
              <g v-for="point in line.points" :key="point.step" class="result-point" role="button" tabindex="0" :aria-label="`${point.profile.label}, ${point.model.label}, step ${point.step}, validation loss ${point.loss.toFixed(4)}`" :aria-pressed="pinned && inspected === point" @mouseenter="inspect(point)" @focus="inspect(point)" @click="inspect(point, true)" @keydown.enter.prevent="inspect(point, true)" @keydown.space.prevent="inspect(point, true)">
                <circle :cx="x(point.x)" :cy="y(point.loss)" :r="trajectory ? 6 : 12" fill="transparent" />
                <circle v-if="inspected === point" :cx="x(point.x)" :cy="y(point.loss)" r="9" :stroke="line.color" fill="white" stroke-width="2"/>
                <rect v-if="point.profile.id === 'analog' && point.endpoint" :x="x(point.x)-4" :y="y(point.loss)-4" width="8" height="8" :fill="line.color"/>
                <circle v-else :cx="x(point.x)" :cy="y(point.loss)" :r="point.endpoint ? 4 : 2.6" :fill="line.color"/>
              </g>
            </g>
          </g>
        </svg>
      </div>
      <div v-if="trajectory" class="plot-legend model-legend"><span v-for="model in models" :key="model.id"><i :style="{ background: colors[model.id] }"></i>{{ model.label }}</span></div>
      <div class="fit-summary"><span v-for="profile in profiles" :key="profile.id" :style="{ color: profile.color }">{{ profile.label }} fit: <MathTex :tex="fitFormula(profile.id)" /></span><span>{{ trajectory ? 'Thick blue/red lines: fits to all five model endpoints · larger markers: recorded endpoints' : 'Markers: recorded endpoints · lines: endpoint fits' }} · shaded region: extrapolation to 10× the largest endpoint</span></div>
      <div class="result-inspection" aria-live="polite" aria-atomic="true">
        <template v-if="inspected"><div class="inspection-head"><strong>{{ inspected.profile.label }} · {{ inspected.model.label }}</strong><button v-if="pinned" type="button" @click="clear">Clear selection</button><span v-else>Click to pin</span></div><dl><div><dt>Validation loss</dt><dd>{{ inspected.loss.toFixed(4) }}</dd></div><div><dt>Evaluation step</dt><dd>{{ inspected.step.toLocaleString('en-US') }}</dd></div><div><dt>Processed tokens</dt><dd>{{ inspected.tokens.toLocaleString('en-US') }}</dd></div><div><dt>Compute (PF-days)</dt><dd>{{ (inspected.tokens * inspected.model.computePerToken).toPrecision(5) }}</dd></div></dl></template>
        <p v-else>Hover or focus a recorded point to inspect its values. Click, tap, or press Enter to pin; Escape clears the selection.</p>
      </div>
      <p class="results-takeaway"><strong>Takeaway.</strong> {{ panel.takeaway }}</p>
      <p class="demo-caption">One seed per condition; power-law fits use all five model endpoints. {{ trajectory ? 'Thin lines connect raw evaluations. Step-zero evaluations are placed at one update’s coordinate for the logarithmic axis; their recorded token count remains zero.' : '' }} Extrapolations are projections beyond the evaluated range. <a href="assets/scaling-results.png">View all five panels</a>.</p>
    </div>
  </div>
</template>
