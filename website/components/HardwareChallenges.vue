<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import katex from 'katex'
import ConverterPlayground from './ConverterPlayground.vue'
import MathTex from './MathTex.vue'
import { pulseResponse, responseBounds, responseFactors, responseBalance, initialPulseWeight } from './pulse-response.js'

const math = tex => katex.renderToString(tex, { throwOnError: true })
const weightMaxLabel = math(String.raw`w_{\max}`)
const weightMinLabel = math(String.raw`w_{\min}`)
const symmetricWeightLabel = math(String.raw`w^\diamond`)
const panels = [{ id: 'converter', label: 'DAC / ADC quantization' }, { id: 'update', label: 'Pulse update' }]
const tabButtons = ref([])

const activePanel = ref('update')

const stateCount = ref(32)
const symmetricPosition = ref(0.5)
const symmetricWeight = computed(() => responseBalance(symmetricPosition.value, beta.value))
const beta = ref(1)
const animationMode = ref('sweep')
const storedWeight = ref(-0.8)
const pulseTrajectory = ref([{ pulses: 0, weight: -0.8, direction: 0 }])
const animationRunning = ref(false)
let animationTimer = null
let animationRunId = 0

const animationPulseCount = computed(() => 4 * stateCount.value)
const animationHalfPulseCount = computed(() => animationPulseCount.value / 2)
const animationSequence = computed(() => animationMode.value === 'sweep'
  ? [...Array(animationHalfPulseCount.value).fill(1), ...Array(animationHalfPulseCount.value).fill(-1)]
  : Array.from({ length: animationPulseCount.value }, (_, index) => index % 2 === 0 ? 1 : -1))
const animationCycleMs = computed(() => 2200 + 650 * Math.log2(stateCount.value))
const pulseIntervalMs = computed(() => Math.max(18, animationCycleMs.value / animationSequence.value.length))
const pulsePlotMax = computed(() => Math.max(animationSequence.value.length, pulseTrajectory.value.length - 1))
const initialWeight = computed(() => initialPulseWeight(beta.value))
const deviceBounds = computed(() => responseBounds(beta.value))
const pulsePlotBounds = computed(() => {
  if (beta.value > 0) return deviceBounds.value
  // Constant responses have no finite zeros. Fit the entire sequence so the
  // linear-case axis remains stable throughout animation and manual stepping.
  let weight = initialWeight.value
  const weights = [weight, ...pulseTrajectory.value.map(point => point.weight)]
  for (const direction of animationSequence.value) {
    weight = pulseResponse(weight, direction, stateCount.value, symmetricPosition.value, beta.value)
    weights.push(weight)
  }
  const low = Math.min(...weights)
  const high = Math.max(...weights)
  const padding = Math.max(0.1, (high - low) * 0.08)
  return { min: low - padding, max: high + padding }
})
const pulsePlotPoints = computed(() => pulseTrajectory.value.map(point => ({
  ...point,
  x: 68 + (point.pulses / pulsePlotMax.value) * 324,
  y: plotY(point.weight),
})))
const pulsePlotPath = computed(() => pulsePlotPoints.value.map(point => `${point.x},${point.y}`).join(' '))
const pulseSwitchX = computed(() => 68 + (animationHalfPulseCount.value / pulsePlotMax.value) * 324)
const symmetricPointY = computed(() => plotY(symmetricWeight.value))

function plotY(weight) {
  const { min, max } = pulsePlotBounds.value
  return 150 - ((weight - min) / (max - min)) * 112
}

const currentResponse = computed(() => responseFactors(storedWeight.value, symmetricPosition.value, beta.value))
const balanceDescription = computed(() => {
  if (beta.value === 0) return symmetricPosition.value === 0.5
    ? 'Constant equal responses; no unique symmetric point.'
    : 'Constant unequal responses; no finite symmetric point. The slider sets directional gain bias.'
  if (symmetricPosition.value === 0) return 'The symmetric point is the lower bound; positive pulses have zero gain.'
  if (symmetricPosition.value === 1) return 'The symmetric point is the upper bound; negative pulses have zero gain.'
  return 'Equal up/down response at the selected position within the weight bounds.'
})
const responseKind = computed(() => beta.value === 0 ? 'Constant response' : 'Linear state-dependent response')
const modeTakeaway = computed(() => {
  if (beta.value === 0) return symmetricPosition.value === 0.5
    ? 'Constant equal responses give straight trajectories; equal opposite pulses cancel exactly.'
    : 'Constant unequal responses give different slopes; balanced pulse signs produce directional drift.'
  if (symmetricPosition.value === 0) return 'The symmetric point is the lower bound. Positive pulses have no response; negative pulses move the weight toward the lower bound.'
  if (symmetricPosition.value === 1) return 'The symmetric point is the upper bound. Negative pulses have no response; positive pulses move the weight toward the upper bound.'
  return animationMode.value === 'sweep'
    ? 'State-dependent responses produce curved trajectories. Reversing pulse direction moves the weight away from saturation.'
    : 'Balanced update noise pushes the weight toward the symmetric point'
})
const responseEquations = [
  String.raw`q_+(w)=2\rho(1-\beta w),\quad q_-(w)=2(1-\rho)(1+\beta w)`,
  String.raw`U_q(w,\Delta w):=\begin{cases}w+\Delta w\cdot q_+(w),&\Delta w\geq 0,\\w+\Delta w\cdot q_-(w),&\Delta w<0.\end{cases}`,
  String.raw`w_{\min}=-1/\beta,\quad w_{\max}=1/\beta,\quad w^\diamond=\frac{2\rho-1}{\beta}\quad(\beta>0)`,
  String.raw`w_0=w_{\min}+0.1(w_{\max}-w_{\min})`,
].map(math)

function pulseWeightUpdate(weight, direction) {
  return pulseResponse(weight, direction, stateCount.value, symmetricPosition.value, beta.value)
}

function selectPanel(panel) {
  activePanel.value = panel
}

async function navigateTabs(event, index) {
  const offsets = { ArrowLeft: -1, ArrowRight: 1, Home: -index, End: panels.length - 1 - index }
  if (!(event.key in offsets)) return
  event.preventDefault()
  const next = (index + offsets[event.key] + panels.length) % panels.length
  selectPanel(panels[next].id)
  await nextTick()
  tabButtons.value[next]?.focus()
}

function applyPulse(direction) {
  const nextWeight = pulseWeightUpdate(storedWeight.value, direction)
  storedWeight.value = nextWeight
  pulseTrajectory.value = [...pulseTrajectory.value, { pulses: pulseTrajectory.value.length, weight: nextWeight, direction }]
}

function stopPulseAnimation() {
  animationRunId += 1
  animationRunning.value = false
  if (animationTimer !== null) {
    window.clearTimeout(animationTimer)
    animationTimer = null
  }
}

function startPulseAnimation() {
  stopPulseAnimation()
  const runId = animationRunId
  let pulseNumber = 0
  resetWeight()
  animationRunning.value = true

  const tick = () => {
    if (runId !== animationRunId || activePanel.value !== 'update') return
    if (pulseNumber < animationSequence.value.length) {
      const direction = animationSequence.value[pulseNumber]
      applyPulse(direction)
      pulseNumber += 1
      animationTimer = window.setTimeout(tick, pulseIntervalMs.value)
      return
    }

    animationTimer = window.setTimeout(() => {
      if (runId !== animationRunId || activePanel.value !== 'update') return
      animationTimer = window.setTimeout(() => {
        if (runId === animationRunId && activePanel.value === 'update') startPulseAnimation()
      }, 500)
    }, 900)
  }

  animationTimer = window.setTimeout(tick, 450)
}

function fireManualPulse(direction) {
  stopPulseAnimation()
  applyPulse(direction)
}

function resetManualSequence() {
  stopPulseAnimation()
  resetWeight()
}

function resetWeight() {
  storedWeight.value = initialWeight.value
  pulseTrajectory.value = [{ pulses: 0, weight: storedWeight.value, direction: 0 }]
}

function restartPulseAnimation() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    stopPulseAnimation()
    resetWeight()
  } else startPulseAnimation()
}

watch(activePanel, panel => {
  if (panel === 'update') restartPulseAnimation()
  else stopPulseAnimation()
})

watch(animationMode, () => {
  resetWeight()
  if (activePanel.value === 'update') restartPulseAnimation()
})

watch([stateCount, symmetricPosition, beta], () => {
  if (activePanel.value === 'update') restartPulseAnimation()
})

onMounted(restartPulseAnimation)
onBeforeUnmount(stopPulseAnimation)
</script>

<template>
  <div class="nonideality-model">
    <div class="nonideality-switcher" role="tablist" aria-label="AIMC hardware non-ideality components">
      <button v-for="(panel, index) in panels" :key="panel.id" :id="`nonideality-tab-${panel.id}`" ref="tabButtons"
        type="button" class="nonideality-tab" role="tab" :aria-controls="`${panel.id}-panel`"
        :aria-selected="activePanel === panel.id" :tabindex="activePanel === panel.id ? 0 : -1"
        @click="selectPanel(panel.id)" @keydown="navigateTabs($event, index)">{{ panel.label }}</button>
    </div>
    <div class="nonideality-grid">
      <article v-if="activePanel === 'converter'" id="converter-panel" class="nonideality-card converter-card" role="tabpanel" aria-labelledby="nonideality-tab-converter" tabindex="0">
        <div class="nonideality-card-head">
          <div>
            <span class="nonideality-index">01 / CONVERTERS</span>
            <h3>DAC / ADC quantization</h3>
          </div>
          <span class="nonideality-chip"><MathTex tex="\varepsilon=Q(x)-x" /></span>
        </div>
        <p class="nonideality-summary">A finite converter maps each element of an input vector onto one of a limited number of voltage or current levels.</p>

        <ConverterPlayground simplified id-prefix="hardware-quantization" />
      </article>

      <article v-else id="update-panel" class="nonideality-card update-card" role="tabpanel" aria-labelledby="nonideality-tab-update" tabindex="0">
        <div class="nonideality-card-head">
          <div>
            <span class="nonideality-index">02 / WEIGHT UPDATE</span>
            <h3>Pulsed conductance update</h3>
          </div>
        </div>
        <p class="nonideality-summary">Tune <MathTex tex="\beta" /> to control the weight response, and move the symmetric point anywhere between the lower and upper weight bounds. Both sequences start 10% above the lower bound.</p>

        <div class="update-layout">
          <div class="update-visual">
            <div class="animation-modes" role="radiogroup" aria-label="Pulse sequence">
              <button type="button" :class="{ selected: animationMode === 'sweep' }" role="radio" :aria-checked="animationMode === 'sweep'" @click="animationMode = 'sweep'"><strong>Pulse sweep</strong></button>
              <button type="button" :class="{ selected: animationMode === 'alternating' }" role="radio" :aria-checked="animationMode === 'alternating'" @click="animationMode = 'alternating'"><strong>Alternating</strong></button>
            </div>

            <div class="pulse-plot-wrap">
              <div class="pulse-plot-head"><strong>Update trajectory</strong><span>{{ pulseTrajectory.length - 1 }} fired pulse{{ pulseTrajectory.length - 1 === 1 ? '' : 's' }}</span></div>
              <svg class="pulse-plot" viewBox="0 0 420 190" role="img" aria-label="Stored weight versus number of fired pulses">
                <line class="plot-grid-line" x1="68" y1="38" x2="392" y2="38" />
                <line class="plot-grid-line" x1="68" y1="94" x2="392" y2="94" />
                <line class="plot-axis-line" x1="68" y1="150" x2="392" y2="150" />
                <line class="plot-axis-line" x1="68" y1="38" x2="68" y2="150" />
                <line v-if="beta > 0" class="symmetric-point-line" x1="68" :y1="symmetricPointY" x2="392" :y2="symmetricPointY" />
                <line v-if="animationMode === 'sweep'" class="pulse-switch-line" :x1="pulseSwitchX" y1="38" :x2="pulseSwitchX" y2="150" />
                <polyline class="pulse-plot-line" :points="pulsePlotPath" />
                <circle v-for="point in pulsePlotPoints" :key="point.pulses" :class="['pulse-plot-point', point.direction > 0 ? 'positive' : point.direction < 0 ? 'negative' : 'initial']" :cx="point.x" :cy="point.y" :r="stateCount > 16 ? 2 : 4" />
                <foreignObject v-if="beta > 0" class="plot-bound" x="24" y="27" width="36" height="22">
                  <div xmlns="http://www.w3.org/1999/xhtml" class="plot-bound-label" v-html="weightMaxLabel"></div>
                </foreignObject>
                <foreignObject v-if="beta > 0" class="plot-bound" x="24" y="139" width="36" height="22">
                  <div xmlns="http://www.w3.org/1999/xhtml" class="plot-bound-label" v-html="weightMinLabel"></div>
                </foreignObject>
                <foreignObject v-if="beta > 0" class="plot-bound plot-symmetry" x="352" :y="symmetricPointY - 24" width="36" height="22">
                  <div xmlns="http://www.w3.org/1999/xhtml" class="plot-bound-label" v-html="symmetricWeightLabel"></div>
                </foreignObject>
                <text class="plot-label" x="68" y="166" text-anchor="middle">0</text>
                <text class="plot-label" x="392" y="166" text-anchor="middle">{{ pulsePlotMax }}</text>
                <text class="plot-note" v-if="animationMode === 'sweep'" :x="pulseSwitchX + 4" y="32">+ → −</text>
                <text class="plot-axis-title" x="230" y="184" text-anchor="middle">fired pulses</text>
                <text class="plot-axis-title" transform="translate(12 94) rotate(-90)" text-anchor="middle">weight w</text>
              </svg>
            </div>

            <p v-if="beta === 0" class="plot-range-note">β = 0: no finite response zeros; the vertical range fits the pulse sequence.</p>
            <div class="pulse-actions" aria-label="Pulse playback and manual controls">
              <button type="button" class="playback-button" @click="animationRunning ? stopPulseAnimation() : startPulseAnimation()">{{ animationRunning ? 'Pause' : 'Replay' }}</button>
              <button type="button" class="pulse-button positive" :disabled="currentResponse.up <= 0" @click="fireManualPulse(1)">Step +</button>
              <button type="button" class="pulse-button negative" :disabled="currentResponse.down <= 0" @click="fireManualPulse(-1)">Step −</button>
              <button type="button" class="playback-button" @click="resetManualSequence">Reset</button>
            </div>
            <p class="animation-note">{{ animationRunning ? 'Playing pulse sequence' : 'Paused · step manually or replay the sequence' }}</p>
          </div>
          <div class="update-controls" aria-label="Pulse response parameters">
            <h4>Device response</h4>
            <span class="response-kind" aria-live="polite">{{ responseKind }}</span>
            <label>
              <span># device states <output>{{ stateCount }}</output></span>
              <input v-model.number="stateCount" type="range" min="8" max="128" step="1" aria-label="Number of device states">
            </label>
            <label>
              <span><span>Symmetric point <span v-html="symmetricWeightLabel"></span></span><output>{{ Math.round(symmetricPosition * 100) }}%</output></span>
              <input v-model.number="symmetricPosition" type="range" min="0" max="1" step="0.01" aria-label="Symmetric point position">
              <div class="asymmetry-scale" aria-hidden="true"><span v-html="weightMinLabel"></span><span v-html="weightMaxLabel"></span></div>
            </label>
            <label>
              <span>Response slope <MathTex tex="\beta" /> <output>{{ beta.toFixed(2) }}</output></span>
              <div class="nonlinearity-scale" aria-hidden="true"><span>0 · constant</span><span>1 · state dependent</span></div>
              <input v-model.number="beta" type="range" min="0" max="1" step="0.05" aria-label="Response slope">
            </label>
            <p class="balance-note">{{ balanceDescription }}</p>
          </div>
        </div>
        <div class="mode-takeaway" aria-live="polite">
          <span class="takeaway-label">{{ animationMode === 'sweep' ? 'Pulse sweep' : 'Alternating' }} · takeaway</span>
          <p>{{ modeTakeaway }}<template v-if="animationMode === 'alternating' && beta > 0 && symmetricPosition > 0 && symmetricPosition < 1">&nbsp;<span v-html="symmetricWeightLabel"></span>.</template></p>
          <p v-if="animationMode === 'alternating' && beta > 0 && symmetricPosition > 0 && symmetricPosition < 1" class="takeaway-detail">Alternating signs illustrate balanced update noise. Finite pulses oscillate around a nearby two-step cycle; smaller pulses approach the symmetric point more closely.</p>
        </div>
        <details class="response-formulation">
          <summary>Pulse response formulation</summary>
          <div class="response-formulation-body">
            <p><MathTex tex="\beta\in[0,1]" /> controls state dependence. The slider position <MathTex tex="\rho\in[0,1]" /> selects the symmetric point from <MathTex tex="w_{\min}" /> to <MathTex tex="w_{\max}" />, without moving the bounds.</p>
            <div v-for="(equation, index) in responseEquations" :key="index" class="response-equation" v-html="equation"></div>
            <p>Each pulse requests <MathTex tex="\Delta w=\pm2/N" /> and applies <MathTex tex="U_q" /> directly. The weight bounds are the zeros of <MathTex tex="q_-" /> and <MathTex tex="q_+" />; no projection is applied. At <MathTex tex="\rho=0" /> or <MathTex tex="1" />, one pulse direction has zero gain and the symmetric point is the corresponding bound. At <MathTex tex="\beta=0" />, no finite response-zero bounds exist: the initial weight is <MathTex tex="-0.8" /> and the plot fits the sequence. The update rule follows <a href="https://arxiv.org/pdf/2406.12774#page=5" target="_blank" rel="noopener">Section 2.2</a>.</p>
          </div>
        </details>
      </article>
    </div>
  </div>
</template>

<style scoped>
.nonideality-model { --hw-easing: cubic-bezier(.22, 1, .36, 1); }
.nonideality-switcher { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; border-bottom: 1px solid var(--border); }
.nonideality-tab { min-width: 0; min-height: 64px; margin-bottom: -1px; padding: 14px 10px; border: 1px solid var(--border); border-radius: 8px 8px 0 0; background: var(--panel); color: var(--muted); font: inherit; font-size: 13px; line-height: 1.4; letter-spacing: .6px; text-transform: uppercase; cursor: pointer; }
.nonideality-tab:hover { color: var(--ink); }
.nonideality-tab[aria-selected="true"] { border-bottom-color: var(--paper); background: var(--paper); color: var(--ink); font-weight: 600; }
.nonideality-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.nonideality-grid { display: block; margin-top: 18px; }
.nonideality-card { min-width: 0; padding: 20px; border: 1px solid var(--border); border-radius: 10px; background: var(--paper); }
.nonideality-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.nonideality-index { display: block; color: var(--accent); font-size: 10px; letter-spacing: 1.1px; }
.nonideality-card h3 { margin: 5px 0 0; font-size: 21px; line-height: 1.15; }
.nonideality-chip { flex: 0 0 auto; padding: 5px 7px; border: 1px solid #e9caca; border-radius: 4px; color: var(--accent); font-family: Georgia, serif; font-size: 12px; white-space: nowrap; }
.nonideality-summary { margin: 12px 0 18px; color: var(--muted); font-size: 14px; line-height: 1.55; }
.nonideality-controls, .update-controls { display: grid; gap: 10px; }
.nonideality-controls label, .update-controls label { display: block; color: var(--muted); font-size: 12px; }
.nonideality-controls label > span, .update-controls label > span { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 5px; }
.nonideality-controls output, .update-controls output { color: var(--ink); font-variant-numeric: tabular-nums; }
.nonideality-model input[type="range"] { display: block; width: 100%; accent-color: var(--accent); cursor: grab; }
.nonideality-model input[type="range"]:active { cursor: grabbing; }
.vector-readout { margin-top: 18px; padding: 13px 12px 11px; border: 1px solid var(--border); border-radius: 7px; background: var(--paper); }
.vector-readout-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 12px; font-size: 12px; }
.vector-readout-head span { color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; }
.vector-rows { display: grid; gap: 8px; }
.vector-row { display: grid; grid-template-columns: 22px minmax(80px, 1fr) 86px 42px; align-items: center; gap: 7px; min-width: 0; font-size: 11px; font-variant-numeric: tabular-nums; }
.vector-label { color: var(--muted); }
.vector-track { position: relative; height: 18px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); background: linear-gradient(90deg, transparent 49.7%, var(--border) 49.7%, var(--border) 50.3%, transparent 50.3%); }
.vector-zero { position: absolute; top: 0; bottom: 0; left: 50%; border-left: 1px solid var(--border); }
.vector-marker { position: absolute; top: 2px; width: 3px; height: 14px; border-radius: 2px; transform: translateX(-50%); }
.input-marker { background: var(--blue); }
.quantized-marker { background: var(--accent); }
.vector-error { position: absolute; top: 7px; height: 4px; min-width: 2px; border-radius: 3px; background: var(--orange); }
.vector-values { overflow: hidden; color: var(--muted); white-space: nowrap; }
.vector-error-value { color: var(--orange); text-align: right; }
.vector-error-value.clipped { color: var(--accent); font-weight: 600; }
.vector-axis { display: flex; justify-content: space-between; margin: 5px 42px 0 29px; color: var(--muted); font-size: 10px; }
.vector-legend { display: flex; flex-wrap: wrap; gap: 11px; margin-top: 10px; color: var(--muted); font-size: 10px; }
.vector-legend span { display: inline-flex; align-items: center; gap: 4px; }
.legend-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; }
.legend-line { display: inline-block; width: 10px; height: 3px; border-radius: 2px; background: var(--orange); }
.converter-stats { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 13px; color: var(--muted); font-size: 11px; }
.converter-stats span { flex: 1 1 80px; }
.converter-stats strong { display: block; color: var(--ink); font-size: 15px; font-variant-numeric: tabular-nums; }
.update-layout { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(190px, 1fr); gap: 22px; align-items: start; }
.update-visual { min-width: 0; }
.pulse-actions { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 12px; }
.pulse-button, .playback-button { min-height: 36px; padding: 7px 12px; border: 1px solid var(--border); border-radius: 5px; background: var(--paper); color: var(--muted); font: inherit; font-size: 12px; cursor: pointer; }
.pulse-button.positive { color: var(--accent); }
.pulse-button.negative { color: var(--blue); }
.pulse-button:hover:not(:disabled), .playback-button:hover { border-color: var(--accent); }
.pulse-button:disabled { cursor: not-allowed; opacity: .4; }
.pulse-button:focus-visible, .playback-button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.animation-modes { display: grid; grid-template-columns: 1fr 1fr; align-items: stretch; gap: 7px; margin-top: 0; }
.animation-modes button { display: flex; align-items: center; justify-content: center; min-height: 46px; padding: 10px 12px; font: inherit; border: 1px solid var(--border); border-radius: 5px; background: var(--paper); color: var(--muted); text-align: center; cursor: pointer; }
.animation-modes button:hover, .animation-modes button:focus-visible { border-color: var(--accent); color: var(--ink); }
.animation-modes button.selected { border-color: var(--accent); background: var(--selected); color: var(--accent); }
.animation-modes button strong { font-size: 15px; font-weight: 600; }
.pulse-plot-wrap { margin-top: 12px; padding: 13px 12px 7px; border: 1px solid var(--border); border-radius: 7px; background: var(--paper); }
.pulse-plot-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 4px; font-size: 12px; }
.pulse-plot-head span { color: var(--muted); font-variant-numeric: tabular-nums; }
.nonideality-model .plot-range-note { margin: 8px 0 0; color: var(--muted); font-size: 11px; line-height: 1.5; }
.pulse-plot { display: block; width: 100%; height: auto; overflow: visible; }
.plot-grid-line { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 4; }
.plot-axis-line { stroke: var(--border); stroke-width: 1.3; }
.symmetric-point-line { stroke: var(--blue); stroke-width: 1.2; stroke-dasharray: 2 4; }
.plot-symmetry .plot-bound-label { color: var(--blue); }
.pulse-switch-line { stroke: var(--orange); stroke-width: 1.2; stroke-dasharray: 2 4; opacity: .8; }
.pulse-plot-line { fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
.pulse-plot-point { fill: var(--paper); stroke-width: 2; }
.pulse-plot-point.initial { stroke: var(--blue); }
.pulse-plot-point.positive { stroke: var(--accent); }
.pulse-plot-point.negative { stroke: var(--blue); }
.plot-label, .plot-axis-title, .plot-note { fill: var(--muted); font-family: Inter, 'Segoe UI', sans-serif; font-size: 10px; }
.plot-bound { overflow: visible; }
.plot-bound-label { display: flex; align-items: center; justify-content: flex-end; height: 100%; color: var(--muted); font-size: 10px; line-height: 1; }
.plot-bound-label :deep(.katex) { font-size: 1.1em; }
.plot-axis-title { font-size: 9px; letter-spacing: .2px; }
.plot-note { fill: var(--accent); font-size: 9px; }
.update-controls { padding: 16px; border: 1px solid var(--border); border-radius: 7px; background: var(--panel); gap: 20px; }
.update-controls h4 { margin: 0; color: var(--ink); font-size: 13px; }
.response-kind { margin-top: -12px; color: var(--blue); font-size: 12px; }
.mode-takeaway { margin-top: 18px; padding: 14px 16px; border-left: 3px solid var(--blue); border-radius: 0 6px 6px 0; background: var(--panel); }
.takeaway-label { display: block; margin-bottom: 5px; color: var(--blue); font-size: 11px; letter-spacing: .5px; text-transform: uppercase; }
.nonideality-model .mode-takeaway p { margin: 0; color: var(--ink); font-size: 14px; line-height: 1.6; }
.nonideality-model .mode-takeaway .takeaway-detail { margin-top: 7px; color: var(--muted); font-size: 12px; }
.balance-note { margin: 0; color: var(--muted); font-size: 11px; line-height: 1.6; }
.response-formulation { margin: 20px 0 0; padding: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.response-formulation-body { padding: 16px 18px; }
.response-formulation summary { color: var(--ink); font-size: 13px; cursor: pointer; font-weight: 600; }
.nonideality-model .response-formulation-body p { color: var(--muted); margin: 0 0 12px; font-size: 12px; line-height: 1.65; }
.response-equation { overflow-x: auto; padding: 7px 0; margin-bottom: 7px; font-size: 13px; color: #000; text-align: center; }
.nonideality-model .response-formulation-body p:last-child { margin-bottom: 0; }
.response-equation :deep(.katex) { font-size: 1.05em; }
.asymmetry-scale, .nonlinearity-scale { display: flex; justify-content: space-between; gap: 8px; margin: 0 0 4px; color: var(--muted); font-size: 10px; }
.asymmetry-scale span, .nonlinearity-scale span { white-space: nowrap; }
.animation-note { margin: 13px 0 0; color: var(--muted); font-size: 11px; text-align: center; }
@container (max-width: 660px) {
  .update-layout { grid-template-columns: minmax(0, 1fr); gap: 16px; }
  .nonideality-card { padding: 16px 13px; }
  .response-formulation-body { padding: 14px; }
  .animation-modes { grid-template-columns: 1fr 1fr; }
  .nonideality-tab { font-size: 11px; padding: 12px 6px; letter-spacing: .3px; }
}
@media (max-width: 760px) { .update-layout { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 430px) {
  .nonideality-card { padding: 16px 13px; }
  .nonideality-card-head { display: block; }
  .nonideality-chip { display: inline-block; margin-top: 10px; }
  .vector-row { grid-template-columns: 20px minmax(65px, 1fr) 72px 38px; gap: 5px; font-size: 10px; }
  .vector-values { font-size: 9px; }
  .animation-modes { grid-template-columns: 1fr 1fr; }
}
@media (prefers-reduced-motion: reduce) { .nonideality-model *, .nonideality-model *::before, .nonideality-model *::after { transition: none !important; } }
</style>
