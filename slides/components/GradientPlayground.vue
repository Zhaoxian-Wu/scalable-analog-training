<script setup>
import { computed, ref, watch, onUnmounted } from 'vue'
import { residualTrace } from './demoMath.mjs'
const increment = ref(0.006), quantum = ref(0.025), signed = ref(true), cursor = ref(40), playing = ref(false)
const steps = 40
let timer
const traces = computed(() => residualTrace(increment.value, quantum.value, signed.value, steps))
const top = computed(() => Math.max(...traces.value.ideal, ...traces.value.direct, ...traces.value.programmed, quantum.value) * 1.15)
const bottom = computed(() => Math.min(0, ...traces.value.ideal, ...traces.value.direct, ...traces.value.programmed) - quantum.value * 0.5)
const y = value => 190 - (value - bottom.value) / (top.value - bottom.value) * 157
const path = values => values.slice(0, cursor.value + 1).map((value, i) => `${i ? 'L' : 'M'} ${46 + i / steps * 488} ${y(value)}`).join(' ')
const number = value => value.toFixed(3)
const stop = () => { clearInterval(timer); playing.value = false }
const play = () => {
  if (playing.value) { stop(); return }
  if (cursor.value === steps) cursor.value = 0
  playing.value = true
  timer = setInterval(() => { cursor.value++; if (cursor.value >= steps) stop() }, 220)
}
const step = () => { stop(); cursor.value = cursor.value === steps ? 1 : cursor.value + 1 }
const reset = () => { stop(); increment.value = 0.006; quantum.value = 0.025; signed.value = true; cursor.value = 40 }
watch([increment, quantum, signed], () => { stop(); cursor.value = 40 })
onUnmounted(stop)
</script>

<template>
  <div class="playground" @click.stop @keydown.stop @pointerdown.stop>
    <div class="playground-controls">
      <div class="panel-kicker">PROGRAMMING CONTROLS</div>
      <label for="update-increment">Update magnitude <output>{{ number(increment) }}</output></label>
      <input id="update-increment" v-model.number="increment" type="range" min="0.001" max="0.02" step="0.001" />
      <label for="device-quantum">Device step Δ <output>{{ number(quantum) }}</output></label>
      <input id="device-quantum" v-model.number="quantum" type="range" min="0.005" max="0.05" step="0.005" />
      <div class="button-row"><button :aria-pressed="!signed" @click="signed = false">Positive</button><button :aria-pressed="signed" @click="signed = true">Signed</button></div>
      <div class="button-row"><button @click="step">Step</button><button @click="play">{{ playing ? 'Pause' : 'Play' }}</button><button @click="reset">Reset</button></div>
      <p>Step {{ cursor }} · {{ traces.counts[cursor] }} pulses</p>
      <div class="toy-badge">CONCEPTUAL · UNIFORM STEPS · NO NOISE</div>
    </div>
    <div class="playground-chart">
      <div class="chart-legend"><span class="legend ideal">Desired sum</span><span class="legend accumulated">With residual</span><span class="legend direct">Direct truncation</span></div>
      <svg viewBox="0 0 570 230" role="img" aria-label="Signed cumulative updates: desired sum, residual-buffer pulse transfers, and direct pulse truncation">
        <g v-for="tick in 4" :key="tick">
          <line x1="46" :y1="y(bottom + (tick - 1) / 3 * (top - bottom))" x2="534" :y2="y(bottom + (tick - 1) / 3 * (top - bottom))" stroke="var(--grid)" stroke-dasharray="3 5" />
          <text x="40" :y="y(bottom + (tick - 1) / 3 * (top - bottom)) + 4" text-anchor="end" fill="var(--muted)">{{ number(bottom + (tick - 1) / 3 * (top - bottom)) }}</text>
        </g>
        <path :d="path(traces.ideal)" fill="none" stroke="var(--blue)" stroke-width="2" stroke-dasharray="5 4" />
        <path :d="path(traces.programmed)" fill="none" stroke="var(--accent)" stroke-width="3" />
        <path :d="path(traces.direct)" fill="none" stroke="var(--orange)" stroke-width="2.5" />
        <line :x1="46 + cursor / steps * 488" y1="33" :x2="46 + cursor / steps * 488" y2="190" stroke="var(--ink)" opacity=".2" />
        <text x="46" y="210" fill="var(--muted)">0</text><text x="534" y="210" text-anchor="end" fill="var(--muted)">40 updates</text>
      </svg>
      <div class="chart-stats" aria-live="polite">
        <div><span>Desired total</span><b>{{ number(traces.ideal[cursor]) }}</b></div>
        <div><span>Programmed</span><b class="accent">{{ number(traces.programmed[cursor]) }}</b></div>
        <div><span>Residual retained</span><b>{{ number(traces.residuals[cursor]) }}</b></div>
      </div>
    </div>
  </div>
  <div class="small-note">Desired sum = programmed increment + retained residual in this idealized model · Open-loop physical writes may deviate</div>
</template>
