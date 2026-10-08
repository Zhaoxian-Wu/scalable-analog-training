<script setup>
import { computed, ref } from 'vue'
import { converterSignal, quantize } from './demoMath.mjs'
const rail = ref(0.35), bits = ref(4), mode = ref('narrow')
const max = Math.max(...converterSignal.map(Math.abs))
const result = computed(() => quantize(converterSignal, rail.value, bits.value))
const preset = name => { mode.value = name; rail.value = name === 'narrow' ? 0.35 : name === 'wide' ? 3 : max }
const reset = () => { bits.value = 4; preset('narrow') }
const x = i => 40 + i / (converterSignal.length - 1) * 492
const y = value => 117 - value / 3.2 * 91
const path = values => values.map((value, i) => `${i ? 'L' : 'M'} ${x(i)} ${y(value)}`).join(' ')
</script>

<template>
  <div class="playground converter-demo" @click.stop @keydown.stop @pointerdown.stop>
    <div class="playground-controls">
      <div class="panel-kicker">EFFECTIVE CONVERTER RAIL</div>
      <label for="converter-rail">Range ±C <output>{{ rail.toFixed(2) }}</output></label>
      <input id="converter-rail" v-model.number="rail" type="range" min="0.2" max="3" step="0.01" @input="mode = 'custom'" />
      <label for="converter-bits">Resolution <output>{{ bits }} bits</output></label>
      <input id="converter-bits" v-model.number="bits" type="range" min="2" max="9" step="1" />
      <div class="button-row"><button :aria-pressed="mode === 'narrow'" @click="preset('narrow')">Too narrow</button><button :aria-pressed="mode === 'wide'" @click="preset('wide')">Too wide</button></div>
      <div class="button-row"><button :aria-pressed="mode === 'max'" @click="preset('max')">Max aligned</button><button @click="reset">Reset</button></div>
      <div class="toy-badge">SYNTHETIC SIGNAL · INPUT QUANTIZER</div>
    </div>
    <div class="playground-chart">
      <div class="chart-legend"><span class="legend ideal">Original signal</span><span class="legend accumulated">Reconstructed</span><span class="legend direct">Clipped input</span></div>
      <svg viewBox="0 0 570 230" role="img" aria-label="Original and quantized signals with adjustable quantization rails; orange marks show inputs outside the rail">
        <rect x="40" :y="y(rail)" width="492" :height="y(-rail) - y(rail)" fill="var(--callout)" />
        <line v-for="v in [-rail, rail]" :key="v" x1="40" :y1="y(v)" x2="532" :y2="y(v)" stroke="var(--accent)" stroke-dasharray="5 4" opacity=".6" />
        <line x1="40" y1="117" x2="532" y2="117" stroke="var(--grid)" />
        <path :d="path(converterSignal)" fill="none" stroke="var(--blue)" stroke-width="1.8" />
        <path :d="path(result.reconstructed)" fill="none" stroke="var(--accent)" stroke-width="2" />
        <g v-for="(value, i) in converterSignal" :key="i"><circle v-if="Math.abs(value) > rail" :cx="x(i)" :cy="y(value)" r="3.8" fill="var(--orange)" /></g>
        <text x="34" :y="y(rail) + 4" text-anchor="end" fill="var(--accent)">+C</text><text x="34" :y="y(-rail) + 4" text-anchor="end" fill="var(--accent)">−C</text>
        <text x="286" y="225" text-anchor="middle" fill="var(--muted)">Signal coordinates · fixed input across all settings</text>
      </svg>
      <div class="chart-stats" aria-live="polite"><div><span>Clipped coordinates</span><b>{{ result.clipped }} / 56</b></div><div><span>Quantization step</span><b>{{ result.delta.toFixed(3) }}</b></div><div><span>Normalized MSE</span><b class="accent">{{ result.nmse.toFixed(4) }}</b></div></div>
    </div>
  </div>
  <div class="small-note">Prescaling changes the effective digital rail while the physical converter range stays fixed · Max alignment eliminates input clipping here</div>
</template>
