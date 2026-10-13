<script setup>
import { ref } from 'vue'
import data from './manuscriptData.json'
const depth = ref(3)
const x = loss => 182 + (loss - 1.6) / 1.05 * 525
</script>
<template>
  <div class="results-demo optimizer-results" @click.stop @keydown.stop @pointerdown.stop>
    <div class="results-toolbar"><div class="button-row"><button v-for="(n, i) in [2,4,6,8]" :key="n" :aria-pressed="depth === i" @click="depth = i">{{ n }} blocks</button></div><div class="chart-legend"><span class="legend ideal">Digital</span><span class="legend accumulated">Analog S</span></div></div>
    <svg viewBox="0 0 840 286" role="img" :aria-label="`Digital and analog S validation loss for seven optimizers at depth ${[2,4,6,8][depth]}`">
      <g v-for="value in [1.6,1.8,2,2.2,2.4,2.6]" :key="value"><line :x1="x(value)" y1="14" :x2="x(value)" y2="247" stroke="var(--grid)" /><text :x="x(value)" y="268" fill="var(--muted)" text-anchor="middle">{{ value.toFixed(1) }}</text></g>
      <g v-for="(row, i) in data.optimizers" :key="row.name">
        <text x="147" :y="30+i*33" text-anchor="end" fill="var(--ink)">{{ row.name }}</text>
        <line :x1="x(row.Digital[depth])" :x2="x(row.S[depth])" :y1="26+i*33" :y2="26+i*33" stroke="var(--muted)" stroke-width="2" />
        <circle :cx="x(row.Digital[depth])" :cy="26+i*33" r="5" fill="var(--blue)" /><circle :cx="x(row.S[depth])" :cy="26+i*33" r="5" fill="var(--accent)" />
        <text x="808" :y="30+i*33" text-anchor="end" fill="var(--accent)">{{ row.S[depth].toFixed(3) }}</text>
      </g>
    </svg>
  </div>
  <div class="small-note">Shakespeare · 5,000 updates · seed 1337 · fixed hyperparameters, zero decay · S denotes fixed logical mapping with managed analog reads · This screen evaluates compatibility, not a universal optimizer ranking</div>
</template>
