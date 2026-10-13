<script setup>
import { computed, ref } from 'vue'
import { mappingStats } from './demoMath.mjs'
const width = ref(512), policy = ref('ours')
const stats = computed(() => mappingStats(width.value, policy.value))
const densityPath = (sigma, domain) => Array.from({length:121}, (_, i) => {
  const z = -domain + 2 * domain * i / 120
  const density = Math.exp(-0.5 * (z / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI))
  return `${i ? 'L' : 'M'} ${28 + i / 120 * 282} ${161 - density * (domain === 1 ? 31 : 8)}`
}).join(' ')
const reset = () => { width.value = 512; policy.value = 'ours' }
</script>

<template>
  <div class="mapping-demo" @click.stop @keydown.stop @pointerdown.stop>
    <div class="mapping-toolbar">
      <div class="button-row"><button :aria-pressed="policy === 'fixed'" @click="policy = 'fixed'">Fixed logical range</button><button :aria-pressed="policy === 'native'" @click="policy = 'native'">Native AbsMax</button><button :aria-pressed="policy === 'ours'" @click="policy = 'ours'">Proposed mapping</button><button @click="reset">Reset</button></div>
      <label for="mapping-width">Layer width D <b>{{ width }}</b><input id="mapping-width" v-model.number="width" type="range" min="128" max="2048" step="128" /></label>
    </div>
    <div class="mapping-panels">
      <div><div class="panel-kicker">SIGNED CONDUCTANCE DEVIATION</div>
        <svg viewBox="0 0 338 190" role="img" aria-label="Physical target density around the reference conductance, with fixed device bounds">
          <line v-for="pos in [28, 310]" :key="pos" :x1="pos" y1="16" :x2="pos" y2="161" stroke="var(--orange)" stroke-dasharray="4 4" />
          <path :d="densityPath(stats.physical, 1)" fill="none" stroke="var(--accent)" stroke-width="3" />
          <line x1="28" y1="161" x2="310" y2="161" stroke="var(--grid)" />
          <text x="28" y="181" fill="var(--muted)">−1</text><text x="169" y="181" fill="var(--muted)" text-anchor="middle">0</text><text x="310" y="181" fill="var(--muted)" text-anchor="end">+1</text>
        </svg>
        <p>σ<sub>physical</sub> = <b>{{ stats.physical.toFixed(3) }}</b></p>
      </div>
      <div><div class="panel-kicker">LOGICAL WEIGHT</div>
        <svg viewBox="0 0 338 190" role="img" aria-label="Logical target density compared with the width-dependent variance-preserving target">
          <path :d="densityPath(stats.target, 0.7)" fill="none" stroke="var(--blue)" stroke-width="2" stroke-dasharray="5 4" />
          <path :d="densityPath(stats.logical, 0.7)" fill="none" stroke="var(--accent)" stroke-width="3" />
          <line x1="28" y1="161" x2="310" y2="161" stroke="var(--grid)" />
          <text x="28" y="181" fill="var(--muted)">−0.7</text><text x="169" y="181" fill="var(--muted)" text-anchor="middle">0</text><text x="310" y="181" fill="var(--muted)" text-anchor="end">+0.7</text>
        </svg>
        <p>σ<sub>logical</sub> = <b>{{ stats.logical.toFixed(3) }}</b><br>Target = <b>{{ stats.target.toFixed(3) }}</b></p>
      </div>
    </div>
    <div class="mapping-summary" aria-live="polite"><b>s = {{ stats.scale.toFixed(3) }}</b><span>{{ policy === 'ours' ? 'Fixed occupancy; logical scale ∝ 1 / √D' : policy === 'fixed' ? 'Useful occupancy; incorrect logical scale' : 'Correct logical scale; narrow occupancy' }}</span></div>
  </div>
  <div class="small-note">Conceptual Gaussian target densities before clipping · Native AbsMax uses an extreme-value approximation; these curves are not experimental histograms</div>
</template>
