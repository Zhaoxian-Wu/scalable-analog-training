<script setup>
import { computed, ref } from 'vue'
import data from '../assets/gradient-results-data.json'
import { gradientChart } from './gradient-plots.js'
import MathTex from './MathTex.vue'
const selected = ref(0)
const inspected = ref(null)
const panels = [
  { label: 'Precision sensitivity', title: 'Protect the weight-gradient path',
    takeaway: 'FP8 weights and activations barely change validation loss, but FP8 gradient-path representations impair convergence.',
    detail: 'This digital diagnostic quantizes error signals and parameter gradients; arithmetic, reductions, and accumulation remain FP32.' },
  { label: 'Gradient cancellation', title: 'The useful update is a small signed residual',
    takeaway: 'Opposing sample contributions largely cancel across all six measured projection types. Their net gradient is much smaller than the terms being summed.',
    detail: 'The cancellation factor κ = Σ|gᵦ| / |Σgᵦ| measures this imbalance. Arithmetic means are sensitive to nearly zero sums.' },
  { label: 'Accumulation error', title: 'Cancellation amplifies rounding error',
    takeaway: 'Rounding after every addition increasingly distorts the gradient as cancellation grows. Even FP16 loses fidelity in strong cancellation regimes.',
    detail: 'FP32 preserves the residual much more accurately than FP16 or FP8 in this sequential accumulation diagnostic.' },
  { label: 'Transfer fragmentation', title: 'Consolidate before programming',
    takeaway: 'Larger micro-batches consolidate more contributions digitally and invoke fewer quantized transfers, reducing the realized update error.',
    detail: 'A controlled single-step test holds total batch size at 128 on a symmetric finite-state device to isolate transfer error.' },
]
const panel = computed(() => panels[selected.value])
const chart = computed(() => gradientChart(data, selected.value))
function select(index) {
  inspected.value = null
  selected.value = index
}
function inspect(event) {
  const mark = event.target.closest('[data-gradient-point]')
  if (mark) inspected.value = chart.value.inspections[Number(mark.dataset.gradientPoint)]
}
function navigate(event) {
  let next = selected.value
  if (event.key === 'ArrowRight') next = (next + 1) % panels.length
  else if (event.key === 'ArrowLeft') next = (next + panels.length - 1) % panels.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = panels.length - 1
  else return
  event.preventDefault()
  select(next)
  event.currentTarget.querySelectorAll('[role="tab"]')[next].focus()
}
</script>

<template>
  <div class="precision-diagnostics">
    <div class="evidence-tabs" role="tablist" aria-label="Why gradients need higher fidelity" @keydown="navigate">
      <button v-for="(item, i) in panels" :id="`precision-tab-${i}`" :key="item.label" type="button" role="tab" :aria-selected="selected === i" :tabindex="selected === i ? 0 : -1" aria-controls="precision-panel" @click="select(i)"><span>{{ String.fromCharCode(97 + i) }}</span>{{ item.label }}</button>
    </div>
    <div id="precision-panel" class="evidence-panel" role="tabpanel" :aria-labelledby="`precision-tab-${selected}`">
      <div class="evidence-takeaway" aria-live="polite" aria-atomic="true">
        <span class="domain-badge">Takeaway {{ selected + 1 }} / 4</span>
        <h3>{{ panel.title }}</h3><p>{{ panel.takeaway }}</p><p class="evidence-detail"><template v-if="selected === 1">The cancellation factor <MathTex tex="\kappa=\frac{\sum_b |g_b|}{|\sum_b g_b|}" /> measures this imbalance. Arithmetic means are sensitive to nearly zero sums.</template><template v-else>{{ panel.detail }}</template></p>
      </div>
      <div class="gradient-chart-scroll" @mouseover="inspect" @focusin="inspect" @click="inspect" v-html="chart.svg"></div>
      <p class="gradient-inspection" aria-live="polite" aria-atomic="true">{{ inspected || 'Hover, tap, or focus a bar or point to inspect its mean and four-seed range.' }}</p>
    </div>
  </div>
</template>
