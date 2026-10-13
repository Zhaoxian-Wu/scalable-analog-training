<script setup>
import { computed, ref, useId } from 'vue'
import MathTex from './MathTex.vue'
const omega = ref(3), widthExponent = ref(7)
const width = computed(() => 2 ** widthExponent.value)
const tau = 1
const sigma = computed(() => 1 / Math.sqrt(width.value))
const scale = computed(() => omega.value * sigma.value / tau)
const id = useId()
const physicalStd = computed(() => tau / omega.value)
const pdf = (value, std) => Math.exp(-0.5 * (value / std) ** 2) / (std * Math.sqrt(2 * Math.PI))
const plots = computed(() => [
  { key: 'logical', title: 'Logical weights', range: 0.6, ymax: 15, std: sigma.value, color: '#1764aa', formula: String.raw`\operatorname{Std}(W) = D^{-1/2} = ${sigma.value.toFixed(3)}`, axis: String.raw`W` },
  { key: 'physical', title: 'Physical conductance deviations', range: 1.5, ymax: 4, std: physicalStd.value, color: '#b31b1b', formula: String.raw`\operatorname{Std}(\widehat W-\widehat W^\diamond)=\frac{\tau}{\omega}=${physicalStd.value.toFixed(3)}`, axis: String.raw`\widehat W-\widehat W^\diamond` },
])
const x = (value, plot) => 43 + (value + plot.range) / (2 * plot.range) * 276
const y = (value, plot) => 185 - value / plot.ymax * 151
const curve = plot => Array.from({ length: 241 }, (_, i) => {
  const value = -plot.range + 2 * plot.range * i / 240
  return `${i ? 'L' : 'M'}${x(value, plot)},${y(pdf(value, plot.std), plot)}`
}).join(' ')
const area = plot => `${curve(plot)} L319,185 L43,185 Z`
const scaleFormula = computed(() => String.raw`s=\frac{\omega D^{-1/2}}{\tau}=${scale.value.toFixed(3)}`)
</script>
<template>
  <div class="weight-distributions">
    <div class="weight-distribution-controls">
      <label><span>Layer width <MathTex tex="D" /><output>{{ width }}</output></span><input v-model.number="widthExponent" type="range" min="5" max="10" step="1" aria-label="Weight mapping layer width" /></label>
      <label><span>Inverse occupancy <MathTex tex="\omega" /><output>{{ omega.toFixed(1) }}</output></span><input v-model.number="omega" type="range" min="1.2" max="8" step="0.1" aria-label="Weight mapping omega" /></label>
    </div>
    <div class="weight-distribution-plots">
      <div v-for="plot in plots" :key="plot.key" class="weight-distribution-plot">
        <h4>{{ plot.title }}</h4><div class="weight-distribution-formula"><MathTex :tex="plot.formula" /></div>
        <svg viewBox="0 0 340 213" role="img" :aria-label="`${plot.title}: Gaussian pre-clipping target, standard deviation ${plot.std.toFixed(4)}, omega ${omega}, width ${width}`">
          <defs><clipPath :id="`${id}-${plot.key}`"><rect x="43" y="24" width="276" height="161"/></clipPath><clipPath v-if="plot.key === 'physical'" :id="`${id}-tails`"><rect x="43" y="24" :width="x(-tau, plot)-43" height="161"/><rect :x="x(tau, plot)" y="24" :width="319-x(tau, plot)" height="161"/></clipPath></defs>
          <g v-for="v in [0, plot.ymax / 2, plot.ymax]" :key="v"><line x1="43" :y1="y(v, plot)" x2="319" :y2="y(v, plot)" stroke="var(--border)"/><text x="35" :y="y(v, plot)+4" text-anchor="end">{{ v }}</text></g>
          <g :clip-path="`url(#${id}-${plot.key})`"><path :d="area(plot)" :fill="plot.color" opacity=".12"/><path v-if="plot.key === 'physical'" :d="area(plot)" fill="var(--analog)" opacity=".5" :clip-path="`url(#${id}-tails)`"/><path :d="curve(plot)" fill="none" :stroke="plot.color" stroke-width="2.5"/></g>
          <g v-if="plot.key === 'physical'"><line v-for="bound in [-tau,tau]" :key="bound" :x1="x(bound,plot)" y1="24" :x2="x(bound,plot)" y2="185" stroke="var(--analog)" stroke-dasharray="5 4"/></g>
          <path d="M43 24 V185 H319" fill="none" stroke="var(--muted)"/>
          <text v-for="v in [-plot.range, 0, plot.range]" :key="v" :x="x(v,plot)" y="206" text-anchor="middle">{{ v }}</text>
          <text x="12" y="103" transform="rotate(-90 12 103)" text-anchor="middle">Density</text>
        </svg>
        <div class="weight-distribution-axis"><MathTex :tex="plot.axis" /></div>
      </div>
    </div>
    <div class="weight-distribution-takeaway" aria-live="polite" aria-atomic="true"><MathTex :tex="scaleFormula" /><p>Increasing <MathTex tex="\omega" /> narrows the physical distribution and increases <MathTex tex="s" />, while the logical initialization stays fixed at a given <MathTex tex="D" />. Increasing <MathTex tex="D" /> narrows the logical distribution and decreases <MathTex tex="s" />, while physical occupancy stays fixed.</p></div>
    <p class="demo-caption">Gaussian initialization targets before clipping, with <MathTex tex="\tau=1" />. Dashed lines mark <MathTex tex="\pm\tau" />; orange tails would be clipped when programming the array. The axes stay fixed while the controls change.</p>
  </div>
</template>
