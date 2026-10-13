<script setup>
import { computed, ref } from 'vue'
import { sampleVector, quantize, optimalRail, aciqRail } from './converter-math.js'
import MathTex from './MathTex.vue'
const props = defineProps({ simplified: { type: Boolean, default: false }, idPrefix: { type: String, default: 'alignment' } })
const controlId = name => `${props.idPrefix}-${name}`
const distribution = ref('gaussian'), size = ref(32), bits = ref(5), seed = ref(1337)
const mode = ref(props.simplified ? 'custom' : 'max'), logarithmic = ref(false)
const signal = computed(() => sampleVector(distribution.value, size.value, seed.value))
const maximum = computed(() => Math.max(...signal.value.map(Math.abs)))
const customRail = ref(maximum.value)
const methods = computed(() => props.simplified ? [] : [
  { id: 'narrow', label: 'Too narrow', rail: maximum.value * 0.35 },
  { id: 'wide', label: 'Too wide', rail: maximum.value * 2 },
  { id: 'max', label: 'Max aligned (CA)', rail: maximum.value },
  { id: 'optimal', label: 'Optimal Rail', rail: optimalRail(signal.value, bits.value) },
  { id: 'gaussian', label: 'ACIQ-Gaussian', rail: aciqRail(signal.value, bits.value, 'gaussian') },
  { id: 'laplacian', label: 'ACIQ-Laplacian', rail: aciqRail(signal.value, bits.value, 'laplacian') },
].map(method => ({ ...method, result: quantize(signal.value, method.rail, bits.value) })))
const selected = computed(() => methods.value.find(method => method.id === mode.value))
const rail = computed(() => selected.value ? selected.value.rail : customRail.value)
const railLimit = computed(() => Math.max(maximum.value * 2, customRail.value, ...methods.value.map(method => method.rail)) * 1.1)
const result = computed(() => selected.value ? selected.value.result : quantize(signal.value, rail.value, bits.value))
const preset = name => { mode.value = name }
const setRail = event => { customRail.value = Number(event.target.value); mode.value = 'custom' }
const reset = () => { distribution.value = 'gaussian'; size.value = 32; bits.value = 5; seed.value = 1337; customRail.value = maximum.value; mode.value = props.simplified ? 'custom' : 'max'; logarithmic.value = false }
const scientific = value => value === 0 ? '0' : value >= 0.001 ? value.toFixed(4) : value.toExponential(2)
const explanations = {
  narrow: 'A narrow rail clips large coordinates; their information is lost before the matrix pass.',
  wide: 'A wide rail avoids clipping but spreads conversion levels apart, reducing resolution around typical values.',
  optimal: 'Optimal Rail minimizes the actual reconstruction MSE for this displayed vector and bit depth. It may accept some clipping to reduce total error.',
  gaussian: 'ACIQ-Gaussian fits a zero-mean Gaussian using the vector’s RMS and selects the rail from its analytical clipping-plus-rounding error model.',
  laplacian: 'ACIQ-Laplacian fits a zero-mean Laplace distribution using the vector’s mean absolute value and selects its analytical rail.',
  custom: 'Your custom rail controls the balance between clipping and rounding error. Compare its measured error with the six reference methods below.',
}
const vectorWidth = computed(() => Math.max(405, size.value * 10 + 85))
const plotWidth = computed(() => vectorWidth.value - 85)
const extent = computed(() => Math.max(maximum.value, rail.value) * 1.15)
const x = i => 52 + (i + 0.5) * plotWidth.value / size.value
const y = value => 140 - value / extent.value * 112
const barWidth = computed(() => plotWidth.value / size.value * 0.38)
const coordinateTicks = computed(() => [...new Set([0, Math.floor(size.value / 4), Math.floor(size.value / 2), Math.floor(size.value * 3 / 4), size.value - 1])])
const errorRows = computed(() => mode.value === 'custom' ? [...methods.value, { id: 'custom', label: 'Custom rail', rail: rail.value, result: result.value }] : methods.value)
const errorBounds = computed(() => {
  const errors = errorRows.value.map(method => Math.max(method.result.nmse, 1e-12))
  return logarithmic.value ? { lo: 10 ** Math.floor(Math.log10(Math.min(...errors))), hi: 10 ** Math.ceil(Math.log10(Math.max(...errors)) + 0.05) }
    : { lo: 0, hi: Math.max(...errors) * 1.15 }
})
const errorX = value => 156 + (logarithmic.value
  ? (Math.log10(Math.max(value, 1e-12)) - Math.log10(errorBounds.value.lo)) / Math.log10(errorBounds.value.hi / errorBounds.value.lo)
  : value / errorBounds.value.hi) * 340
const errorTicks = computed(() => logarithmic.value
  ? [errorBounds.value.lo, Math.sqrt(errorBounds.value.lo * errorBounds.value.hi), errorBounds.value.hi]
  : [0, errorBounds.value.hi / 2, errorBounds.value.hi])
</script>

<template>
  <div class="converter-playground">
    <div class="converter-controls">
      <div class="converter-control"><label :for="controlId('distribution')">Input distribution</label><select :id="controlId('distribution')" v-model="distribution"><option value="gaussian">Gaussian</option><option value="laplacian">Laplacian</option></select></div>
      <div class="converter-control"><label :for="controlId('size')">Vector size <output>{{ size }}</output></label><input :id="controlId('size')" v-model.number="size" type="range" min="8" max="256" step="8" /></div>
      <div class="converter-control"><label :for="controlId('bits')">Resolution <output>{{ bits }} bits</output></label><input :id="controlId('bits')" v-model.number="bits" type="range" min="2" max="9" step="1" /></div>
      <div class="converter-control"><label :for="controlId('rail')">Rail <MathTex tex="\pm C" /> <output>{{ rail.toFixed(3) }}</output></label><input :id="controlId('rail')" :value="rail" type="range" :min="maximum * 0.02" :max="railLimit" :step="maximum / 1000" @input="setRail" /></div>
      <div v-if="!simplified" class="converter-sample-actions"><button type="button" @click="seed += 1">New sample</button><button type="button" @click="reset">Reset</button></div>
      <div v-if="!simplified" class="converter-pipeline"><span>Prescale digitally</span><b><MathTex tex="x/C" /></b><span>Fixed physical DAC rail</span><b><MathTex tex="[-1,1]" /></b><span>Restore the digital scale</span><b><MathTex tex="C Q_{K,1}(x/C)" /></b></div>
    </div>
    <div class="converter-chart">
      <div v-if="!simplified" class="converter-presets" role="group" aria-label="Rail method"><button v-for="method in methods" :key="method.id" type="button" :aria-pressed="mode === method.id" @click="preset(method.id)">{{ method.label }}</button></div>
      <div class="converter-figure-heading"><h4>Vector coordinates</h4><span><template v-if="!simplified">{{ selected?.label || 'Custom rail' }} · </template><MathTex :tex="`C = ${rail.toFixed(3)}`" /></span></div>
      <div class="plot-legend"><span><i class="original-key"></i>Original</span><span><i class="reconstructed-key"></i>Reconstructed</span><span><i class="clipped-key"></i>Clipped input</span></div>
      <div class="converter-vector-scroll" tabindex="0" aria-label="Vector bar plot; scroll horizontally for large vectors">
        <svg class="converter-vector-plot" :viewBox="`0 0 ${vectorWidth} 295`" :style="{ minWidth: `${vectorWidth}px` }" role="group" :aria-label="`${size} ${distribution} input coordinates and their reconstruction using ${selected?.label || 'a custom rail'}, with ${bits} bits`">
          <g v-for="value in [-extent / 1.15, 0, extent / 1.15]" :key="value"><line x1="52" :y1="y(value)" :x2="vectorWidth - 33" :y2="y(value)" stroke="var(--border)"/><text x="44" :y="y(value) + 4" text-anchor="end">{{ value.toFixed(1) }}</text></g>
          <rect x="52" :y="y(rail)" :width="plotWidth" :height="y(-rail) - y(rail)" fill="var(--primary-soft)" opacity=".6" />
          <line v-for="v in [-rail, rail]" :key="v" x1="52" :y1="y(v)" :x2="vectorWidth - 33" :y2="y(v)" stroke="var(--primary-dark)" stroke-dasharray="5 4" />
          <text :x="vectorWidth - 27" :y="y(rail) + 4">+C</text><text :x="vectorWidth - 27" :y="y(-rail) + 4">−C</text>
          <line x1="52" :y1="y(0)" :x2="vectorWidth - 33" :y2="y(0)" stroke="var(--muted)" />
          <g v-for="(value, i) in signal" :key="i" class="converter-coordinate" tabindex="0" :aria-label="`Coordinate ${i + 1}: original ${value.toFixed(4)}, reconstructed ${result.reconstructed[i].toFixed(4)}${Math.abs(value) > rail ? ', clipped' : ''}`">
            <title>Coordinate {{ i + 1 }} · original {{ value.toFixed(4) }} · reconstructed {{ result.reconstructed[i].toFixed(4) }}</title>
            <rect class="converter-original-bar" :x="x(i) - barWidth" :y="Math.min(y(0), y(value))" :width="barWidth" :height="Math.abs(y(value) - y(0))" :fill="Math.abs(value) > rail ? 'var(--analog)' : 'var(--primary-dark)'" />
            <rect class="converter-reconstructed-bar" :x="x(i)" :y="Math.min(y(0), y(result.reconstructed[i]))" :width="barWidth" :height="Math.abs(y(result.reconstructed[i]) - y(0))" fill="#b31b1b" />
          </g>
          <text v-for="i in coordinateTicks" :key="i" :x="x(i)" y="270" text-anchor="middle">{{ i + 1 }}</text>
          <text :x="vectorWidth / 2" y="290" text-anchor="middle">Vector coordinate · same sample for every rail method</text>
        </svg>
      </div>
      <p class="converter-scroll-hint">Hover or focus a coordinate for its values. Scroll horizontally when the vector extends beyond the plot.</p>
      <div class="converter-stats" aria-live="polite" aria-atomic="true"><div><span>Clipped</span><b>{{ result.clipped }} / {{ signal.length }}</b></div><div><span>Step size</span><b>{{ rail ? result.delta.toPrecision(3) : '0' }}</b></div><div><span>Normalized MSE</span><b>{{ scientific(result.nmse) }}</b></div></div>
      <p class="converter-explanation" aria-live="polite"><template v-if="simplified">The rail sets the input range: a narrow rail clips large coordinates; a wide rail spreads the conversion levels apart.</template><template v-else-if="mode === 'max'">CA sets <MathTex tex="C=\lVert x\rVert_\infty" />. It uses the full DAC range without input clipping and cheaply approximates rail optimization; it need not minimize finite-bit MSE.</template><template v-else>{{ explanations[mode] }}</template></p>
    </div>
  </div>
  <div v-if="!simplified" class="converter-error-comparison">
    <div class="converter-figure-heading"><h4>Reconstruction error by rail method</h4><label><input v-model="logarithmic" type="checkbox" /> Log error axis</label></div>
    <p class="converter-error-caption">Same {{ size }}-coordinate {{ distribution === 'gaussian' ? 'Gaussian' : 'Laplacian' }} sample, {{ bits }} bits, and quantizer. Select a bar to apply its rail; scroll horizontally on small screens.</p>
    <div class="converter-error-scroll">
      <svg class="converter-error-plot" :viewBox="`0 0 600 ${errorRows.length * 40 + 55}`" role="group" aria-label="Normalized MSE comparison of rail methods; lower error is better">
        <g v-for="tick in errorTicks" :key="tick"><line :x1="errorX(tick)" y1="8" :x2="errorX(tick)" :y2="errorRows.length * 40" stroke="var(--border)"/><text :x="errorX(tick)" :y="errorRows.length * 40 + 21" text-anchor="middle">{{ scientific(tick) }}</text></g>
        <g v-for="(method, i) in errorRows" :key="method.id" class="converter-error-row" role="button" tabindex="0" :aria-label="`${method.label}, rail ${method.rail.toFixed(4)}, normalized MSE ${scientific(method.result.nmse)}`" :aria-pressed="mode === method.id" @click="preset(method.id)" @keydown.enter.prevent="preset(method.id)" @keydown.space.prevent="preset(method.id)">
          <rect x="0" :y="i * 40 + 3" width="600" height="37" :fill="mode === method.id ? 'var(--primary-soft)' : 'transparent'" rx="4" />
          <text x="146" :y="i * 40 + 27" text-anchor="end">{{ method.label }}</text>
          <rect x="156" :y="i * 40 + 11" :width="Math.max(1, errorX(method.result.nmse) - 156)" height="22" :fill="mode === method.id ? 'var(--primary-dark)' : '#9caec6'" rx="2" />
          <text :x="Math.min(510, errorX(method.result.nmse) + 8)" :y="i * 40 + 27">{{ scientific(method.result.nmse) }}</text>
        </g>
        <text x="326" :y="errorRows.length * 40 + 48" text-anchor="middle">Normalized MSE · {{ logarithmic ? 'log scale' : 'linear scale' }} · lower is better</text>
      </svg>
    </div>
    <p class="demo-caption">Optimal Rail minimizes the displayed vector’s MSE over positive scalar rails. ACIQ uses fitted, zero-mean Gaussian or Laplace statistics and the authors’ analytical error model; all bars evaluate the same endpoint quantizer. <a href="https://github.com/submission2019/cnn-quantization/blob/master/mse_analysis.py" rel="noreferrer">ACIQ model source</a>.</p>
  </div>
</template>
