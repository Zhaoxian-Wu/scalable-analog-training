<script setup>
import { computed, ref, useId } from 'vue'
import katex from 'katex'

const uid = `mvm-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`
const hovered = ref(null)
const focused = ref(null)
const active = computed(() => hovered.value ?? focused.value)
const math = (tex) => katex.renderToString(tex, { throwOnError: false })
// Match the diagram's voltage/input red and current/output blue in every state
const voltage = (tex) => `\\textcolor{#B31B1B}{${tex}}`
const current = (tex) => `\\textcolor{#315FC1}{${tex}}`
const forwardFormula = `${current('y')}=\\widehat W ${voltage('x')}`
const backwardFormula = `${current('\\delta_{\\mathrm{in}}')}=\\widehat W^{\\mathsf T}${voltage('\\delta_{\\mathrm{out}}')}`
const subs = ['₁', '₂', '₃']
const cells = Array.from({ length: 9 }, (_, n) => ({
  i: Math.floor(n / 3) + 1, j: n % 3 + 1,
  x: 130 + Math.floor(n / 3) * 78 - 22, y: 100 + (n % 3) * 54,
}))
const cellPath = 'M0 0H4L7 -4L11 4L15 -4L19 4L23 -4L27 0H31.11'
const detail = computed(() => {
  const target = active.value
  if (target?.kind === 'cell') return {
    title: 'Ohm’s law', text: 'Input voltage × stored conductance gives the cell current',
    formula: `${current(`I_{${target.i}${target.j}}`)}=\\widehat W_{${target.i}${target.j}}${voltage(`V_{${target.j}}`)}`,
    note: `Cell (${target.i}, ${target.j}) multiplies one input by one weight`,
  }
  if (target?.kind === 'output') return {
    title: 'Kirchhoff’s law', text: 'Cell currents add along the output column',
    formula: `${current(`I_{${target.i}}`)}=\\sum_j\\widehat W_{${target.i}j}${voltage('V_j')}`,
    note: `Column ${target.i} sums the products in parallel`,
  }
  if (target?.kind === 'dac') return {
    title: 'Digital → analog', text: 'The DAC converts digital inputs into row voltages',
    formula: `${voltage('V')}=\\operatorname{DAC}(${voltage('x')})`, note: 'DAC · Digital-to-analog converter',
  }
  if (target?.kind === 'adc') return {
    title: 'Analog → digital', text: 'The ADC converts summed currents into digital outputs',
    formula: `${current('y')}=\\operatorname{ADC}(${current('I')})`, note: 'ADC · Analog-to-digital converter',
  }
  return null
})
</script>

<template>
  <div class="mvm-engine">
    <div class="mvm-circuit">
      <svg class="mvm-crossbar" viewBox="0 0 330 340" role="group" aria-label="Analog crossbar matrix–vector multiplication" :aria-describedby="`${uid}-desc`">
        <desc :id="`${uid}-desc`">DAC drives three input rows · Conductance cells multiply inputs · Currents flow through cells and sum down output columns into the ADC · Hover or focus cells, outputs, and converters for explanations</desc>
        <defs>
          <pattern :id="`${uid}-converter`" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(40)">
            <rect width="6" height="6" fill="#fff" />
            <path d="M0 0 V6" stroke="#ecd2d2" stroke-width="2" />
          </pattern>
          <marker :id="`${uid}-arrow`" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0 0L8 4L0 8Z" fill="var(--blue)" />
          </marker>
        </defs>
        <foreignObject x="84" y="5" width="226" height="28">
          <div xmlns="http://www.w3.org/1999/xhtml" class="mvm-weight-label">Conductance <span v-html="math('\\widehat W')" /></div>
        </foreignObject>
        <rect class="crossbar-frame" x="84" y="66" width="226" height="182" rx="8" fill="var(--paper)" stroke="var(--border)" />
        <g v-for="j in 3" :key="`row-${j}`" class="mvm-wires">
          <text x="5" :y="105 + (j - 1) * 54" fill="var(--accent)">x{{ subs[j - 1] }}</text>
          <line x1="58" :y1="100 + (j - 1) * 54" x2="298" :y2="100 + (j - 1) * 54" stroke="var(--accent)" stroke-width="3.4" stroke-opacity=".75" />
          <path class="mvm-current-flow input-flow" :d="`M58 ${100 + (j - 1) * 54} H298`" :style="{ animationDelay: `${-(j - 1) * .3}s` }" />
          <text x="65" :y="90 + (j - 1) * 54" fill="var(--accent)">V{{ subs[j - 1] }}</text>
        </g>
        <g v-for="i in 3" :key="`output-${i}`" class="mvm-target mvm-output" :class="{ selected: active?.kind === 'output' && active.i === i }"
          role="button" tabindex="0" :aria-label="`Output column ${i}: Kirchhoff’s law`"
          @pointerenter="hovered = { kind: 'output', i }" @pointerleave="hovered = null"
          @focus="focused = { kind: 'output', i }" @blur="focused = null">
          <rect class="output-hit" :x="118 + (i - 1) * 78" y="82" width="32" height="192" />
          <line class="output-wire" :x1="130 + (i - 1) * 78" y1="84" :x2="130 + (i - 1) * 78" y2="273" stroke="var(--blue)" :marker-end="`url(#${uid}-arrow)`" />
          <path class="mvm-current-flow output-flow" :d="`M${130 + (i - 1) * 78} 84 V270`" :style="{ animationDelay: `${-(i - 1) * .4}s` }" />
          <text class="output-label" :x="138 + (i - 1) * 78" y="262" fill="var(--blue)">I{{ subs[i - 1] }}</text>
        </g>
        <g v-for="cell in cells" :key="`cell-${cell.i}-${cell.j}`" class="mvm-target mvm-cell" :class="{ selected: active?.kind === 'cell' && active.i === cell.i && active.j === cell.j }"
          role="button" tabindex="0" :aria-label="`Cell (${cell.i}, ${cell.j}): Ohm’s law`"
          @pointerenter="hovered = { kind: 'cell', ...cell }" @pointerleave="hovered = null"
          @focus="focused = { kind: 'cell', ...cell }" @blur="focused = null">
          <rect class="cell-hit" :x="cell.x - 5" :y="cell.y - 7" width="33" height="36" rx="5" />
          <g :transform="`translate(${cell.x} ${cell.y}) rotate(45)`" class="mvm-device">
            <path :d="cellPath" class="device-body" />
            <path :d="cellPath" class="mvm-current-flow cell-flow" :style="{ animationDelay: `${-(cell.i + cell.j) * .25}s` }" />
          </g>
          <circle :cx="cell.x" :cy="cell.y" r="3.2" fill="var(--accent)" />
          <circle :cx="cell.x + 22" :cy="cell.y + 22" r="3.2" fill="var(--blue)" />
        </g>
        <g class="mvm-target mvm-dac" :class="{ selected: active?.kind === 'dac' }" role="button" tabindex="0" aria-label="DAC: digital-to-analog converter"
          @pointerenter="hovered = { kind: 'dac' }" @pointerleave="hovered = null" @focus="focused = { kind: 'dac' }" @blur="focused = null">
          <rect class="converter-bar" x="26" y="58" width="32" height="177" rx="5" :fill="`url(#${uid}-converter)`" />
          <text transform="translate(47 147) rotate(-90)" text-anchor="middle">DAC</text>
        </g>
        <g class="mvm-target mvm-adc" :class="{ selected: active?.kind === 'adc' }" role="button" tabindex="0" aria-label="ADC: analog-to-digital converter"
          @pointerenter="hovered = { kind: 'adc' }" @pointerleave="hovered = null" @focus="focused = { kind: 'adc' }" @blur="focused = null">
          <rect class="converter-bar" x="91" y="278" width="220" height="28" rx="5" :fill="`url(#${uid}-converter)`" />
          <text x="201" y="297" text-anchor="middle">ADC</text>
        </g>
        <g v-for="i in 3" :key="`y-${i}`" class="mvm-wires">
          <line :x1="130 + (i - 1) * 78" y1="306" :x2="130 + (i - 1) * 78" y2="315" stroke="var(--blue)" stroke-width="3.4" />
          <text :x="121 + (i - 1) * 78" y="334" fill="var(--blue)">y{{ subs[i - 1] }}</text>
        </g>
      </svg>
    </div>
    <div class="mvm-explanation" aria-live="polite" aria-atomic="true">
      <div v-if="detail" class="mvm-detail">
        <h3>{{ detail.title }}</h3>
        <p>{{ detail.text }}</p>
        <div class="mvm-formula" v-html="math(detail.formula)" />
        <div class="mvm-detail-note">{{ detail.note }}</div>
      </div>
      <div v-else class="mvm-default">
        <div class="mvm-pass"><b>Forward</b><div class="mvm-formula" v-html="math(forwardFormula)" /></div>
        <div class="mvm-pass"><b>Backward</b><div class="mvm-formula" v-html="math(backwardFormula)" /></div>
        <div class="mvm-efficiency"><strong><span>10×–100×</span> more energy-efficient than GPUs</strong><small></small></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mvm-crossbar { width: 100%; height: auto; }
.mvm-crossbar text { font-size: 14px; }
.mvm-weight-label { display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 17px; color: var(--ink); }
.mvm-wires, .mvm-device, .mvm-cell circle { pointer-events: none; }
.mvm-target { cursor: pointer; outline: none; }
.cell-hit, .output-hit { fill: transparent; stroke: transparent; stroke-width: 1.5; }
.mvm-cell.selected .cell-hit, .mvm-cell:focus-visible .cell-hit { fill: var(--selected); stroke: var(--accent); }
.output-wire { stroke-width: 3.4; stroke-opacity: .75; transition: stroke-width .15s, stroke-opacity .15s; }
.output-wire, .output-label { pointer-events: none; }
.mvm-output.selected .output-wire, .mvm-output:focus-visible .output-wire { stroke-width: 6; stroke-opacity: 1; }
.mvm-output.selected .output-flow, .mvm-output:focus-visible .output-flow { stroke-width: 7; }
.device-body { fill: none; stroke: var(--ink); stroke-width: 3.2; stroke-linejoin: round; }
.mvm-current-flow { fill: none; stroke-width: 5.6; stroke-linecap: round; stroke-dasharray: 1 26; animation: mvm-current 1.3s linear infinite; pointer-events: none; }
.input-flow { stroke: var(--accent); }
.output-flow { stroke: var(--blue); }
.cell-flow { stroke: var(--orange); stroke-width: 5.2; }
@keyframes mvm-current { to { stroke-dashoffset: -27; } }
.converter-bar { stroke: #9b7777; stroke-width: 1.4; }
.mvm-dac text, .mvm-adc text { fill: #26364b; font-size: 16px; font-weight: 650; pointer-events: none; }
.mvm-target.selected .converter-bar, .mvm-target:focus-visible .converter-bar { stroke: var(--accent); stroke-width: 2.2; }
.mvm-interaction-hint { margin-top: 2px; text-align: center; font-size: 10px; color: var(--muted); }
.mvm-explanation { min-width: 0; padding-left: 15px; border-left: 1px solid var(--border); }
.mvm-explanation .mvm-detail h3 { color: var(--accent); margin: 0 0 12px; }
.mvm-explanation p { color: var(--muted); line-height: 1.2; margin: 0 0 20px; }
.mvm-formula { margin-top: 7px; font-size: var(--font-body); white-space: nowrap; }
.mvm-pass + .mvm-pass { margin-top: 18px; }
.mvm-pass b { font-size: var(--font-body); font-weight: 600; }
.mvm-detail-note { margin-top: 20px; font-size: 14px; line-height: 1.35; color: var(--muted); }
.mvm-efficiency { border-top: 1px solid var(--border); margin-top: 22px; padding-top: 16px; }
.mvm-efficiency strong { display: block; font-size: var(--font-body); line-height: 1.2; font-weight: 650; }
.mvm-efficiency span { color: var(--accent); white-space: nowrap; }
.mvm-efficiency small { display: block; margin-top: 8px; color: var(--muted); font-size: 11px; }
@media (prefers-reduced-motion: reduce) { .mvm-current-flow { animation: none; } }
</style>
