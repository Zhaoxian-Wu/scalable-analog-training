<script setup>
import { computed, nextTick, ref, useId } from 'vue'

const props = defineProps({
  initialView: {
    type: String,
    default: 'prior',
    validator: value => ['digital', 'prior', 'ours'].includes(value),
  },
})

const uid = `overview-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`
const selected = ref(props.initialView)
const hovered = ref(null)
const focused = ref(null)
const pinned = ref(null)
const tabButtons = ref([])
const activeId = computed(() => hovered.value ?? focused.value ?? pinned.value)
const views = [
  {
    id: 'digital', label: 'Digital mixed precision', citation: 'Micikevicius et al.',
    title: 'Keep training in digital',
    intro: 'Reduced-precision matrix arithmetic works alongside full-precision master weights and optimizer states',
    points: ['Digital matrix passes', 'FP32 master weights'],
    takeaway: 'Separate matrix and update precision',
  },
  {
    id: 'prior', label: 'MP baseline', citation: 'Nandakumar et al., 2020',
    title: 'Compute in the crossbar',
    intro: 'A DAC–crossbar–ADC path executes matrix passes · Digital gradients and a buffer drive pulsed SGD updates',
    points: ['Analog matrix passes', 'Digital SGD + residuals'],
    takeaway: 'Preserve batch-gradient precision digitally',
  },
  {
    id: 'ours', label: 'This work', citation: 'Proposed architecture',
    title: 'Co-design the boundary',
    intro: 'Coordinate analog matrix passes and digital updates with logical-space optimizers, width-aware mapping, and converter alignment',
    points: ['Logical-space updates', 'Mapping + converters'],
    takeaway: 'Coordinate mapping, updates, and converters',
  },
]
const view = computed(() => views.find(item => item.id === selected.value))
const isAnalog = computed(() => selected.value !== 'digital')
const isOurs = computed(() => selected.value === 'ours')
const node = (id, title, sub, x, y, w, h, domain, kind, description, equation) =>
  ({ id, title, sub, x, y, w, h, domain, kind, description, equation })
const nodes = computed(() => {
  const common = [
    node('gradient', 'W-grad', 'digital accumulation', 364, 254, 129, 47, 'digital', 'gradient',
      'Form outer products of activations and errors, then accumulate the batch weight gradient digitally', 'G = (1/B) Σ δᵦ xᵦᵀ'),
    node('optimizer', selected.value === 'prior' ? 'SGD' : 'Adam', 'logical update', 364, 182, 129, 42, 'digital', 'optimizer',
      selected.value === 'prior'
        ? 'The illustrated prior design turns the accumulated weight gradient into an SGD update'
        : 'Apply the optimizer digitally · In this work, its states and desired updates stay in logical weight coordinates',
      'G → ΔW'),
  ]
  if (!isAnalog.value) return [
    node('weight', 'Weight · FP16', 'reduced-precision MVM', 99, 126, 134, 92, 'digital', 'array',
      'Use a reduced-precision weight representation for selected forward and backward matrix arithmetic', 'y = Wx · δᵢₙ = Wᵀδₒᵤₜ'),
    node('master', 'Weight · FP32', 'master weights', 364, 86, 129, 73, 'digital', 'buffer',
      'Apply updates to the full-precision master weights, then quantize a working copy for matrix arithmetic', 'W₃₂ ← W₃₂ + ΔW'),
    ...common,
  ]
  return [
    node('dac', 'DAC', 'voltage input', 112, 69, 106, 38, 'converter', 'converter',
      isOurs.value ? 'Digitally rescale activations or backward errors, then convert them to voltages within the fixed DAC rails'
        : 'Convert digital activations or backward errors to crossbar input voltages using the baseline converter range',
      isOurs.value ? 'x/C → DAC' : 'x → DAC'),
    node('crossbar', 'Crossbar', 'conductance weights', 101, 119, 128, 62, 'analog', 'array',
      'Stored conductances execute forward and transposed backward MVMs; W-grad stays digital',
      'y = Wx · δᵢₙ = Wᵀδₒᵤₜ'),
    node('adc', 'ADC', 'digitized current', 112, 192, 106, 38, 'converter', 'converter',
      'Digitize the summed crossbar output current · Fixed ADC rails limit range and conversion resolution', 'current → digital'),
    node('buffer', 'Residual buffer', 'retain small updates', 364, 102, 129, 55, 'digital', 'buffer',
      'Keep small increments in a digital residual · Once the device threshold is crossed, issue open-loop pulses; actual writes remain imperfect',
      'H ← H + ΔW → pulses'),
    node('mapping', 'Mapping', isOurs.value ? 's ∝ D⁻¹ᐟ²' : 'fixed logical scale', 27, 299, 115, 39, 'digital', 'mapping',
      isOurs.value ? 'Use a width-aware scale to preserve logical initialization while occupying a fixed fraction of the physical device range'
        : 'Convert the physical crossbar response to logical weights using the baseline mapping scale',
      'W = s (Ŵ − Ŵᵈ)'),
    ...common,
    ...(isOurs.value ? [
      node('scaling', 'Scaling', '‖x‖∞', 38, 125, 54, 63, 'digital', 'mapping',
        'Use the largest input magnitude to select the effective DAC rail · Restore the input scale after digitizing the result', 'C = ‖x‖∞'),
      node('bm', 'BM', 'retry', 245, 165, 46, 37, 'digital', 'feedback',
        'If the ADC clips, reduce the digital input and repeat the MVM · Restore the retry factor digitally after a successful read', 'x → x/2 → retry'),
      node('output', 'OUT', 'restore scale', 112, 242, 106, 32, 'digital', 'mapping',
        'Restore the digital input scaling and logical-to-conductance mapping after the analog read', 'y = sC · digitized response'),
    ] : []),
  ]
})
const active = computed(() => nodes.value.find(item => item.id === activeId.value))
// Keep full technical descriptions above; use concise prose for projected text.
const summaries = {
  gradient: 'Form outer products, then accumulate the batch gradient digitally',
  optimizer: 'Apply the optimizer in logical weight coordinates',
  weight: 'Use reduced-precision weights for digital matrix passes',
  master: 'Update FP32 master weights; quantize a working copy',
  dac: 'Convert digital inputs to voltages within the DAC rails',
  crossbar: 'Stored conductances perform forward and backward MVMs',
  adc: 'Digitize currents within the ADC range',
  buffer: 'Retain small updates; issue pulses at the device threshold',
  mapping: 'Match logical initialization to the physical conductance range',
  scaling: 'Align input magnitude to the DAC range; restore scale digitally',
  bm: 'Reduce the input and retry when the ADC clips',
  output: 'Restore input and weight scales after the analog read',
}
const activeSummary = computed(() => {
  if (selected.value === 'prior' && active.value?.id === 'optimizer')
    return 'Turn the digital gradient into an SGD update'
  if (selected.value === 'prior' && active.value?.id === 'mapping')
    return 'Convert physical responses to logical weights using a fixed scale'
  return summaries[active.value?.id]
})
const paths = computed(() => [
  { id: 'input', d: isAnalog.value ? 'M165 27 V69' : 'M165 27 V126', targets: ['dac', 'weight', 'scaling'] },
  { id: 'activation', d: 'M165 38 H516 V319 H429 V301', targets: ['gradient'] },
  { id: 'error', d: 'M235 343 V319 H429', targets: ['gradient'] },
  { id: 'backward', d: 'M235 319 V57', targets: ['crossbar', 'weight'] },
  { id: 'gradient', d: 'M429 254 V224', targets: ['gradient', 'optimizer'] },
  { id: 'update', d: 'M429 182 V159', targets: ['optimizer', 'buffer', 'master'] },
  { id: 'program', d: isAnalog.value ? 'M364 130 H331 V146 H229' : 'M364 122 H270 V158 H233', targets: ['buffer', 'master', 'crossbar', 'weight'] },
  ...(isAnalog.value ? [
    { id: 'read', d: 'M165 107 V119 M165 181 V192', targets: ['dac', 'crossbar', 'adc'] },
    { id: 'output', d: isOurs.value ? 'M165 230 V242 M165 274 V343' : 'M165 230 V343', targets: ['adc', 'output', 'mapping'] },
    { id: 'mapping', d: 'M142 319 H165', targets: ['mapping'] },
    ...(isOurs.value ? [
      { id: 'feedback', d: 'M218 213 H268 V202 M268 165 V88 H218', targets: ['bm', 'adc', 'dac'] },
      { id: 'scale', d: 'M65 188 V258 H112', targets: ['scaling', 'output'] },
    ] : []),
  ] : [{ id: 'output', d: 'M165 218 V343', targets: ['weight'] }]),
])

function selectView(id) {
  selected.value = id
  hovered.value = focused.value = pinned.value = null
}
function togglePin(id) {
  pinned.value = pinned.value === id ? null : id
}
async function navigateTabs(event, index) {
  const offsets = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: views.length - 1 - index }
  if (!(event.key in offsets)) return
  event.preventDefault()
  const next = (index + offsets[event.key] + views.length) % views.length
  selectView(views[next].id)
  await nextTick()
  tabButtons.value[next]?.focus()
}
</script>

<template>
  <div class="overview-comparison" @click.stop @pointerdown.stop @keydown.stop @keydown.esc="pinned = null">
    <div class="comparison-tabs" role="tablist" aria-label="Training architecture">
      <button v-for="(item, index) in views" :id="`${uid}-tab-${item.id}`" :key="item.id" ref="tabButtons"
        role="tab" :aria-selected="selected === item.id" :aria-controls="`${uid}-panel`"
        :tabindex="selected === item.id ? 0 : -1" @click="selectView(item.id)" @keydown="navigateTabs($event, index)">
        <span class="tab-number">0{{ index + 1 }}</span>{{ item.label }}
      </button>
    </div>

    <div :id="`${uid}-panel`" class="comparison-body" role="tabpanel" :aria-labelledby="`${uid}-tab-${selected}`">
      <div class="comparison-figure">
        <div class="figure-meta"><span>{{ view.citation }}</span><div class="comparison-legend"><span class="digital">Digital</span><span v-if="isAnalog" class="analog">Analog</span></div></div>
        <svg :key="selected" class="architecture-svg" viewBox="0 0 540 353" role="group" :aria-label="`${view.label} training diagram; focus or hover a component to explore`">
          <defs>
            <marker :id="`${uid}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke" /></marker>
            <pattern :id="`${uid}-converter`" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(40)"><rect width="6" height="6" fill="#fff" /><path d="M0 0 V6" stroke="#ecd2d2" stroke-width="2" /></pattern>
          </defs>
          <rect class="mvm-boundary" x="23" y="54" :width="isAnalog ? 283 : 246" height="233" rx="10" />
          <text class="boundary-label" x="36" y="276">{{ isAnalog ? 'ANALOG MVM' : 'DIGITAL MVM' }}</text>
          <text class="signal-label" x="110" y="18">Activation x</text>
          <text class="signal-label" x="140" y="352">Output</text>
          <text class="signal-label" x="218" y="352">Error δ</text>
          <text class="command-label" x="281" :y="isAnalog ? 119 : 110">{{ isAnalog ? 'pulses' : 'quantize' }}</text>

          <g class="connections" fill="none" stroke-width="1.7" :marker-end="`url(#${uid}-arrow)`">
            <path v-for="path in paths" :key="path.id" :d="path.d" :class="{ energized: path.targets.includes(activeId) }" />
          </g>
          <g v-for="item in nodes" :key="item.id" class="architecture-node" :class="[item.domain, { 'is-active': activeId === item.id, 'is-pinned': pinned === item.id }]"
            :transform="`translate(${item.x}, ${item.y})`" role="button" tabindex="0" :aria-label="`${item.title}: ${item.description}`" :aria-pressed="pinned === item.id"
            @mouseenter="hovered = item.id" @mouseleave="hovered = null" @focus="focused = item.id" @blur="focused = null"
            @click="togglePin(item.id)" @keydown.enter.prevent="togglePin(item.id)" @keydown.space.prevent="togglePin(item.id)">
            <title>{{ item.title }} — {{ item.description }}</title>
            <rect class="node-frame" :width="item.w" :height="item.h" rx="8" :fill="item.domain === 'converter' ? `url(#${uid}-converter)` : undefined" />
            <text class="node-title" :class="{ compact: item.w < 70 }" :x="item.w / 2" :y="item.h > 60 ? 22 : 16">{{ item.title }}</text>
            <text class="node-subtitle" :x="item.w / 2" :y="item.h > 60 ? 39 : 29">{{ item.sub }}</text>

            <!-- Glyphs animate only while a component is hovered, focused, or pinned. -->
            <g v-if="item.kind === 'array'" class="array-glyph" :transform="`translate(${item.w / 2 - 24}, ${item.h - 17})`">
              <path d="M0 0 H48 M0 6 H48 M0 12 H48 M6 -3 V15 M18 -3 V15 M30 -3 V15 M42 -3 V15" />
              <circle v-for="i in 4" :key="i" class="array-cell" :cx="i * 12 - 6" cy="6" r="2.5" :style="{ animationDelay: `${i * .12}s` }" />
            </g>
            <g v-else-if="item.kind === 'buffer'" class="buffer-glyph" :transform="`translate(${item.w / 2 - 22}, ${item.h - 12})`">
              <rect v-for="i in 5" :key="i" :x="(i - 1) * 9" y="0" width="6" height="5" rx="1" :style="{ animationDelay: `${i * .14}s` }" />
            </g>
            <path v-else-if="item.kind === 'feedback'" class="feedback-glyph" :d="`M${item.w - 8} 9 a5 5 0 1 1 -4 -5 l0 3`" />
            <path v-else-if="item.kind === 'converter'" class="converter-glyph" :d="`M8 ${item.h - 8} h4 v-4 h4 v-4 h4`" />
            <circle v-else class="signal-dot" :cx="7" :cy="item.h - 7" r="2.5" />
          </g>
          <g class="junctions"><circle cx="165" cy="38" r="3" /><circle cx="235" cy="319" r="3" /><circle cx="429" cy="319" r="3" /></g>
        </svg>
        <div class="figure-hint">Hover or focus to explore · Click to pin a component</div>
      </div>

      <aside class="comparison-copy">
        <div class="copy-kicker">{{ selected === 'ours' ? 'OUR DESIGN' : 'TRAINING PARTITION' }}</div>
        <h3 v-if="!active">{{ view.title }}</h3>
        <p v-if="!active" class="comparison-intro">{{ view.takeaway }}</p>
        <ul v-if="!active" class="comparison-points"><li v-for="point in view.points" :key="point">{{ point }}</li></ul>
        <div v-if="active" class="component-detail inspecting" aria-live="polite" aria-atomic="true">
          <div class="detail-heading"><strong>{{ active?.title ?? 'Key idea' }}</strong><span v-if="active">{{ active.domain === 'converter' ? 'CONVERSION' : active.domain.toUpperCase() }}</span></div>
          <div v-if="active" class="detail-equation">{{ active.equation }}</div>
          <p>{{ activeSummary }}</p>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.overview-comparison { margin-top: 12px; }
.comparison-tabs { display: flex; gap: 7px; padding-bottom: 13px; }
.comparison-tabs button { display: flex; align-items: center; gap: 8px; padding: 8px 13px; border: 1px solid var(--border); border-radius: 6px; background: white; font-size: 11px; color: var(--muted); cursor: pointer; transition: background .2s, border-color .2s; }
.tab-number { font-size: 9px; opacity: .65; }
.comparison-tabs button:hover { background: var(--panel); }
.comparison-tabs button[aria-selected="true"] { color: var(--accent); background: var(--selected); border-color: var(--accent); }
.comparison-tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.comparison-body { display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px; align-items: start; }
.comparison-figure { border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px 8px; background: #fdfdfd; }
.figure-meta { display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 9px; color: var(--muted); }
.comparison-legend { display: flex; gap: 12px; }
.comparison-legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
.comparison-legend .digital::before { background: #dce7f7; border: 1px solid #315fc1; }
.comparison-legend .analog::before { background: #f6dddd; border: 1px solid #b31b1b; }
.architecture-svg { display: block; width: 100%; height: 285px; margin: 8px 0 3px; animation: diagram-arrive .25s ease-out; }
.mvm-boundary { fill: #f4f4f4; stroke: #a6abb3; stroke-width: 1.2; stroke-dasharray: 5 4; }
.boundary-label { font-size: 8px; fill: #777; letter-spacing: 1px; }
.signal-label { font-size: 12px; fill: var(--ink); }
.command-label { font-size: 10px; fill: var(--muted); }
.connections { stroke: #848b95; }
.connections path { transition: stroke .2s; }
.connections .energized { stroke: var(--accent); stroke-dasharray: 6 4; animation: signal-flow .65s linear infinite; }
.junctions { fill: #848b95; }
.architecture-node { cursor: pointer; outline: none; }
.node-frame { stroke-width: 1.4; transition: stroke .2s, filter .2s; }
.digital .node-frame { fill: #e8effa; stroke: #6c87b4; }
.analog .node-frame { fill: #f6dddd; stroke: #b87878; }
.converter .node-frame { stroke: #9b7777; }
.architecture-node.is-active .node-frame { stroke: var(--accent); stroke-width: 2.2; filter: drop-shadow(0 2px 3px #b31b1b22); }
.architecture-node:focus-visible .node-frame { stroke: var(--accent); stroke-width: 2.5; stroke-dasharray: 4 2; }
.is-pinned .node-frame { stroke-width: 2.2; }
.node-title { text-anchor: middle; font-size: 13px; font-weight: 600; fill: #26364b; pointer-events: none; }
.node-title.compact { font-size: 11px; }
.node-subtitle { text-anchor: middle; font-size: 8.5px; fill: #526171; pointer-events: none; }
.array-glyph { stroke: #8699b0; stroke-width: .7; fill: #8699b0; pointer-events: none; }
.array-cell { stroke: none; }
.buffer-glyph { fill: #8ba0bf; }
.feedback-glyph, .converter-glyph { fill: none; stroke: #8396b3; stroke-width: 1.4; }
.signal-dot { fill: #8ba0bf; }
.is-active .array-cell { fill: var(--accent); animation: cell-pulse .8s ease-in-out infinite alternate; }
.is-active .buffer-glyph rect { fill: var(--accent); animation: buffer-fill .9s ease-in-out infinite; }
.is-active .feedback-glyph { stroke: var(--accent); stroke-dasharray: 4 2; animation: signal-flow .6s linear infinite; }
.is-active .converter-glyph { stroke: var(--accent); animation: cell-pulse .7s ease-in-out infinite alternate; }
.is-active .signal-dot { fill: var(--accent); animation: dot-travel 1.1s ease-in-out infinite; }
.figure-hint { font-size: 8.5px; color: var(--muted); text-align: center; }
.comparison-copy { padding-top: 5px; }
.copy-kicker { color: var(--accent); font-size: 9px; letter-spacing: 1.4px; font-weight: 600; }
.overview-comparison .comparison-copy h3 { font-size: var(--font-body); line-height: 1.2; margin: 8px 0 10px; letter-spacing: -.4px; }
.overview-comparison .comparison-intro { color: var(--muted); font-size: var(--font-body); line-height: 1.2; margin: 0; }
.comparison-points { padding-left: 15px; margin: 10px 0 12px; font-size: var(--font-body); color: var(--ink); line-height: 1.2; }
.comparison-points li { margin: 6px 0; padding-left: 2px; line-height: 1.2; }
.component-detail { border-top: 2px solid var(--border); padding: 12px 0 0; }
.component-detail.inspecting { border-color: var(--accent); }
.detail-heading { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.detail-heading strong { color: var(--accent); font-size: var(--font-body); font-weight: 600; }
.detail-heading span { font-size: 8px; color: var(--muted); letter-spacing: .8px; }
.detail-equation { font-size: var(--font-body); color: var(--blue); margin-top: 8px; }
.overview-comparison .component-detail p { font-size: var(--font-body); color: var(--muted); line-height: 1.2; margin: 8px 0 0; }
@keyframes signal-flow { to { stroke-dashoffset: -20; } }
@keyframes cell-pulse { from { opacity: .35; } to { opacity: 1; } }
@keyframes buffer-fill { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
@keyframes dot-travel { 0% { transform: translateX(0); opacity: .3; } 60% { opacity: 1; } 100% { transform: translateX(28px); opacity: 0; } }
@keyframes diagram-arrive { from { opacity: .3; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
