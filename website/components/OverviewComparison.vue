<script setup>
import { computed, nextTick, ref, useId } from 'vue'
import katex from 'katex'

const math = tex => katex.renderToString(tex, { throwOnError: false })
const adamEquation = String.raw`\begin{aligned}
m_k &= \beta_1 m_{k-1} + (1-\beta_1)G_k \\
v_k &= \beta_2 v_{k-1} + (1-\beta_2)(G_k \odot G_k) \\
\widehat m_k &= \frac{m_k}{1-\beta_1^k} \\
\widehat v_k &= \frac{v_k}{1-\beta_2^k} \\
\Delta W_k &= -\eta_k\frac{\widehat m_k}{\sqrt{\widehat v_k}+\epsilon}
\end{aligned}`
const residualEquation = String.raw`\begin{aligned}
H_{k+\frac12} &= H_k + \Delta W_k \\
N_k &= \operatorname{trunc}\!\left(\frac{H_{k+\frac12}}{s\,\Delta w_{\min}}\right) \\
H_{k+1} &= H_{k+\frac12} - s\,N_k\Delta w_{\min}
\end{aligned}`
const bmEquation = String.raw`\begin{aligned}
u &= x/C, \qquad C=\lVert x\rVert_\infty \\
r_n &= \mathcal{A}(u/2^n), \qquad n=0,1,2,\ldots \\
y &\approx sC\,2^{n_*}r_{n_*}
\end{aligned}`
const logicalWeightNotation = [
  [String.raw`W,\widehat W,\widehat W^\diamond`, 'Logical weight matrix, physical conductance array, and fixed reference array'],
  ['s', 'Scale converting conductance differences into logical weights'],
]
const mvmNotation = [
  ['W,x,y', 'Logical weight matrix, input activation vector, and output vector'],
  [String.raw`\delta_{\mathrm{in}},\delta_{\mathrm{out}}`, 'Backpropagated input and output error vectors'],
  [String.raw`(\cdot)^{\mathsf T}`, 'Matrix or vector transpose'],
]
const sgdNotation = [
  ['k,G_k', 'Training-step index and batch gradient in logical-weight units'],
  [String.raw`\eta_k,\Delta W_k`, 'Learning rate and signed logical-weight increment at step k'],
]
const notationByNode = {
  gradient: [
    ['G,B,b', 'Batch gradient, batch size, and sample index'],
    [String.raw`x_b,\delta_b`, 'Input activations and output errors for sample b'],
    [String.raw`(\cdot)^{\mathsf T}`, 'Transpose; each outer product has the shape of the weight matrix'],
  ],
  optimizer: [
    ...sgdNotation,
    ['m_k,v_k', 'First and second gradient moments; both start at zero'],
    [String.raw`\beta_1,\beta_2`, 'Exponential decay factors for the two moments, each in [0, 1)'],
    [String.raw`\widehat m_k,\widehat v_k`, 'Bias-corrected first and second moments'],
    [String.raw`\epsilon`, 'Positive constant that stabilizes division'],
    [String.raw`\odot`, 'Elementwise multiplication; division and square root are also elementwise'],
  ],
  weight: mvmNotation,
  master: [[String.raw`W_{32},\Delta W`, 'FP32 master weights and the signed logical-weight increment']],
  dac: [['x,C', 'Input vector and its largest absolute entry; a zero input can bypass the read'], ['DAC', 'Digital-to-analog conversion']],
  crossbar: mvmNotation,
  adc: [['ADC', 'Analog-to-digital conversion of summed output currents']],
  buffer: [
    ['k', 'Training-step index; k + ½ labels accumulation before pulse transfer'],
    [String.raw`H_k,\Delta W_k`, 'Residual buffer and requested increment, both in logical-weight units'],
    ['s', 'Logical-to-conductance mapping scale'],
    [String.raw`\Delta w_{\min}`, 'Nominal conductance increment of one pulse'],
    ['N_k', 'Signed integer pulse count, computed separately for each weight'],
    [String.raw`\operatorname{trunc}`, 'Elementwise truncation toward zero; the untransferred remainder stays in H'],
  ],
  mapping: [...logicalWeightNotation, ['D', 'Number of inputs to the layer; the proposed initialization scale is proportional to 1/√D']],
  scaling: [['x,C', 'Input vector and its maximum absolute entry; a zero input can bypass the read'], [String.raw`\lVert x\rVert_\infty`, 'Infinity norm: the largest absolute component of x']],
  bm: [
    ['x,C,u', 'Original input, maximum absolute input, and normalized input x/C; bypass the read for a zero vector'],
    [String.raw`\mathcal A`, 'One DAC–crossbar–ADC read with fixed converter rails, including quantization and read noise'],
    ['n,r_n', 'Retry index (zero for the first attempt) and its digitized physical response'],
    ['n_*', 'First attempt without detected ADC saturation, subject to a configured retry limit'],
    ['s,y', 'Logical mapping scale and restored logical output vector'],
  ],
  output: [['s,C', 'Logical mapping scale and input normalization factor'], ['y', 'Logical output restored from the digitized physical response']],
}

const props = defineProps({
  initialView: {
    type: String,
    default: 'ours',
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
    node('gradient', 'W-grad', 'digital accumulation', 364, 315, 129, 47, 'digital', 'gradient',
      'Form outer products of activations and errors, then accumulate the batch weight gradient digitally', String.raw`G = \frac{1}{B} \sum_{b=1}^{B} \delta_b x_b^{\mathsf{T}}`),
    node('optimizer', selected.value === 'prior' ? 'SGD' : 'Adam', 'logical update', 364, 227, 129, 42, 'digital', 'optimizer',
      selected.value === 'prior'
        ? 'The illustrated prior design turns the accumulated weight gradient into an SGD update'
        : 'Compute first and second moments digitally, starting from zero · Bias-correct the moments and apply elementwise preconditioning to produce a signed logical-weight increment',
      selected.value === 'prior' ? String.raw`\Delta W_k = -\eta_k G_k` : adamEquation),
  ]
  if (!isAnalog.value) return [
    node('weight', 'Weight · FP16', 'reduced-precision MVM', 99, 126, 134, 92, 'digital', 'array',
      'Use a reduced-precision weight representation for selected forward and backward matrix arithmetic', String.raw`y = Wx, \quad \delta_{\mathrm{in}} = W^{\mathsf{T}} \delta_{\mathrm{out}}`),
    node('master', 'Weight · FP32', 'master weights', 364, 86, 129, 73, 'digital', 'buffer',
      'Apply updates to the full-precision master weights, then quantize a working copy for matrix arithmetic', String.raw`W_{32} \leftarrow W_{32} + \Delta W`),
    ...common,
  ]
  return [
    node('dac', 'DAC', 'voltage input', 101, 69, 128, 38, 'converter', 'converter',
      isOurs.value ? 'Digitally rescale activations or backward errors, then convert them to voltages within the fixed DAC rails'
        : 'Convert digital activations or backward errors to crossbar input voltages using the baseline converter range',
      isOurs.value ? String.raw`x/C \to \mathrm{DAC}` : String.raw`x \to \mathrm{DAC}`),
    node('crossbar', 'Crossbar', 'conductance weights', 101, 119, 128, 88, 'analog', 'array',
      'Stored conductances execute forward and transposed backward MVMs; W-grad stays digital',
      String.raw`y = Wx, \quad \delta_{\mathrm{in}} = W^{\mathsf{T}} \delta_{\mathrm{out}}`),
    node('adc', 'ADC', 'digitized current', 101, 224, 128, 38, 'converter', 'converter',
      'Digitize the summed crossbar output current · Fixed ADC rails limit range and conversion resolution', String.raw`\mathrm{current} \to \mathrm{digital}`),
    node('buffer', 'Residual buffer', 'retain small updates', 364, 135, 129, 55, 'digital', 'buffer',
      'Accumulate increments in logical-weight units · Divide by the nominal logical pulse size and truncate toward zero to select signed pulse counts · Retain the untransferred residual · Actual open-loop writes remain imperfect',
      residualEquation),
    node('mapping', 'Mapping', isOurs.value ? String.raw`s \propto D^{-1/2}` : 'fixed logical scale', 27, 365, 115, 39, 'digital', 'mapping',
      isOurs.value ? 'Use a width-aware scale to preserve logical initialization while occupying a fixed fraction of the physical device range'
        : 'Convert the physical crossbar response to logical weights using the baseline mapping scale',
      String.raw`W = s (\widehat{W} - \widehat{W}^{\diamond})`),
    ...common,
    ...(isOurs.value ? [
      node('scaling', 'Scaling', String.raw`\lVert x \rVert_{\infty}`, 38, 125, 54, 44, 'digital', 'mapping',
        'Use the largest input magnitude to select the effective DAC rail · Restore the input scale after digitizing the result', String.raw`C = \lVert x \rVert_{\infty}`),
      node('bm', 'BM', 'retry', 239, 195, 58, 37, 'digital', 'feedback',
        'Bound management uses ADC saturation feedback to select an input scale that fits the fixed output range', bmEquation),
      node('output', 'OUT', 'restore scale', 101, 280, 128, 38, 'digital', 'mapping',
        'Restore the digital input scaling and logical-to-conductance mapping after the analog read', String.raw`y = sC \cdot \mathrm{digitized\ response}`),
    ] : []),
  ]
})
const active = computed(() => nodes.value.find(item => item.id === activeId.value))
const activeNotation = computed(() => {
  if (active.value?.id === 'optimizer' && selected.value === 'prior') return sgdNotation
  if (active.value?.id === 'mapping' && !isOurs.value) return logicalWeightNotation
  return notationByNode[active.value?.id] ?? []
})
const paths = computed(() => [
  { id: 'input', d: 'M165 27 V443', targets: ['dac', 'weight', 'scaling', 'output', 'mapping'] },
  { id: 'activation', d: 'M165 38 H228 Q235 50 242 38 H516 V407 H429 V362', targets: ['gradient'] },
  { id: 'error', d: 'M235 443 V407 H429', targets: ['gradient'] },
  { id: 'backward', d: 'M235 407 V5', targets: ['crossbar', 'weight'] },
  { id: 'gradient', d: 'M429 315 V269', targets: ['gradient', 'optimizer'] },
  { id: 'update', d: isAnalog.value ? 'M429 227 V190' : 'M429 227 V159', targets: ['optimizer', 'buffer', 'master'] },
  { id: 'program', d: isOurs.value ? 'M364 163 H275 Q268 151 261 163 H229' : isAnalog.value ? 'M364 163 H229' : 'M344 122 H271', targets: ['buffer', 'master', 'crossbar', 'weight'] },
  ...(isAnalog.value ? [
    // This free-standing MVM arrow is separate from the covered activation/output path.
    { id: 'read', d: 'M165 62 V333', targets: ['dac', 'crossbar', 'adc', 'output'] },
    { id: 'mapping', d: 'M142 384 H165', targets: ['mapping'] },
    ...(isOurs.value ? [
      { id: 'scale-input', d: 'M165 62 H65 V125', targets: ['dac', 'scaling'] },
      { id: 'feedback', d: 'M229 243 H268 V232 M268 195 V88 H229', targets: ['bm', 'adc', 'dac'] },
      { id: 'scale', d: 'M65 169 V299 H101', targets: ['scaling', 'output'] },
    ] : []),
  ] : [{ id: 'read', d: 'M165 118 V233', targets: ['weight'] }]),
])
// The MVM region covers external signal paths; its internal wiring stays visible.
const externalSignalIds = new Set(['input', 'activation', 'error', 'backward'])
const backgroundPaths = computed(() => paths.value.filter(path => externalSignalIds.has(path.id)))
const foregroundPaths = computed(() => paths.value.filter(path => !externalSignalIds.has(path.id)))
// Clear the wire underneath each bridge so crossings cannot be read as junctions.
const crossingJumps = computed(() => [
  { id: 'activation', d: 'M228 38 Q235 50 242 38', surface: '#fdfdfd' },
  ...(isOurs.value ? [{ id: 'program', d: 'M275 163 Q268 151 261 163', surface: '#f4f4f4' }] : []),
].map(jump => ({ ...jump, targets: paths.value.find(path => path.id === jump.id).targets })))

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
        {{ item.label }}
      </button>
    </div>

    <div :id="`${uid}-panel`" class="comparison-body" role="tabpanel" :aria-labelledby="`${uid}-tab-${selected}`">
      <div class="comparison-figure">
        <div class="figure-meta"><span>{{ view.citation }}</span><div class="comparison-legend"><span class="digital">Digital</span><span v-if="isAnalog" class="analog">Analog</span></div></div>
        <svg :key="selected" class="architecture-svg" viewBox="0 0 540 470" role="group" :aria-label="`${view.label} training diagram; focus or hover a component to explore`">
          <defs>
            <marker :id="`${uid}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke" /></marker>
            <pattern :id="`${uid}-converter`" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(40)"><rect width="6" height="6" fill="#fff" /><path d="M0 0 V6" stroke="#ecd2d2" stroke-width="2" /></pattern>
          </defs>
          <g class="connections" fill="none" stroke-width="1.7">
            <path v-for="path in backgroundPaths" :key="path.id" :d="path.d" :marker-end="path.id === 'error' ? undefined : `url(#${uid}-arrow)`" :class="{ energized: path.targets.includes(activeId) }" />
          </g>
          <rect class="mvm-boundary" :x="isAnalog ? 23 : 85" :y="isAnalog ? 54 : 111" :width="isAnalog ? 283 : 174" :height="isAnalog ? 286 : 135" rx="10" />
          <text class="boundary-label" :x="isAnalog ? 36 : 210" :y="isAnalog ? 330 : 239">{{ isAnalog ? 'ANALOG MVM' : 'MVM' }}</text>
          <foreignObject x="98" y="0" width="134" height="23">
            <div xmlns="http://www.w3.org/1999/xhtml" class="signal-label signal-math">Activation <span v-html="math('x')" /></div>
          </foreignObject>
          <text class="signal-label" x="165" y="461" text-anchor="middle">Output</text>
          <foreignObject x="205" y="447" width="70" height="23">
            <div xmlns="http://www.w3.org/1999/xhtml" class="signal-label signal-math">Error <span v-html="math(String.raw`\delta`)" /></div>
          </foreignObject>
          <text class="command-label" x="281" :y="isAnalog ? 153 : 144">{{ isAnalog ? 'pulses' : 'quantize' }}</text>

          <g class="connections" fill="none" stroke-width="1.7">
            <path v-for="path in foregroundPaths" :key="path.id" :d="path.d" :marker-end="`url(#${uid}-arrow)`" :class="{ energized: path.targets.includes(activeId) }" />
          </g>
          <g class="connections" fill="none" stroke-width="1.7" aria-hidden="true">
            <g v-for="jump in crossingJumps" :key="jump.id">
              <path :d="jump.d" :stroke="jump.surface" stroke-width="5" stroke-linecap="round" />
              <path :d="jump.d" :class="{ energized: jump.targets.includes(activeId) }" />
            </g>
          </g>
          <g v-for="item in nodes" :key="item.id" class="architecture-node" :class="[item.domain, { 'is-active': activeId === item.id, 'is-pinned': pinned === item.id }]"
            :transform="`translate(${item.x}, ${item.y})`" role="button" tabindex="0" :aria-label="`${item.title}: ${item.description}`" :aria-pressed="pinned === item.id"
            @mouseenter="hovered = item.id" @mouseleave="hovered = null" @focus="focused = item.id" @blur="focused = null"
            @click="togglePin(item.id)" @keydown.enter.prevent="togglePin(item.id)" @keydown.space.prevent="togglePin(item.id)">
            <title>{{ item.title }} — {{ item.description }}</title>
            <rect class="node-frame" :width="item.w" :height="item.h" rx="8" :fill="item.domain === 'converter' ? `url(#${uid}-converter)` : undefined" />
            <text class="node-title" :class="{ compact: item.w < 70 }" :x="item.kind === 'feedback' ? (item.w - 18) / 2 : item.w / 2" :y="item.h > 60 ? 22 : 16">{{ item.title }}</text>
            <foreignObject v-if="item.id === 'scaling' || (item.id === 'mapping' && isOurs)" x="0" :y="item.h > 60 ? 28 : 20" :width="item.w" height="22">
              <div xmlns="http://www.w3.org/1999/xhtml" class="node-math" v-html="math(item.sub)" />
            </foreignObject>
            <text v-else class="node-subtitle" :x="item.w / 2" :y="item.h > 60 ? 39 : 29">{{ item.sub }}</text>

            <!-- Glyphs animate only while a component is hovered, focused, or pinned. -->
            <g v-if="item.kind === 'array'" class="array-glyph" :transform="`translate(${item.w / 2 - 24}, ${item.h - 34})`">
              <path d="M0 0 H48 M0 12 H48 M0 24 H48 M6 -4 V28 M24 -4 V28 M42 -4 V28" />
              <circle v-for="i in 9" :key="i" class="array-cell" :cx="6 + ((i - 1) % 3) * 18" :cy="Math.floor((i - 1) / 3) * 12" r="2.5" :style="{ animationDelay: `${i * .12}s` }" />
            </g>
            <g v-else-if="item.kind === 'buffer'" class="buffer-glyph" :transform="`translate(${item.w / 2 - 22}, ${item.h - 12})`">
              <rect v-for="i in 5" :key="i" :x="(i - 1) * 9" y="0" width="6" height="5" rx="1" :style="{ animationDelay: `${i * .14}s` }" />
            </g>
            <path v-else-if="item.kind === 'feedback'" class="feedback-glyph" :d="`M${item.w - 8} 9 a5 5 0 1 1 -4 -5 l0 3`" />
            <path v-else-if="item.kind === 'converter'" class="converter-glyph" :d="`M8 ${item.h - 8} h4 v-4 h4 v-4 h4`" />
            <circle v-else-if="item.id !== 'scaling'" class="signal-dot" :cx="7" :cy="item.h - 7" r="2.5" />
          </g>
          <g class="junctions">
            <circle cx="165" cy="38" r="3" /><circle cx="235" cy="407" r="3" /><circle cx="429" cy="407" r="3" />
            <circle v-if="isOurs" cx="165" cy="62" r="3" />
            <circle v-if="isAnalog" cx="165" cy="384" r="3" />
          </g>
        </svg>
        <div class="figure-hint">Hover or focus to explore · Click to pin a component</div>
      </div>

      <aside class="comparison-copy">
        <div class="copy-kicker">{{ selected === 'ours' ? 'OUR DESIGN' : 'TRAINING PARTITION' }}</div>
        <h3 v-if="!active">{{ view.title }}</h3>
        <p v-if="!active" class="comparison-intro">{{ view.intro }}</p>
        <ul v-if="!active" class="comparison-points"><li v-for="point in view.points" :key="point">{{ point }}</li></ul>
        <div v-if="active" class="component-detail inspecting" aria-live="polite" aria-atomic="true">
          <div class="detail-heading"><strong>{{ active?.title ?? 'Key idea' }}</strong><span v-if="active">{{ active.domain === 'converter' ? 'CONVERSION' : active.domain.toUpperCase() }}</span></div>
          <div v-if="active" class="detail-equation" v-html="math(active.equation)" />
          <p>{{ active.description }}</p>
          <div v-if="active.id === 'bm'" class="bm-explanation">
            <ol>
              <li>Read the array with the normalized input and check whether any ADC channel saturates.</li>
              <li>If it saturates, halve the DAC input and repeat the entire MVM. Continue until no saturation is detected or the retry limit is reached.</li>
              <li>After a successful read, multiply the response by the accumulated power of two, then restore the input and logical-weight scales.</li>
            </ol>
            <p>For example, a response of 3.2 exceeds an ADC rail of ±1. Two halvings produce 0.8; digital multiplication by 4 restores 3.2 before the other scales are applied.</p>
            <p>Each retry repeats the array read and conversions. Output rescaling also amplifies read noise and ADC quantization error. This feedback differs from Scaling, which selects the initial input scale before the read.</p>
            <p class="detail-citation">Source: <a href="https://doi.org/10.3389/fnins.2017.00538" target="_blank" rel="noopener">Gokmen, Onen &amp; Haensch (2017), Training Deep Convolutional Neural Networks with Resistive Cross-Point Devices</a>, “Noise and bound management techniques.”</p>
          </div>
          <dl v-if="activeNotation.length" class="detail-notation" aria-label="Equation notation">
            <div v-for="[symbol, definition] in activeNotation" :key="symbol">
              <dt v-html="math(symbol)" />
              <dd>{{ definition }}</dd>
            </div>
          </dl>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.overview-comparison { margin-top: 12px; }
.comparison-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
.comparison-tabs button { display: flex; align-items: center; justify-content: center; min-width: 0; margin-bottom: -1px; padding: 12px 10px; border: 1px solid var(--border); border-radius: 6px 6px 0 0; background: var(--panel); font-size: 11px; line-height: 1.4; text-align: center; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); cursor: pointer; transition: background .2s, color .2s; }
.comparison-tabs button:hover { color: var(--ink); }
.comparison-tabs button[aria-selected="true"] { color: var(--ink); background: white; border-bottom-color: white; font-weight: 600; }
.comparison-tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.comparison-body { display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px; align-items: start; }
.comparison-figure { border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px 8px; background: #fdfdfd; }
.figure-meta { display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 9px; color: var(--muted); }
.comparison-legend { display: flex; gap: 12px; }
.comparison-legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
.comparison-legend .digital::before { background: #dce7f7; border: 1px solid #315fc1; }
.comparison-legend .analog::before { background: #f6dddd; border: 1px solid #b31b1b; }
.architecture-svg { display: block; width: 100%; height: auto; margin: 8px 0 3px; animation: diagram-arrive .25s ease-out; }
.mvm-boundary { fill: #f4f4f4; stroke: #a6abb3; stroke-width: 1.2; stroke-dasharray: 5 4; }
.boundary-label { font-size: 8px; fill: #777; letter-spacing: 1px; }
.signal-label { font-size: 12px; fill: var(--ink); }
.signal-math { text-align: center; color: var(--ink); }
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
.node-math { text-align: center; font-size: 13px; line-height: 1.2; color: #526171; pointer-events: none; }
.node-math :deep(.katex) { font-size: 1em; }
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
.detail-equation { font-size: var(--font-body); color: #000; text-align: center; margin-top: 8px; overflow-x: auto; padding: 8px 0; }
.detail-equation :deep(.katex) { font-size: 1em; }
.detail-notation { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 14px; margin: 16px 0 0; padding-top: 14px; border-top: 1px solid var(--border); font-size: 13px; line-height: 1.5; }
.detail-notation > div { display: contents; }
.detail-notation dt { color: #000; }
.detail-notation dd { margin: 0; color: var(--muted); }
.detail-notation :deep(.katex) { font-size: 1em; }
.bm-explanation ol { padding-left: 22px; margin: 12px 0; font-size: var(--font-body); line-height: 1.6; color: var(--muted); }
.bm-explanation li { margin: 6px 0; }
.overview-comparison .component-detail .detail-citation { font-size: 13px; }
.overview-comparison .component-detail p { font-size: var(--font-body); color: var(--muted); line-height: 1.2; margin: 8px 0 0; }
@keyframes signal-flow { to { stroke-dashoffset: -20; } }
@keyframes cell-pulse { from { opacity: .35; } to { opacity: 1; } }
@keyframes buffer-fill { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }
@keyframes dot-travel { 0% { transform: translateX(0); opacity: .3; } 60% { opacity: 1; } 100% { transform: translateX(28px); opacity: 0; } }
@keyframes diagram-arrive { from { opacity: .3; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
