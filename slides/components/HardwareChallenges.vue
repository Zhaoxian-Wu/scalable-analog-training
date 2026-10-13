<script setup>
import { ref, useId } from 'vue'

const active = ref(null)
const instanceId = useId()
const challenges = [
  {
    id: 'converters', title: 'Quantization-like error', subtitle: 'in DAC / ADC',
    description: 'Finite DAC / ADC levels introduce rounding error',
    consequence: 'Distorts both matrix passes',
    formula: 'ε = Q(z) − z',
    diagram: 'Ideal linear conversion and a staircase approximation with a highlighted rounding error',
  },
  {
    id: 'granularity', title: 'Limited update granularity',
    description: 'Each pulse changes conductance by a finite step',
    consequence: 'Small updates may be below one pulse step',
    formula: 'One pulse ≈ Δwₘᵢₙ',
    diagram: 'Programming pulses produce finite conductance steps; one pulse step is marked as delta w min',
  },
  {
    id: 'asymmetry', title: 'Asymmetric updates',
    description: 'Positive and negative pulses can differ',
    consequence: 'Opposite pulses need not cancel',
    formula: '|Δw₊| ≠ |Δw₋|',
    diagram: 'A positive pulse raises the weight and a negative pulse lowers it by a smaller amount, leaving an offset',
  },
]
const toggle = id => { active.value = active.value === id ? null : id }
const panelId = id => `${instanceId}-${id}`

// Deliberately coarse, deterministic schematics rather than device measurements
const converterSteps = 'M 38 176 H 56 V 157 H 91 V 138 H 126 V 119 H 161 V 100 H 196 V 81 H 231 V 62 H 248'
</script>

<template>
  <div class="hardware-challenges" @click.stop @pointerdown.stop @keydown.esc.stop.prevent="active = null">
    <div class="hardware-challenge-row" :class="{ 'has-selection': active !== null }">
      <article
        v-for="(challenge, index) in challenges" :key="challenge.id"
        class="hardware-challenge"
        :class="{ expanded: active === challenge.id, collapsed: active !== null && active !== challenge.id }"
        @click="toggle(challenge.id)"
      >
        <button
          class="hardware-challenge-toggle" :aria-expanded="active === challenge.id"
          :aria-controls="panelId(challenge.id)" @click.stop="toggle(challenge.id)"
          @keydown.space.stop @keydown.enter.stop
        >
          <span class="hardware-challenge-index">0{{ index + 1 }}</span>
          <span class="hardware-challenge-title">{{ challenge.title }}<span v-if="challenge.subtitle">{{ challenge.subtitle }}</span></span>
          <span class="hardware-challenge-icon" aria-hidden="true">{{ active === challenge.id ? '−' : '+' }}</span>
        </button>

        <div class="hardware-challenge-body" :aria-hidden="active !== null && active !== challenge.id">
          <div class="hardware-challenge-visual">
            <svg viewBox="0 0 280 230" role="img" :aria-label="challenge.diagram">
              <template v-if="challenge.id === 'converters'">
                <path class="hw-axis" d="M 38 36 V 176 H 248" />
                <text class="hw-label" x="38" y="23">Converted signal</text>
                <path class="hw-ideal" d="M 38 176 L 248 62" />
                <path class="hw-actual" :d="converterSteps" />
                <path class="hw-error" d="M 176 101 V 119 M 170 101 H 182 M 170 119 H 182" />
                <text class="hw-label hw-error-label" x="185" y="131">error</text>
                <text class="hw-label" x="143" y="200" text-anchor="middle">Input signal</text>
                <path class="hw-ideal hw-swatch" d="M 41 220 H 59" /><text class="hw-legend" x="65" y="224">Ideal</text>
                <path class="hw-actual hw-swatch" d="M 141 220 H 159" /><text class="hw-legend" x="165" y="224">DAC / ADC</text>
              </template>

              <template v-else-if="challenge.id === 'granularity'">
                <text class="hw-label" x="38" y="23">Programming pulses</text>
                <path class="hw-pulses" d="M 38 55 H 65 V 34 H 78 V 55 H 120 V 34 H 133 V 55 H 175 V 34 H 188 V 55 H 230" />
                <path class="hw-axis" d="M 38 83 V 176 H 248" />
                <text class="hw-label" x="38" y="78">Conductance</text>
                <path class="hw-ideal" d="M 38 176 L 214 101" stroke-dasharray="5 4" />
                <path class="hw-actual" d="M 38 176 H 65 V 151 H 120 V 126 H 175 V 101 H 230" />
                <path class="hw-error" d="M 236 126 V 151 M 231 126 H 241 M 231 151 H 241" />
                <text class="hw-label hw-error-label" x="242" y="144">Δw</text>
                <text class="hw-label" x="143" y="200" text-anchor="middle">Update commands</text>
                <path class="hw-ideal hw-swatch" d="M 41 220 H 59" stroke-dasharray="5 4" /><text class="hw-legend" x="65" y="224">Target</text>
                <path class="hw-actual hw-swatch" d="M 141 220 H 159" /><text class="hw-legend" x="165" y="224">Pulsed</text>
              </template>

              <template v-else>
                <text class="hw-label" x="38" y="23">Stored weight</text>
                <path class="hw-axis" d="M 38 36 V 176 H 248" />
                <path class="hw-ideal" d="M 38 157 H 248" stroke-dasharray="5 4" />
                <path class="hw-actual" d="M 58 157 L 136 65 L 214 119" />
                <circle v-for="point in [[58, 157], [136, 65], [214, 119]]" :key="point[0]" :cx="point[0]" :cy="point[1]" r="4" class="hw-dot" />
                <text class="hw-label" x="84" y="89">+ pulse</text>
                <text class="hw-label" x="169" y="75">− pulse</text>
                <path class="hw-error" d="M 234 119 V 157 M 229 119 H 239 M 229 157 H 239" />
                <text class="hw-label hw-error-label" x="172" y="146">offset</text>
                <text class="hw-label" x="58" y="199" text-anchor="middle">Start</text>
                <text class="hw-label" x="214" y="199" text-anchor="middle">After + / −</text>
                <path class="hw-ideal hw-swatch" d="M 41 220 H 59" stroke-dasharray="5 4" /><text class="hw-legend" x="65" y="224">Return</text>
                <path class="hw-actual hw-swatch" d="M 141 220 H 159" /><text class="hw-legend" x="165" y="224">Actual</text>
              </template>
            </svg>
          </div>

          <div class="hardware-challenge-detail-track">
            <div :id="panelId(challenge.id)" class="hardware-challenge-detail" :aria-hidden="active !== challenge.id" :inert="active !== challenge.id">
              <p>{{ challenge.description }}</p>
              <div class="hardware-challenge-formula">{{ challenge.formula }}</div>
              <p class="hardware-challenge-consequence">{{ challenge.consequence }}</p>
            </div>
          </div>
        </div>
      </article>
    </div>
    <div class="hardware-challenge-hint">{{ active === null ? 'Click a challenge to explore' : 'Click another title to switch · Click the open title or press Esc to reset' }}</div>
  </div>
</template>

<style scoped>
.hardware-challenges { --hw-easing: cubic-bezier(.22, 1, .36, 1); }
.hardware-challenge-row { display: flex; align-items: center; gap: 14px; height: 354px; }
.hardware-challenge { flex: 1 1 0; min-width: 0; height: 326px; padding: 16px; overflow: hidden; border: 1px solid var(--border); border-top: 3px solid var(--accent); border-radius: 10px; background: var(--panel); transition: flex-grow .48s var(--hw-easing), height .48s var(--hw-easing), background .3s ease; }
.hardware-challenge.expanded { flex-grow: 4.3; height: 354px; background: var(--paper); }
.hardware-challenge.collapsed { flex-grow: 1; height: 136px; padding: 12px; background: var(--panel); }
.hardware-challenge-toggle { display: grid; grid-template-columns: 1fr 20px; align-content: start; gap: 8px; width: 100%; height: 114px; padding: 0; border: 0; background: transparent; color: var(--ink); text-align: left; cursor: pointer; }
.hardware-challenge-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 5px; border-radius: 3px; }
.hardware-challenge-index { grid-column: 1 / -1; color: var(--accent); font-size: 11px; letter-spacing: 1px; }
.hardware-challenge-title { font-size: var(--font-body); line-height: 1.15; font-weight: 550; transition: font-size .48s var(--hw-easing); }
.hardware-challenge-title > span { display: block; }
.hardware-challenge-icon { color: var(--accent); font-size: 22px; line-height: 1; }
.collapsed .hardware-challenge-title { font-size: 18px; }
.collapsed .hardware-challenge-icon, .collapsed .hardware-challenge-index { display: none; }
.collapsed .hardware-challenge-toggle { display: flex; align-items: center; height: 106px; }
.expanded .hardware-challenge-toggle { height: 86px; }
.hardware-challenge-body { display: grid; grid-template-columns: minmax(0, 1fr) 0fr; align-items: center; height: 174px; opacity: 1; transition: grid-template-columns .48s var(--hw-easing), opacity .18s ease, transform .48s var(--hw-easing); }
.expanded .hardware-challenge-body { grid-template-columns: minmax(0, 1.05fr) 1fr; height: 230px; }
.collapsed .hardware-challenge-body { opacity: 0; transform: translateY(15px) scale(.92); pointer-events: none; visibility: hidden; transition: opacity .14s ease, transform .48s var(--hw-easing), visibility 0s .18s; }
.hardware-challenge-visual { position: relative; min-width: 0; height: 174px; }
/* Scale the complete SVG so Chromium paints text and paths at the same size */
.hardware-challenge-visual svg { position: absolute; top: 0; left: 50%; display: block; width: 280px; max-width: none; height: 230px; overflow: visible; transform: translateX(-50%) scale(.7565); transform-origin: center top; }
.expanded .hardware-challenge-visual { height: 230px; }
.expanded .hardware-challenge-visual svg { transform: translateX(-50%) scale(.98); }
.hardware-challenge-detail-track { min-width: 0; overflow: hidden; }
.hardware-challenge-detail { min-width: 222px; padding-left: 16px; opacity: 0; transform: translateX(12px); visibility: hidden; transition: opacity .16s ease, transform .32s var(--hw-easing), visibility 0s .16s; }
.expanded .hardware-challenge-detail { opacity: 1; transform: translateX(0); visibility: visible; transition-delay: .16s; }
.hardware-challenge-detail p { margin: 0; font-size: var(--font-body); line-height: 1.2; }
.hardware-challenge-formula { margin: 12px 0; color: var(--accent); font-size: var(--font-body); font-weight: 550; }
.hardware-challenge-detail .hardware-challenge-consequence { color: var(--muted); }
.hardware-challenge-hint { height: 18px; margin-top: 10px; color: var(--muted); font-size: 12px; text-align: center; }
.hw-axis { fill: none; stroke: var(--border); stroke-width: 1.4; }
.hw-ideal { fill: none; stroke: var(--blue); stroke-width: 2; }
.hw-actual { fill: none; stroke: var(--accent); stroke-width: 3; stroke-linejoin: round; }
.hw-error { fill: none; stroke: var(--orange); stroke-width: 1.6; }
.hw-pulses { fill: none; stroke: var(--accent); stroke-width: 2; }
.hw-dot { fill: var(--accent); }
.hardware-challenge-visual svg text, .hw-swatch { visibility: hidden; }
.expanded .hardware-challenge-visual svg text, .expanded .hw-swatch { visibility: visible; }
.hw-label { fill: var(--muted); font-size: 14px; font-family: Inter, 'Segoe UI', sans-serif; }
.hw-legend { fill: var(--muted); font-size: 13px; }
.hw-error-label { fill: var(--orange); }
@media (prefers-reduced-motion: reduce) { .hardware-challenges *, .hardware-challenges *::before, .hardware-challenges *::after { transition: none !important; } }
</style>
