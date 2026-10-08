<script setup>
import { computed, ref, shallowRef, useId } from 'vue'
import studies from './trainingScaleData.json'

const frame = { left: 58, right: 625, top: 20, bottom: 332 }
const x = value => frame.left + (Math.log10(value) - Math.log10(5)) / (Math.log10(30_000_000_000) - Math.log10(5)) * (frame.right - frame.left)
const y = value => frame.bottom - (Math.log10(value) - Math.log10(5)) / (Math.log10(1_000_000_000) - Math.log10(5)) * (frame.bottom - frame.top)
const decades = Array.from({ length: 11 }, (_, i) => i)
const gridLines = computed(() => decades.flatMap(decade => Array.from({ length: 9 }, (_, i) => (i + 1) * 10 ** decade)))
const groups = computed(() => {
  const grouped = new Map()
  studies.forEach(study => {
    const key = `${study.weights}/${study.tokens}`
    if (!grouped.has(key)) grouped.set(key, { x: x(study.tokens), y: y(study.weights), studies: [] })
    grouped.get(key).studies.push(study)
  })
  return [...grouped.values()]
})
const thisWork = studies.find(study => study.id === 'this-work')
const active = ref(thisWork)
const activeGroup = shallowRef(null)
const pinned = ref(false)
const detailsVisible = ref(false)
const detailsId = useId()
const hoveredGroup = shallowRef(null)
const showGroup = (group, pin = false) => {
  if (pinned.value && !pin) return
  activeGroup.value = group
  active.value = group.studies[0]
  if (pin && detailsVisible.value) pinned.value = true
}
const nearestGroup = event => {
  const matrix = event.currentTarget.getScreenCTM()
  if (!matrix) return null
  const cursor = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
  let nearest = null
  let distance = 14
  groups.value.forEach(group => {
    const candidate = Math.hypot(group.x - cursor.x, group.y - cursor.y)
    if (candidate < distance) { distance = candidate; nearest = group }
  })
  return nearest
}
const inspectPointer = event => {
  const group = nearestGroup(event)
  hoveredGroup.value = group
  if (group) showGroup(group)
  else resetHover()
}
const pinPointer = event => {
  const group = nearestGroup(event)
  if (group) showGroup(group, true)
}
const resetHover = () => {
  hoveredGroup.value = null
  if (pinned.value && detailsVisible.value) return
  active.value = thisWork
  activeGroup.value = null
}
const selectStudy = study => { active.value = study; pinned.value = true }
const clearPin = () => { pinned.value = false; resetHover() }
const toggleDetails = () => {
  detailsVisible.value = !detailsVisible.value
  if (!detailsVisible.value) clearPin()
}
const compact = value => new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value)
const evidence = study => study.evidence === 'H' ? 'Physical hardware' : 'Simulation'
const regime = study => study.regime === 'T' ? 'Transfer learning' : 'From scratch'
const pointLabel = group => `${group.studies.map(study => study.label).join('; ')} · ${group.studies[0].weights.toLocaleString()} analog-mapped weights and ${group.studies[0].tokens.toLocaleString()} dataset tokens`
const notes = computed(() => {
  const result = []
  if (active.value.evidence === 'H') result.push('Task executed with physical arrays; external control may still be used')
  else result.push('Architecture-level or algorithmic simulation evidence')
  if (active.value.partialMappingExcluded) result.push('A larger partial-mapping result from this paper is excluded')
  return result.join(' · ')
})
const methodLabels = [
  ['AGAD', 'Rasch et al., 2024', 200_000, 4_300_000, 2_000_000],
  ['Residual Learning', 'Wu et al., 2025', 200_000, 235_000, 120_000],
  ['Tiki-Taka', 'Gokmen & Haensch, 2020', 240_000, 235_000, 7_200],
  ['Analog MP', 'Nandakumar et al., 2020', 240_000, 199_000, 430],
]
</script>

<template>
  <div class="training-scale-demo" :class="{ 'details-open': detailsVisible }" @click.stop @keydown.stop @keydown.esc="clearPin" @pointerdown.stop @mouseleave="resetHover">
    <div class="training-scale-plot">
    <button class="scale-details-toggle" type="button" :aria-label="detailsVisible ? 'Hide study details' : 'Show study details'" :title="detailsVisible ? 'Hide study details' : 'Show study details'" :aria-expanded="detailsVisible" :aria-controls="detailsId" @click="toggleDetails">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M15 3v18" />
        <path :d="detailsVisible ? 'm8 9 3 3-3 3' : 'm10 15-3-3 3-3'" />
      </svg>
    </button>
      <svg viewBox="0 0 680 372" role="group" aria-label="Interactive logarithmic scatter plot of analog-mapped weights versus dataset tokens for 34 analog training study entries" @pointermove="inspectPointer" @pointerleave="resetHover" @click="pinPointer">
        <defs>
          <marker id="slide-scale-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1 9 5 1 9" fill="none" stroke="var(--muted)" stroke-width="1.5" /></marker>
        </defs>
        <template v-for="value in gridLines" :key="`grid-${value}`">
          <line v-if="value >= 5 && value <= 3e10" :x1="x(value)" :x2="x(value)" :y1="frame.top" :y2="frame.bottom" :class="value / 10 ** Math.floor(Math.log10(value)) === 1 ? 'scale-grid' : 'scale-minor-grid'" />
          <line v-if="value >= 5 && value <= 1e9" :x1="frame.left" :x2="frame.right" :y1="y(value)" :y2="y(value)" :class="value / 10 ** Math.floor(Math.log10(value)) === 1 ? 'scale-grid' : 'scale-minor-grid'" />
        </template>
        <line :x1="frame.left" :x2="frame.left" :y1="frame.top" :y2="frame.bottom" class="scale-axis" />
        <line :x1="frame.left" :x2="frame.right" :y1="frame.bottom" :y2="frame.bottom" class="scale-axis" />
        <g v-for="tick in [[1e3, '1k'], [1e6, '1M'], [1e9, '1B']]" :key="`x-${tick[0]}`">
          <line :x1="x(tick[0])" :x2="x(tick[0])" :y1="frame.bottom" :y2="frame.bottom + 5" class="scale-axis" />
          <text :x="x(tick[0])" :y="frame.bottom + 21" text-anchor="middle" class="scale-axis-label">{{ tick[1] }}</text>
          <line :x1="frame.left - 5" :x2="frame.left" :y1="y(tick[0])" :y2="y(tick[0])" class="scale-axis" />
          <text :x="frame.left - 10" :y="y(tick[0]) + 4" text-anchor="end" class="scale-axis-label">{{ tick[1] }}</text>
        </g>
        <text :x="(frame.left + frame.right) / 2" y="368" text-anchor="middle" class="scale-axis-title">Dataset tokens</text>
        <text x="13" y="176" text-anchor="middle" transform="rotate(-90 13 176)" class="scale-axis-title">Analog-mapped weights</text>
        <line :x1="frame.left" :x2="frame.right" :y1="y(34e6)" :y2="y(34e6)" class="scale-prior-guide" />
        <line :x1="frame.left" :x2="frame.right" :y1="y(123_532_032)" :y2="y(123_532_032)" class="scale-current-guide" />
        <line :x1="x(240_000)" :x2="x(2_467_430_400)" :y1="y(250e6)" :y2="y(250e6)" class="scale-gap" marker-start="url(#slide-scale-arrow)" marker-end="url(#slide-scale-arrow)" />
        <text :x="(x(240_000) + x(2_467_430_400)) / 2" :y="y(250e6) - 8" text-anchor="middle" class="scale-gap-label">10,300× dataset-token scale</text>
        <line :x1="x(2e6)" :x2="x(2e6)" :y1="y(34e6)" :y2="y(123_532_032)" class="scale-gap" marker-start="url(#slide-scale-arrow)" marker-end="url(#slide-scale-arrow)" />
        <text :x="x(2e6) + 10" :y="(y(34e6) + y(123_532_032)) / 2 + 3" class="scale-gap-label">3.6×</text>
        <g v-for="(group, index) in groups" :key="index" class="scale-point" :class="{ active: detailsVisible && activeGroup === group, hovered: hoveredGroup === group, current: group.studies.some(study => study.id === 'this-work') }" tabindex="0" role="button" :aria-label="pointLabel(group)" :aria-pressed="detailsVisible && pinned && activeGroup === group" @focus="showGroup(group)" @keydown.enter.prevent="showGroup(group, true)" @keydown.space.prevent="showGroup(group, true)">
          <title>{{ pointLabel(group) }}</title>
          <circle :cx="group.x" :cy="group.y" r="14" fill="transparent" />
          <circle :cx="group.x" :cy="group.y" r="11" class="scale-point-halo" />
          <circle v-if="group.studies.some(study => study.id === 'this-work')" :cx="group.x" :cy="group.y" r="9" class="scale-point-ring" />
          <circle :cx="group.x" :cy="group.y" :r="group.studies.some(study => study.id === 'this-work') ? 6 : 5" class="scale-point-dot" />
        </g>
        <g v-for="item in methodLabels" :key="item[0]" class="scale-method">
          <line :x1="x(item[2]) + 6" :y1="y(item[3])" :x2="x(7e6) - 7" :y2="y(item[4])" class="scale-leader" />
          <text :x="x(7e6)" :y="y(item[4]) + 3" class="scale-method-label">{{ item[0] }}</text>
          <text :x="x(7e6)" :y="y(item[4]) + 16" class="scale-reference-label">{{ item[1] }}</text>
        </g>
        <text :x="x(2_467_430_400) - 14" :y="y(123_532_032) + 18" text-anchor="end" class="scale-method-label">This work</text>
      </svg>
    </div>

    <div class="training-scale-sidebar">
      <Transition name="scale-sidebar">
        <div v-show="!detailsVisible" class="scale-overview" :aria-hidden="detailsVisible"><slot /></div>
      </Transition>
      <Transition name="scale-sidebar">
    <aside v-show="detailsVisible" :id="detailsId" class="training-scale-details" :aria-hidden="!detailsVisible" :inert="!detailsVisible" aria-live="polite">
      <div class="scale-detail-kicker">{{ pinned ? 'PINNED STUDY' : 'HOVERED STUDY' }}</div>
      <button v-if="pinned" class="scale-unpin" type="button" @click="clearPin">Reset</button>
      <h3>{{ active.method && active.method !== 'This work' ? `${active.method} · ` : '' }}{{ active.label }}</h3>
      <p class="scale-paper-title">{{ active.title }}</p>
      <div v-if="activeGroup && activeGroup.studies.length > 1" class="scale-overlap">
        <span>{{ activeGroup.studies.length }} overlapping studies</span>
        <button v-for="study in activeGroup.studies" :key="study.id" type="button" :aria-pressed="active.id === study.id" @click="selectStudy(study)">{{ study.label }}</button>
      </div>
      <dl>
        <dt>Analog weights</dt><dd>{{ compact(active.weights) }}</dd>
        <dt>Dataset tokens</dt><dd>{{ compact(active.tokens) }}</dd>
        <dt>Model</dt><dd>{{ active.model }}</dd>
        <dt>Dataset</dt><dd>{{ active.dataset }}</dd>
        <dt>Evidence</dt><dd>{{ evidence(active) }}</dd>
        <dt>Training</dt><dd>{{ regime(active) }}</dd>
      </dl>
      <p class="scale-detail-note">{{ notes }}</p>
    </aside>
      </Transition>
    </div>
  </div>
</template>
