<script setup>
import { computed, ref } from 'vue'
import data from './manuscriptData.json'
const scale = ref(4), stage = ref(3)
const names = ['MpSGD', 'MpAdam', '+ alignment', '+ initialization']
const row = computed(() => data.bridge[scale.value])
const y = value => 207 - value / 12 * 163
</script>
<template>
  <div class="results-demo" @click.stop @keydown.stop @pointerdown.stop>
    <div class="results-toolbar"><div class="button-row"><button v-for="(label, i) in names" :key="label" :aria-pressed="stage === i" @click="stage = i">{{ label }}</button></div><label>Model <select v-model.number="scale" aria-label="Ablation model scale"><option v-for="(item, i) in data.bridge" :key="item.name" :value="i">{{ item.name }} · {{ item.params }}</option></select></label></div>
    <svg viewBox="0 0 840 278" role="img" :aria-label="`Measured ablation validation losses for ${row.name}; digital reference ${row.losses[4]}`">
      <g v-for="value in [0, 3, 6, 9, 12]" :key="value"><line x1="64" :y1="y(value)" x2="800" :y2="y(value)" stroke="var(--grid)"/><text x="52" :y="y(value)+4" fill="var(--muted)" text-anchor="end">{{ value }}</text></g>
      <text x="18" y="126" fill="var(--muted)" transform="rotate(-90 18 126)" text-anchor="middle">Validation loss ↓</text>
      <line x1="64" :y1="y(row.losses[4])" x2="800" :y2="y(row.losses[4])" stroke="var(--blue)" stroke-width="2" stroke-dasharray="5 5" />
      <g v-for="(name, i) in names" :key="name" :opacity="i <= stage ? 1 : 0.12">
        <rect :x="107 + i * 174" :y="y(row.losses[i])" width="90" :height="207 - y(row.losses[i])" rx="4" :fill="i === 3 ? 'var(--accent)' : i === 2 ? 'var(--soft-blue)' : 'var(--orange)'" />
        <text :x="152 + i * 174" :y="y(row.losses[i]) - 10" text-anchor="middle" fill="var(--ink)" font-size="15">{{ row.losses[i].toFixed(4) }}</text>
        <text :x="152 + i * 174" y="235" text-anchor="middle" fill="var(--ink)">{{ name }}</text>
      </g>
      <text x="800" y="263" text-anchor="end" fill="var(--blue)">Digital reference: {{ row.losses[4].toFixed(4) }}</text>
    </svg>
  </div>
  <div class="small-note">Reported OpenWebText endpoints · one seed per condition · Buttons reveal measured stages</div>
</template>
