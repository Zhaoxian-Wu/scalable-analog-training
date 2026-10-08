<script setup>
import { ref } from 'vue'
const phase = ref(0)
const phases = [
  { title: 'Forward', formula: 'y = Wx', text: 'Convert inputs, read the array, digitize outputs', analog: true },
  { title: 'Backward', formula: 'δin = Wᵀδout', text: 'Read stored weights in the transposed direction', analog: true },
  { title: 'Gradient', formula: 'G = (1/B) Σ δb xbᵀ', text: 'Form digital gradients; apply the logical optimizer', analog: false },
  { title: 'Update', formula: 'H ← H + ΔW → pulses', text: 'Retain residuals; transfer consolidated pulses', analog: false },
]
</script>
<template>
  <div class="flow-demo" @click.stop @keydown.stop @pointerdown.stop>
    <div class="button-row"><button v-for="(item, i) in phases" :key="item.title" :aria-pressed="phase === i" @click="phase = i">{{ i + 1 }} · {{ item.title }}</button></div>
    <div class="flow-grid">
      <div class="flow-box" :class="{ active: phases[phase].analog }"><div class="panel-kicker">ANALOG DOMAIN</div><h3>Conductance arrays</h3><div class="flow-symbol">W &nbsp; / &nbsp; Wᵀ</div><p>Forward / backward<br>matrix passes</p></div>
      <div class="flow-bridge"><span :class="{ active: phase < 3 }">signals →</span><span :class="{ active: phase === 3 }">← pulses</span></div>
      <div class="flow-box" :class="{ active: !phases[phase].analog }"><div class="panel-kicker">DIGITAL DOMAIN</div><h3>Gradients & optimizer</h3><div class="flow-symbol">G &nbsp; → &nbsp; ΔW &nbsp; → &nbsp; H</div><p>Digital updates<br>Nonlinear operations</p></div>
    </div>
    <div class="flow-current" aria-live="polite"><strong>{{ phases[phase].formula }}</strong><span>{{ phases[phase].text }}</span></div>
  </div>
</template>
