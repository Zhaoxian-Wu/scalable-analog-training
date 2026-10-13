<template>
  <div class="crossbar-visual">
    <div class="crossbar-label">IN-MEMORY MATRIX ENGINE</div>
    <svg viewBox="0 0 350 320" role="img" aria-label="Conceptual crossbar: horizontal input lines, conductance cells, and vertical output lines">
      <rect x="57" y="42" width="249" height="244" rx="20" fill="var(--panel)" stroke="var(--border)" />
      <g v-for="row in 6" :key="`row-${row}`">
        <text x="12" :y="60 + row * 32" fill="var(--muted)" font-size="10" font-family="monospace">x{{ row }}</text>
        <line x1="37" :y1="56 + row * 32" x2="322" :y2="56 + row * 32" stroke="var(--accent)" stroke-opacity=".55" stroke-width="1.5" />
      </g>
      <g v-for="col in 6" :key="`col-${col}`">
        <line :x1="58 + col * 32" y1="28" :x2="58 + col * 32" y2="300" stroke="var(--blue)" stroke-opacity=".48" stroke-width="1.5" />
        <text :x="52 + col * 32" y="316" fill="var(--blue)" font-size="10" font-family="monospace">y{{ col }}</text>
        <rect v-for="row in 6" :key="row" :x="51 + col * 32" :y="49 + row * 32" width="14" height="14" rx="3" :fill="(row + col) % 3 === 0 ? 'var(--accent)' : 'var(--cell)'" :stroke="(row + col) % 3 === 0 ? 'var(--accent-soft)' : 'var(--border)'" />
      </g>
    </svg>
    <div class="crossbar-caption"><span class="status-dot"></span> Weights stored as conductances</div>
  </div>
</template>
