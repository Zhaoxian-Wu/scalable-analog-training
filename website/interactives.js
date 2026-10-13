import { createApp } from 'vue';
import katex from 'katex';
import renderMathInElement from 'katex/dist/contrib/auto-render.mjs';
import 'katex/dist/katex.min.css';
import CrossbarMvm from './components/CrossbarMvm.vue';
import HardwareChallenges from './components/HardwareChallenges.vue';
import OverviewComparison from './components/OverviewComparison.vue';
import PrecisionDiagnostics from './components/PrecisionDiagnostics.vue';
import ConverterPlayground from './components/ConverterPlayground.vue';
import ScalingResults from './components/ScalingResults.vue';
import WeightDistributions from './components/WeightDistributions.vue';

for (const [id, component, props] of [
  ['analog-mvm-demo', CrossbarMvm, {}],
  ['hardware-imperfections-demo', HardwareChallenges, {}],
  ['mp-architecture-demo', OverviewComparison, { initialView: 'ours' }],
  ['gradient-diagnostics-demo', PrecisionDiagnostics, {}],
  ['converter-alignment-demo', ConverterPlayground, {}],
  ['scaling-results-demo', ScalingResults, {}],
  ['weight-distributions-demo', WeightDistributions, {}],
]) {
  const element = document.getElementById(id);
  if (element) createApp(component, props).mount(element);
}

function renderFormula(element) {
  katex.render(element.dataset.math, element, { displayMode: element.dataset.mathDisplay !== 'inline', throwOnError: true });
}
for (const element of document.querySelectorAll('[data-math]')) renderFormula(element);
document.addEventListener('math-content-changed', event => renderFormula(event.detail.element));
renderMathInElement(document.querySelector('.project-content'), {
  delimiters: [{ left: '\\(', right: '\\)', display: false }, { left: '\\[', right: '\\]', display: true }],
  ignoredTags: ['script', 'style', 'textarea', 'pre', 'code', 'option', 'svg'],
  ignoredClasses: ['katex'],
  throwOnError: true,
});
