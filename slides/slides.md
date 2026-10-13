---
theme: default
layout: default
title: 'Making Analog Training Scale'
titleTemplate: '%s'
info: 'A 22-slide research talk with a cover and 3 appendix slides on scalable mixed-signal AIMC training. Interactive teaching examples and manuscript experiment results are labeled separately.'
author: 'Zhaoxian Wu, Tayfun Gokmen, Omobayode Fagbohungbe, T. Patrick Xiao, Tianyi Chen'
colorSchema: light
aspectRatio: 16/9
canvasWidth: 980
transition: fade
routerMode: hash
class: cover-slide
drawings:
  persist: false
fonts:
  sans: Inter
  mono: monospace
  provider: none
monaco: false
record: false
wakeLock: false
---

# Making Analog Training Scale

<p class="hero-copy">Co-designing Mapping, Optimizer, and Converters</p>

<div class="cover-publication">
<div class="cover-presenter"><strong>Zhaoxian Wu</strong><sup>1</sup></div>
<div class="cover-authors"><span>Tayfun Gokmen<sup>2</sup></span><span>Omobayode Fagbohungbe<sup>2</sup></span><span>T. Patrick Xiao<sup>3</sup></span><span>Tianyi Chen<sup>1</sup></span></div>
<div class="cover-affiliations"><span><sup>1</sup> Cornell Tech and Cornell University</span><span><sup>2</sup> IBM T. J. Watson Research Center</span><span><sup>3</sup> Sandia National Laboratories</span></div>
</div>

<div class="cover-logos" aria-label="Author institutions">
<img class="cornell-logo" src="/logos/cornell.svg" alt="Cornell University logo" />
<img class="ibm-logo" src="/logos/ibm.svg" alt="IBM logo" />
<img class="sandia-logo" src="/logos/sandia.svg" alt="Sandia National Laboratories logo" />
</div>

<!-- <div class="hero-equation">Analog matrix passes.<br>Digital gradient fidelity.</div>
<div class="tags"><span>123.6M parameters</span><span>Training from scratch</span><span>AIHWKit simulations</span></div> -->

<!-- Temporarily hidden cover illustration; retain for later use.
<CrossbarDiagram />
-->

<!-- Temporarily hidden resource buttons; retain URLs and website styling.
<div class="publication-links">
<a class="project-button" href="https://github.com/Zhaoxian-Wu/scalable-analog-training" target="_blank" rel="noopener noreferrer"><span class="button-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="7" cy="5" r="2"/><circle cx="7" cy="19" r="2"/><circle cx="17" cy="5" r="2"/><path d="M7 7v10M17 7v3a4 4 0 0 1-4 4H7"/></svg></span><span class="button-label">Code</span></a>
<a class="project-button" href="https://arxiv.org/abs/2609.36584" target="_blank" rel="noopener noreferrer"><span class="button-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M5 3h14v18H5zM8 7h8M8 11h8M8 15h5"/></svg></span><span class="button-label">Paper</span></a>
</div>
-->

<!--
Opening: Can analog arrays participate in training modern models despite finite precision and imperfect writes?
The central claim is algorithmic feasibility at the evaluated scales, not measured chip efficiency.
The author list, affiliations, and code URL follow tex/iclr2026/main.tex as of 2026-10-07.
Public preprint: https://arxiv.org/abs/2609.36584. The interactive examples are conceptual.
-->

---

# Scaling AI training demands increasing power

<div class="figure-split training-power-split">
<figure class="paper-figure training-power-figure"><img src="/figures/frontier_training_power.svg" alt="Training power draw versus publication date for 495 models from Epoch AI. The 61 frontier models are highlighted, and Epoch's original fitted trend shows 2.1 times annual growth. Power uses a logarithmic axis in watts." /></figure>
<div class="side-story training-power-story">
<div class="metric-number">≈2×</div>
<div class="metric-caption">training power demand each year</div>
<div class="mini-rule" aria-hidden="true"></div>
<p>Power demand keeps growing despite more efficient hardware</p>
</div>
</div>
<div class="callout compact">Growing workloads strengthen the case for efficient training hardware</div>
<div class="small-note source-note">Estimated training power draw · Power is measured in W, not total energy · Redrawn from Luke Emberson and Robi Rahman, <a href="https://epoch.ai/data-insights/power-usage-trend" target="_blank" rel="noopener noreferrer">Epoch AI</a> (CC BY; data updated November 24, 2025)</div>

<!--
Frontier models are defined by Epoch AI as among the top 10 by training compute at release.
The source chart includes a 2.1×/year fitted trend; ≈2× is the rounded audience takeaway.
The plot is redrawn with matplotlib from Epoch's chart JSON and CSV downloaded on 2026-10-07.
All 495 original point coordinates, the 61 frontier assignments, axis limits, and the original
100-point fitted trend are preserved. Labels, colors, and layout are adapted for this deck.
Regenerate with slides/plot_training_power.py; source snapshots are in slides/data/.
These are estimates of training power demand, not measured energy per run or dollar costs.
Hardware efficiency improvements have not offset frontier workload growth in Epoch's analysis.
Transition: this growing power demand motivates analog matrix–vector multiplication in memory.
This figure motivates efficient hardware; it does not quantify savings from our AIMC simulations.
-->

---

# Accelerate MVM via analog circuits


<div class="architecture-grid mvm-architecture">
<div class="panel digital-panel">
<h3 class="mvm-panel-heading">Digital computing</h3>
<div class="mvm-digital-body">
<div class="memory-path"><span>Memory</span><b>⇅</b><span>Processor</span></div>
<p>Move weights for each MVM</p>
</div>
</div>
<div class="bridge-arrow">→</div>
<div class="panel analog-panel">
<h3 class="mvm-panel-heading">Analog in-memory computing (AIMC)</h3>
<CrossbarMvm />
</div>
</div>

<div class="small-note source-note">S Jain, et al. “A heterogeneous and programmable compute-in-memory accelerator architecture for analog-AI using dense 2-D mesh” · <a href="https://doi.org/10.1109/TVLSI.2022.3221390" target="_blank" rel="noopener noreferrer"><b>TVLSI (2022): 114–127</b></a><br>S Ambrogio, et al. “An analog-AI chip for energy-efficient speech recognition and transcription” · <a href="https://www.nature.com/articles/s41586-023-06337-5" target="_blank" rel="noopener noreferrer"><b>Nature 620, 768–775 (2023)</b></a></div>

<!--
Explain the circuit in two steps: each device multiplies by Ohm’s law, and column currents sum by Kirchhoff’s law.
Rows carry V_j proportional to x_j; the device at input row j and output column i represents the conductance-domain weight widehat W_ij.
I_i is proportional to y_i after weight/input scaling and conversion. The equations are an ideal read model.
Signed logical weights require differential conductances or another encoding; the drawing omits that detail and suppresses logical/physical scaling.
Hover or focus any cell, output current, DAC, or ADC to explain each stage; leaving restores forward/backward equations.
Moving dashes show conventional current flowing along input rows, through 45-degree devices, and down output columns.
The 10–100× energy-efficiency statement expresses inference potential, not a universal GPU comparison or a result of our training experiments.
Jain reports projected 40–140× inference energy efficiency over A100; Ambrogio demonstrates speech inference hardware.
The Jain citation retains the requested 2022 online-publication year; the journal issue is January 2023.
Parallelism is within an array read; converters, settling, precision, and tiling affect system performance.
The transpose pass swaps drive/sense directions with supporting peripheral circuitry.
Forward/backward refers to analog-mapped linear layers; attention-score products and nonlinear operations stay digital.

Full wording retained for presenting:
Stored conductances multiply inputs; circuit currents sum the products in parallel.
Memory ⇄ processor
Weights ⇄ Compute
Read stored weights into arithmetic units for each matrix operation.
Multiply · Ohm’s law
Input voltages × stored conductances
Sum · Kirchhoff’s current law
Each column collects its cell currents
Why it accelerates MVM: parallel multiply–accumulate in the array, with reduced weight movement.
Training uses both directions: forward $y=Wx$ and backward $\delta_{\mathrm{in}}=W^{\mathsf T}\delta_{\mathrm{out}}$.
-->

---

# Training encounters imperfect hardware

<HardwareChallenges />
<div class="small-note">Conceptual schematics · Converter resolution affects matrix reads · Pulse granularity and asymmetry affect weight updates</div>

<!--
Three challenges: quantization-like error in DAC/ADC, limited update granularity, and asymmetric updates.
Default: three titles and schematics side by side, with detail text hidden.
Click a title to expand it while the other two shrink to titles only.
Click another title to switch directly; click the open title or press Escape to restore the overview.
Converter plot: ideal conversion versus discrete levels, with a rounding error marked.
Pulse plot: finite conductance changes per programming pulse; the uniform steps are illustrative.
Asymmetry plot: opposite pulses need not return to the starting weight; response magnitudes can depend on conductance state.
These diagrams are conceptual, not measured device curves or training results.
Finite write granularity is distinct from converter precision.
Read errors enter forward/backward passes; write errors change stored weights used in later passes.
-->

---

# Analog training has faced a scaling gap

<TrainingScaleChart>
  <p class="scale-intro">Common prior benchmarks:<br><strong>MNIST and CIFAR-10</strong></p>
</TrainingScaleChart>
<div class="scale-takeaway">What prevents analog training from scaling?</div>

<!--
This figure is a literature landscape, not a controlled head-to-head comparison.
The appendix distinguishes task-level physical hardware evidence from simulations calibrated to device measurements.
Do not describe the 123.6M result as a fabricated-chip demonstration or claim results at 1B parameters.

Full wording retained for presenting:
What is the key bottleneck preventing analog training from converging at scale?
-->

---

# Scaling needs an accurate batch gradient

<p class="lead">Three major matrix operations in a Transformer linear layer</p>

<div class="three-grid training-operation-cards">
<div class="panel digital-panel"><div class="panel-kicker">01 · FORWARD</div><h3>Compute activations</h3><div class="operation-equation">

$y = Wx$

</div><p>Weight W<br>Forward activation x</p></div>
<div class="panel digital-panel"><div class="panel-kicker">02 · BACKWARD</div><h3>Propagate errors</h3><div class="operation-equation">

$\delta_{\mathrm{in}} = W^{\mathsf T}\delta$

</div><p>Weight W<br>Backward signal δ</p></div>
<div class="panel analog-panel"><div class="panel-kicker">03 · W-GRAD</div><h3>Form batch gradients</h3><div class="operation-equation">

$G = \frac{1}{B}\sum_b \delta_b x_b^{\mathsf T}$

</div><p>Outer products<br>+ batch accumulation</p></div>
</div>
<div class="analog-signal-strip"><b>Analog candidates</b><span>Weights W · Activations (x, δ) · Gradients G</span></div>
<div class="callout">Gradient fidelity is a key bottleneck as networks scale</div>
<div class="small-note">Linear-layer schematic · Backward activations denote propagated error signals · W-grad reduces contributions across batch and token positions</div>

<!--
Answer the question on displayed slide 04: accurate batch-gradient formation is a key scaling bottleneck.
A Transformer linear layer has three compute-intensive matrix operations: forward, backward, and W-grad.
The formulas use a column-vector convention and average per-example contributions; token positions also enter the reduction.
Weights, activations (both forward x and backward error delta), and parameter gradients G can each use an analog representation or path.
Distinguish the transposed backward MVM from the outer-product reduction that forms W-grad.
This is an architectural motivation, not a claim that every prior study failed for one experimentally isolated reason.
Next: use a simplified FP16-to-FP8 digital diagnostic to ask which signal path is most precision-sensitive.
-->

---

# Low precision hurts the gradient path most

<p class="lead">FP16 → FP8 as a simplified proxy for analog precision loss</p>

<div class="figure-split precision-split">
<figure class="paper-figure precision-first"><svg viewBox="0 0 650 555" role="img" aria-label="Panel a: FP8 weights and activations preserve validation loss near 1.85, while FP8 gradient-path quantization raises it to 4.184"><defs><clipPath id="precision-panel-a"><rect width="650" height="555" /></clipPath></defs><image href="/figures/precision_cancellation_four_panels.png" width="2673" height="555" clip-path="url(#precision-panel-a)" /></svg></figure>
<div class="side-story">
<div class="evidence-tag">DIGITAL SHAKESPEARE EXPERIMENT</div>
<h3>Same precision · Different effects</h3>
<p>Weights / forward activations:<br>validation loss stays near 1.85</p>
<p>W-grad path:<br>validation loss rises to 4.18</p>
<div class="callout compact">Focus precision on W-grad formation</div>
</div>
</div>
<div class="small-note">Digital 8-block Shakespeare Transformer · Four-seed means and min–max whiskers · W-grad condition quantizes backward errors δ and parameter gradients G · Arithmetic, accumulation, and optimizer state remain FP32</div>

<!--
Read panel (a) from weights to activations to W-grad. This is a simplified value-precision proxy, not an analog-hardware simulation.
Selected values are round-tripped through FP16 or emulated FP8 E4M3FN; arithmetic, reductions, accumulation, and optimizer state remain FP32.
Weight-access and forward-activation quantization change validation loss little in this controlled diagnostic.
The W-grad condition quantizes both backward errors entering leaf-module backward computations and the accumulated parameter gradients supplied to AdamW.
Therefore it does not isolate accumulation precision or establish that every backward-activation quantization scheme is safe.
The marked loss increase motivates protecting the gradient path when assigning computations to analog hardware.
It is not a universal claim about FP8 training recipes or direct evidence that analog W-grad must fail.
Transition: why is the final batch gradient so sensitive? Opposing contributions leave a small residual.
-->

---

# Cancellation makes a small gradient fragile

<div class="cancellation-split">
<div class="cancellation-example">
<div class="panel-kicker">TWO OPPOSING GRADIENT CONTRIBUTIONS</div>
<div class="cancellation-row"><span>FP16</span><div>

$100.25 - 99.75 = \mathbf{0.50}$

</div></div>
<div class="cancellation-row quantized"><span>FP8</span><div>

$104 - 96 = \mathbf{8.00}$

</div></div>
<p class="cancellation-rounding">Round each term to FP8 E4M3</p>
<div class="cancellation-error"><span>Relative error</span><div>

$\frac{|8.00-0.50|}{0.50} = \mathbf{1500\%}$

</div></div>
<div class="cancellation-ratio">

$\kappa = \frac{\sum_n |g_n|}{|\sum_n g_n|}$

<div class="cancellation-ratio-caption">Large κ → small net signal</div>
</div>
</div>
<div class="cancellation-evidence">
<figure class="paper-figure precision-cancellation"><svg viewBox="656 0 650 555" role="img" aria-label="Panel b: mean gradient cancellation factors from 472 to 1052 across Q, K, V, O, and MLP up/down projections of a Shakespeare Transformer"><defs><clipPath id="precision-panel-b"><rect x="656" width="650" height="555" /></clipPath></defs><image href="/figures/precision_cancellation_four_panels.png" width="2673" height="555" clip-path="url(#precision-panel-b)" /></svg></figure>
<p>Transformer gradients show strong cancellation</p>
</div>
</div>
<div class="callout">Small net gradients need high-precision accumulation → MP baseline</div>
<div class="small-note">Example: nearest-value FP8 E4M3 rounding without rescaling · Plot: four-seed arithmetic means and min–max whiskers at a fixed Transformer checkpoint · Means are sensitive to near-zero sums; κ measures cancellation, not stochastic SNR</div>

<!--
Start with two large opposing contributions (they can also be sums from two microbatches), not two separate optimizer steps.
Both 100.25 and 99.75 are exactly representable in FP16. In FP8 E4M3 the adjacent representable values are 96 and 104.
Nearest-value rounding without rescaling gives 104 and 96; the difference changes from 0.50 to 8.00.
Each term has less than 4% relative error, but the net gradient has 1500% relative error. The example has kappa = 200 / 0.50 = 400.
Quantizing contributions before cancellation is distinct from quantizing a completed high-precision sum.
Panel (b) measures kappa across query, key, value, attention output, and MLP up/down projections in a fixed eight-block Shakespeare Transformer checkpoint.
Large kappa means the net signal is small relative to the magnitudes of opposing terms, so numerical perturbations can overwhelm it.
Use this as a signal-to-error explanation; kappa is not a measured stochastic signal-to-noise ratio.
The plotted arithmetic means are sensitive to nearly canceled coordinates and are not typical per-coordinate values.
High-precision accumulation protects these cancellations; digital W-grad motivates the established mixed-precision computational-memory baseline illustrated on the next slide.
-->

---

# Mixed-precision Training Architecture

<OverviewComparison initial-view="prior" />

<div class="small-note">Diagrams are conceptual; animations illustrate signal flow</div>

<!--
Start on the middle MP baseline tab: analog matrix passes, digital gradient accumulation, and residual programming.
The digital gradient accumulator and residual buffer are already present in prior analog mixed precision.
Hover or keyboard-focus any component to animate its signal path and explain its role; click to pin it.
The three tabs remain available for comparison; the complete proposed architecture returns after converter alignment.
Transition: explain how the residual buffer retains small updates until a programming pulse can be issued.
-->

---

# Keep the residual · Trigger a pulse

<div class="inline-equation residual-equations">
<div class="residual-add">

$$ H^{\mathrm{pre}}=H+\Delta W $$

</div><div>

$$ n=\operatorname{trunc}\!\left(\frac{H^{\mathrm{pre}}}{s\Delta w_{\min}}\right) $$

</div><div>

$$ H^{\mathrm{next}}=H^{\mathrm{pre}}-sn\Delta w_{\min} $$

</div>
</div>

<GradientPlayground />

<!--
Demo: start with the completed signed example, then Step or Play. Compare direct pulse truncation with retained residuals.
Positive mode is useful for first explaining threshold crossing; signed mode shows cancellation before a pulse.
The scalar demo folds s into the device step Δ. Its exact conservation identity assumes uniform, noiseless pulse response.
The physical device need not realize the nominal nΔ; the residual tracks requested pulses, not measured write error.

Full wording retained for presenting:
A small desired increment survives until it can cross the programming threshold.
-->

---

# Match logical scale to physical range

<div class="inline-equation">

$$ W=s(\widehat W-\widehat W^\diamond),\qquad \sigma_W\propto D^{-1/2},\qquad s=\frac{\omega\sigma_W}{\tau} $$

</div>

<MappingPlayground />

<!--
Demo: select Fixed logical range, then Native AbsMax, then Proposed mapping; move D from 128 to 2048.
Blue dashed is the width-dependent logical target, green is the selected policy. Physical bounds stay fixed.
The curves illustrate Gaussian pre-clipping targets; they do not reproduce the empirical initialization histograms.
Reference subtraction permits signed logical weights even though physical conductances are non-negative.

Full wording retained for presenting:
Change layer width and compare the logical and physical target distributions.
-->

---

# Mapping improves training across depth

<figure class="paper-figure mapping-evidence"><img src="/figures/initial_scaling_three_panels.png" alt="Measured physical and logical initialization distributions and Shakespeare validation loss across Transformer depth for three mapping controls" /></figure>
<div class="three-grid insight-cards">
<div><b>Fixed logical range</b><p>Wrong logical scale</p></div>
<div><b>Native AbsMax</b><p>Narrow device occupancy</p></div>
<div><b class="accent">Proposed mapping</b><p>Both scales preserved</p></div>
</div>
<div class="small-note">Shakespeare task · These controls isolate mapping choices rather than reproduce the cited prior training systems in full</div>

<!--
Connect this empirical figure to the preceding conceptual distributions. Explain the rightmost panel last.
All three methods improve with depth here; the proposed mapping gives the strongest result within these matched controls.

Full wording retained for presenting:
The measured controls expose the logical-conditioning and physical-occupancy tradeoff.
Broad physical occupancy; incorrect width-dependent logical scale.
Logical scale is preserved; physical deviations are concentrated.
Reconciles both requirements and achieves lower loss in these controls.
-->

---

# Converter ranges trade clipping for resolution

<p class="lead">Change the rail and bit depth; keep the signal fixed</p>

<ConverterPlayground />

<div class="callout compact">Narrow rails clip; wide rails quantize coarsely</div>

<!--
Demo sequence: Too narrow, Too wide, Max aligned at 4 bits. Then increase to 8 bits.
These are deterministic synthetic inputs and a uniform saturating quantizer, not Transformer activation measurements.
Max alignment prevents input clipping; it need not minimize total error at every finite precision.
The quantizer uses 2^b levels and rounds exact ties to the nearest even interval index, as in the manuscript.

Full wording retained for presenting:
Keep the signal fixed. Change the rail and bit depth to see the distortion.
Too narrow → overload error. Too wide → coarse quantization. Alignment uses the available levels more effectively.
-->

---

# Rescale digitally · Respect physical rails

<div class="converter-pipeline"><div><small>DIGITAL</small><b>x / C</b><span>Prescale</span></div><i>→</i><div><small>FIXED INPUT RAIL</small><b>DAC</b><span>Voltage</span></div><i>→</i><div><small>ANALOG</small><b>Ŵ − Ŵ◇</b><span>MVM</span></div><i>→</i><div><small>OUTPUT RAIL</small><b>ADC</b><span>Digitize</span></div><i>→</i><div><small>DIGITAL</small><b>× sC</b><span>Restore</span></div></div>

<div class="inline-equation pipeline-equation">

$$ y=sC\,\operatorname{ADC}\!\left((\widehat W-\widehat W^\diamond)\operatorname{DAC}(x/C)+\xi\right) $$

</div>
<div class="two-grid">
<div class="panel"><div class="panel-kicker">INPUT ALIGNMENT</div><p>C = ‖x‖∞<br>Prescale each input vector</p></div>
<div class="panel"><div class="panel-kicker">OUTPUT-RANGE DESIGN</div><p>Matrix-specific ADC rails<br>c<sub>out</sub> = 6</p></div>
</div>

<!--
The effective input rail is adjusted by digital prescaling; the physical DAC rail itself need not be reconfigured.
Do not imply that max-aligning the digital input automatically prevents ADC clipping.
Reference subtraction in this expression is an effective operation; this is not a detailed signed-weight circuit schematic.

Full wording retained for presenting:
Input alignment and output-range design solve different boundary constraints.
Prescale input
Convert to voltage
Crossbar read
Digitize current
Restore units
Choose C from the input vector. Max alignment uses C = ‖x‖∞ without assuming a fixed distribution.
ADC sees the physical current. The scaling recipe uses matrix-specific output rails with c out = 6.
-->

---

# Why max alignment works

<p class="lead">Asymptotically optimal within the norm–rail family</p>

<div class="theorem-strip"><div><span class="panel-kicker">ASYMPTOTIC RESULT</span>

$$ \frac{R_K(\infty,1;\mathcal V)}{r_K(\mathcal V)}\longrightarrow 1\quad\text{as }K\to\infty $$

</div><p>Fixed dimension<br>Absolutely continuous inputs<br>Finite second moment</p></div>

<figure class="paper-figure converter-evidence"><img src="/figures/gaussian_laplacian_normalization_nmse_bits_and_dimension_b6_dual_aciq.png" alt="Gaussian and Laplacian reconstruction distortion versus converter precision and dimension for max alignment and comparison rails" /></figure>
<div class="small-note">At D = 48, max alignment is within 10% of the calibrated optimum from 4 bits onward for the two tested product laws · The theorem is not a finite-bit guarantee for arbitrary inputs</div>

<!--
State the family restriction: rails are A times a p-norm, with norm order and multiplier chosen for a fixed distribution.
K denotes quantization intervals, not bit count. The asymptotic result holds at fixed dimension and distribution.
Distortion dimension bounds are O(log D / K^2) for Gaussian tails and O(log^2 D / K^2) for Laplace-type tails, with distribution constants.

Full wording retained for presenting:
A low-overhead policy approaches the optimum within the analyzed norm–rail family.
Fixed dimension; absolutely continuous joint input distribution with finite second moment. Gaussian and Laplace tails give mild logarithmic dimension dependence.
-->

---

# Mixed-precision Training Architecture

<OverviewComparison initial-view="ours" />

<div class="small-note">Diagrams are conceptual; animations illustrate signal flow</div>

<!--
Start on This work to recap the complete proposed architecture after mapping and converter alignment.
This work coordinates logical-space optimizers, width-aware mapping, and converter alignment.
Analog matrix passes, digital gradient accumulation, and residual programming extend the established MP baseline.
Hover or keyboard-focus components to explain their roles; click to pin, or switch tabs to compare architectures.
BM retries a clipped read with a smaller digital input; it does not directly rescale the analog ADC input.
Transition: evaluate this coordinated recipe with matched Transformer training from scratch.
-->

---

# Matched Transformer training from scratch

<p class="lead">OpenWebText · token budget ≈ 20N</p>

<div class="evaluation-grid">
<div class="panel experiment-table">
<div class="panel-kicker">DECODER-ONLY GPT · CONTEXT 1,024</div>

| Model | Blocks | Width | Parameters |
| :--- | ---: | ---: | ---: |
| Tiny | 2 | 128 | 6.83M |
| Small | 4 | 256 | 16.01M |
| Medium | 6 | 384 | 29.92M |
| Large | 8 | 512 | 50.91M |
| XLarge | 12 | 768 | 123.55M |

</div>
</div>
<div class="small-note">One seed (1337), no array-size tiling limits · Digital baseline uses BF16 autocasting; the analog simulation path uses FP32</div>

<!--
Point out that this is an architecture-level simulation calibrated to device measurements, not physical task execution.
The configuration has substantial digital state and relatively high converter precision. Those costs are not assessed here.
All schedules share 50,257-token vocabulary, tied embedding/head weights, and context length 1,024.

Full wording retained for presenting:
Five scales on OpenWebText, with a token budget of approximately 20N.
-->

---

# Matched training protocol

<div class="evaluation-details protocol-details">
<div><b>Optimization</b><p>AdamW, zero decay · warm-up + cosine<br>491,520 tokens per update</p></div>
<div><b>Hardware</b><p>AIHWKit Li-ECRAM · 16-bit DAC / 9-bit ADC<br>Finite steps and asymmetric pulsed writes</p></div>
<div><b>Partition</b><p>Analog: dense maps + tied vocabulary projection<br>Digital: attention-score products + nonlinear operations</p></div>
</div>
<div class="small-note">One seed (1337), no array-size tiling limits · Digital baseline uses BF16 autocasting; the analog simulation path uses FP32</div>

<!--
These are the matched protocol details for the five model scales on the preceding slide.
This is an architecture-level simulation calibrated to device measurements, not a fabricated-chip demonstration.
Digital state and converter energy, area, and latency are not assessed by these training results.
-->

---

# Stable training through 123.6M parameters

<p class="lead">All five analog schedules learn stably</p>

<div class="figure-split training-split">
<figure class="paper-figure"><img src="/figures/scaling_loss_vs_tokens_local_digital_vs_sout.png" alt="Measured OpenWebText validation loss trajectories versus processed tokens for five matched digital and analog model scales" /></figure>
<div class="side-story">
<div class="evidence-tag">XLARGE · COMPLETED TRAINING</div>
<h3>XLarge validation loss</h3>
<p>Finite-precision reads and imperfect pulsed writes</p>
<div class="endpoint-pair"><div><span>Analog</span><b class="accent">3.3950</b></div><div><span>Digital</span><b>3.2452</b></div></div>
<p>A validation-loss gap remains</p>
</div>
</div>
<div class="small-note">OpenWebText, one seed per condition · Horizontal axis counts processed tokens rather than wall-clock time</div>

<!--
Use solid/dashed line labels from the original figure to distinguish the two paths.
The claim is stable learning over completed schedules. It does not establish identical sample efficiency or hardware throughput.
The endpoint values here are taken from the reported bridge table.

Full wording retained for presenting:
Loss decreases across all five evaluated analog training schedules.
Non-idealities do not prevent learning at these scales
Forward/backward conversion errors and imperfect pulsed writes are included in the analog path.
XLarge analog loss
Matched digital loss
A persistent validation-loss gap remains.
-->

---

# Analog loss improves with model scale

<p class="lead">Preliminary fits over 6.8M–123.6M parameters</p>

<div class="figure-split scaling-split">
<figure class="paper-figure"><img src="/figures/scaling_law_d20n_local_digital_vs_sout_params.png" alt="Five observed analog and digital endpoint losses versus parameters, with fitted power laws and dashed extrapolations" /></figure>
<div class="side-story">
<div class="exponent-pair"><div><span>Analog</span><b class="accent">0.231</b></div><div><span>Digital</span><b>0.238</b></div></div>

$$ L\propto N^{-\alpha} $$

<p>Similar slopes; a persistent loss gap</p>


</div>
</div>
<div class="small-note">Five single-seed endpoints; dashed lines extrapolate · N and token budget covary · Fits omit an irreducible-loss term and do not isolate independent parameter/data exponents</div>

<!--
Clearly identify the final observed model and the dashed extrapolated region, including the untested 1B scale.
Compute-axis figures use a shared algorithmic work proxy; none of these plots is an energy or wall-clock comparison.

Full wording retained for presenting:
Preliminary slopes are close over the evaluated 6.8M–123.6M range.
Comparable fitted slopes coexist with an absolute loss gap.
Five single-seed endpoints. Dashed lines are extrapolations beyond observations.
N and token budget covary; these fits do not isolate independent parameter and data exponents.
Two-parameter log–log fits omit an irreducible-loss term and serve as preliminary guides, rather than established asymptotic scaling laws.
-->

---

# One logical interface · Diverse optimizers

<p class="lead">Seven optimizers · four Transformer depths</p>

<OptimizerChart />

<div class="callout compact">Beyond AdamW; optimizer choice still matters</div>

<!--
This is the Shakespeare compatibility screen, not the main OpenWebText scaling experiment.
Dots show Digital and S only; the manuscript additionally reports SFB. S retains managed reads; SFB replaces them with explicit periphery.
Hyperparameters are fixed across the screen. There is no per-optimizer tuning or statistically supported universal ranking.
Use the block-count buttons to show that the same interface remains trainable across the tested depths.

Full wording retained for presenting:
Digital update rules can share the same residual and physical-transfer interface.
Compatibility extends beyond AdamW; the optimizer still affects the achieved loss.
-->

---

# Preserve fidelity where training needs it

<div class="three-grid closing-cards">
<div class="panel"><div class="card-index">01</div><h3>Protect gradients</h3><p>Preserve precision-sensitive updates</p></div>
<div class="panel"><div class="card-index">02</div><h3>Align scales</h3><p>Match logical and physical ranges</p></div>
<div class="panel"><div class="card-index">03</div><h3>Validate scale</h3><p>Test larger models and real hardware</p></div>
</div>
<div class="closing-line">Analog passes · Digital gradients · Coordinated boundaries</div>

<!--
Close with the division of labor, then the co-design requirement, then the empirical milestone.
The study does not establish a measured hardware efficiency advantage. The 16-bit DAC and 9-bit ADC configuration may incur substantial physical costs.
Future work should quantify seed variability, realistic tiling, and lower-precision operation before extrapolating to larger model scales or efficiency claims.

Full wording retained for presenting:
The co-design supports improving loss through 123.6M parameters in the evaluated simulations.
Digital W-grad, logical optimization, and residuals preserve precision-sensitive updates.
Weight mapping and converter alignment maintain usable physical and logical signal scales.
Stable trajectories and preliminary loss trends motivate further validation and physical optimization.
Analog matrix passes. Digital gradient fidelity. Coordinated mapping and converters.
-->

---

# Evidence and next steps

<div class="two-grid evidence-limits">
<div class="panel"><div class="panel-kicker">ESTABLISHED HERE</div><h3>Stable simulated training</h3><p>Learning through 123.6M parameters</p><p>Device-calibrated architecture simulations</p></div>
<div class="panel"><div class="panel-kicker">NEXT EVIDENCE</div><h3>Validate physical scaling</h3><p>Multiple seeds and lower converter precision</p><p>Realistic tiling; energy, area, and latency accounting</p></div>
</div>

<!--
The demonstrated milestone is stable learning through 123.6M parameters in the evaluated simulations.
The study does not establish a measured hardware efficiency advantage.
The 16-bit DAC and 9-bit ADC configuration may incur substantial physical costs.
Quantify seed variability, realistic tiling, and lower-precision operation before extrapolating to larger scales or efficiency claims.
-->

---
appendix: true
---

# Mixed-precision Training Architecture

<OverviewComparison />

<div class="small-note">Diagrams are conceptual; animations illustrate signal flow</div>

<!--
Start on the middle MP baseline tab: analog matrix passes, digital gradient accumulation, and residual programming.
Compare the three tabs: Digital mixed precision → MP baseline → This work.
Hover or keyboard-focus any component to animate its signal path and explain its role; click to pin it.
The digital gradient accumulator and residual buffer are already present in prior analog mixed precision.
This work coordinates logical-space optimizers, width-aware mapping, and converter alignment.
BM retries a clipped read with a smaller digital input; it does not directly rescale the analog ADC input.
-->

---
appendix: true
---

# Analog passes · Digital gradients

<TrainingFlow />

<div class="small-note">Conceptual execution diagram · Extends prior mixed-precision computational-memory training (Nandakumar et al., 2020) with logical-space optimization, mapping, and converter co-design</div>

<!--
Click Forward, Backward, Gradient, Update. Highlight that W-grad is a sum of outer products, not the transposed backward MVM.
Digital gradient handling permits optimizer states and update transformations to remain in logical coordinates.
Array writes are open loop; subsequent gradients are evaluated using the actual resulting conductance state.

Full wording retained for presenting:
Walk through one training iteration across the analog–digital boundary.
-->

---
appendix: true
---

# The components must work together

<p class="lead">Changing the optimizer alone is insufficient here</p>

<AblationChart />

<div class="callout compact">Alignment gives the largest improvement in this protocol</div>

<!--
Demo: start at MpSGD, then MpAdam, + alignment, + initialization. XLarge is selected by default; use the model dropdown to compare smaller scales.
All displayed numbers are copied from the manuscript table, not simulated interactively.
MpSGD uses vanilla momentum-free SGD. Do not attribute its gap to gradient accumulation alone or claim all SGD recipes fail.
The controls form a sequential bridge, not a complete factorial experiment or a proof of additive causal effects.

Full wording retained for presenting:
Reveal the measured bridge from the mixed-precision baseline to the complete recipe.
In this protocol, changing the optimizer alone is insufficient. Converter alignment produces the largest improvement.
-->
