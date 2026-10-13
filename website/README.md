# Scalable Analog Training homepage

A static research homepage modeled on the typography, sticky contents, callouts,
figures, and technical disclosures in the [FFA Theory website](https://github.com/Zhaoxian-Wu/ffa-theory/tree/main/website).

The homepage lives in `website/` on the repository's
[`website` branch](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/website).
In the original research workspace, the branch is checked out at
`code/scalable-analog-training-website`, and `analog-transformer/website` is a
compatibility link to its `website/` directory.

Open `index.html` directly, or serve this directory:

```bash
python -m http.server 8000 --directory website
```

Then visit http://localhost:8000. No package installation is required. Core content
and section navigation work without JavaScript. JavaScript adds the training-step
explorer, interactive training-scale scatter plot, current-section highlighting,
and diagrams adapted from the presentation:

- **Analog MVM:** animated voltage/current paths; hover or focus conductance
  cells, output columns, DAC, and ADC to inspect their equations. Click/tap or
  Enter/Space pins an explanation; Escape clears the pin.
- **AIMC hardware non-ideality model:** two tabs switch between converter
  quantization and pulse updates (selected by default). Converter controls expose
  input-vector size, rail, and bit depth. The pulse panel groups the trajectory,
  playback/manual stepping, and device-response controls, with its exact response
  equations below. The horizon is 4× the device-state setting, split evenly in
  pulse-sweep mode or alternating signs. Both sequences start at
  w_min + 0.1(w_max − w_min). The model uses q_+(w) = 2ρ(1 − βw) and
  q_−(w) = 2(1 − ρ)(1 + βw). The symmetric-point position ρ ranges from 0 to 1,
  giving w^diamond = (2ρ − 1)/β. The response-zero bounds remain −1/β and +1/β
  when this slider moves. Its endpoints put w^diamond exactly at w_min/w_max;
  one pulse direction has zero gain at each endpoint. β controls state dependence.
  Pulses request Δw = ±2/N and apply U_q without projection. At β = 0 no finite
  bounds exist, the initial weight is −0.8, and the vertical range fits the sequence.
  Axes show LaTeX bound and symmetric-point labels without numeric y-axis ticks.
  N ranges from 8 to 128 (default 32); ρ defaults to 0.5 and β to 1.
  The compact formulation is collapsed by default. Alternating mode explains
  attraction toward the selected balance point, finite-pulse oscillation, and the
  endpoint degeneracy. Left/Right/Home/End navigate tabs.
- **MP training architecture:** compare digital mixed precision, the analog MP
  baseline, and this work (the initial view). Hover/focus components to animate
  connected paths; click or Enter/Space pins a component; Escape clears the pin.
  Left/Right/Home/End navigate architecture tabs.
- **Gradient diagnostics:** four labels follow manuscript panels (a–d): precision
  sensitivity, cancellation, accumulation error, and transfer fragmentation.
  Native SVG bars, whiskers, curves, and bands reproduce the archived four-seed
  numerical data, with axes containing every observed seed range. Hover, tap,
  or focus a bar/point for its values. Each panel displays its takeaway and
  experimental context. Arrow keys and Home/End navigate the labels.
  Standalone SVG panels also provide the no-JavaScript fallback; no raster
  crops or screenshots are used for these charts.
- **Converter alignment:** choose Gaussian or Laplacian inputs, vector sizes
  from 8 to 256, and 2–9-bit precision (default 5 bits). Paired coordinate bars show the original
  and reconstructed vector. Compare narrow/wide rails, max alignment (CA),
  Optimal Rail, ACIQ-Gaussian, and ACIQ-Laplacian using selectable NMSE bars
  with an optional logarithmic error axis. Adjusting a custom rail adds its error
  to the comparison. Rail/bit changes preserve the sample; changing size extends
  or truncates it; New sample resamples it; Reset restores all defaults.
  All coordinates fit within the plot; bars narrow as vector size grows.
  Digital prescaling keeps the physical DAC
  range fixed; this example isolates input quantization.
  Optimal Rail sweeps every quantizer-assignment boundary and minimizes the
  quadratic MSE within each interval to find the displayed vector's optimal
  positive scalar rail. This differs from the paper's expected norm–rail objective.
  ACIQ fits zero-mean Gaussian RMS or Laplace mean-absolute statistics and
  minimizes the analytical clipping-plus-rounding model from the
  [authors' source](https://github.com/submission2019/cnn-quantization/blob/master/mse_analysis.py).
  The selected ACIQ method shows its multiplier × estimated scale, the sample
  maximum, and any mismatch between its assumed prior and the input distribution.
  Its population rail can exceed a short sample’s maximum; the sample-optimal
  rail and ACIQ therefore optimize different objectives.
  ACIQ's rounding term is adapted to the manuscript's endpoint quantizer:
  K = 2^bits − 1 intervals, Δ = 2C/K, and approximate rounding MSE C²/(3K²).
  Quantization, sample-optimal rail search, and ACIQ share the same interval count.
  The resolution control displays both bits and K to make this convention explicit.
- **Scaling results:** three views show model-size scaling and token/compute
  trajectories with endpoint fits drawn directly on each plot. Fits always use
  all five model endpoints, including when a trajectory is filtered. Hover/focus a point to inspect exact values; click,
  tap, or Enter/Space pins the selection; Escape clears it. A model selector
  filters trajectories. Fits start at the smallest endpoint and always extend to
  10× the largest; a shaded region identifies this projection beyond the observed
  range. Digital AdamW uses solid lines and MP Analog Adam uses dotted lines in
  both fits and trajectories. Arrow keys and Home/End navigate labels.
- **Weight distributions:** tune inverse occupancy ω and layer width D to see
  Gaussian pre-clipping targets in logical and physical coordinates. The physical
  standard deviation is τ/ω; the logical target is D^(-1/2); s = ωσw/τ compensates
  for the change in occupancy. Fixed axes expose how each distribution changes.
  Orange tails mark values outside the device bounds, with τ = 1 in this example.

The hardware model's **DAC / ADC quantization** tab reuses the converter component
in `simplified` mode: only distribution, vector size, resolution, and rail controls,
with paired coordinate bars and reconstruction statistics. Its state and control
IDs are independent of the full converter playground.

### Math notation

KaTeX is already bundled, including its auto-render extension. Always author
formulations in LaTeX, including inline notation:

- Static HTML: use `\(s\propto D^{-1/2}\)` inline and `\[...\]` for display math.
- Explicit elements: `data-math="..."` uses display math; add
  `data-math-display="inline"` for inline rendering.
- Reactive Vue content: use `<MathTex tex="..." />` or `<MathTex :tex="formula" />`;
  the optional `display` prop selects display mode. This component updates the
  rendered formula when its input changes.

The label **Rail** means the range of the unscaled digital signal. Digital
prescaling realizes it with a fixed physical DAC range; the full playground's
pipeline and caption explain that distinction.


These are conceptual teaching diagrams, not device measurements or timing
simulations. Reduced-motion preferences disable animation and transitions. On
small screens, the architecture diagram scrolls horizontally to keep labels legible.
Without JavaScript, each diagram retains a written explanation.

The training-scale chart supports hover, keyboard focus, and click/tap to keep
details open. A selector exposes nearby or overlapping studies without changing
their coordinates. Details identify hardware versus simulation evidence and
training from scratch versus transfer learning. Small screens can scroll the plot
horizontally. Without JavaScript, the original manuscript figure appears instead.

Refresh its data after editing the manuscript table or bibliography:

```bash
python website/update_training_scale.py
python website/build.py
```

The exporter locates the manuscript in the original research workspace. From a
separate clone, pass `--research-root /path/to/analog-transformer`. A standalone
GitHub clone already includes the chart-data snapshot and does not need the
manuscript to display or build the site.

The exporter checks all 34 entries against the coordinates in
`scripts/plot_prior_training_scale.py` before writing
`assets/training-scale-data.js`. Counts below the image-patch scale are retained
at their actual values. The chart can also be opened directly from the filesystem.

Edit `index.html`, `styles.css`, `app.js`, and `training-scale.js`. Run `python website/build.py` to
refresh the deployable `website/dist/` directory, which can also be used with
GitHub Pages or any static host. The Sites manifest uses `dist` as its public root.

### Rebuild the interactive diagrams

`components/` contains webpage adaptations of `CrossbarMvm.vue`,
`HardwareChallenges.vue`, `OverviewComparison.vue`, and `ConverterPlayground.vue`
from the repository's [`slides` branch](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/slides),
plus the manuscript diagnostic and scaling explorers.
`interactives.css` adapts their layout to the webpage. The bundled JavaScript,
component CSS, and KaTeX fonts in `assets/interactive/` are included so the site
runs directly, including from `file://`, without installing Node dependencies.

After changing a Vue component or `interactives.js`, use Node.js 22.12 or newer:

```bash
cd website
npm ci
npm run build:interactives
cd ..
python3 website/build.py
```

In this workspace, `build-interactives.mjs` can also reuse dependencies already
installed in `slides/node_modules/`, using the `analog-slides` environment:

```bash
conda run --no-capture-output -n analog-slides node website/build-interactives.mjs
python3 website/build.py
```

Pushes to the `website` branch run the GitHub Pages workflow, rebuild the
interactive bundle and `website/dist/`, and deploy the generated artifact.

## Content provenance

- Title, architecture, and numerical claims: `tex/iclr2026/main.tex`.
- Figures: copied from `tex/iclr2026/figures/`; `overview-cmp.pdf` was rendered to PNG.
- Results are labeled preliminary; no fabricated chip, completed billion-parameter
  experiment, measured energy saving, or measured throughput is claimed.
- Authors and affiliations match the current preprint metadata in `tex/iclr2026/main.tex`.
- Author names link to verified homepages where available, following the FFA Theory
  website: [Zhaoxian Wu](https://zhaoxian-wu.github.io/) and
  [Tianyi Chen](https://chen.tech.cornell.edu/).
- Interactive diagram source: the `slides` branch, adapted in `website/components/`.
- Paper: https://arxiv.org/abs/2609.36584. The homepage links to the public preprint
  rather than copying a local manuscript PDF.
- Project repository: https://github.com/Zhaoxian-Wu/scalable-analog-training.
- Fonts optionally load from Google Fonts; local serif and sans-serif fallbacks
  keep the site readable offline. No analytics scripts were copied from the reference.

When manuscript results change, update the copy and figures together, then rebuild.

Refresh the gradient diagnostics from the frozen four-panel figure data
(no experiments or manuscript regeneration):

```bash
python website/update_gradient_results.py
conda run --no-capture-output -n analog-slides node website/build-interactives.mjs
python website/build.py
```

The exporter verifies all 39 plotted four-seed means and min–max ranges before
writing `assets/gradient-results-data.json`. Use `--research-root` from a separate
clone. `components/gradient-plots.js` renders both the interactive plots and the
standalone SVGs during the interactive build, using the bundled data snapshot.

Refresh the scaling explorer from the archived source data of the five-panel
figure (no experiments or manuscript regeneration):

```bash
python website/update_scaling_results.py
conda run --no-capture-output -n analog-slides node website/build-interactives.mjs
python website/build.py
```

The exporter reads `code/E2-scaling-law/assets/scaling_local_ddp2_results.json`
and the model/compute constants in `plot_local_ddp2_scaling.py` from the research
workspace. It verifies all ten final evaluations and six fits before writing
`assets/scaling-results-data.json`. Use `--research-root` from a separate clone;
the bundled snapshot is sufficient for building and displaying the site.
