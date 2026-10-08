# Making Analog Training Scale · first slide deck

A research talk with 22 numbered content slides, an unnumbered cover, and
3 unnumbered appendix slides (26 slides in total), built on the existing
Slidev template and designed for a
20–25 minute presentation to a mixed ML/hardware audience. The deck follows
the manuscript's `tex/iclr2026/main.tex` in the research workspace, with
publication metadata updated on October 7, 2026. The manuscript and experiment
scripts are maintained separately; the deck includes local snapshots of the
figures and data it needs to run.

The deck is tracked in the
[`scalable-analog-training` GitHub repository](https://github.com/Zhaoxian-Wu/scalable-analog-training)
under `slides/`. In the original research workspace,
`/home/zw868/Desktop/analog-transformer/slides` is a compatibility link to
`code/scalable-analog-training/slides`.

The deck uses a Cornell-inspired light palette: white (`#FFFFFF`), near-black
(`#222222`), and carnelian red (`#B31B1B`). These primary colors follow Cornell's
[published brand resources](https://human.cornell.edu/about/communications/brand-resources).
Blue and orange remain comparison colors in teaching diagrams and charts.
Slide titles use carnelian red (`#B31B1B`). All displayed slide text omits
sentence-ending periods, including body text, footnotes, and interactive copy.
Preserve decimal points, initials, abbreviations, URLs, code, and math.
Follow the [slide editing guide](../GUIDES/slide-editing.md) for this convention
and the deck's layout rules before making changes.
Theme font sizes in `style.css` match a 13⅓ × 7½ inch PowerPoint template
with 32 pt titles and 24 pt body text. That slide is 960 × 540 points; scaling
to the 980 px Slidev canvas gives 32.6667 px content titles and
24.5 px regular text. Paragraphs, leads, callouts, panel text, and main equations
share the body size. Large metrics retain emphasis; chart annotations, controls,
and references have separate sizes. Footer notes and page numbers remain 9 px.
These are proportional PowerPoint equivalents, not literal CSS point sizes.
Content slides start with the title, without section labels above it, to leave
more room for figures and text.
Content-slide titles stay fixed at the top. Main content is vertically centered as a group
between the reserved title space and footer. Footnotes, references, and page
numbers stay fixed at the bottom. The cover retains its separate alignment,
centering its title, subtitle, author details, and institution logos together.
Figures and results from our paper do not need visible source references;
retain external attributions and relevant experimental context.
This is a custom Slidev adaptation; official presentation resources are available
from [Cornell University Library](https://www.library.cornell.edu/about/staff/central-departments/library-communications/library-branding/#powerpoint)
and [Cornell Tech](https://brand.tech.cornell.edu/sandbox/).

The centered title slide places the Cornell, IBM, and Sandia logos at the bottom.
It highlights presenter Zhaoxian Wu, with the four coauthors on a single line
below and all affiliations on another single line. The cover uses independent
`--font-cover-*` variables in `style.css`: title 42 px, subtitle 24 px, presenter
30 px, coauthors 20 px, and affiliations 16 px. Adjusting `--font-body` or
`--font-title` changes content slides without changing the cover. The crossbar illustration
and Code/Paper buttons are temporarily commented out in `slides.md`; their
markup, URLs, and website button styling are retained for reuse. The footer
shows footnotes and references together on the left, when present, and the
slide number and total on the right. The cover and appendix slides are excluded
from both counts: the first content slide shows 01 / 22 and the last shows
22 / 22. Appendices show `Appendix` in place of a page count; the cover has no
footer. Appendix slides are marked with `appendix: true` in their frontmatter,
and the footer computes the main-slide total from that metadata.
Use `small-note` for footnotes and
`source-note` for references; combine them in one element when both are present.
Long notes wrap within the footer area while leaving room for the page count.

Layouts accommodate the large body text with concise projected copy and fuller
presenter notes. Result plots take the wider column; interactive charts use more
vertical space, and the residual equations share one row above the controls.
The precision diagnostics use explicit SVG clipping to show only their intended
panels. If you change font sizes again, check the interactive states as well as
the initial view; narrow columns and formulas need room to wrap or expand.

The story moves from rising frontier training power demand to the AIMC opportunity
and precision bottleneck, followed by the
mixed-signal partition, weight mapping, converter alignment, and completed
Transformer experiments. Every slide includes presenter notes.

## Open with live reload

Use the dedicated Conda environment `analog-slides`, which provides Node.js 22
and npm. In a terminal, run:

```bash
conda activate analog-slides
cd slides # From the scalable-analog-training repository root
npm run dev
```

Open <http://localhost:3030/#/1>. Keep the terminal running. Saving changes to
`slides.md`, `style.css`, or a Vue component updates the presentation in the
browser. Stop the server with **Ctrl+C**.

`vite.config.ts` enables polling so edits from the IDE and shared workspace
also refresh slide order and appendix metadata. Restart `npm run dev` once
after changing this configuration. If the preview still shows the old slide
order or total, stop and restart the server, then reload the browser; the main
deck currently has 22 counted pages.

Use arrow keys or Space to navigate. Interactive controls handle their own
keyboard and pointer events. Presenter view is available at
<http://localhost:3030/#/presenter/1>.

### One-time setup on another machine

The environment is already installed on this machine. To recreate it elsewhere,
run these commands from the repository root:

```bash
conda env create -f slides/environment.yml
conda activate analog-slides
cd slides
npm ci
npm run dev
```

The Conda `nodejs` package includes npm. Slidev requires Node.js **22.12.0 or
newer**; `environment.yml` selects a compatible Node.js 22 release.

If `conda activate` is unavailable in a Bash terminal, load Conda first:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate analog-slides
```

On other machines, use the `etc/profile.d/conda.sh` file from that Conda
installation. You can also launch without activating the environment:

```bash
cd slides # From the scalable-analog-training repository root
conda run --no-capture-output -n analog-slides npm run dev
```

## Interactive slides

| Slide | Interaction | Suggested live sequence |
| --- | --- | --- |
| 3 (displayed as 02) | AIMC matrix–vector multiplication | Hover or focus a cell → output current → DAC → ADC; leaving restores forward/backward equations and projected inference efficiency |
| 4 (displayed as 03) | Hardware challenges | Click one of three titles → selected diagram expands with details, other cards shrink to titles → click the open title or Esc to reset |
| 5 | Analog-training scale landscape | Icon beside the plot smoothly expands study details (hidden by default) → hover across studies → click to pin while details are open; closing clears the selection and restores the benchmark sentence |
| 9 (displayed as 08) | MP baseline architecture | Starts on MP baseline after the cancellation example; hover components or click to pin |
| 10 (displayed as 09) | Signed residual accumulation | Reset → Step or Play → Positive/Signed; adjust device step |
| 11 (displayed as 10) | Logical/physical mapping | Compare three policies; change layer width |
| 13 (displayed as 12) | Converter clipping and resolution | Too narrow → Too wide → Max aligned; increase bits |
| 16 (displayed as 15) | Complete proposed architecture | Starts on This work after max alignment; switch tabs to compare architectures |
| 21 (displayed as 20) | Measured optimizer compatibility | Compare 2, 4, 6, and 8 Transformer blocks |
| Appendix 1 (route 24) | Manuscript architecture comparison | Starts on the middle MP baseline tab → compare Digital mixed precision and This work; hover components or click to pin |
| Appendix 2 (route 25) | Training data flow | Forward → Backward → Gradient → Update |
| Appendix 3 (route 26) | Measured component ablation | MpSGD → MpAdam → alignment → initialization; select model |

The appendices follow `Evidence and next steps`, preserving the content,
interactions, and presenter notes of the former slides 9, 10, and 21.

Residual, mapping, and converter examples are **conceptual, deterministic
teaching models**, not training results. Residual programming assumes uniform
noiseless steps. Mapping curves show Gaussian pre-clipping targets; Native
AbsMax uses an extreme-value approximation. The converter example uses a
synthetic signal and uniform saturating quantization.

Interactive slides start with useful static views for screenshots/export.
The AIMC diagram on slide 3 continuously shows current flowing through rows,
45-degree conductance devices, and output columns; hover or Tab to focus a cell,
output, DAC, or ADC for its explanation. Reduced-motion preferences disable the
current animation. Its energy-efficiency callout describes projected inference
potential from external literature, not measured training efficiency in this work.
The hardware-challenge slide follows the MVM explanation and precedes the scale landscape.
Its DAC/ADC staircase, finite pulse steps, and unequal positive/negative responses
are conceptual schematics, not device measurements. The default view hides prose
details; selecting a title expands its card and shrinks the other two to titles.
Enter/Space activates focused titles; Escape restores the overview. Reduced-motion
preferences disable the card transitions.
Other teaching-model animation begins when Play is pressed. On the architecture slides, component
and signal-path animations run while hovered, focused, or pinned; reduced-motion
preferences disable these animations. The left diagram and short introduction
on the right switch together. Use Left/Right or Home/End within the architecture
tabs, Tab to focus components, Enter/Space to pin, and Escape to clear the pin.

Displayed slides 05–07 explain the gradient-fidelity motivation:
three linear-layer matrix operations and their signal types → an FP16/FP8
precision diagnostic using panel (a) → a cancellation example and panel (b).
The FP8 W-grad condition quantizes both backward errors and parameter gradients;
arithmetic and accumulation remain FP32. It is a simplified value-precision
diagnostic, not an analog-hardware experiment. The cancellation example rounds
100.25 and 99.75 to FP8 E4M3 values 104 and 96, changing a difference of 0.50 to
8.00 (1500% relative error). Panel (b)'s arithmetic mean cancellation factors
are sensitive to nearly canceled coordinates; they do not measure stochastic SNR.
The MP baseline architecture follows displayed slide 07; the complete proposed
architecture follows the original slide 13 (`Why max alignment works`, now
displayed as 14). Both copies and Appendix 1 share
`components/OverviewComparison.vue`; `initial-view="prior"` selects MP baseline
and `initial-view="ours"` selects This work. Omitting the prop retains the
appendix's MP baseline default. Each instance keeps its own tab selection and
hover/focus/pin state; all three tabs remain interactive.

## Evidence and scope

- Manuscript PNG figures are copied into `public/figures/` and served locally.
- Slide 2 redraws Epoch AI's [frontier training power data](https://epoch.ai/data-insights/power-usage-trend)
  under CC BY, credited to Luke Emberson and Robi Rahman. The source snapshots in
  `data/epoch_training_power_chart.json` and `data/epoch_training_power.csv` were
  downloaded October 7, 2026; the page dates the CSV to November 24, 2025. Original
  sources: [chart JSON](https://epoch.ai/ancient_graphs/datahub/power-usage-trend.json)
  and [CSV](https://epoch.ai/data/charts/power-usage-trend/training_power_draw.csv).
  `plot_training_power.py` validates the 495 models against the CSV and preserves
  exact chart coordinates, the 61 frontier assignments, axis limits, and the
  original 100-point fitted trend. Colors, labels, and layout are adapted.
  Regenerate `public/figures/frontier_training_power.svg` with
  `python plot_training_power.py` in a Python environment with matplotlib.
  The SVG is drawn at 610 × 370 pixels, keeping font shapes
  intact. The ≈2× callout rounds the plotted 2.1× annual trend; these are power
  estimates, not total training energy. The original PNG is retained as a reference.
- `components/manuscriptData.json` contains five bridge-ablation rows from
  `main.tex` and seven optimizer rows from
  `tex/iclr2026/tables/optimizer-s-sfb-results.tex`.
- `components/trainingScaleData.json` contains the 34 study entries behind the
  interactive scale landscape. To refresh it, run
  `python website/update_training_scale.py` from the original research workspace
  root; the exporter checks all coordinates against
  `scripts/plot_prior_training_scale.py`. These research scripts are maintained
  separately and are not needed to run or build this deck.
- Result controls reveal reported data, without computing hypothetical
  outcomes. Reconcile the JSON and figure snapshots when results change.
- Main scaling experiments are single-seed, device-calibrated architectural
  simulations through 123.55M parameters (rounded to 123.6M in the talk), with
  16-bit DACs, 9-bit ADCs, and no array-size tiling limits.
- Loss gaps, preliminary fits, and extrapolations are labeled. The deck makes
  no measured chip-efficiency, energy, or throughput claim.

## Build the webpage

From the `slides` folder with `analog-slides` activated:

```bash
npm run build
python3 -m http.server 3031 --directory dist
```

Open <http://localhost:3031>. Serve the build over HTTP rather than opening
`dist/index.html` directly. For subdirectory deployment, build with the matching
base path, for example `npm run build -- --base /slides/`.

Optional PDF export: install `playwright-chromium` as a development dependency,
then run `npm run export`. Export requires a working Chromium installation.

## Edit

To tune the training-power chart's vertical size without stretching its text,
regenerate it from the repository root with a Python environment containing matplotlib:

```bash
python slides/plot_training_power.py --height 370 --top 0.93 --bottom 0.12 --font-scale 1.15
```

`--height` sets the canvas height in slide pixels; the slide frame follows the
SVG automatically. Increasing `--top` or decreasing `--bottom` gives the plotted
axes more of that canvas. `--font-scale` controls label sizes independently;
changing height alone keeps fonts fixed. `WIDTH_PX` in the script matches the
610px chart column in `style.css`. Leave room for the legend, axis
labels, and slide source note when choosing values. On this machine, the script
can be run with `/home/zw868/.conda/envs/aihwkit-cuda-dev/bin/python`.

- `slides.md`: content, equations, ordering, and presenter notes.
- `style.css`: shared light palette for layouts and SVG diagrams, plus research-deck layouts.
- `global-top.vue`: shared footer and slide count.
- `components/`: interactive diagrams, teaching math, and reported data.
- `public/figures/`: manuscript figure snapshots.

Slidev and UnoCSS remain pinned to the template's verified versions. Preserve
`package-lock.json` for reproducible installs. Math uses KaTeX through Slidev;
all figure assets and fonts are local or system-provided.
