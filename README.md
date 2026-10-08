# Scalable Analog Training

Research on scalable mixed-signal training in analog in-memory computing.
Training code will be released soon.

The [research presentation](slides/README.md) is available in `slides/`, including
Slidev source, interactive diagrams, figure assets, and data snapshots.

To run it from this repository's root:

```bash
conda env create -f slides/environment.yml
conda activate analog-slides
cd slides
npm ci
npm run dev
```

Open <http://localhost:3030/#/1>. Run `npm run build` from `slides/` to generate
the static presentation in `slides/dist/`.

Read the [slide editing guide](GUIDES/slide-editing.md) before editing the deck.
Dependencies, generated builds, exports, and local experiment inventories are
excluded from Git; source files, assets, data snapshots, and the npm lockfile
are included.
