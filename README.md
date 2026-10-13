# Scalable Analog Training · Slides

Presentation source for the Scalable Analog Training project. The deck is kept
on the `slides` branch so that the repository's `main` branch can remain focused
on released research code.

- [Research code (`main`)](https://github.com/Zhaoxian-Wu/scalable-analog-training)
- [Project homepage (`website`)](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/website)
- [Slide documentation](slides/README.md)

## Develop locally

```bash
conda env create -f slides/environment.yml
conda activate analog-slides
cd slides
npm ci
npm run dev
```

Open <http://localhost:3030/#/1>. Build the static presentation with:

```bash
cd slides
npm run build -- --base /scalable-analog-training/slides/
```

Read the [slide editing guide](GUIDES/slide-editing.md) before editing the deck.
Dependencies, generated builds, exports, and local experiment inventories are
excluded from Git; source files, assets, data snapshots, and the npm lockfile
are included.
