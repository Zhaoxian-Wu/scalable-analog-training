# Scalable Analog Training · Slides

Presentation source for the Scalable Analog Training project. The deck is kept
on the `slides` branch so that the repository's `main` branch can remain focused
on released research code.

- [Research code (`main`)](https://github.com/Zhaoxian-Wu/scalable-analog-training)
- [Project homepage (`website`)](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/website)
- [Slide documentation](slides/README.md)
- [View slides online](https://zhaoxian-wu.github.io/scalable-analog-training/slides/)

## GitHub Pages

Pushes to `slides` publish the presentation at <https://zhaoxian-wu.github.io/scalable-analog-training/slides/>. The workflow calls the shared Pages workflow on `website`, which builds the project homepage and the slides into one artifact. Pushes to `website` also rebuild both, so each deployment keeps the presentation available alongside the homepage.

Repository **Settings → Pages → Build and deployment → Source** must be **GitHub Actions**. The `github-pages` environment must allow deployments from both `website` and `slides` under **Settings → Environments → github-pages → Deployment branches and tags**. Pull requests build the slides for validation without deploying.

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
