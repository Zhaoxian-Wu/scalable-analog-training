# Scalable Analog Training · Website

Project-homepage source for Scalable Analog Training. The site is kept on the
`website` branch so that the repository's `main` branch can remain focused on
released research code.

- [Live project homepage](https://zhaoxian-wu.github.io/scalable-analog-training/)
- [Research code (`main`)](https://github.com/Zhaoxian-Wu/scalable-analog-training)
- [Presentation source (`slides`)](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/slides)
- [Website documentation](website/README.md)

## Develop locally

The static content can be served without installing dependencies:

```bash
python3 -m http.server 8000 --directory website
```

After changing a Vue component or `website/interactives.js`, rebuild everything:

```bash
cd website
npm ci
npm run build:interactives
cd ..
python3 website/build.py
```

The generated `website/dist/` directory is ignored by Git. Pushes to this branch
build and deploy that directory through GitHub Actions.
