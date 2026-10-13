// Compile the slide-derived Vue diagrams into a standalone, file://-safe bundle.
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { gradientChart, gradientOverview } from './components/gradient-plots.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const gradientData = JSON.parse(await readFile(path.join(root, 'assets/gradient-results-data.json'), 'utf8'));
for (let index = 0; index < 4; index++) {
  await writeFile(path.join(root, `assets/gradient-panel-${String.fromCharCode(97 + index)}.svg`), gradientChart(gradientData, index).svg + '\n');
}
await writeFile(path.join(root, 'assets/gradient-precision.svg'), gradientOverview(gradientData) + '\n');
let resolve = createRequire(import.meta.url);
try {
  resolve.resolve('vite');
} catch {
  // Reuse the deck's installed tools when working in the combined checkout.
  resolve = createRequire(path.join(root, '../slides/package.json'));
}
const { build } = await import(pathToFileURL(resolve.resolve('vite')).href);
const { default: vue } = await import(pathToFileURL(resolve.resolve('@vitejs/plugin-vue')).href);
await build({
  configFile: false,
  root,
  base: './',
  plugins: [vue()],
  resolve: {
    alias: {
      vue: resolve.resolve('vue/dist/vue.runtime.esm-bundler.js'),
      'katex/contrib/auto-render': resolve.resolve('katex/contrib/auto-render'),
      katex: path.dirname(resolve.resolve('katex/package.json')),
    },
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
    __VUE_OPTIONS_API__: false,
    __VUE_PROD_DEVTOOLS__: false,
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false,
  },
  build: {
    outDir: 'assets/interactive',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    cssCodeSplit: false,
    rollupOptions: {
      input: path.join(root, 'interactives.js'),
      output: {
        format: 'iife',
        entryFileNames: 'diagrams.js',
        assetFileNames: asset => asset.names.some(name => name.endsWith('.css'))
          ? 'diagrams.css' : '[name]-[hash][extname]',
      },
    },
  },
});
