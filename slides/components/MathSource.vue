<script setup lang="ts">
// Keep source outside slides.md so its math preprocessor leaves it literal.
const source = String.raw`An update uses learning rate $\eta$.

$$
\begin{aligned}
\mathbf{g}_t &= \nabla_{\mathbf{W}}
  \mathcal{L}(\mathbf{W}_t) \\
\mathbf{W}_{t+1} &= \mathbf{W}_t
  - \eta\mathbf{g}_t
\end{aligned}
$$`
</script>

<template>
  <pre class="latex-source"><code>{{ source }}</code></pre>
</template>
