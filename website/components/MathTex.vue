<script setup>
import { computed } from 'vue'
import katex from 'katex'
const props = defineProps({ tex: { type: String, required: true }, display: { type: Boolean, default: false } })
const rendered = computed(() => katex.renderToString(props.tex, { displayMode: props.display, throwOnError: true }))
</script>
<template><span class="math-tex" v-html="rendered"></span></template>
