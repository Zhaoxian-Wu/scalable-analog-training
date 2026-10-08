<script setup>
import { computed } from 'vue'
import { useNav } from '@slidev/client'

const { slides, currentSlideRoute, currentSlideNo } = useNav()
const contentSlides = computed(() => slides.value.filter(slide => slide.no > 1 && !slide.meta?.slide?.frontmatter.appendix))
const pageNumber = computed(() => contentSlides.value.findIndex(slide => slide.no === currentSlideNo.value) + 1)
const isAppendix = computed(() => currentSlideRoute.value?.meta?.slide?.frontmatter.appendix === true)
</script>

<template>
  <footer v-if="pageNumber || isAppendix" class="deck-footer">
    <span v-if="isAppendix">Appendix</span>
    <span v-else>{{ String(pageNumber).padStart(2, '0') }} <span class="footer-separator">/</span> {{ String(contentSlides.length).padStart(2, '0') }}</span>
  </footer>
</template>
