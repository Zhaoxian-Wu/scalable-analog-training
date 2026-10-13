import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    watch: {
      // Poll so edits from the IDE and shared workspace refresh slide metadata.
      usePolling: true,
      interval: 300,
    },
  },
})
