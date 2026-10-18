import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';

export default defineConfig({
  // compiles 'use gpu' TypeScript functions to WGSL
  plugins: [typegpu({ include: [/\.m?[jt]sx?$/] })],
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  server: { host: 'localhost', port: 5173 },
});
