import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

/**
 * onnxruntime-web が public/ の .mjs ファイルを import() で動的にロードする。
 * Vite 開発サーバーは public/ のファイルを ESモジュールとして import することを
 * ブロックするため、カスタムプラグインで解決・ロードをバイパスする。
 */
function ortWasmPlugin(): Plugin {
  return {
    name: 'ort-wasm-mjs-bypass',
    enforce: 'pre',
    resolveId(id) {
      if (/ort-wasm.*\.mjs/.test(id)) {
        const filename = path.basename(id)
        const filePath = path.resolve(__dirname, 'public', filename)
        if (fs.existsSync(filePath)) {
          return filePath
        }
      }
    },
    load(id) {
      if (/ort-wasm.*\.mjs/.test(id) && fs.existsSync(id)) {
        return fs.readFileSync(id, 'utf-8')
      }
    },
  }
}

export default defineConfig({
  plugins: [ortWasmPlugin(), react()],
  server: {
    headers: {
      // SharedArrayBuffer (ONNX Runtime WASM マルチスレッド) に必要
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
  },
})
