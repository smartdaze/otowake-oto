/**
 * ONNX Runtime Web を使った音源分離エンジン
 *
 * アーキテクチャ:
 *   1. JavaScript: STFT (前処理) — 波形 → magnitude テンソル
 *   2. ONNX: NN コア — encoder / transformer / decoder
 *   3. JavaScript: iSTFT (後処理) — freq_out → 波形再構成 + time_out 加算
 */

import * as ort from 'onnxruntime-web';
import type { ModelConfig } from './model-config.ts';
import {
  htdemucsSpec,
  specToMagnitude,
  freqOutToWaveform,
} from './dsp.ts';
import type { SpecParams } from './dsp.ts';

export type ProgressCallback = (progress: number, message: string) => void;

export interface StemResult {
  name: string;
  channels: Float32Array[]; // [ch0, ch1]
}

export interface SeparationResult {
  stems: StemResult[];
  sampleRate: number;
}

/* ---- ONNX Runtime 初期設定 ---- */

// WASM ファイルのパスを明示的に指定（Vercel等でバンドルパスと異なる場合の対策）
ort.env.wasm.wasmPaths = '/';

// proxy: true はバンドル全体を Worker に読み込むため、
// React 等の DOM コードが Worker 内で document を参照してエラーになる。
// proxy を使わず、numThreads のみでマルチスレッド WASM 演算を行う。
ort.env.wasm.proxy = false;

if (typeof SharedArrayBuffer !== 'undefined') {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
} else {
  ort.env.wasm.numThreads = 1;
}

/* ================================================================ */

export class OnnxSeparator {
  private session: ort.InferenceSession | null = null;
  private config: ModelConfig | null = null;
  private loadedModelName: string | null = null;

  /* ---------- モデル読み込み ---------- */

  async loadModel(config: ModelConfig, onProgress?: ProgressCallback): Promise<void> {
    if (this.session && this.loadedModelName === config.name) return;
    this.dispose();
    this.config = config;

    const modelUrl = `/models/${config.modelFile}`;
    onProgress?.(0, 'モデルをダウンロード中...');

    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(
        `モデルファイルが見つかりません: /models/${config.modelFile}\n` +
        'tools/export_onnx.py でモデルを変換し、public/models/ に配置してください。',
      );
    }

    const contentLength = Number(response.headers.get('Content-Length') || 0);
    const reader = response.body!.getReader();
    const chunks: Uint8Array[] = [];
    let receivedLength = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedLength += value.length;
      if (contentLength > 0) {
        onProgress?.(Math.round((receivedLength / contentLength) * 50), 'モデルをダウンロード中...');
      }
    }

    const modelData = new Uint8Array(receivedLength);
    let offset = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, offset);
      offset += chunk.length;
    }

    onProgress?.(55, 'ONNXセッションを初期化中...');

    this.session = await ort.InferenceSession.create(modelData.buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    this.loadedModelName = config.name;
    onProgress?.(60, 'モデルの準備完了');
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  getLoadedModelName(): string | null {
    return this.loadedModelName;
  }

  /* ---------- 分離処理 ---------- */

  async separate(
    audioData: Float32Array,
    channels: number,
    length: number,
    stems: string[],
    removeMode: boolean,
    onProgress?: ProgressCallback,
  ): Promise<SeparationResult> {
    if (!this.session || !this.config) {
      throw new Error('モデルが読み込まれていません');
    }

    const cfg = this.config;
    const sr = cfg.sampleRate;
    const numSources = cfg.stems.length;
    const segmentLength = Math.round(cfg.segment * sr);
    const overlap = 0.25;
    const stride = Math.round((1 - overlap) * segmentLength);

    // STFT パラメータ (メタデータから取得、または計算)
    const specParams: SpecParams = {
      nfft: cfg.nfft ?? 4096,
      hopLength: cfg.hopLength ?? 1024,
      numFreqBins: cfg.numFreqBins ?? 2048,
      specTimeFrames: cfg.specTimeFrames ?? Math.ceil(segmentLength / (cfg.hopLength ?? 1024)),
      specPad: cfg.specPad ?? ((cfg.hopLength ?? 1024) / 2 * 3),
    };

    onProgress?.(5, '音声データを準備中...');

    // テント窓
    const weight = new Float32Array(segmentLength);
    const half = Math.floor(segmentLength / 2);
    for (let i = 0; i < half; i++) weight[i] = (i + 1) / half;
    for (let i = half; i < segmentLength; i++) weight[i] = (segmentLength - i) / (segmentLength - half);

    // パディング
    const padLen = segmentLength - stride;
    const totalPadLen = length + 2 * padLen;

    const paddedInput = new Float32Array(channels * totalPadLen);
    for (let ch = 0; ch < channels; ch++) {
      const srcOff = ch * length;
      const dstOff = ch * totalPadLen + padLen;
      for (let i = 0; i < length; i++) {
        paddedInput[dstOff + i] = audioData[srcOff + i];
      }
      for (let i = 0; i < padLen; i++) {
        paddedInput[ch * totalPadLen + padLen - 1 - i] =
          audioData[srcOff + Math.min(i + 1, length - 1)];
      }
      for (let i = 0; i < padLen; i++) {
        paddedInput[ch * totalPadLen + padLen + length + i] =
          audioData[srcOff + Math.max(length - 2 - i, 0)];
      }
    }

    const numSegments = Math.max(1, Math.ceil((totalPadLen - segmentLength) / stride) + 1);

    // 出力バッファ
    const output = new Float32Array(numSources * channels * length);
    const sumWeight = new Float32Array(length);

    onProgress?.(10, `楽器分離を実行中... (0/${numSegments})`);

    for (let seg = 0; seg < numSegments; seg++) {
      const start = seg * stride;
      const end = Math.min(start + segmentLength, totalPadLen);
      const actualLen = end - start;

      // セグメント切り出し
      const segMix = new Float32Array(channels * segmentLength);
      for (let ch = 0; ch < channels; ch++) {
        const srcOff = ch * totalPadLen + start;
        for (let i = 0; i < actualLen; i++) {
          segMix[ch * segmentLength + i] = paddedInput[srcOff + i];
        }
      }

      // ---- 1. STFT (JavaScript) ----
      const channelSpecs = [];
      for (let ch = 0; ch < channels; ch++) {
        const chSignal = segMix.subarray(ch * segmentLength, (ch + 1) * segmentLength);
        channelSpecs.push(htdemucsSpec(chSignal, specParams));
      }
      const mag = specToMagnitude(channelSpecs);

      // ---- 2. ONNX 推論 ----
      const Fr = specParams.numFreqBins;
      const T_spec = specParams.specTimeFrames;
      const magTensor = new ort.Tensor('float32', mag, [1, channels * 2, Fr, T_spec]);
      const mixTensor = new ort.Tensor('float32', segMix, [1, channels, segmentLength]);

      const feeds: Record<string, ort.Tensor> = {
        mag: magTensor,
        mix: mixTensor,
      };
      const results = await this.session.run(feeds);
      const freqOutData = results['freq_out'].data as Float32Array;
      const timeOutData = results['time_out'].data as Float32Array;

      // ---- 3. iSTFT (JavaScript) ----
      const freqWaveform = freqOutToWaveform(
        freqOutData,
        numSources,
        channels,
        specParams,
        segmentLength,
      );

      // freq + time を合成
      // freqWaveform: [S * C * segmentLength]
      // timeOutData:  [1 * S * C * segmentLength] (バッチ次元含む)

      // overlap-add 蓄積
      const outStart = start - padLen;
      for (let src = 0; src < numSources; src++) {
        for (let ch = 0; ch < channels; ch++) {
          const freqOff = (src * channels + ch) * segmentLength;
          const timeOff = (src * channels + ch) * segmentLength; // 1バッチなので同じオフセット
          const bufOff = (src * channels + ch) * length;
          for (let i = 0; i < actualLen; i++) {
            const outIdx = outStart + i;
            if (outIdx >= 0 && outIdx < length) {
              const sample = freqWaveform[freqOff + i] + timeOutData[timeOff + i];
              output[bufOff + outIdx] += sample * weight[i];
            }
          }
        }
      }

      for (let i = 0; i < actualLen; i++) {
        const outIdx = outStart + i;
        if (outIdx >= 0 && outIdx < length) {
          sumWeight[outIdx] += weight[i];
        }
      }

      const pct = 10 + Math.round(((seg + 1) / numSegments) * 75);
      onProgress?.(pct, `楽器分離を実行中... (${seg + 1}/${numSegments})`);
      await yieldToEventLoop();
    }

    // 重み正規化
    for (let src = 0; src < numSources; src++) {
      for (let ch = 0; ch < channels; ch++) {
        const off = (src * channels + ch) * length;
        for (let i = 0; i < length; i++) {
          if (sumWeight[i] > 0) output[off + i] /= sumWeight[i];
        }
      }
    }

    onProgress?.(90, '結果を生成中...');

    // ステム抽出
    const allStems = cfg.stems;
    const resultStems: StemResult[] = [];

    if (removeMode) {
      const ch0 = new Float32Array(length);
      const ch1 = new Float32Array(length);
      for (let src = 0; src < numSources; src++) {
        if (stems.includes(allStems[src])) continue;
        const off0 = (src * channels + 0) * length;
        const off1 = (src * channels + 1) * length;
        for (let i = 0; i < length; i++) {
          ch0[i] += output[off0 + i];
          ch1[i] += output[off1 + i];
        }
      }
      resultStems.push({
        name: `no_${stems.join('_')}`,
        channels: [ch0, ch1],
      });
    } else {
      for (let src = 0; src < numSources; src++) {
        if (!stems.includes(allStems[src])) continue;
        const off0 = (src * channels + 0) * length;
        const off1 = channels > 1 ? (src * channels + 1) * length : off0;
        resultStems.push({
          name: allStems[src],
          channels: [
            output.slice(off0, off0 + length),
            output.slice(off1, off1 + length),
          ],
        });
      }
    }

    onProgress?.(95, '完了準備中...');
    return { stems: resultStems, sampleRate: sr };
  }

  /* ---------- リソース解放 ---------- */

  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.config = null;
    this.loadedModelName = null;
  }
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
