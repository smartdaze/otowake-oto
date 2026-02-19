/**
 * ブラウザ内 DSP (Digital Signal Processing) ユーティリティ
 *
 * HTDemucs の _spec / _ispec に対応する STFT / iSTFT を JavaScript で実装。
 * PyTorch の torch.stft / torch.istft と互換性のある出力を生成する。
 *
 * パラメータ (HTDemucs デフォルト):
 *   nfft = 4096, hop_length = 1024, window = hann(4096)
 *   normalized = true, center = true
 */

/* ============================================================
 * Radix-2 Cooley-Tukey FFT
 * ============================================================ */

/** ビット反転インデックスの事前計算 */
function bitReversePermutation(n: number): Uint32Array {
  const bits = Math.log2(n) | 0;
  const perm = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let rev = 0;
    let val = i;
    for (let b = 0; b < bits; b++) {
      rev = (rev << 1) | (val & 1);
      val >>= 1;
    }
    perm[i] = rev;
  }
  return perm;
}

/**
 * インプレース radix-2 FFT (Cooley-Tukey)
 * @param re 実部 (長さ n, n は 2 の冪)
 * @param im 虚部 (長さ n)
 * @param inverse true なら iFFT (1/N スケーリング含む)
 */
function fftInPlace(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  const perm = bitReversePermutation(n);

  // ビット反転並べ替え
  for (let i = 0; i < n; i++) {
    const j = perm[i];
    if (j > i) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // バタフライ演算
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angleStep = sign * 2 * Math.PI / size;
    const wRe = Math.cos(angleStep);
    const wIm = Math.sin(angleStep);

    for (let start = 0; start < n; start += size) {
      let curRe = 1.0;
      let curIm = 0.0;
      for (let j = 0; j < halfSize; j++) {
        const a = start + j;
        const b = start + j + halfSize;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] = re[a] + tRe;
        im[a] = im[a] + tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/* ============================================================
 * Hann window
 * ============================================================ */

function hannWindow(size: number): Float64Array {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return w;
}

/* ============================================================
 * STFT (Short-Time Fourier Transform)
 *
 * PyTorch 互換: center=true, normalized=true, pad_mode='reflect'
 *
 * 出力: { real, imag } 各 Float32Array[numFreqs * numFrames]
 *       numFreqs = nfft/2 + 1, row-major [freq][frame]
 * ============================================================ */

export interface StftResult {
  real: Float32Array;   // [numFreqs * numFrames]
  imag: Float32Array;
  numFreqs: number;     // nfft/2 + 1
  numFrames: number;
}

/**
 * 1D 信号に対する STFT
 * @param signal 入力信号 (Float32Array)
 * @param nfft FFT サイズ
 * @param hopLength ホップ長
 * @returns StftResult
 */
export function stft(signal: Float32Array, nfft: number, hopLength: number): StftResult {
  const normFactor = 1.0 / Math.sqrt(nfft);  // normalized=true
  const window = hannWindow(nfft);
  const numFreqs = nfft / 2 + 1;

  // center padding (reflect)
  const padSize = nfft / 2;
  const paddedLen = signal.length + 2 * padSize;
  const padded = new Float64Array(paddedLen);

  // reflect pad left
  for (let i = 0; i < padSize; i++) {
    padded[padSize - 1 - i] = signal[Math.min(i + 1, signal.length - 1)];
  }
  // copy signal
  for (let i = 0; i < signal.length; i++) {
    padded[padSize + i] = signal[i];
  }
  // reflect pad right
  for (let i = 0; i < padSize; i++) {
    padded[padSize + signal.length + i] = signal[Math.max(signal.length - 2 - i, 0)];
  }

  const numFrames = Math.floor((paddedLen - nfft) / hopLength) + 1;

  const real = new Float32Array(numFreqs * numFrames);
  const imag = new Float32Array(numFreqs * numFrames);

  const frameRe = new Float64Array(nfft);
  const frameIm = new Float64Array(nfft);

  for (let t = 0; t < numFrames; t++) {
    const offset = t * hopLength;

    // 窓掛け
    for (let i = 0; i < nfft; i++) {
      frameRe[i] = padded[offset + i] * window[i];
      frameIm[i] = 0;
    }

    // FFT
    fftInPlace(frameRe, frameIm, false);

    // 正の周波数のみ保存 (normalized)
    for (let f = 0; f < numFreqs; f++) {
      real[f * numFrames + t] = (frameRe[f] * normFactor) as number;
      imag[f * numFrames + t] = (frameIm[f] * normFactor) as number;
    }
  }

  return { real, imag, numFreqs, numFrames };
}

/* ============================================================
 * iSTFT (Inverse Short-Time Fourier Transform)
 *
 * PyTorch 互換: center=true, normalized=true
 * ============================================================ */

/**
 * iSTFT — 複素スペクトログラムから時間領域信号を再構成
 * @param real 実部 [numFreqs * numFrames] (row-major [freq][frame])
 * @param imag 虚部
 * @param numFreqs 周波数ビン数 (nfft/2+1)
 * @param numFrames 時間フレーム数
 * @param hopLength ホップ長
 * @param outputLength 出力信号の目標長 (center=true のトリミング前)
 * @returns Float32Array 時間領域信号
 */
export function istft(
  real: Float32Array,
  imag: Float32Array,
  numFreqs: number,
  numFrames: number,
  hopLength: number,
  outputLength: number,
): Float32Array {
  const nfft = (numFreqs - 1) * 2;
  const normFactor = Math.sqrt(nfft);  // normalized=true の逆
  const window = hannWindow(nfft);

  // overlap-add バッファ
  const rawLength = (numFrames - 1) * hopLength + nfft;
  const outBuf = new Float64Array(rawLength);
  const winBuf = new Float64Array(rawLength);

  const frameRe = new Float64Array(nfft);
  const frameIm = new Float64Array(nfft);

  for (let t = 0; t < numFrames; t++) {
    // 正の周波数を取得 + 正規化逆変換
    for (let f = 0; f < numFreqs; f++) {
      frameRe[f] = real[f * numFrames + t] * normFactor;
      frameIm[f] = imag[f * numFrames + t] * normFactor;
    }

    // 共役対称性で負の周波数を復元
    for (let f = 1; f < numFreqs - 1; f++) {
      frameRe[nfft - f] = frameRe[f];
      frameIm[nfft - f] = -frameIm[f];
    }

    // iFFT
    fftInPlace(frameRe, frameIm, true);

    // 窓掛け + overlap-add
    const offset = t * hopLength;
    for (let i = 0; i < nfft; i++) {
      outBuf[offset + i] += frameRe[i] * window[i];
      winBuf[offset + i] += window[i] * window[i];
    }
  }

  // 窓正規化
  for (let i = 0; i < rawLength; i++) {
    if (winBuf[i] > 1e-8) {
      outBuf[i] /= winBuf[i];
    }
  }

  // center=true のトリミング
  const padSize = nfft / 2;
  const result = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    result[i] = outBuf[padSize + i];
  }
  return result;
}

/* ============================================================
 * HTDemucs 用 _spec / _ispec ヘルパー
 *
 * Python 版の _spec / _ispec のパディング・トリミングロジックを再現
 * ============================================================ */

export interface SpecParams {
  nfft: number;
  hopLength: number;
  numFreqBins: number;      // nfft/2 (trim 後)
  specTimeFrames: number;   // le (trim 後)
  specPad: number;          // hop_length // 2 * 3
}

/**
 * _spec 相当: 波形 → magnitude テンソル (ONNX 入力用)
 *
 * @param signal 1ch の波形 Float32Array
 * @param params STFT パラメータ
 * @returns { mag: Float32Array[C*2 * Fr * T], specReal, specImag } specReal/specImag は iSTFT 用
 */
export interface SpecResult {
  /** magnitude [Fr * T] — 1 チャネル分の real part */
  real: Float32Array;
  /** magnitude [Fr * T] — 1 チャネル分の imag part */
  imag: Float32Array;
  numFreqs: number;  // = numFreqBins (trim 後, nfft/2)
  numFrames: number; // = specTimeFrames (trim 後, le)
}

/**
 * HTDemucs._spec 相当の前処理
 *
 * 1. reflect pad
 * 2. STFT
 * 3. freq trim (:-1) と time trim (2:2+le)
 *
 * @param signal 1ch の波形
 * @param params SpecParams
 */
export function htdemucsSpec(signal: Float32Array, params: SpecParams): SpecResult {
  const { nfft, hopLength, numFreqBins, specTimeFrames, specPad } = params;
  const le = specTimeFrames;
  const T = signal.length;

  // _spec の reflect パディング
  const padLeft = specPad;
  const padRight = specPad + le * hopLength - T;
  const paddedLen = T + padLeft + padRight;
  const padded = new Float32Array(paddedLen);

  // reflect pad left
  for (let i = 0; i < padLeft; i++) {
    padded[padLeft - 1 - i] = signal[Math.min(i + 1, T - 1)];
  }
  // copy signal
  for (let i = 0; i < T; i++) {
    padded[padLeft + i] = signal[i];
  }
  // reflect pad right
  for (let i = 0; i < padRight; i++) {
    padded[padLeft + T + i] = signal[Math.max(T - 2 - i, 0)];
  }

  // STFT (center=true, normalized=true)
  const stftResult = stft(padded, nfft, hopLength);

  // freq trim: 除去最後のビン (nfft/2+1 → nfft/2)
  // stftResult.numFreqs = nfft/2+1, 最後のビンを除外
  // time trim: [2:2+le]
  const Fr = numFreqBins;  // nfft/2
  const real = new Float32Array(Fr * le);
  const imag = new Float32Array(Fr * le);

  for (let f = 0; f < Fr; f++) {
    for (let t = 0; t < le; t++) {
      const srcIdx = f * stftResult.numFrames + (t + 2);
      real[f * le + t] = stftResult.real[srcIdx];
      imag[f * le + t] = stftResult.imag[srcIdx];
    }
  }

  return { real, imag, numFreqs: Fr, numFrames: le };
}

/**
 * _magnitude 相当: スペクトログラムを ONNX 入力の magnitude 形式に変換
 *
 * view_as_real(z).permute(0,1,4,2,3).reshape(B, C*2, Fr, T)
 * → チャネル c の real/imag を交互に配置: [c0_real, c0_imag, c1_real, c1_imag, ...][Fr][T]
 *
 * @param channelSpecs 各チャネルの SpecResult
 * @returns Float32Array [(C*2) * Fr * T] — row-major
 */
export function specToMagnitude(channelSpecs: SpecResult[]): Float32Array {
  const C = channelSpecs.length;
  const Fr = channelSpecs[0].numFreqs;
  const T = channelSpecs[0].numFrames;
  const mag = new Float32Array(C * 2 * Fr * T);

  for (let c = 0; c < C; c++) {
    const spec = channelSpecs[c];
    const realOff = (c * 2) * Fr * T;
    const imagOff = (c * 2 + 1) * Fr * T;
    for (let i = 0; i < Fr * T; i++) {
      mag[realOff + i] = spec.real[i];
      mag[imagOff + i] = spec.imag[i];
    }
  }

  return mag;
}

/**
 * ONNX 出力の freq_out → iSTFT で波形を再構成
 *
 * freq_out [S, C*2, Fr, T] → view(S, C, 2, Fr, T) → permute(S, C, Fr, T, 2)
 * → pad freq (+1), pad time (+4) → iSTFT
 *
 * @param freqOut Float32Array [S * C*2 * Fr * T]
 * @param numSources S
 * @param audioChannels C
 * @param params SpecParams
 * @param segmentLength 元波形のサンプル数 T_original
 * @returns Float32Array [S * C * T_original]
 */
export function freqOutToWaveform(
  freqOut: Float32Array,
  numSources: number,
  audioChannels: number,
  params: SpecParams,
  segmentLength: number,
): Float32Array {
  const S = numSources;
  const C = audioChannels;
  const Fr = params.numFreqBins;   // nfft/2 = 2048
  const T = params.specTimeFrames; // le = 336
  const hl = params.hopLength;     // 1024

  // _ispec パラメータ
  const specPad = hl / 2 * 3;     // 1536
  const le_ispec = hl * Math.ceil(segmentLength / hl) + 2 * specPad;

  const result = new Float32Array(S * C * segmentLength);

  for (let s = 0; s < S; s++) {
    for (let c = 0; c < C; c++) {
      // freq_out のレイアウト: [s, c*2, Fr, T] と [s, c*2+1, Fr, T]
      const realChIdx = s * C * 2 + c * 2;
      const imagChIdx = s * C * 2 + c * 2 + 1;

      // パディング: freq +1 (最後のビンをゼロ追加), time +4 (前後2フレームずつ)
      const paddedFr = Fr + 1;  // nfft/2 + 1 = 2049
      const paddedT = T + 4;    // le + 4

      const paddedReal = new Float32Array(paddedFr * paddedT);
      const paddedImag = new Float32Array(paddedFr * paddedT);

      // freq_out から取得して padding 配置
      // time: 2 フレームオフセット (前に 2 フレーム追加)
      for (let f = 0; f < Fr; f++) {
        for (let t = 0; t < T; t++) {
          const srcReal = freqOut[realChIdx * Fr * T + f * T + t];
          const srcImag = freqOut[imagChIdx * Fr * T + f * T + t];
          paddedReal[f * paddedT + (t + 2)] = srcReal;
          paddedImag[f * paddedT + (t + 2)] = srcImag;
        }
      }
      // 最後の周波数ビン (Fr) はゼロのまま
      // 前後 2 フレーム (0,1 と T+2,T+3) はゼロのまま

      // iSTFT
      const waveform = istft(paddedReal, paddedImag, paddedFr, paddedT, hl, le_ispec);

      // _ispec のトリミング: [specPad : specPad + segmentLength]
      const outOff = (s * C + c) * segmentLength;
      for (let i = 0; i < segmentLength; i++) {
        result[outOff + i] = waveform[specPad + i];
      }
    }
  }

  return result;
}
