/**
 * ブラウザ内音声処理ユーティリティ
 *
 * Web Audio API を利用した音声デコード・リサンプル・WAV エンコード
 */

export interface AudioInfo {
  file_name: string;
  format: string;
  sample_rate: number;
  channels: number;
  duration: number;
  duration_str: string;
  file_size: number;
  file_size_mb: number;
}

/* ---------- デコード ---------- */

/**
 * File を AudioBuffer にデコードする。
 * ブラウザの AudioContext が対応するすべてのフォーマットを扱える。
 */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new AudioContext();
  try {
    return await ctx.decodeAudioData(arrayBuffer);
  } finally {
    await ctx.close();
  }
}

/* ---------- リサンプル ---------- */

export async function resampleAudio(
  audioBuffer: AudioBuffer,
  targetSampleRate: number,
): Promise<AudioBuffer> {
  if (audioBuffer.sampleRate === targetSampleRate) return audioBuffer;

  const channels = audioBuffer.numberOfChannels;
  const targetLength = Math.ceil(audioBuffer.duration * targetSampleRate);
  const offlineCtx = new OfflineAudioContext(channels, targetLength, targetSampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);
  return offlineCtx.startRendering();
}

/* ---------- AudioBuffer → Float32Array (チャネル分離形式) ---------- */

/**
 * AudioBuffer から [ch0_all_samples, ch1_all_samples, ...] の flat Float32Array を返す。
 * モノラル入力はステレオに複製する。
 */
export function audioBufferToChannelData(
  audioBuffer: AudioBuffer,
): { data: Float32Array; channels: number; length: number } {
  const numCh = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;

  if (numCh === 1) {
    const mono = audioBuffer.getChannelData(0);
    const data = new Float32Array(2 * length);
    data.set(mono, 0);
    data.set(mono, length);
    return { data, channels: 2, length };
  }

  // ステレオ (or 多チャネル → 先頭 2ch を使う)
  const ch = Math.min(numCh, 2);
  const data = new Float32Array(ch * length);
  for (let c = 0; c < ch; c++) {
    data.set(audioBuffer.getChannelData(c), c * length);
  }
  return { data, channels: ch, length };
}

/* ---------- WAV エンコード ---------- */

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function clamp(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

/**
 * 複数チャネルの Float32Array 配列を 32-bit float WAV Blob にエンコードする。
 */
export function encodeWav(
  channelData: Float32Array[],
  sampleRate: number,
): Blob {
  const channels = channelData.length;
  const numSamples = channelData[0].length;
  const bytesPerSample = 4; // 32-bit float
  const dataSize = numSamples * channels * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true); // IEEE float
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 32, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < channels; ch++) {
      view.setFloat32(offset, clamp(channelData[ch][i]), true);
      offset += 4;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/* ---------- AudioInfo ---------- */

export function getAudioInfo(file: File, audioBuffer: AudioBuffer): AudioInfo {
  const duration = audioBuffer.duration;
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  const ext = file.name.split('.').pop()?.toUpperCase() ?? 'Unknown';

  return {
    file_name: file.name,
    format: ext,
    sample_rate: audioBuffer.sampleRate,
    channels: audioBuffer.numberOfChannels,
    duration,
    duration_str: `${minutes}:${seconds.toString().padStart(2, '0')}`,
    file_size: file.size,
    file_size_mb: Math.round((file.size / (1024 * 1024)) * 100) / 100,
  };
}
