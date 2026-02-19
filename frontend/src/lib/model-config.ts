/**
 * ONNXモデルの静的設定
 *
 * ルール（絶対遵守）:
 *   - 日本語で思考・応答
 *   - music-separator / music-separator-web は変更しない
 *   - OtoWake-ONNX 内のみで作業
 *   - ランディングページは現状維持
 */

export interface ModelConfig {
  name: string;
  displayName: string;
  stems: string[];
  sampleRate: number;
  /** セグメント長（秒） — Demucs apply_model の segment パラメータ */
  segment: number;
  modelFile: string;
  /** STFT パラメータ (export_onnx.py の JSON メタデータから) */
  nfft?: number;
  hopLength?: number;
  numFreqBins?: number;
  specTimeFrames?: number;
  specPad?: number;
}

export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  htdemucs: {
    name: 'htdemucs',
    displayName: 'HTDemucs (4ステム)',
    stems: ['drums', 'bass', 'other', 'vocals'],
    sampleRate: 44100,
    segment: 7.8,
    modelFile: 'htdemucs.onnx',
    nfft: 4096,
    hopLength: 1024,
    numFreqBins: 2048,
    specTimeFrames: 336,
    specPad: 1536,
  },
  htdemucs_ft: {
    name: 'htdemucs_ft',
    displayName: 'HTDemucs Fine-tuned (4ステム)',
    stems: ['drums', 'bass', 'other', 'vocals'],
    sampleRate: 44100,
    segment: 7.8,
    modelFile: 'htdemucs_ft.onnx',
    nfft: 4096,
    hopLength: 1024,
    numFreqBins: 2048,
    specTimeFrames: 336,
    specPad: 1536,
  },
  htdemucs_6s: {
    name: 'htdemucs_6s',
    displayName: 'HTDemucs 6 Sources',
    stems: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
    sampleRate: 44100,
    segment: 7.8,
    modelFile: 'htdemucs_6s.onnx',
    nfft: 4096,
    hopLength: 1024,
    numFreqBins: 2048,
    specTimeFrames: 336,
    specPad: 1536,
  },
  hdemucs_mmi: {
    name: 'hdemucs_mmi',
    displayName: 'HDemucs MMI (4ステム)',
    stems: ['drums', 'bass', 'other', 'vocals'],
    sampleRate: 44100,
    segment: 7.8,
    modelFile: 'hdemucs_mmi.onnx',
    nfft: 4096,
    hopLength: 1024,
    numFreqBins: 2048,
    specTimeFrames: 336,
    specPad: 1536,
  },
};

export const DEFAULT_MODEL = 'htdemucs_6s';

export const SUPPORTED_AUDIO_EXTENSIONS = [
  '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.webm',
];
