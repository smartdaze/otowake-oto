declare module 'soundtouchjs' {
  export class SoundTouch {
    tempo: number;
    rate: number;
    pitch: number;
    pitchSemitones: number;
  }

  export class WebAudioBufferSource {
    constructor(buffer: AudioBuffer);
    extract(target: Float32Array, numFrames: number, position: number): number;
  }

  export class SimpleFilter {
    constructor(source: WebAudioBufferSource, pipe: SoundTouch, callback?: () => void);
    extract(target: Float32Array, numFrames: number): number;
    sourcePosition: number;
  }
}
