"""
Demucsモデルのラッパークラス

音楽ファイルの楽器分離を実行するためのDemucsモデルのラッパーを提供します。
"""

import os
import torch
import torchaudio
import multiprocessing
from pathlib import Path
from typing import List, Callable, Optional, Dict
import logging

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemucsModel:
    """Demucsモデルを使用した音楽分離を実行するクラス"""

    AVAILABLE_MODELS = {
        'htdemucs': ['drums', 'bass', 'other', 'vocals'],
        'htdemucs_ft': ['drums', 'bass', 'other', 'vocals'],
        'htdemucs_6s': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
        'hdemucs_mmi': ['drums', 'bass', 'other', 'vocals'],
    }

    def __init__(self, model_name: str = 'htdemucs_6s'):
        self.model_name = model_name
        self.model = None

        if torch.cuda.is_available():
            self.device = 'cuda'
            device_name = 'NVIDIA GPU (CUDA)'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.zeros(1).to('mps')
                self.device = 'mps'
                device_name = 'Mac GPU (Metal Performance Shaders)'
            except Exception:
                self.device = 'cpu'
                device_name = 'CPU'
        else:
            self.device = 'cpu'
            device_name = 'CPU'

        logger.info(f"デバイス: {self.device} ({device_name})")
        logger.info(f"モデル: {model_name}")

    def load_model(self):
        if self.model is None:
            logger.info(f"{self.model_name}をロード中...")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            if hasattr(self.model, 'sources'):
                logger.info(f"モデルのトラック順序: {self.model.sources}")
            logger.info("モデルのロードが完了しました")

    def get_available_stems(self) -> List[str]:
        return self.AVAILABLE_MODELS.get(
            self.model_name,
            ['drums', 'bass', 'other', 'vocals']
        )

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        stems: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        remove_mode: bool = False,
        filename_prefix: str = ''
    ) -> Dict[str, str]:
        if self.model is None:
            self.load_model()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if stems is None:
            stems = self.get_available_stems()

        try:
            if progress_callback:
                progress_callback(10, "音声ファイルを読み込み中...")

            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.device)

            if sr != self.model.samplerate:
                logger.info(f"サンプルレートを変換: {sr}Hz -> {self.model.samplerate}Hz")
                resampler = torchaudio.transforms.Resample(
                    sr, self.model.samplerate
                ).to(self.device)
                wav = resampler(wav)
                sr = self.model.samplerate

            if progress_callback:
                progress_callback(30, "楽器分離を実行中...")

            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    wav.unsqueeze(0),
                    device=self.device,
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=True
                )[0]

            if progress_callback:
                progress_callback(70, "分離したトラックを保存中...")

            output_files = {}
            model_stems = self.model.sources if hasattr(self.model, 'sources') else self.get_available_stems()

            if not filename_prefix:
                filename_prefix = Path(audio_path).stem

            if remove_mode:
                mixed_audio = None
                for i, stem_name in enumerate(model_stems):
                    if stem_name not in stems:
                        if mixed_audio is None:
                            mixed_audio = sources[i].clone()
                        else:
                            mixed_audio += sources[i]

                if mixed_audio is not None:
                    output_file = output_path / f"{filename_prefix}_no_{'_'.join(stems)}.wav"
                    save_audio(mixed_audio.cpu(), str(output_file), sr, clip='clamp', as_float=True, bits_per_sample=24)
                    output_files[f"no_{'_'.join(stems)}"] = str(output_file)
            else:
                for i, stem_name in enumerate(model_stems):
                    if stem_name in stems:
                        output_file = output_path / f"{filename_prefix}_{stem_name}.wav"
                        save_audio(sources[i].cpu(), str(output_file), sr, clip='clamp', as_float=True, bits_per_sample=24)
                        output_files[stem_name] = str(output_file)

            if progress_callback:
                progress_callback(100, "分離完了！")

            return output_files

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def estimate_duration(self, audio_path: str) -> float:
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 0.0
