"""
オーディオファイル管理（Web版）
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

import torchaudio
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioManager:
    """オーディオファイルの管理・検証クラス"""

    SUPPORTED_FORMATS = {
        '.mp3': 'MP3',
        '.wav': 'WAV',
        '.flac': 'FLAC',
        '.m4a': 'M4A',
        '.ogg': 'OGG',
        '.aac': 'AAC',
    }

    SUPPORTED_EXTENSIONS = set(SUPPORTED_FORMATS.keys())

    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in AudioManager.SUPPORTED_FORMATS

    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        if not os.path.exists(file_path):
            return False, "ファイルが存在しません"

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "ファイルが空です"

        if file_size > 500 * 1024 * 1024:
            return False, "ファイルが大きすぎます（500MB以上）"

        if not AudioManager.is_supported_format(file_path):
            ext = Path(file_path).suffix
            supported = ', '.join(AudioManager.SUPPORTED_FORMATS.keys())
            return False, f"サポートされていないフォーマット: {ext}\nサポート: {supported}"

        try:
            info = torchaudio.info(file_path)
            if info.num_frames == 0:
                return False, "オーディオデータがありません"
        except Exception as e:
            return False, f"オーディオファイルの読み込みに失敗: {str(e)}"

        return True, "OK"

    @staticmethod
    def get_audio_info(file_path: str) -> Dict:
        try:
            info = torchaudio.info(file_path)
            file_size = os.path.getsize(file_path)
            duration = info.num_frames / info.sample_rate
            minutes = int(duration // 60)
            seconds = int(duration % 60)

            return {
                'file_name': Path(file_path).name,
                'format': AudioManager.SUPPORTED_FORMATS.get(Path(file_path).suffix.lower(), 'Unknown'),
                'sample_rate': info.sample_rate,
                'channels': info.num_channels,
                'duration': duration,
                'duration_str': f"{minutes}:{seconds:02d}",
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.error(f"オーディオ情報の取得に失敗: {e}")
            return {}

    @staticmethod
    def convert_to_format(
        input_file: str,
        output_file: str,
        format: str = 'mp3',
        bitrate: str = '320k'
    ) -> bool:
        try:
            cmd = ['ffmpeg', '-y', '-i', input_file]

            if format == 'mp3':
                cmd.extend(['-codec:a', 'libmp3lame', '-b:a', bitrate, '-q:a', '0'])
            elif format == 'flac':
                cmd.extend(['-codec:a', 'flac', '-compression_level', '8'])
            elif format == 'ogg':
                cmd.extend(['-codec:a', 'libvorbis', '-q:a', '10'])

            cmd.append(output_file)

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

            if result.returncode != 0:
                logger.error(f"ffmpegエラー: {result.stderr.decode('utf-8', errors='ignore')}")
                return False

            return os.path.exists(output_file)

        except subprocess.TimeoutExpired:
            logger.error("変換タイムアウト（5分超過）")
            return False
        except Exception as e:
            logger.error(f"変換に失敗: {e}")
            return False
