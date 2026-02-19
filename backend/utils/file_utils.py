"""
ファイル操作ユーティリティ（Web版）
"""

import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_unique_filename(directory: str, base_name: str, extension: str) -> str:
    path = Path(directory)
    file_path = path / f"{base_name}{extension}"

    if not file_path.exists():
        return str(file_path)

    counter = 1
    while True:
        file_path = path / f"{base_name}_{counter}{extension}"
        if not file_path.exists():
            return str(file_path)
        counter += 1


def clean_directory(directory: str):
    path = Path(directory)
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"ディレクトリを削除: {directory}")


def sanitize_filename(filename: str) -> str:
    """ファイル名をサニタイズ"""
    # パストラバーサル防止
    filename = Path(filename).name
    # 危険な文字を除去
    keepchars = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()
