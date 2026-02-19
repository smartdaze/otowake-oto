"""
分離処理ジョブ管理（Web版）

QThread→threading.Thread に変更し、WebSocket経由で進捗を通知する。
"""

import os
import time
import uuid
import asyncio
import threading
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
from collections import deque
import logging

from models.demucs_wrapper import DemucsModel
from services.audio_manager import AudioManager

logger = logging.getLogger(__name__)

# ジョブの保存先ベースパス
JOBS_BASE_DIR = Path(__file__).parent.parent / "jobs"
UPLOADS_DIR = JOBS_BASE_DIR / "uploads"
OUTPUTS_DIR = JOBS_BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class Job:
    """1つの分離ジョブを表すクラス"""

    def __init__(self, job_id: str, file_path: str, file_name: str):
        self.job_id = job_id
        self.file_path = file_path
        self.file_name = file_name
        self.status = "ready"  # ready, queued, processing, completed, error
        self.progress = 0
        self.message = ""
        self.audio_info: Dict = {}
        self.tracks: Dict[str, str] = {}  # {track_name: file_path}
        self.error: Optional[str] = None
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "file_name": self.file_name,
            "audio_info": self.audio_info,
            "tracks": {k: Path(v).name for k, v in self.tracks.items()} if self.tracks else None,
            "error": self.error,
        }


class JobManager:
    """ジョブの管理とキュー処理"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.queue: deque = deque()
        self.is_processing = False
        self._lock = threading.Lock()
        self._ws_connections: Dict[str, List[Any]] = {}  # job_id -> [websocket, ...]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._demucs_models: Dict[str, DemucsModel] = {}

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def create_job(self, file_path: str, file_name: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id, file_path, file_name)

        # オーディオ情報を取得
        audio_info = AudioManager.get_audio_info(file_path)
        job.audio_info = audio_info

        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def delete_job(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False

        # ファイルをクリーンアップ
        if os.path.exists(job.file_path):
            os.remove(job.file_path)
        output_dir = OUTPUTS_DIR / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return True

    def register_ws(self, job_id: str, ws):
        if job_id not in self._ws_connections:
            self._ws_connections[job_id] = []
        self._ws_connections[job_id].append(ws)

    def unregister_ws(self, job_id: str, ws):
        if job_id in self._ws_connections:
            self._ws_connections[job_id] = [
                w for w in self._ws_connections[job_id] if w is not ws
            ]

    def _notify_ws(self, job_id: str, data: dict):
        """WebSocket接続に通知を送信（スレッドセーフ）"""
        if job_id not in self._ws_connections or not self._loop:
            return

        for ws in self._ws_connections.get(job_id, []):
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json(data), self._loop
                )
            except Exception as e:
                logger.warning(f"WebSocket送信エラー: {e}")

    def start_separation(
        self,
        job_id: str,
        model_name: str,
        stems: List[str],
        output_format: str,
        bitrate: str,
        remove_mode: bool,
    ):
        job = self.get_job(job_id)
        if job is None:
            return

        job.status = "queued"
        self.queue.append({
            "job_id": job_id,
            "model_name": model_name,
            "stems": stems,
            "output_format": output_format,
            "bitrate": bitrate,
            "remove_mode": remove_mode,
        })

        self._process_queue()

    def _process_queue(self):
        with self._lock:
            if self.is_processing or not self.queue:
                return
            self.is_processing = True

        task = self.queue.popleft()
        thread = threading.Thread(target=self._run_separation, args=(task,), daemon=True)
        thread.start()

    def _run_separation(self, task: dict):
        job_id = task["job_id"]
        job = self.get_job(job_id)
        if job is None:
            with self._lock:
                self.is_processing = False
            self._process_queue()
            return

        try:
            job.status = "processing"
            job.progress = 0
            job.message = "初期化中..."
            self._notify_ws(job_id, {"type": "progress", "progress": 0, "message": "初期化中..."})

            model_name = task["model_name"]
            start_time = time.time()

            # モデルをキャッシュから取得または新規作成
            if model_name not in self._demucs_models:
                self._demucs_models[model_name] = DemucsModel(model_name=model_name)
            demucs_model = self._demucs_models[model_name]

            # 出力ディレクトリ
            output_dir = str(OUTPUTS_DIR / job_id)
            os.makedirs(output_dir, exist_ok=True)

            filename_prefix = Path(job.file_path).stem

            def progress_callback(progress: int, message: str):
                # 推定残り時間
                if progress > 0:
                    elapsed = time.time() - start_time
                    estimated_total = elapsed * 100 / progress
                    remaining = estimated_total - elapsed
                    if remaining > 60:
                        eta = f"残り約{int(remaining/60)}分"
                    else:
                        eta = f"残り約{int(remaining)}秒"
                    message = f"{message} ({eta})"

                job.progress = progress
                job.message = message
                self._notify_ws(job_id, {
                    "type": "progress",
                    "progress": progress,
                    "message": message,
                })

            # 分離実行
            output_files = demucs_model.separate(
                audio_path=job.file_path,
                output_dir=output_dir,
                stems=task["stems"],
                progress_callback=progress_callback,
                remove_mode=task["remove_mode"],
                filename_prefix=filename_prefix,
            )

            # フォーマット変換
            output_format = task["output_format"]
            if output_format != "wav":
                self._notify_ws(job_id, {
                    "type": "progress",
                    "progress": 85,
                    "message": f"{output_format.upper()}に変換中...",
                })

                converted = {}
                for stem_name, wav_file in output_files.items():
                    out_file = wav_file.replace('.wav', f'.{output_format}')
                    if AudioManager.convert_to_format(wav_file, out_file, output_format, task["bitrate"]):
                        converted[stem_name] = out_file
                        os.remove(wav_file)
                    else:
                        converted[stem_name] = wav_file
                output_files = converted

            # 完了
            job.status = "completed"
            job.progress = 100
            job.message = "分離完了！"
            job.tracks = output_files

            track_names = list(output_files.keys())
            self._notify_ws(job_id, {
                "type": "complete",
                "tracks": {k: Path(v).name for k, v in output_files.items()},
            })

        except Exception as e:
            logger.error(f"分離エラー (job {job_id}): {e}", exc_info=True)
            job.status = "error"
            job.error = str(e)
            self._notify_ws(job_id, {"type": "error", "message": str(e)})

        finally:
            with self._lock:
                self.is_processing = False
            # 次のキューを処理
            self._process_queue()

    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """古いジョブをクリーンアップ"""
        now = time.time()
        to_delete = [
            job_id for job_id, job in self.jobs.items()
            if (job.status in ("completed", "error")) and (now - job.created_at > max_age_seconds)
        ]
        for job_id in to_delete:
            self.delete_job(job_id)
            logger.info(f"古いジョブを削除: {job_id}")


# シングルトンインスタンス
job_manager = JobManager()
