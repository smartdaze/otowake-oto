"""
分離処理のAPIエンドポイント
"""

import os
import zipfile
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from schemas.separation import SeparationRequest, JobStatus, ModelInfo, UploadResponse
from services.separator import job_manager, UPLOADS_DIR, OUTPUTS_DIR
from services.audio_manager import AudioManager
from utils.file_utils import sanitize_filename
from models.demucs_wrapper import DemucsModel

router = APIRouter(prefix="/api")

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """利用可能なモデル一覧を取得"""
    return [
        ModelInfo(name=name, stems=stems)
        for name, stems in DemucsModel.AVAILABLE_MODELS.items()
    ]


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """音声ファイルをアップロード"""
    # ファイル名をサニタイズ
    safe_name = sanitize_filename(file.filename or "audio.mp3")

    # 拡張子チェック
    ext = Path(safe_name).suffix.lower()
    if ext not in AudioManager.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないフォーマット: {ext}"
        )

    # ファイルを一時保存
    import uuid
    temp_name = f"{uuid.uuid4()}{ext}"
    file_path = str(UPLOADS_DIR / temp_name)

    # サイズチェックしながら書き込み
    total_size = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="ファイルが大きすぎます（500MB以上）")
            f.write(chunk)

    # オーディオファイルの検証
    is_valid, message = AudioManager.validate_audio_file(file_path)
    if not is_valid:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=message)

    # ジョブを作成
    job = job_manager.create_job(file_path, safe_name)

    return UploadResponse(
        job_id=job.job_id,
        file_name=safe_name,
        audio_info=job.audio_info,
    )


@router.post("/separate")
async def start_separation(request: SeparationRequest):
    """分離処理を開始"""
    job = job_manager.get_job(request.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")

    if job.status not in ("ready",):
        raise HTTPException(status_code=400, detail=f"ジョブの状態が不正です: {job.status}")

    # モデルの検証
    if request.model_name not in DemucsModel.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"不明なモデル: {request.model_name}")

    # トラックの検証
    available = DemucsModel.AVAILABLE_MODELS[request.model_name]
    invalid = [s for s in request.stems if s not in available]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"モデル '{request.model_name}' は以下のトラックをサポートしていません: {invalid}"
        )

    # 分離処理を開始
    job_manager.start_separation(
        job_id=request.job_id,
        model_name=request.model_name,
        stems=request.stems,
        output_format=request.output_format,
        bitrate=request.bitrate,
        remove_mode=request.remove_mode,
    )

    return {"status": "started", "job_id": request.job_id}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """ジョブの状態を取得"""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return JobStatus(**job.to_dict())


@router.get("/jobs/{job_id}/download/{track}")
async def download_track(job_id: str, track: str):
    """分離済みトラックをダウンロード"""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="ジョブが完了していません")

    file_path = job.tracks.get(track)
    if file_path is None or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"トラック '{track}' が見つかりません")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@router.get("/jobs/{job_id}/download-all")
async def download_all_tracks(job_id: str):
    """全トラックをZIPでダウンロード"""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="ジョブが完了していません")

    # ZIPファイルを作成
    zip_path = str(OUTPUTS_DIR / job_id / f"{Path(job.file_name).stem}_separated.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for track_name, file_path in job.tracks.items():
            if os.path.exists(file_path):
                zf.write(file_path, Path(file_path).name)

    return FileResponse(
        path=zip_path,
        filename=Path(zip_path).name,
        media_type="application/zip",
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """ジョブを削除"""
    if not job_manager.delete_job(job_id):
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return {"status": "deleted"}
