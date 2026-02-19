"""
OtoWake Web - FastAPIエントリーポイント
"""

import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routers.separation import router as separation_router
from services.separator import job_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    logger.info("OtoWake Web を起動中...")

    # イベントループをJobManagerに設定
    loop = asyncio.get_event_loop()
    job_manager.set_event_loop(loop)

    # 定期クリーンアップタスクを開始
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # シャットダウン
    cleanup_task.cancel()
    logger.info("OtoWake Web をシャットダウンしました")


async def periodic_cleanup():
    """定期的に古いジョブをクリーンアップ"""
    while True:
        await asyncio.sleep(600)  # 10分ごと
        try:
            job_manager.cleanup_old_jobs(max_age_seconds=3600)
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


app = FastAPI(
    title="OtoWake Web",
    description="AI-Powered Stem Separation API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIルーター
app.include_router(separation_router)


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocketで進捗をリアルタイム配信"""
    await websocket.accept()

    job = job_manager.get_job(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": "ジョブが見つかりません"})
        await websocket.close()
        return

    # WebSocket接続を登録
    job_manager.register_ws(job_id, websocket)

    # 現在の状態を即座に送信
    if job.status == "completed":
        await websocket.send_json({
            "type": "complete",
            "tracks": {k: str(v).split("/")[-1] for k, v in job.tracks.items()} if job.tracks else {},
        })
    elif job.status == "error":
        await websocket.send_json({"type": "error", "message": job.error or "不明なエラー"})
    elif job.status in ("processing", "queued"):
        await websocket.send_json({
            "type": "progress",
            "progress": job.progress,
            "message": job.message,
        })

    try:
        # クライアントが切断するまで維持
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unregister_ws(job_id, websocket)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# 静的ファイル配信（Docker本番環境用）
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """React SPAのフォールバックルーティング"""
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
