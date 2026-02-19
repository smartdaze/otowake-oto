"""
Pydanticスキーマ定義
"""

from typing import List, Optional, Dict
from pydantic import BaseModel


class SeparationRequest(BaseModel):
    job_id: str
    model_name: str = 'htdemucs_6s'
    stems: List[str] = ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']
    output_format: str = 'mp3'
    bitrate: str = '320k'
    remove_mode: bool = False


class JobStatus(BaseModel):
    job_id: str
    status: str  # uploading, ready, queued, processing, completed, error
    progress: int = 0
    message: str = ''
    file_name: Optional[str] = None
    audio_info: Optional[Dict] = None
    tracks: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    stems: List[str]


class UploadResponse(BaseModel):
    job_id: str
    file_name: str
    audio_info: Dict
