"""
녹음 데이터 모델
Pydantic 스키마 정의
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RecordingStatus(str, Enum):
    """녹음 상태 열거형"""
    UPLOADED = "uploaded"       # 업로드 완료
    TRANSCRIBING = "transcribing"  # 텍스트 변환 중
    TRANSCRIBED = "transcribed"   # 텍스트 변환 완료
    SUMMARIZING = "summarizing"   # 요약 중
    COMPLETED = "completed"       # 처리 완료
    ERROR = "error"               # 오류 발생


class RecordingBase(BaseModel):
    """녹음 기본 스키마"""
    title: Optional[str] = Field(None, description="녹음 제목")
    duration: Optional[float] = Field(None, description="녹음 길이 (초)")


class RecordingCreate(RecordingBase):
    """녹음 생성 스키마"""
    pass


class RecordingUpdate(BaseModel):
    """녹음 업데이트 스키마"""
    title: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    key_points: Optional[list[str]] = None
    status: Optional[RecordingStatus] = None


class Recording(RecordingBase):
    """녹음 응답 스키마"""
    id: str = Field(..., description="녹음 고유 ID")
    file_url: str = Field(..., description="오디오 파일 URL")
    file_name: str = Field(..., description="파일명")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    mime_type: str = Field(..., description="MIME 타입")
    status: RecordingStatus = Field(
        default=RecordingStatus.UPLOADED,
        description="처리 상태"
    )
    transcript: Optional[str] = Field(None, description="텍스트 변환 결과")
    summary: Optional[str] = Field(None, description="요약 결과")
    key_points: Optional[list[str]] = Field(None, description="핵심 포인트")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

    class Config:
        from_attributes = True


class RecordingList(BaseModel):
    """녹음 목록 응답 스키마"""
    items: list[Recording]
    total: int
    page: int
    page_size: int


class TranscriptionResponse(BaseModel):
    """텍스트 변환 응답 스키마"""
    recording_id: str
    transcript: str
    status: RecordingStatus


class SummaryResponse(BaseModel):
    """요약 응답 스키마"""
    recording_id: str
    summary: str
    key_points: list[str]
    status: RecordingStatus
