"""
녹음 API 라우터
음성 녹음 업로드, 변환, 요약 관련 엔드포인트
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.recording import (
    Recording,
    RecordingList,
    RecordingStatus,
    SummaryResponse,
    SummaryType,
    TranscriptionResponse,
)
from app.services.storage import get_storage_service
from app.services.stt import get_stt_service
from app.services.llm import get_llm_service, SummaryType as LLMSummaryType


router = APIRouter(prefix="/recordings", tags=["recordings"])


@router.post("/upload", response_model=Recording, status_code=status.HTTP_201_CREATED)
async def upload_recording(
    file: UploadFile = File(..., description="오디오 파일"),
    title: Optional[str] = Form(None, description="녹음 제목"),
    duration: Optional[float] = Form(None, description="녹음 길이 (초)")
):
    """
    오디오 파일 업로드

    - 지원 형식: webm, mp3, wav, m4a, ogg
    - 최대 파일 크기: 25MB (Groq Whisper 제한)
    """
    # 파일 유효성 검사
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일명이 없습니다."
        )

    # 지원하는 오디오 형식 확인
    allowed_extensions = {"webm", "mp3", "wav", "m4a", "ogg", "flac"}
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
        )

    # 파일 읽기
    content = await file.read()
    file_size = len(content)

    # 파일 크기 검사 (25MB)
    max_size = 25 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일 크기가 25MB를 초과합니다."
        )

    try:
        storage = get_storage_service()

        # Supabase Storage에 파일 업로드
        file_path, file_url = await storage.upload_file(
            file_content=content,
            file_name=file.filename,
            mime_type=file.content_type or "audio/webm"
        )

        # 데이터베이스에 레코드 생성
        recording = await storage.create_recording(
            file_url=file_url,
            file_name=file.filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type or "audio/webm",
            title=title,
            duration=duration
        )

        return recording

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 업로드 실패: {str(e)}"
        )


@router.post("/{recording_id}/transcribe", response_model=TranscriptionResponse)
async def transcribe_recording(recording_id: str):
    """
    녹음 파일을 텍스트로 변환 (Groq Whisper API)

    - 한국어 자동 인식
    - 변환 결과는 데이터베이스에 저장됨
    """
    storage = get_storage_service()
    stt = get_stt_service()

    # 녹음 조회
    recording = await storage.get_recording(recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="녹음을 찾을 수 없습니다."
        )

    # 이미 변환된 경우
    if recording.transcript:
        return TranscriptionResponse(
            recording_id=recording_id,
            transcript=recording.transcript,
            status=recording.status
        )

    try:
        # 상태 업데이트: 변환 중
        await storage.update_recording(
            recording_id,
            status=RecordingStatus.TRANSCRIBING
        )

        # 로컬 파일에서 오디오 데이터 읽기
        audio_content, file_name = await storage.get_file_content_by_recording_id(recording_id)

        # STT 실행
        transcript = await stt.transcribe(
            audio_content=audio_content,
            file_name=file_name
        )

        # 결과 저장
        updated = await storage.update_recording(
            recording_id,
            transcript=transcript,
            status=RecordingStatus.TRANSCRIBED
        )

        return TranscriptionResponse(
            recording_id=recording_id,
            transcript=transcript,
            status=RecordingStatus.TRANSCRIBED
        )

    except Exception as e:
        # 오류 발생 시 상태 업데이트
        await storage.update_recording(
            recording_id,
            status=RecordingStatus.ERROR
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"텍스트 변환 실패: {str(e)}"
        )


@router.post("/{recording_id}/summarize", response_model=SummaryResponse)
async def summarize_recording(
    recording_id: str,
    summary_type: SummaryType = SummaryType.GENERAL
):
    """
    녹음 텍스트를 요약 (Cerebras API + LangChain 템플릿)

    - 먼저 transcribe 엔드포인트로 텍스트 변환 필요
    - 요약 유형에 따라 다른 템플릿 사용
    - 유형: general(일반), meeting(회의), lecture(강의), interview(인터뷰), memo(메모)
    """
    storage = get_storage_service()
    llm = get_llm_service()

    # 녹음 조회
    recording = await storage.get_recording(recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="녹음을 찾을 수 없습니다."
        )

    # 텍스트 변환 확인
    if not recording.transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="먼저 텍스트 변환(transcribe)을 수행해주세요."
        )

    # 이미 요약된 경우 (같은 유형인 경우에만)
    if recording.summary and recording.summary_type == summary_type:
        return SummaryResponse(
            recording_id=recording_id,
            summary=recording.summary,
            summary_type=summary_type,
            key_points=recording.key_points or [],
            extra_data=recording.extra_data,
            status=recording.status
        )

    try:
        # 상태 업데이트: 요약 중
        await storage.update_recording(
            recording_id,
            status=RecordingStatus.SUMMARIZING
        )

        # LLM 요약 실행 (LangChain 템플릿 사용)
        llm_summary_type = LLMSummaryType(summary_type.value)
        summary, key_points, extra_data = await llm.summarize(
            recording.transcript,
            llm_summary_type
        )

        # 결과 저장
        await storage.update_recording(
            recording_id,
            summary=summary,
            summary_type=summary_type.value,
            key_points=key_points,
            extra_data=extra_data,
            status=RecordingStatus.COMPLETED
        )

        return SummaryResponse(
            recording_id=recording_id,
            summary=summary,
            summary_type=summary_type,
            key_points=key_points,
            extra_data=extra_data,
            status=RecordingStatus.COMPLETED
        )

    except Exception as e:
        # 오류 발생 시 상태 업데이트
        await storage.update_recording(
            recording_id,
            status=RecordingStatus.ERROR
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"요약 생성 실패: {str(e)}"
        )


@router.get("", response_model=RecordingList)
async def list_recordings(
    page: int = 1,
    page_size: int = 20
):
    """
    녹음 목록 조회

    - 최신순 정렬
    - 페이지네이션 지원
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    storage = get_storage_service()
    recordings, total = await storage.list_recordings(page, page_size)

    return RecordingList(
        items=recordings,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{recording_id}", response_model=Recording)
async def get_recording(recording_id: str):
    """
    단일 녹음 조회
    """
    storage = get_storage_service()
    recording = await storage.get_recording(recording_id)

    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="녹음을 찾을 수 없습니다."
        )

    return recording


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(recording_id: str):
    """
    녹음 삭제

    - 데이터베이스 레코드와 스토리지 파일 모두 삭제
    """
    storage = get_storage_service()

    # 존재 여부 확인
    recording = await storage.get_recording(recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="녹음을 찾을 수 없습니다."
        )

    # 삭제 실행
    success = await storage.delete_recording(recording_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="녹음 삭제 실패"
        )

    return None
