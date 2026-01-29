"""
STT (Speech-to-Text) 서비스
Groq Whisper API를 사용한 음성 텍스트 변환
- 25MB 제한에 맞춰 자동 분할 처리
"""

import httpx
import io
import subprocess
import tempfile
import os
from typing import Optional
from pathlib import Path

from app.config import get_settings

# Groq Whisper 제한
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
CHUNK_DURATION_SECONDS = 600  # 10분 단위로 분할


class STTService:
    """
    Groq Whisper API를 사용한 STT 서비스
    음성 파일을 텍스트로 변환
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.GROQ_API_KEY
        self.api_url = settings.GROQ_API_URL
        self.model = settings.GROQ_MODEL

    async def transcribe(
        self,
        audio_content: bytes,
        file_name: str,
        language: str = "ko"
    ) -> str:
        """
        오디오 파일을 텍스트로 변환
        25MB 초과 시 자동 분할 처리

        Args:
            audio_content: 오디오 파일 바이너리 데이터
            file_name: 파일명 (확장자 포함)
            language: 언어 코드 (기본: 한국어)

        Returns:
            변환된 텍스트

        Raises:
            Exception: API 호출 실패 시
        """
        file_size = len(audio_content)

        # 25MB 초과 시 분할 처리
        if file_size > MAX_FILE_SIZE:
            return await self._transcribe_chunked(audio_content, file_name, language)

        return await self._transcribe_single(audio_content, file_name, language)

    async def _transcribe_single(
        self,
        audio_content: bytes,
        file_name: str,
        language: str = "ko"
    ) -> str:
        """단일 파일 STT 처리"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        # multipart/form-data로 파일 전송
        files = {
            "file": (file_name, audio_content, "audio/webm"),
        }
        data = {
            "model": self.model,
            "language": language,
            "response_format": "text"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                files=files,
                data=data
            )

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"Groq API 오류 (상태 코드: {response.status_code}): {error_detail}"
                )

            # response_format이 "text"이면 응답이 바로 텍스트
            return response.text.strip()

    async def _transcribe_chunked(
        self,
        audio_content: bytes,
        file_name: str,
        language: str = "ko"
    ) -> str:
        """
        긴 오디오 파일을 청크로 분할하여 STT 처리
        ffmpeg을 사용하여 오디오를 10분 단위로 분할
        """
        ext = file_name.split(".")[-1].lower() if "." in file_name else "wav"
        transcripts = []

        with tempfile.TemporaryDirectory() as temp_dir:
            # 원본 파일 저장
            input_path = Path(temp_dir) / f"input.{ext}"
            with open(input_path, "wb") as f:
                f.write(audio_content)

            # 오디오 길이 확인
            duration = self._get_audio_duration(str(input_path))
            if duration is None:
                # ffprobe 실패 시 단일 처리 시도
                return await self._transcribe_single(audio_content, file_name, language)

            # 청크 분할
            chunk_paths = self._split_audio(
                str(input_path),
                temp_dir,
                CHUNK_DURATION_SECONDS
            )

            # 각 청크 STT 처리
            for i, chunk_path in enumerate(chunk_paths):
                with open(chunk_path, "rb") as f:
                    chunk_content = f.read()

                chunk_name = f"chunk_{i}.{ext}"
                transcript = await self._transcribe_single(chunk_content, chunk_name, language)
                if transcript:
                    transcripts.append(transcript)

        # 결과 병합
        return " ".join(transcripts)

    def _get_audio_duration(self, file_path: str) -> Optional[float]:
        """ffprobe로 오디오 길이 확인"""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def _split_audio(
        self,
        input_path: str,
        output_dir: str,
        chunk_duration: int
    ) -> list[str]:
        """ffmpeg으로 오디오를 청크로 분할"""
        ext = input_path.split(".")[-1]
        output_pattern = str(Path(output_dir) / f"chunk_%03d.{ext}")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", input_path,
                    "-f", "segment",
                    "-segment_time", str(chunk_duration),
                    "-c", "copy",
                    "-y",
                    output_pattern
                ],
                capture_output=True,
                timeout=300
            )
        except Exception as e:
            raise Exception(f"오디오 분할 실패: {str(e)}")

        # 생성된 청크 파일 목록 반환
        chunk_files = sorted(Path(output_dir).glob(f"chunk_*.{ext}"))
        return [str(f) for f in chunk_files]

    async def transcribe_from_url(
        self,
        audio_url: str,
        file_name: str,
        language: str = "ko"
    ) -> str:
        """
        URL에서 오디오 파일을 다운로드하여 텍스트로 변환

        Args:
            audio_url: 오디오 파일 URL
            file_name: 파일명
            language: 언어 코드

        Returns:
            변환된 텍스트
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 오디오 파일 다운로드
            response = await client.get(audio_url)
            if response.status_code != 200:
                raise Exception(f"오디오 파일 다운로드 실패: {audio_url}")

            audio_content = response.content

        # 텍스트 변환
        return await self.transcribe(audio_content, file_name, language)


# 싱글톤 인스턴스
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    """STTService 싱글톤 인스턴스 반환"""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
