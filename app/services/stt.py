"""
STT (Speech-to-Text) 서비스
Groq Whisper API를 사용한 음성 텍스트 변환
"""

import httpx
from typing import Optional

from app.config import get_settings


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

        Args:
            audio_content: 오디오 파일 바이너리 데이터
            file_name: 파일명 (확장자 포함)
            language: 언어 코드 (기본: 한국어)

        Returns:
            변환된 텍스트

        Raises:
            Exception: API 호출 실패 시
        """
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
