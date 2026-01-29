"""
애플리케이션 설정 모듈
환경 변수를 통한 설정 관리
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    환경 변수 또는 .env 파일에서 설정을 로드
    """

    # Groq API 설정 (Whisper STT)
    GROQ_API_KEY: str
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/audio/transcriptions"
    GROQ_MODEL: str = "whisper-large-v3"

    # Cerebras API 설정 (LLM 요약)
    CEREBRAS_API_KEY: str
    CEREBRAS_API_URL: str = "https://api.cerebras.ai/v1/chat/completions"
    CEREBRAS_MODEL: str = "llama-3.3-70b"

    # 로컬 스토리지 설정
    STORAGE_PATH: str = "./data"

    # 앱 설정
    APP_NAME: str = "Voice Recording Summarization API"
    DEBUG: bool = False

    # CORS 설정
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """
    설정 인스턴스를 캐싱하여 반환
    매번 새로 로드하지 않고 캐시된 인스턴스 사용
    """
    return Settings()
