"""
FastAPI 메인 애플리케이션
음성 녹음 요약 API 서버
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.recordings import router as recordings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 수명 주기 관리
    시작 시 초기화, 종료 시 정리 작업 수행
    """
    # 시작 시 실행
    settings = get_settings()
    print(f"🚀 {settings.APP_NAME} 시작")
    print(f"📡 Groq Whisper 모델: {settings.GROQ_MODEL}")
    print(f"🧠 Cerebras 모델: {settings.CEREBRAS_MODEL}")
    print(f"💾 로컬 저장소: {settings.STORAGE_PATH}")

    yield

    # 종료 시 실행
    print("👋 서버 종료")


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 팩토리
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="""
## 음성 녹음 요약 API

음성 파일을 업로드하고, 텍스트로 변환한 후, AI로 요약합니다.

### 기능
- **업로드**: 음성 파일 업로드 (webm, mp3, wav, m4a, ogg)
- **텍스트 변환**: Groq Whisper API를 사용한 음성-텍스트 변환
- **요약**: Cerebras LLM을 사용한 텍스트 요약 및 핵심 포인트 추출

### 워크플로우
1. `/api/recordings/upload` - 파일 업로드
2. `/api/recordings/{id}/transcribe` - 텍스트 변환
3. `/api/recordings/{id}/summarize` - 요약 생성
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(recordings_router, prefix="/api")

    # 정적 파일 서빙 (업로드된 오디오 파일)
    files_path = Path(settings.STORAGE_PATH) / "files"
    files_path.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(files_path)), name="files")

    # 헬스 체크 엔드포인트
    @app.get("/health", tags=["health"])
    async def health_check():
        """서버 상태 확인"""
        return {"status": "healthy", "service": settings.APP_NAME}

    return app


# 애플리케이션 인스턴스 생성
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
