"""
로컬 파일 시스템 스토리지 서비스
파일 저장 및 녹음 메타데이터 관리 (JSON 파일 기반)
"""

import json
import uuid
import aiofiles
import aiofiles.os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.recording import Recording, RecordingStatus


class StorageService:
    """
    로컬 파일 시스템 스토리지 서비스 클래스
    - 파일 업로드/다운로드/삭제
    - 녹음 메타데이터 CRUD (JSON 파일)
    """

    def __init__(self):
        settings = get_settings()
        self.base_path = Path(settings.STORAGE_PATH)
        self.files_path = self.base_path / "files"
        self.db_path = self.base_path / "db.json"

        # 디렉토리 생성
        self.files_path.mkdir(parents=True, exist_ok=True)

        # DB 파일 초기화
        if not self.db_path.exists():
            self._save_db_sync({"recordings": {}})

    def _save_db_sync(self, data: dict):
        """동기적으로 DB 저장 (초기화용)"""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    async def _load_db(self) -> dict:
        """DB 파일 로드"""
        async with aiofiles.open(self.db_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def _save_db(self, data: dict):
        """DB 파일 저장"""
        async with aiofiles.open(self.db_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    async def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        mime_type: str
    ) -> tuple[str, str]:
        """
        파일을 로컬 스토리지에 업로드

        Args:
            file_content: 파일 바이너리 데이터
            file_name: 원본 파일명
            mime_type: MIME 타입

        Returns:
            (file_path, file_url) 튜플
        """
        # 고유한 파일 경로 생성
        file_id = str(uuid.uuid4())
        ext = file_name.split(".")[-1] if "." in file_name else "webm"
        file_path = f"{file_id}.{ext}"
        full_path = self.files_path / file_path

        # 파일 저장
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(file_content)

        # 로컬 URL (API에서 서빙할 경로)
        file_url = f"/files/{file_path}"

        return file_path, file_url

    async def get_file_path(self, file_path: str) -> Path:
        """파일의 실제 경로 반환"""
        return self.files_path / file_path

    async def get_file_content(self, file_path: str) -> bytes:
        """파일 내용 읽기"""
        full_path = self.files_path / file_path
        async with aiofiles.open(full_path, "rb") as f:
            return await f.read()

    async def get_file_content_by_recording_id(self, recording_id: str) -> tuple[bytes, str]:
        """
        녹음 ID로 파일 내용 읽기
        Returns: (file_content, file_name)
        """
        db = await self._load_db()
        data = db["recordings"].get(recording_id)
        if not data:
            raise FileNotFoundError(f"Recording not found: {recording_id}")

        file_path = data.get("file_path")
        if not file_path:
            raise FileNotFoundError(f"File path not found for recording: {recording_id}")

        content = await self.get_file_content(file_path)
        return content, data.get("file_name", "audio.wav")

    async def delete_file(self, file_path: str) -> bool:
        """
        로컬 스토리지에서 파일 삭제

        Args:
            file_path: 삭제할 파일 경로

        Returns:
            삭제 성공 여부
        """
        try:
            full_path = self.files_path / file_path
            if full_path.exists():
                await aiofiles.os.remove(full_path)
            return True
        except Exception:
            return False

    async def create_recording(
        self,
        file_url: str,
        file_name: str,
        file_path: str,
        file_size: int,
        mime_type: str,
        title: Optional[str] = None,
        duration: Optional[float] = None
    ) -> Recording:
        """
        녹음 레코드 생성

        Args:
            file_url: 파일 URL
            file_name: 원본 파일명
            file_path: 스토리지 파일 경로
            file_size: 파일 크기
            mime_type: MIME 타입
            title: 녹음 제목 (선택)
            duration: 녹음 길이 (선택)

        Returns:
            생성된 Recording 객체
        """
        now = datetime.utcnow().isoformat()
        recording_id = str(uuid.uuid4())

        data = {
            "id": recording_id,
            "file_url": file_url,
            "file_name": file_name,
            "file_path": file_path,
            "file_size": file_size,
            "mime_type": mime_type,
            "title": title or file_name,
            "duration": duration,
            "status": RecordingStatus.UPLOADED.value,
            "transcript": None,
            "summary": None,
            "summary_type": None,
            "key_points": None,
            "extra_data": None,
            "created_at": now,
            "updated_at": now
        }

        # DB에 저장
        db = await self._load_db()
        db["recordings"][recording_id] = data
        await self._save_db(db)

        return self._to_recording(data)

    async def get_recording(self, recording_id: str) -> Optional[Recording]:
        """
        녹음 레코드 조회

        Args:
            recording_id: 녹음 ID

        Returns:
            Recording 객체 또는 None
        """
        db = await self._load_db()
        data = db["recordings"].get(recording_id)

        if data:
            return self._to_recording(data)
        return None

    async def list_recordings(
        self,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[list[Recording], int]:
        """
        녹음 목록 조회

        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지 크기

        Returns:
            (녹음 목록, 총 개수) 튜플
        """
        db = await self._load_db()
        all_recordings = list(db["recordings"].values())

        # 최신순 정렬
        all_recordings.sort(key=lambda x: x["created_at"], reverse=True)

        total = len(all_recordings)

        # 페이지네이션
        offset = (page - 1) * page_size
        paginated = all_recordings[offset:offset + page_size]

        recordings = [self._to_recording(item) for item in paginated]
        return recordings, total

    async def update_recording(
        self,
        recording_id: str,
        **kwargs
    ) -> Optional[Recording]:
        """
        녹음 레코드 업데이트

        Args:
            recording_id: 녹음 ID
            **kwargs: 업데이트할 필드들

        Returns:
            업데이트된 Recording 객체 또는 None
        """
        db = await self._load_db()

        if recording_id not in db["recordings"]:
            return None

        # status가 Enum인 경우 문자열로 변환
        if "status" in kwargs and hasattr(kwargs["status"], "value"):
            kwargs["status"] = kwargs["status"].value

        kwargs["updated_at"] = datetime.utcnow().isoformat()

        # 업데이트
        db["recordings"][recording_id].update(kwargs)
        await self._save_db(db)

        return self._to_recording(db["recordings"][recording_id])

    async def delete_recording(self, recording_id: str) -> bool:
        """
        녹음 레코드 및 파일 삭제

        Args:
            recording_id: 녹음 ID

        Returns:
            삭제 성공 여부
        """
        db = await self._load_db()

        if recording_id not in db["recordings"]:
            return False

        recording_data = db["recordings"][recording_id]

        # 파일 삭제
        file_path = recording_data.get("file_path")
        if file_path:
            await self.delete_file(file_path)

        # 레코드 삭제
        del db["recordings"][recording_id]
        await self._save_db(db)

        return True

    def _to_recording(self, data: dict) -> Recording:
        """
        딕셔너리를 Recording 객체로 변환

        Args:
            data: 데이터베이스 레코드

        Returns:
            Recording 객체
        """
        created_at = data["created_at"]
        updated_at = data["updated_at"]

        # 문자열인 경우 datetime으로 변환
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

        # summary_type 처리
        from app.models.recording import SummaryType
        summary_type = data.get("summary_type")
        if summary_type and isinstance(summary_type, str):
            summary_type = SummaryType(summary_type)

        return Recording(
            id=data["id"],
            file_url=data["file_url"],
            file_name=data["file_name"],
            file_size=data["file_size"],
            mime_type=data["mime_type"],
            title=data.get("title"),
            duration=data.get("duration"),
            status=RecordingStatus(data["status"]),
            transcript=data.get("transcript"),
            summary=data.get("summary"),
            summary_type=summary_type,
            key_points=data.get("key_points"),
            extra_data=data.get("extra_data"),
            created_at=created_at,
            updated_at=updated_at
        )


# 싱글톤 인스턴스
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """StorageService 싱글톤 인스턴스 반환"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
