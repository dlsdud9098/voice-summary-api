"""
LLM 서비스
Cerebras API를 사용한 텍스트 요약
"""

import json
import httpx
from typing import Optional

from app.config import get_settings


class LLMService:
    """
    Cerebras API를 사용한 LLM 서비스
    텍스트 요약 및 핵심 포인트 추출
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.CEREBRAS_API_KEY
        self.api_url = settings.CEREBRAS_API_URL
        self.model = settings.CEREBRAS_MODEL

    async def summarize(self, transcript: str) -> tuple[str, list[str]]:
        """
        텍스트를 요약하고 핵심 포인트를 추출

        Args:
            transcript: 요약할 텍스트 (음성 변환 결과)

        Returns:
            (요약문, 핵심 포인트 리스트) 튜플

        Raises:
            Exception: API 호출 실패 시
        """
        # 요약을 위한 시스템 프롬프트
        system_prompt = """당신은 음성 녹음 내용을 요약하는 전문가입니다.
주어진 텍스트를 분석하여 다음 JSON 형식으로 응답해주세요:

{
    "summary": "전체 내용을 2-3문장으로 요약한 내용",
    "key_points": [
        "핵심 포인트 1",
        "핵심 포인트 2",
        "핵심 포인트 3"
    ]
}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 요약은 명확하고 간결하게 작성합니다.
3. 핵심 포인트는 3-5개로 정리합니다.
4. 중요한 정보나 행동 항목을 우선적으로 포함합니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요."""

        user_prompt = f"""다음 음성 녹음 내용을 요약해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"Cerebras API 오류 (상태 코드: {response.status_code}): {error_detail}"
                )

            result = response.json()

        # 응답에서 메시지 추출
        content = result["choices"][0]["message"]["content"]

        # JSON 파싱
        try:
            # JSON 블록이 있으면 추출
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            parsed = json.loads(json_str)
            summary = parsed.get("summary", "요약을 생성할 수 없습니다.")
            key_points = parsed.get("key_points", [])

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 전체 내용을 요약으로 사용
            summary = content
            key_points = []

        return summary, key_points


# 싱글톤 인스턴스
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """LLMService 싱글톤 인스턴스 반환"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
