"""
LLM 서비스
LangChain을 사용한 템플릿 기반 텍스트 요약
"""

import json
import httpx
import re
from enum import Enum
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from app.config import get_settings


# 예약어 (커스텀 필드에서 사용 불가)
RESERVED_FIELDS = {"summary", "key_points", "id", "status", "error", "transcript"}

# 검증 상수
MAX_EXTRA_FIELDS = 5
MAX_FIELD_NAME_LENGTH = 20
ALLOWED_FIELD_PATTERN = re.compile(r'^[\w\s가-힣]+$')  # 한글, 영문, 숫자, 공백, _


def validate_extra_fields(extra_fields: list[str]) -> tuple[bool, str]:
    """
    커스텀 필드 검증

    Returns:
        (성공 여부, 에러 메시지)
    """
    if len(extra_fields) > MAX_EXTRA_FIELDS:
        return False, f"추가 필드는 최대 {MAX_EXTRA_FIELDS}개까지 가능합니다."

    for field in extra_fields:
        field = field.strip()
        if not field:
            return False, "빈 필드명은 허용되지 않습니다."
        if len(field) > MAX_FIELD_NAME_LENGTH:
            return False, f"필드명은 최대 {MAX_FIELD_NAME_LENGTH}자까지 가능합니다: {field}"
        if field.lower() in RESERVED_FIELDS:
            return False, f"예약어는 필드명으로 사용할 수 없습니다: {field}"
        if not ALLOWED_FIELD_PATTERN.match(field):
            return False, f"필드명에 허용되지 않는 문자가 포함되어 있습니다: {field}"

    return True, ""


class SummaryType(str, Enum):
    """요약 유형"""
    GENERAL = "general"      # 일반
    MEETING = "meeting"      # 회의
    LECTURE = "lecture"      # 강의
    INTERVIEW = "interview"  # 인터뷰
    MEMO = "memo"            # 메모/아이디어


# 요약 유형별 시스템 프롬프트 템플릿
SUMMARY_TEMPLATES = {
    SummaryType.GENERAL: {
        "system": """당신은 음성 녹음 내용을 요약하는 전문가입니다.
주어진 텍스트를 분석하여 다음 JSON 형식으로 응답해주세요:

{{
    "summary": "전체 내용을 2-3문장으로 요약한 내용",
    "key_points": [
        "핵심 포인트 1",
        "핵심 포인트 2",
        "핵심 포인트 3"
    ]
}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 요약은 명확하고 간결하게 작성합니다.
3. 핵심 포인트는 3-5개로 정리합니다.
4. 중요한 정보나 행동 항목을 우선적으로 포함합니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요.""",
        "human": """다음 음성 녹음 내용을 요약해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""
    },

    SummaryType.MEETING: {
        "system": """당신은 비즈니스 회의록을 정리하는 전문가입니다.
회의 내용을 분석하여 다음 JSON 형식으로 응답해주세요:

{{
    "summary": "회의의 주요 목적과 결론을 2-3문장으로 요약",
    "key_points": [
        "논의된 주요 안건 또는 결정 사항"
    ],
    "action_items": [
        "담당자와 함께 후속 조치 사항"
    ],
    "decisions": [
        "회의에서 내려진 주요 결정"
    ],
    "next_steps": [
        "다음 단계 또는 후속 회의 계획"
    ]
}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 누가 무엇을 하기로 했는지 명확하게 기록합니다.
3. 결정 사항과 미결 사항을 구분합니다.
4. 마감일이 언급되었다면 포함합니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요.""",
        "human": """다음 회의 녹음 내용을 정리해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""
    },

    SummaryType.LECTURE: {
        "system": """당신은 교육 콘텐츠를 정리하는 전문가입니다.
강의/강연 내용을 분석하여 다음 JSON 형식으로 응답해주세요:

{{
    "summary": "강의의 핵심 주제와 목적을 2-3문장으로 요약",
    "key_points": [
        "강의에서 다룬 핵심 개념"
    ],
    "concepts": [
        {{
            "term": "개념/용어",
            "explanation": "간단한 설명"
        }}
    ],
    "examples": [
        "강의에서 언급된 예시나 사례"
    ],
    "study_tips": [
        "학습자를 위한 팁이나 강조 사항"
    ]
}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 전문 용어는 쉽게 풀어서 설명합니다.
3. 학습에 도움이 되도록 체계적으로 정리합니다.
4. 예시와 개념을 연결하여 이해를 돕습니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요.""",
        "human": """다음 강의 녹음 내용을 정리해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""
    },

    SummaryType.INTERVIEW: {
        "system": """당신은 인터뷰 내용을 정리하는 전문가입니다.
인터뷰 내용을 분석하여 다음 JSON 형식으로 응답해주세요:

{{
    "summary": "인터뷰의 주요 내용을 2-3문장으로 요약",
    "key_points": [
        "인터뷰에서 나온 핵심 답변이나 인사이트"
    ],
    "questions_answers": [
        {{
            "question": "질문 내용",
            "answer": "답변 요약"
        }}
    ],
    "quotes": [
        "인상적인 발언이나 인용구"
    ],
    "follow_up": [
        "추가로 확인이 필요한 사항"
    ]
}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 질문과 답변을 명확하게 구분합니다.
3. 인터뷰이의 의견이나 경험을 정확하게 전달합니다.
4. 중요한 발언은 가능한 원문 그대로 인용합니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요.""",
        "human": """다음 인터뷰 녹음 내용을 정리해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""
    },

    SummaryType.MEMO: {
        "system": """당신은 아이디어와 메모를 정리하는 전문가입니다.
음성 메모 내용을 분석하여 다음 JSON 형식으로 응답해주세요:

{{
    "summary": "메모의 핵심 내용을 1-2문장으로 요약",
    "key_points": [
        "주요 아이디어나 생각"
    ],
    "ideas": [
        "기록된 아이디어 목록"
    ],
    "todos": [
        "해야 할 일이나 확인 사항"
    ],
    "reminders": [
        "기억해야 할 사항"
    ]
}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 아이디어는 명확하고 실행 가능하게 정리합니다.
3. 할 일과 리마인더를 구분합니다.
4. 간결하지만 핵심을 놓치지 않습니다.
5. 반드시 유효한 JSON 형식으로만 응답해주세요.""",
        "human": """다음 음성 메모 내용을 정리해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""
    },
}


class LLMService:
    """
    LangChain을 활용한 LLM 서비스
    템플릿 기반 텍스트 요약 및 핵심 포인트 추출
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.CEREBRAS_API_KEY
        self.api_url = settings.CEREBRAS_API_URL
        self.model = settings.CEREBRAS_MODEL

    def _create_prompt(self, summary_type: SummaryType) -> ChatPromptTemplate:
        """
        요약 유형에 따른 프롬프트 템플릿 생성
        """
        template = SUMMARY_TEMPLATES.get(summary_type, SUMMARY_TEMPLATES[SummaryType.GENERAL])

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template["system"]),
            HumanMessagePromptTemplate.from_template(template["human"])
        ])

    def _create_custom_prompt(
        self,
        summary_type: SummaryType,
        extra_fields: list[str]
    ) -> ChatPromptTemplate:
        """
        커스텀 필드가 포함된 프롬프트 템플릿 생성
        """
        # 기본 템플릿 가져오기
        base_template = SUMMARY_TEMPLATES.get(summary_type, SUMMARY_TEMPLATES[SummaryType.GENERAL])

        # 커스텀 필드를 JSON 스키마에 추가
        extra_fields_json = ",\n    ".join([
            f'{json.dumps(field, ensure_ascii=False)}: ["관련 내용을 추출하여 리스트로 작성"]'
            for field in extra_fields
        ])

        # 시스템 프롬프트 수정 (기존 JSON 스키마에 커스텀 필드 추가)
        system_prompt = f"""당신은 음성 녹음 내용을 분석하는 전문가입니다.
주어진 텍스트를 분석하여 다음 JSON 형식으로 응답해주세요:

{{{{
    "summary": "전체 내용을 2-3문장으로 요약한 내용",
    "key_points": [
        "핵심 포인트 1",
        "핵심 포인트 2",
        "핵심 포인트 3"
    ],
    {extra_fields_json}
}}}}

규칙:
1. 반드시 한국어로 작성해주세요.
2. 요약은 명확하고 간결하게 작성합니다.
3. 핵심 포인트는 3-5개로 정리합니다.
4. 추가 필드({', '.join(extra_fields)})도 내용에서 관련 정보를 추출하여 작성합니다.
5. 추가 필드에 해당하는 내용이 없으면 빈 배열 []로 응답합니다.
6. 반드시 유효한 JSON 형식으로만 응답해주세요."""

        human_prompt = """다음 음성 녹음 내용을 분석해주세요:

---
{transcript}
---

JSON 형식으로 응답해주세요."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])

    async def summarize(
        self,
        transcript: str,
        summary_type: SummaryType = SummaryType.GENERAL,
        extra_fields: Optional[list[str]] = None
    ) -> tuple[str, list[str], dict]:
        """
        텍스트를 요약하고 핵심 포인트를 추출

        Args:
            transcript: 요약할 텍스트 (음성 변환 결과)
            summary_type: 요약 유형 (general, meeting, lecture, interview, memo)
            extra_fields: 커스텀 추가 필드 목록 (선택)

        Returns:
            (요약문, 핵심 포인트 리스트, 추가 데이터 딕셔너리) 튜플

        Raises:
            Exception: API 호출 실패 시
            ValueError: extra_fields 검증 실패 시
        """
        # 커스텀 필드 검증
        if extra_fields:
            is_valid, error_msg = validate_extra_fields(extra_fields)
            if not is_valid:
                raise ValueError(error_msg)

        # LangChain 프롬프트 생성
        if extra_fields:
            prompt = self._create_custom_prompt(summary_type, extra_fields)
        else:
            prompt = self._create_prompt(summary_type)
        messages = prompt.format_messages(transcript=transcript)

        # 메시지를 API 형식으로 변환
        api_messages = []
        for msg in messages:
            role = "system" if msg.type == "system" else "user"
            api_messages.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": api_messages,
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

            # 추가 데이터 (요약 유형별 특수 필드)
            extra_data = {}
            for key in parsed:
                if key not in ["summary", "key_points"]:
                    extra_data[key] = parsed[key]

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 전체 내용을 요약으로 사용
            summary = content
            key_points = []
            extra_data = {}

        return summary, key_points, extra_data


# 싱글톤 인스턴스
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """LLMService 싱글톤 인스턴스 반환"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
