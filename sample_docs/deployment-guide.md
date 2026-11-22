# 배포 및 실행 가이드

## 로컬 실행 방법
Python 3.11 이상 환경에서 가상환경을 생성한 뒤 의존성을 설치합니다.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 환경 변수 설정
`.env.example`을 `.env`로 복사한 뒤 사용할 LLM 제공자를 선택합니다.
- `DOCQA_LLM_PROVIDER=offline`: API 키 없이 검색 근거 요약 모드로 실행
- `DOCQA_LLM_PROVIDER=gemini`: `GEMINI_API_KEY` 필요
- `DOCQA_LLM_PROVIDER=openai`: `OPENAI_API_KEY` 필요

## Docker 배포
컨테이너로 실행하려면 포트 `8000`을 열어 `uvicorn docqa.api:app`을 구동합니다. 헬스 체크 엔드포인트는 `/api/health`입니다.
