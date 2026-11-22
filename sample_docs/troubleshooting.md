# 트러블슈팅 가이드

## 검색 결과가 나오지 않을 때
인덱싱된 문서가 없거나 질문 키워드가 문서에 전혀 포함되지 않은 경우 빈 결과가 반환됩니다. `docqa index ./sample_docs` 명령으로 문서를 다시 색인화하세요.

## PDF 파일이 읽히지 않을 때
PDF 텍스트 추출에는 `pypdf` 패키지가 필요합니다. 이미지로만 스캔된 PDF는 OCR 전처리 후 텍스트나 Markdown으로 변환해야 검색할 수 있습니다.

## API 키 오류 (401 Unauthorized)
`GEMINI_API_KEY` 또는 `OPENAI_API_KEY`가 만료되었거나 공백이 포함되었는지 `.env` 파일을 확인하세요. 키 없이 테스트하려면 `DOCQA_LLM_PROVIDER=offline`으로 전환하면 됩니다.
