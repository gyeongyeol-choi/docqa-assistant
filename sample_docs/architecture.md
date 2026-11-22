# DocQA 시스템 아키텍처

## 전체 파이프라인
DocQA Assistant는 개인 노트와 기술 문서를 색인화하고, 질문에 대해 근거 출처와 함께 답변을 생성하는 RAG(Retrieval-Augmented Generation) 파이프라인입니다.

1. **문서 로더(Loader)**: Markdown, Text, PDF 파일을 읽어 섹션 단위로 분리하고 오버랩(Overlap)을 두어 청킹합니다.
2. **하이브리드 검색기(Hybrid Retriever)**: BM25 어휘 검색과 선택적 밀집 벡터(Dense Embedding) 검색을 Reciprocal Rank Fusion(RRF)으로 결합합니다.
3. **LLM 생성기(Generator)**: 검색된 상위 청크만 컨텍스트로 주입해 환각(Hallucination)을 줄이고 `[1]`, `[2]` 형태의 출처를 표기합니다.

## 저장소 및 캐싱
색인된 청크와 임베딩 벡터는 로컬 JSON 파일(`.docqa_index.json`)에 저장됩니다. 문서가 변경되지 않았다면 재시작 시 디스크에서 즉시 로드합니다.
