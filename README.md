# Medical QA Chatbot

의료 지식 기반의 질의응답 챗봇 서비스입니다. RAG(Retrieval-Augmented Generation) 파이프라인과 FastAPI 백엔드로 구성되어 있으며, 멀티 쿼리 생성과 ReRank를 통해 높은 품질의 답변을 제공합니다.

---

## 아키텍처

```
사용자 질문
    │
    ▼
FastAPI 서버 (REST / WebSocket)
    │
    ▼
InferenceService → Predictor
    │
    ▼
RAGModel
 ├─ Multi-Query 생성 (OpenAI GPT)     ← 질문을 3개의 다양한 쿼리로 확장
 ├─ ChromaDB 벡터 검색                ← HuggingFace에서 DB 다운로드 (BAAI/bge-m3)
 ├─ ReRank (zeroentropy/zerank-2)     ← CrossEncoder로 상위 문서 선별
 └─ 답변 생성 (OpenAI GPT)
```

---

## 주요 기능

- **Multi-Query RAG**: 사용자 질문을 3개의 다양한 관점의 쿼리로 변환하여 검색 품질 향상
- **ReRank**: CrossEncoder 모델(`zeroentropy/zerank-2`)로 검색된 문서를 재순위화하여 가장 관련성 높은 문서 선별
- **HuggingFace 연동**: ChromaDB 벡터 데이터베이스를 HuggingFace Hub에서 자동 다운로드
- **REST API**: HTTP POST 엔드포인트로 질의응답 제공
- **WebSocket API**: 실시간 스트리밍 방식의 질의응답 지원

---

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 백엔드 프레임워크 | FastAPI, Uvicorn |
| LLM | OpenAI GPT (gpt-4o-mini 기본) |
| 임베딩 모델 | BAAI/bge-m3 (HuggingFace) |
| 벡터 데이터베이스 | ChromaDB |
| ReRank 모델 | zeroentropy/zerank-2 (CrossEncoder) |
| RAG 프레임워크 | LangChain |
| 캐싱 | Redis |

---

## 프로젝트 구조

```
medical-qa-chatbot/
├── model/
│   └── rag_model.py          # RAG 모델 코어 (Multi-Query + ReRank)
├── fastapi/
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI 앱 진입점
│       ├── api/
│       │   ├── inference.py  # POST /inference 엔드포인트
│       │   ├── ws.py         # WebSocket /ws/inference 엔드포인트
│       │   └── health.py     # 헬스 체크
│       ├── models/
│       │   ├── loader.py     # RAG 모델 싱글턴 로더
│       │   └── predictor.py  # 예측 인터페이스
│       ├── services/
│       │   ├── inference_service.py  # 추론 서비스
│       │   └── cache_service.py      # Redis 캐시 서비스
│       ├── schemas/
│       │   ├── request.py    # 요청 스키마
│       │   └── response.py   # 응답 스키마
│       └── core/
│           ├── config.py     # 앱 설정
│           └── redis.py      # Redis 연결
└── requirements.txt          # RAG 모델 의존성
```

---

## 시작하기

### 사전 요구사항

- Python 3.10+
- OpenAI API 키
- HuggingFace 토큰 (private 리포지토리 접근 시)

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...        # HuggingFace private 리포지토리 접근 시 필요
```

### 의존성 설치

```bash
# RAG 모델 의존성
pip install -r requirements.txt

# FastAPI 서버 의존성
pip install -r fastapi/requirements.txt
```

CUDA 환경에서 실행하려면 `requirements.txt`에서 PyTorch 인덱스 URL을 수정하세요.

```
# CUDA 12.1 예시
--extra-index-url https://download.pytorch.org/whl/cu121
```

### 서버 실행

```bash
cd fastapi
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 시작 시 HuggingFace Hub에서 ChromaDB 파일이 자동으로 다운로드됩니다.

---

## API 명세

### REST API

#### 질의응답

```
POST /inference
```

**요청**

```json
{
  "question": "당뇨병의 초기 증상은 무엇인가요?"
}
```

**응답**

```json
{
  "answer": "당뇨병의 초기 증상으로는 ..."
}
```

#### 헬스 체크

```
GET /health
```

### WebSocket API

```
WS /ws/inference
```

**클라이언트 → 서버**

```json
{
  "content": "고혈압에 좋은 음식은 무엇인가요?"
}
```

**서버 → 클라이언트**

```json
{
  "type": "answer",
  "content": "고혈압 환자에게 권장되는 음식으로는 ..."
}
```

---

## RAG 파이프라인 상세

1. **Multi-Query 생성**: 사용자 질문을 OpenAI GPT로 3개의 서로 다른 검색 쿼리로 확장하여 벡터 검색의 한계를 보완합니다.
2. **벡터 검색**: 각 쿼리로 ChromaDB에서 유사 문서를 검색 (기본 k=10)하고 중복을 제거합니다.
3. **ReRank**: CrossEncoder 모델이 질문과 각 문서의 관련성 점수를 계산하여 상위 3개 문서를 선별합니다.
4. **답변 생성**: 선별된 문서를 컨텍스트로 OpenAI GPT가 최종 답변을 생성합니다.
