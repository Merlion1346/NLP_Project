# NLP_Project — RAG Chatbot

FastAPI + FAISS + llama.cpp 기반의 **RAG(Retrieval-Augmented Generation) 챗봇**입니다.  
로컬 LLM에 문서 검색 컨텍스트를 주입하여 답변하며, 실험 조건(도메인·난이도·답변 형식)을 UI에서 직접 제어할 수 있습니다.

---

## 아키텍처

```
Browser
  │
  ▼
FastAPI (main.py)  ──[FAISS 검색]──▶  vectorstore/
  │                                    (index.faiss / index.pkl)
  │  [컨텍스트 주입]
  ▼
llama.cpp 서버 (외부, :30004)
```

---

## 프로젝트 구조

```
nlp_project/
├── main.py                  # FastAPI 서버 (RAG + 웹 UI + OpenAI 프록시)
├── tools_vectordb/
│   └── create_faissDB.py    # FAISS 벡터스토어 생성·업데이트 도구
├── txt/                     # 벡터스토어로 변환할 원본 문서 (.txt / .md)
├── vectorstore/             # 생성된 FAISS 인덱스 (index.faiss, index.pkl)
└── requirements.txt
```

---

## 설치

```bash
# 가상환경 생성 (선택)
python -m venv .venv && source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

> **GPU 환경**: `requirements.txt`에 `faiss-gpu-cu12`가 기본값으로 설정되어 있습니다.  
> **CPU 환경**: `faiss-gpu-cu12` 줄을 주석 처리하고 `faiss-cpu` 줄의 주석을 해제하세요.

---

## 빠른 시작

### 1. 벡터스토어 생성

`txt/` 폴더에 `.txt` 또는 `.md` 파일을 넣고 실행합니다.

```bash
python tools_vectordb/create_faissDB.py
```

`vectorstore/index.faiss`, `vectorstore/index.pkl`이 생성됩니다.

### 2. llama.cpp 서버 실행 (별도 터미널)

```bash
llama-server --model <모델 경로> --port 30004
```

### 3. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속 시 채팅 UI가 열립니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **RAG** | 사용자 질문으로 FAISS 벡터 DB를 검색, 관련 문서를 LLM 컨텍스트에 주입 |
| **스트리밍** | SSE(Server-Sent Events)로 실시간 토큰 스트리밍 |
| **웹 UI** | `/` 경로에서 채팅 인터페이스 제공 (RAG 토글·파라미터 조절 포함) |
| **실험 조건** | 도메인(Known/Unknown) · 난이도(Low/Medium/High) · 답변 형식(연역/귀납/자유) 제어 |
| **OpenAI 호환 프록시** | `/v1/*` 경로를 llama.cpp 서버로 투명하게 프록시 |

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 채팅 UI |
| GET | `/health` | llama.cpp 연결 상태 확인 |
| POST | `/chat` | 채팅 요청 (RAG + 스트리밍 지원) |
| GET | `/models` | 사용 모델 목록 |
| GET/POST | `/v1/*` | llama.cpp OpenAI 호환 프록시 |

### `/chat` 요청 예시

```json
{
  "messages": [{"role": "user", "content": "질문 내용"}],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true,
  "use_rag": true,
  "domain": "unknown",
  "difficulty": "medium",
  "layout": "free"
}
```

---

## 벡터스토어 관리 (`create_faissDB.py`)

| 함수 | 설명 |
|---|---|
| `create_vectorstore_from_files()` | 폴더 내 파일로 벡터스토어 새로 생성 (기존 덮어쓰기) |
| `update_vectorstore_from_files()` | 기존 벡터스토어에 새 파일 추가, 없으면 새로 생성 |
| `load_vectorstore()` | 저장된 벡터스토어 로드 |
| `file_to_documents()` | 단일 파일을 LangChain Document로 변환 |

- **임베딩 모델**: `intfloat/multilingual-e5-small` (한국어 포함 다국어 지원)
- **청크 설정**: 기본 1,000자 / overlap 200자

---

## 설정 값

`main.py` 상단에서 변경할 수 있습니다.

```python
LLAMA_CPP_BASE_URL = "http://localhost:30004"   # llama.cpp 서버 주소
MODEL_NAME         = "unsloth/Qwen3.5-0.8B"     # 모델 이름
EMBEDDING_MODEL    = "intfloat/multilingual-e5-small"
VECTOR_STORE_PATH  = "vectorstore"
TOP_K              = 3                           # 검색할 문서 수
```
