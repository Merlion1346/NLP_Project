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
llama-server  (:30004)   ← Qwen3.5-0.8B   (채팅/생성)
llama-embedding (:30005) ← BGE-M3          (임베딩)
```

---

## 프로젝트 구조

```
nlp_project/
├── main.py                  # FastAPI 서버 (RAG + 웹 UI + OpenAI 프록시)
├── setup_models.sh          # llama.cpp 빌드·모델 다운로드·systemd 서비스 설치
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

## 모델 서버 설정 (`setup_models.sh`)

`setup_models.sh`를 실행하면 아래 작업이 자동으로 수행됩니다.

1. llama.cpp 클론 및 빌드 (`llama-server` 바이너리 생성)
2. Hugging Face에서 모델 다운로드
   - `unsloth/Qwen3.5-0.8B-GGUF` — 채팅/생성용
   - `ggml-org/bge-m3-Q8_0-GGUF` — 임베딩용
3. systemd 서비스 파일 생성 및 활성화

```bash
sudo bash setup_models.sh
```

> `sudo`가 필요한 이유: `/etc/systemd/system/`에 서비스 파일을 작성하기 때문입니다.

### 설치되는 systemd 서비스

| 서비스 | 포트 | 모델 | 설명 |
|---|---|---|---|
| `llama-server` | 30004 | Qwen3.5-0.8B | 채팅 완성 (chat completions) |
| `llama-embedding` | 30005 | BGE-M3 | 텍스트 임베딩 |

### 서비스 관리 명령어

```bash
# 상태 확인
sudo systemctl status llama-server
sudo systemctl status llama-embedding

# 재시작
sudo systemctl restart llama-server
sudo systemctl restart llama-embedding

# 실시간 로그
sudo journalctl -u llama-server      -f
sudo journalctl -u llama-embedding   -f
```

---

## 빠른 시작

### 1. 모델 서버 설치 및 실행

```bash
sudo bash setup_models.sh
```

### 2. 벡터스토어 생성

`txt/` 폴더에 `.txt` 또는 `.md` 파일을 넣고 실행합니다.

```bash
python tools_vectordb/create_faissDB.py
```

`vectorstore/index.faiss`, `vectorstore/index.pkl`이 생성됩니다.

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

- **임베딩 모델**: `BAAI/bge-m3` (한국어 포함 다국어 지원)
- **청크 설정**: 기본 1,000자 / overlap 200자

---

## 설정 값

`main.py` 상단에서 변경할 수 있습니다.

```python
LLAMA_CPP_BASE_URL = "http://localhost:30004"   # llama-server 주소 (채팅)
EMBEDDING_BASE_URL = "http://localhost:30005"   # llama-embedding 주소 (임베딩)
MODEL_NAME         = "unsloth/Qwen3.5-0.8B"     # 생성 모델 이름
EMBEDDING_MODEL    = "BAAI/bge-m3"              # 임베딩 모델 이름
VECTOR_STORE_PATH  = "vectorstore"
TOP_K              = 3                           # 검색할 문서 수
```
