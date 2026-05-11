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
원격 llama-server  (:30004)   ← Qwen3.6-35B-A3B   (채팅/생성)
원격 llama-embedding (:30005) ← BGE-M3             (임베딩)
```

---

## 프로젝트 구조

```
nlp_project/
├── main.py                  # FastAPI 서버 (RAG + 웹 UI + OpenAI 프록시)
├── setup_models.sh          # llama.cpp 빌드·모델 다운로드·systemd 서비스 설치
├── test_chatbot.py          # 챗봇 평가 스크립트 (객관식 문항 자동 응답 수집)
├── grade_llm.py             # Gemini LLM 기반 채점 스크립트 (Accuracy/Precision/Recall/F1)
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

## 답변 형식 프롬프트 (`layout`)

`/chat` 요청의 `layout` 파라미터로 LLM의 답변 구조를 제어합니다.

### 두괄식 (`deductive`)

결론 → 근거 → 요약 순서로 답변합니다.

```
1. 첫 문장에서 질문에 대한 답변을 명확하게 제시
2. 이후 문장에서 결론의 근거와 이유를 설명
3. 마지막 문장에서 결론을 한 번 더 요약
```

> 첫 문장을 근거 설명(`~때문에`, `~에 따르면`)으로 시작하거나  
> `근거:` · `이유:` 같은 레이블을 앞에 붙이는 것을 금지합니다.

### 미괄식 (`inductive`)

근거 → 추론 → 결론 순서로 답변합니다.

```
1. 참고 문서의 관련 근거를 먼저 나열
2. 근거를 바탕으로 추론 과정 설명
3. 마지막 문장에서만 질문에 대한 답변을 명확하게 제시
```

> 첫 문장에 결론을 쓰거나  
> `결론:` · `답:` · `정답:` 같은 레이블을 앞에 붙이는 것을 금지합니다.

### 자유형식 (`free`)

형식 제약 없이 모델이 자유롭게 답변합니다.

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

서버 주소는 `api_keys.env`에서 로드됩니다. 아래 항목을 설정하세요.

```env
LLAMA_CPP_BASE_URL="http://<서버주소>:30004"
EMBEDDING_BASE_URL="http://<서버주소>:30005"
```

나머지 값은 `main.py` 상단에서 변경할 수 있습니다.

```python
MODEL_NAME         = "unsloth/Qwen3.6.-35B-A3B"   # 생성 모델 이름
EMBEDDING_MODEL    = "BAAI/bge-m3"                 # 임베딩 모델 이름
VECTOR_STORE_PATH  = "vectorstore"
TOP_K              = 3                              # 검색할 문서 수
```

> **주의**: `api_keys.env`는 `.gitignore`에 등록되어 있어 Git에 업로드되지 않습니다.

---

## 챗봇 평가 (`test_chatbot.py`)

객관식 문항을 챗봇에 자동으로 질의하고 응답을 Excel로 저장합니다.  
답변 형식(두괄식·미괄식·자유형식) × 난이도(Low·Medium·High) 조합으로 실험합니다.

```bash
# 기본 실행 (RAG ON)
python test_chatbot.py

# RAG 없이
python test_chatbot.py --no-rag

# Known Parametric 모드
python test_chatbot.py --domain known

# 중단된 실행 이어하기
python test_chatbot.py --resume results_rag_*.xlsx

# 특정 문항만 실행
python test_chatbot.py --question-ids L-01 L-02
```

출력 파일: `results_rag_<타임스탬프>.xlsx` (`결과` + `요약` 시트)

---

## LLM 채점 (`grade_llm.py`)

Gemini API를 사용해 챗봇 응답을 채점하고 분류 평가 지표를 계산합니다.  
단순 선택지 매칭이 아니라 응답 전문을 LLM이 읽고 정오 판단 + 근거 품질을 평가합니다.

### 사전 조건

`api_keys.env`에 Google API 키를 설정합니다.

```env
GOOGLE_API_KEY="your-api-key"
```

### 실행

```bash
# chatgpt 결과 채점
python grade_llm.py --results results_rag_chat_gpt_*.xlsx --answers ANSWER_only_chatgpt.xlsx

# claude 결과 채점
python grade_llm.py --results results_rag_*.xlsx --answers ANSWER_only_claude.xlsx

# 옵션 지정
python grade_llm.py --results results_*.xlsx --answers ANSWER_only_*.xlsx \
  --model gemini-2.5-flash \
  --concurrency 8 \
  --output my_score.xlsx
```

### 출력 파일 구조 (`llm_scored_*.xlsx`)

| 시트 | 내용 |
|---|---|
| `채점결과` | 원본 데이터 + `LLM선택` · `LLM정오(O/X)` · `LLM근거점수(1~5)` · `LLM평가` |
| `요약` | 형식 × 난이도별 Accuracy / Precision / Recall / F1 |
| `클래스별` | 정답 선택지(A/B/C/D)별 Precision / Recall / F1 + 지지수 |
| `혼동행렬_두괄식` 등 | 형식별 4×4 혼동행렬 |

### LLM 채점 방식

채점은 **3단계**로 이루어집니다.

#### 1단계 — 최종 선택지 추출

Gemini는 챗봇의 응답 전문을 읽고 **최종적으로 선택한 답안(A/B/C/D)** 을 추출합니다.

- 응답 중간에 여러 선택지가 언급되더라도, 결론부에서 최종 확정한 선택지를 기준으로 합니다.
- 응답에 명시적 선택지가 없거나 판단 불가능한 경우 `?`로 표기합니다.
- `?`로 표기된 항목은 평가 지표(Accuracy/Recall 등) 계산 시 분모에서 제외됩니다.

#### 2단계 — 정오(O/X) 판단

추출한 선택지를 정답지의 정답과 직접 비교하여 일치하면 `O`, 불일치하면 `X`로 기록합니다.

#### 3단계 — 근거 품질 점수 (1~5점)

답안의 정오와 무관하게, 응답에 포함된 **추론·근거의 타당성**을 독립적으로 평가합니다.

| 점수 | 정오 | 근거 상태 | 설명 |
|:---:|:---:|---|---|
| **5** | O | 완전히 정확 | 정답이고, 근거·추론 모두 사실에 부합하며 논리적으로 완결됨 |
| **4** | O | 대체로 적절 | 정답이고, 근거가 대부분 적절하나 일부 부정확하거나 누락이 있음 |
| **3** | O | 불충분 / 부분 오류 | 답은 맞지만 근거가 빈약하거나, 핵심 근거 없이 결론만 제시함 |
| **2** | X | 일부 타당 | 오답이지만 추론 과정에 부분적으로 타당한 논리가 포함되어 있음 |
| **1** | X | 완전 부적절 / 환각 | 오답이고 근거도 사실과 전혀 다르거나 환각(hallucination)에 해당함 |

> **참고**: 점수 3은 "운 좋게 맞힌" 경우를 포착하기 위한 구간입니다.  
> 정답을 맞혔더라도 근거가 없거나 잘못되었다면 높은 점수를 부여하지 않습니다.

---

### 평가 지표 정의

A/B/C/D 네 선택지를 **4-class 분류 문제**로 보고 아래 지표를 계산합니다.  
`?`(미추출) 항목은 전체 계산에서 제외됩니다.

| 지표 | 계산식 | 설명 |
|---|---|---|
| **Accuracy** | 정답 수 / 전체 수 | 전체 정답률 |
| **Precision** (클래스별) | TP / (TP + FP) | 모델이 X라고 예측한 것 중 실제로 X인 비율 |
| **Recall** (클래스별) | TP / (TP + FN) | 실제 정답이 X인 것 중 모델이 X라고 맞힌 비율 |
| **F1** (클래스별) | 2 × P × R / (P + R) | Precision과 Recall의 조화평균 |
| **Macro Precision/Recall/F1** | 클래스별 단순 평균 | 클래스 불균형을 보정하지 않은 평균 |

혼동행렬(Confusion Matrix)은 형식별(두괄식·미괄식·자유형식)로 각각 생성되어  
어떤 선택지를 어떤 선택지로 혼동하는지 확인할 수 있습니다.
