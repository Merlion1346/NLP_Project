# NLP_Project

## tools_vectordb/

텍스트 파일을 FAISS 벡터스토어로 변환하고 관리하는 도구 모음입니다.

### 파일 설명

#### `create_faissDB.py`
- `.txt` / `.md` 파일을 읽어 FAISS 벡터스토어를 생성하거나 업데이트합니다.
- 임베딩 모델: `intfloat/multilingual-e5-small` (다국어 지원)
- 주요 함수:
  - `create_vectorstore_from_files()` : 폴더 내 파일로 벡터스토어를 새로 생성 (기존 덮어쓰기)
  - `update_vectorstore_from_files()` : 기존 벡터스토어에 새 파일을 추가. 없으면 새로 생성
  - `load_vectorstore()` : 저장된 벡터스토어 로드
  - `file_to_documents()` : 단일 파일을 LangChain Document로 변환

### 처리 흐름

1. 지정 폴더에서 `.txt` / `.md` 파일 수집
2. 파일을 청크 단위로 분할 (기본: 1000자, overlap 200자)
3. 다국어 임베딩 모델로 벡터화 후 FAISS DB로 저장


## main.py

FastAPI 기반 RAG 챗봇 서버입니다. llama.cpp 백엔드와 FAISS 벡터스토어를 연동하여
스트리밍 채팅 API 및 웹 UI를 제공합니다.

### 주요 기능

- **RAG (Retrieval-Augmented Generation)**: 사용자 질문과 관련된 문서를 벡터 DB에서 검색 후 LLM에 컨텍스트로 주입
- **스트리밍 응답**: SSE(Server-Sent Events) 방식으로 실시간 토큰 스트리밍
- **웹 UI 내장**: `/` 경로에서 채팅 인터페이스 제공
- **실험 조건 제어**: 도메인 / 난이도 / 답변 형식(연역·귀납·자유)을 파라미터로 조절 가능
- **OpenAI 호환 프록시**: `/v1/*` 경로를 llama.cpp 서버로 투명하게 프록시

### 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 채팅 UI |
| GET | `/health` | 서버 상태 확인 |
| POST | `/chat` | 채팅 요청 (RAG + 스트리밍 지원) |
| GET | `/models` | 사용 모델 목록 |
| GET/POST | `/v1/*` | llama.cpp OpenAI 호환 프록시 |

### 실행 방법

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## vectorstore/

`create_faissDB.py`로 생성된 FAISS 벡터스토어 바이너리 파일이 저장되는 폴더입니다.

### 파일 설명

- `index.faiss` : 벡터 인덱스 데이터 (임베딩된 청크 벡터)
- `index.pkl` : 문서 메타데이터 및 텍스트 매핑 정보
