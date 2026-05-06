# NLP_Project


## tools_vtt/

YouTube 영상의 자막(VTT)을 수집하고 정제하는 도구 모음입니다.

### 파일 설명

#### `create_vtt.ipynb`
- `yt_dlp` 라이브러리를 사용해 YouTube 영상의 한국어 자막(.vtt)을 다운로드합니다.
- 영상 본체는 다운로드하지 않고 자막 파일만 추출합니다.

#### `vtt_to_text.py`
- 다운로드한 `.vtt` 자막 파일을 순수 텍스트로 정제합니다.
- 처리 내용:
  - WEBVTT 헤더, 타임스탬프, HTML 태그 제거
  - HTML 엔티티 디코딩
  - Rolling caption 방식으로 인한 중복 문장 제거
- 정제된 텍스트를 `.txt` 파일로 저장하거나 문자열로 반환합니다.

### 사용 흐름

1. `create_vtt.ipynb` → YouTube URL로 `.vtt` 자막 파일 다운로드
2. `vtt_to_text.py` → `.vtt` 파일을 정제된 텍스트로 변환


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
