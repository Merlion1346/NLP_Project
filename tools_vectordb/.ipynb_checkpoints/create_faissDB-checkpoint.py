"""
TXT / MD → FAISS 벡터스토어 변환기
폴더 내 .txt 및 .md 파일을 읽어 FAISS 벡터스토어로 저장합니다.

설치:
    pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers
"""
import glob
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── 설정 ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "intfloat/multilingual-e5-small"
INPUT_PATH        = "txt"
VECTOR_STORE_PATH = "vectorstore"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

# 지원 확장자
SUPPORTED_EXTENSIONS = ("*.txt", "*.md")

# ── 임베딩 모델 (전역 1회 로드) ───────────────────────────────────────────────
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=        ["\n\n", "\n", " "],
        chunk_size=        chunk_size,
        chunk_overlap=     chunk_overlap,
        length_function=   len,
        is_separator_regex=False,
    )


def _chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """주어진 파라미터로 문서를 청킹합니다."""
    return _build_splitter(chunk_size, chunk_overlap).split_documents(documents)


def _collect_files(directory: str) -> List[Path]:
    """디렉터리에서 지원 확장자(.txt, .md) 파일을 재귀 탐색합니다."""
    found: List[Path] = []
    for pattern in SUPPORTED_EXTENSIONS:
        found.extend(
            Path(p) for p in glob.glob(
                str(Path(directory) / "**" / pattern), recursive=True
            )
        )
    # 경로 기준 정렬 & 중복 제거
    return sorted(set(found))


# ── 파일 → Document 변환 ──────────────────────────────────────────────────────
def file_to_documents(file_path: str) -> List[Document]:
    """
    단일 .txt 또는 .md 파일을 읽어 Document 리스트로 변환합니다.
    파일 타입은 확장자로 자동 판별합니다.
    """
    path = Path(file_path)
    ext  = path.suffix.lstrip(".").lower()   # "txt" 또는 "md"

    with open(file_path, encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    return [Document(
        page_content=content,
        metadata={
            "source": path.stem,
            "title":  path.stem,
            "type":   ext,
            "file":   path.name,
        },
    )]


# 하위 호환성을 위한 별칭
def txt_to_documents(txt_path: str) -> List[Document]:
    """(하위 호환) file_to_documents()로 위임합니다."""
    return file_to_documents(txt_path)


# ── 벡터스토어 로드 ───────────────────────────────────────────────────────────
def load_vectorstore(path: str = VECTOR_STORE_PATH) -> FAISS:
    """저장된 FAISS 벡터스토어를 로드합니다."""
    return FAISS.load_local(path, EMBEDDINGS, allow_dangerous_deserialization=True)


# ── 파일 폴더 → 벡터스토어 생성 ──────────────────────────────────────────────
def create_vectorstore_from_files(
    input_dir:    str = INPUT_PATH,
    output_path:  str = VECTOR_STORE_PATH,
    chunk_size:   int = CHUNK_SIZE,
    chunk_overlap:int = CHUNK_OVERLAP,
) -> FAISS:
    """
    폴더 내 모든 .txt / .md 파일을 읽어 FAISS 벡터스토어를 새로 생성합니다.
    기존 벡터스토어가 있으면 덮어씁니다.
    """
    # ── Step 1. 파일 로드 ─────────────────────────────────────────
    print(f"\n[1/3] 파일 로드: {input_dir}")
    files = _collect_files(input_dir)
    if not files:
        raise ValueError(f".txt/.md 파일이 없습니다: {input_dir}")

    all_docs: List[Document] = []
    for fp in files:
        docs = file_to_documents(str(fp))
        print(f"  [{fp.name}] → Document {len(docs)}개")
        all_docs.extend(docs)
    print(f"  총 Document: {len(all_docs)}개")

    if not all_docs:
        raise ValueError("변환된 Document가 없습니다. 파일 내용을 확인하세요.")

    # ── Step 2. 청킹 ──────────────────────────────────────────────
    print("\n[2/3] 청킹")
    all_chunks = _chunk_documents(all_docs, chunk_size, chunk_overlap)
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap} → {len(all_chunks)}개 청크")

    # ── Step 3. 임베딩 & 저장 ────────────────────────────────────
    print(f"\n[3/3] 임베딩 & 벡터스토어 저장 → {output_path}")
    vectorstore = FAISS.from_documents(all_chunks, EMBEDDINGS)
    vectorstore.save_local(output_path)
    print(f"  ✅ 저장 완료: {output_path}/")
    return vectorstore


# ── 파일 폴더 → 기존 벡터스토어에 추가 ──────────────────────────────────────
def update_vectorstore_from_files(
    input_dir:    str = INPUT_PATH,
    output_path:  str = VECTOR_STORE_PATH,
    chunk_size:   int = CHUNK_SIZE,
    chunk_overlap:int = CHUNK_OVERLAP,
) -> FAISS:
    """
    폴더 내 모든 .txt / .md 파일을 읽어 기존 FAISS 벡터스토어에 추가합니다.
    벡터스토어가 없으면 새로 생성합니다.
    """
    # ── Step 1. 파일 로드 & 청킹 ─────────────────────────────────
    print(f"\n[1/3] 파일 로드: {input_dir}")
    files = _collect_files(input_dir)
    if not files:
        raise ValueError(f".txt/.md 파일이 없습니다: {input_dir}")

    all_chunks: List[Document] = []
    for fp in files:
        docs   = file_to_documents(str(fp))
        chunks = _chunk_documents(docs, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  [{fp.name}] → {len(chunks)}개 청크")
    print(f"  총 청크 수: {len(all_chunks)}개")

    # ── Step 2. 벡터스토어 로드 or 생성 ──────────────────────────
    if (Path(output_path) / "index.faiss").exists():
        print(f"\n[2/3] 기존 벡터스토어 로드: {output_path}")
        vectorstore = load_vectorstore(output_path)
    else:
        print(f"\n[2/3] 벡터스토어 없음 → 새로 생성")
        vectorstore = FAISS.from_documents(all_chunks, EMBEDDINGS)
        vectorstore.save_local(output_path)
        print(f"  ✅ 저장 완료: {output_path}/")
        return vectorstore

    # ── Step 3. 문서 추가 & 재저장 ───────────────────────────────
    print("\n[3/3] 문서 추가 & 재저장")
    vectorstore.add_documents(all_chunks)
    vectorstore.save_local(output_path)
    print(f"  ✅ 업데이트 완료: {output_path}/")
    return vectorstore


# 하위 호환성을 위한 별칭
def update_vectorstore_from_txt(
    txt_dir:      str = INPUT_PATH,
    output_path:  str = VECTOR_STORE_PATH,
    chunk_size:   int = CHUNK_SIZE,
    chunk_overlap:int = CHUNK_OVERLAP,
) -> FAISS:
    """(하위 호환) update_vectorstore_from_files()로 위임합니다."""
    return update_vectorstore_from_files(txt_dir, output_path, chunk_size, chunk_overlap)


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    index_file = Path(VECTOR_STORE_PATH) / "index.faiss"
    if index_file.exists():
        vs = load_vectorstore()
    else:
        vs = update_vectorstore_from_files()

    retriever = vs.as_retriever(search_kwargs={"k": 3})
    print("\n[검색 테스트] '주요 개념'")
    results = retriever.invoke("주요 개념")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "")
        print(f"  [{i}] {src} — {doc.page_content[:80]}...")