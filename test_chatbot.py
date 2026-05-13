#!/usr/bin/env python3
"""
test_chatbot.py — 챗봇 모델 평가 스크립트

evaluation/*.xlsx의 객관식 문항을 두괄식 / 미괄식 / 자유형식으로 각각 테스트하고
선택 답변과 전체 응답을 Excel로 저장합니다.

엑셀 컬럼: 난이도 | 질문 | 선택지A | 선택지B | 선택지C | 선택지D | 정답

사용법:
  python test_chatbot.py                              # 기본 실행 (RAG ON, unknown domain)
  python test_chatbot.py --no-rag                     # RAG 없이 실행
  python test_chatbot.py --domain known               # Known Parametric 모드
  python test_chatbot.py --resume results_*.xlsx      # 중단된 실행 이어하기
  python test_chatbot.py --question-ids Q-0001 Q-0002 # 특정 문항만 테스트
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

# ─── 기본 설정 ─────────────────────────────────────────────────────────────────
CHATBOT_URL = "http://localhost:8000"
EXCEL_FILE  = "evaluation/자백의대가_평가문항_300제.xlsx"

LAYOUTS    = ["deductive", "inductive", "free"]
LAYOUT_KO  = {"deductive": "두괄식", "inductive": "미괄식", "free": "자유형식"}
DIFF_MAP   = {"Low": "low", "Medium": "medium", "High": "high"}

TEMPERATURE = 0.01   # 평가 시 결정론적 응답을 위해 낮게 설정
MAX_TOKENS  = 2048
TOP_P       = 0.9
RETRY_MAX   = 2     # 실패 시 재시도 횟수
RETRY_DELAY = 3.0   # 재시도 대기 시간(초)

# ─── 질문 포맷 ─────────────────────────────────────────────────────────────────
def format_question(row: pd.Series) -> str:
    return (
        f"다음 객관식 문제를 풀어주세요.\n\n"
        f"문제: {row['질문']}\n"
        f"A. {row['선택지A']}\n"
        f"B. {row['선택지B']}\n"
        f"C. {row['선택지C']}\n"
        f"D. {row['선택지D']}\n\n"
        "반드시 '정답: A', '정답: B', '정답: C', '정답: D' 형식으로 정답 선택지를 명시한 뒤 이유를 설명해주세요."
    )

FOLLOWUP_PROMPT = "위 문제의 최종 정답은 A, B, C, D 중 무엇인가요? 선택지 하나만 답하세요."


# ─── 응답에서 선택지 추출 ──────────────────────────────────────────────────────
_CHOICE_PATTERNS = [
    r'정답\s*[:：]\s*\**([A-D])\**',
    r'정답[은이]?\s*[:：]?\s*([A-D])',
    r'답[은이]?\s*[:：]?\s*([A-D])',
    r'선택[은이]?\s*[:：]?\s*([A-D])',
    r'([A-D])[번항]\s*[이가]?\s*정답',
    r'([A-D])[번항]\s*입니다',
    r'\b([A-D])[)\.）]',
    r'\b([A-D])\b',
]

def extract_choice(text: str) -> str:
    for pat in _CHOICE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "?"


# ─── 단일 요청 (재시도 포함) ───────────────────────────────────────────────────
async def _post(
    client: httpx.AsyncClient,
    messages: list,
    difficulty: str,
    layout: str,
    domain: str,
    use_rag: bool,
) -> str:
    payload = {
        "messages":    messages,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "top_p":       TOP_P,
        "stream":      False,
        "use_rag":     use_rag,
        "domain":      domain,
        "difficulty":  difficulty,
        "layout":      layout,
    }
    for attempt in range(RETRY_MAX + 1):
        try:
            resp = await client.post(f"{CHATBOT_URL}/chat", json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt < RETRY_MAX:
                await asyncio.sleep(RETRY_DELAY)
            else:
                return f"[오류] {exc}"


async def ask(
    client: httpx.AsyncClient,
    question: str,
    difficulty: str,
    layout: str,
    domain: str,
    use_rag: bool,
) -> str:
    messages = [{"role": "user", "content": question}]
    response = await _post(client, messages, difficulty, layout, domain, use_rag)

    # 선택지가 추출되지 않으면 멀티턴으로 선택지만 재질의
    if not response.startswith("[오류]") and extract_choice(response) == "?":
        followup = await _post(
            client,
            [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response},
                {"role": "user",      "content": FOLLOWUP_PROMPT},
            ],
            difficulty, "free", domain, False,
        )
        if not followup.startswith("[오류]"):
            response = response + f"\n[폴백] {followup}"

    return response


# ─── 헬스 체크 ────────────────────────────────────────────────────────────────
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{CHATBOT_URL}/health")
            data = r.json()
            if data.get("llama_cpp") != "ok":
                print(f"[경고] llama.cpp 서버 상태: {data.get('llama_cpp')}")
            rag_status = data.get("vectorstore", "unknown")
            print(f"[INFO] 서버 연결 OK  |  vectorstore: {rag_status}")
            return True
    except Exception as e:
        print(f"[오류] 서버 연결 실패: {e}")
        return False


# ─── 결과 저장 ────────────────────────────────────────────────────────────────
def save_results(records: list[dict], output_path: Path):
    df = pd.DataFrame(records)

    # 열 순서 정리
    fixed_cols = ["난이도", "문항번호", "질문", "선택지A", "선택지B", "선택지C", "선택지D", "정답"]
    layout_cols = []
    for lk in ["두괄식", "미괄식", "자유형식"]:
        layout_cols += [f"선택_{lk}", f"응답_{lk}"]
    ordered = fixed_cols + [c for c in layout_cols if c in df.columns]
    df = df[ordered]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="결과", index=False)

        # 요약 시트
        summary = _build_summary(df)
        summary.to_excel(writer, sheet_name="요약", index=False)

        # 열 너비 자동 조정
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(
                    (len(str(cell.value)) if cell.value else 0) for cell in col
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

    print(f"[저장] {output_path}")


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layout_ko in ["두괄식", "미괄식", "자유형식"]:
        sel_col = f"선택_{layout_ko}"
        if sel_col not in df.columns:
            continue
        for diff in ["Low", "Medium", "High", "전체"]:
            subset = df if diff == "전체" else df[df["난이도"] == diff]
            counts = subset[sel_col].value_counts().reindex(["A", "B", "C", "D", "?"], fill_value=0)
            # 정답 컬럼이 있으면 정답률 계산
            correct = None
            if "정답" in subset.columns:
                matched = subset[sel_col] == subset["정답"].astype(str).str.strip()
                valid = subset[sel_col] != "?"
                correct = round(matched[valid].sum() / valid.sum() * 100, 1) if valid.sum() > 0 else None
            row = {
                "형식":    layout_ko,
                "난이도":  diff,
                "전체":    len(subset),
                **{f"선택_{k}": int(v) for k, v in counts.items()},
            }
            if correct is not None:
                row["정답률(%)"] = correct
            rows.append(row)
    return pd.DataFrame(rows)


# ─── 메인 평가 루프 ───────────────────────────────────────────────────────────
async def run_evaluation(args):
    # 헬스 체크
    if not await health_check():
        sys.exit(1)

    # 원본 데이터 로드
    df_src = pd.read_excel(EXCEL_FILE)
    # 문항번호 자동 생성 (Q-0001, Q-0002, ...)
    df_src["문항번호"] = [f"Q-{i+1:04d}" for i in range(len(df_src))]
    print(f"[INFO] 총 {len(df_src)}개 문항 ({', '.join(df_src['난이도'].value_counts().to_dict().__repr__()[1:-1].split(', '))})")

    # 특정 문항만 테스트하는 경우 필터
    if args.question_ids:
        df_src = df_src[df_src["문항번호"].isin(args.question_ids)].reset_index(drop=True)
        print(f"[INFO] 필터 적용 후 {len(df_src)}개 문항")

    # 출력 파일명 결정
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "rag" if args.use_rag else "norag"
        output_path = Path(f"results_{tag}_{ts}.xlsx")

    # 재개 모드: 이미 완료된 문항 건너뜀
    done_ids: set[str] = set()
    existing_records: list[dict] = []
    if args.resume and Path(args.resume).exists():
        prev = pd.read_excel(args.resume, sheet_name="결과")
        # 세 형식 모두 응답이 있는 행만 완료로 간주
        completed = prev.dropna(subset=["응답_두괄식", "응답_미괄식", "응답_자유형식"])
        done_ids = set(completed["문항번호"].tolist())
        existing_records = completed.to_dict("records")
        print(f"[재개] 이미 완료된 {len(done_ids)}개 문항 건너뜀")

    total = len(df_src)
    records = list(existing_records)

    async with httpx.AsyncClient() as client:
        for seq, (_, row) in enumerate(df_src.iterrows(), start=1):
            qid = str(row["문항번호"])
            if qid in done_ids:
                continue

            difficulty = DIFF_MAP.get(str(row["난이도"]), "medium")
            question_text = format_question(row)

            print(f"\n[{seq:2d}/{total}] {qid} ({row['난이도']}) — {str(row['질문'])[:40]}...")

            record: dict = {
                "난이도":  row["난이도"],
                "문항번호": qid,
                "질문":    row["질문"],
                "선택지A": row["선택지A"],
                "선택지B": row["선택지B"],
                "선택지C": row["선택지C"],
                "선택지D": row["선택지D"],
                "정답":    row.get("정답", ""),
            }

            for layout in LAYOUTS:
                layout_ko = LAYOUT_KO[layout]
                print(f"  [{layout_ko}] 요청 중 ...", end=" ", flush=True)

                response = await ask(
                    client, question_text, difficulty, layout,
                    domain=args.domain, use_rag=args.use_rag,
                )
                choice = extract_choice(response)
                is_error = response.startswith("[오류]")

                print(f"→ 선택: {choice}" + (" ⚠ 오류" if is_error else ""))

                record[f"선택_{layout_ko}"] = choice
                record[f"응답_{layout_ko}"] = response

            records.append(record)

            # 10문항마다 중간 저장
            if seq % 10 == 0:
                save_results(records, output_path)
                print(f"  [중간 저장] {seq}문항 완료")

    # 최종 저장
    save_results(records, output_path)

    # 콘솔 요약 출력
    print("\n" + "=" * 60)
    print("평가 완료")
    print(f"결과 파일: {output_path}")
    print(f"총 문항:   {len(records)}개")
    print("=" * 60)
    result_df = pd.DataFrame(records)
    for layout_ko in ["두괄식", "미괄식", "자유형식"]:
        col = f"선택_{layout_ko}"
        if col in result_df.columns:
            dist = result_df[col].value_counts().to_dict()
            print(f"[{layout_ko}] {dist}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="챗봇 모델 평가 스크립트")
    p.add_argument("--url",          default=CHATBOT_URL,  help="FastAPI 서버 URL")
    p.add_argument("--input",        default=EXCEL_FILE,   help="입력 Excel 파일")
    p.add_argument("--output",       default=None,         help="출력 Excel 파일명 (미지정 시 자동 생성)")
    p.add_argument("--domain",       default="unknown",    choices=["known", "unknown"],
                                                           help="지식 도메인 (known / unknown)")
    p.add_argument("--no-rag",       dest="use_rag",       action="store_false",
                                                           help="RAG 비활성화")
    p.add_argument("--resume",       default=None,         metavar="FILE",
                                                           help="이전 결과 파일에서 이어하기")
    p.add_argument("--question-ids", nargs="+",            metavar="ID",
                                                           help="테스트할 문항 ID 목록 (예: L-01 M-03)")
    p.set_defaults(use_rag=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CHATBOT_URL = args.url
    EXCEL_FILE  = args.input
    asyncio.run(run_evaluation(args))
