#!/usr/bin/env python3
"""
grade_llm.py — Gemini LLM 기반 챗봇 답안 채점 스크립트

사용법:
  python grade_llm.py --results results_rag_*.xlsx --answers ANSWER_only_claude.xlsx
  python grade_llm.py --results results_rag_chat_gpt_*.xlsx --answers ANSWER_only_chatgpt.xlsx
  python grade_llm.py --results results_*.xlsx --model gemini-2.5-flash --concurrency 8
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm.asyncio import tqdm as atqdm

load_dotenv("api_keys.env")

LAYOUTS_KO   = ["두괄식", "미괄식", "자유형식"]
DIFFICULTIES = ["Low", "Medium", "High"]
CHOICES      = ["A", "B", "C", "D"]

DEFAULT_MODEL       = "gemini-2.5-flash"
DEFAULT_CONCURRENCY = 5

# ─── 채점 프롬프트 ─────────────────────────────────────────────────────────────
GRADE_PROMPT = """\
당신은 객관식 문제 채점 전문가입니다.
아래 문제와 챗봇 응답을 분석하여 정확하게 채점하세요.

## 문제
{question}
A. {A}
B. {B}
C. {C}
D. {D}

## 정답
{correct} — {correct_content}

## 챗봇 응답
{response}

## 채점 지시
1. 챗봇이 **최종적으로 선택한 답안 (A/B/C/D)** 을 추출하세요.
   - 응답에 명시적 선택지가 없거나 알 수 없으면 "?" 로 표기하세요.
2. 추출한 선택지가 정답과 일치하는지 판단하세요.
3. 추론/근거의 타당성을 1~5점으로 평가하세요.
   - 5: 정답이고 근거도 완전히 정확
   - 4: 정답이고 근거가 대체로 적절
   - 3: 답은 맞지만 근거가 불충분하거나 부분적으로 잘못됨
   - 2: 오답이지만 일부 근거는 타당
   - 1: 오답이고 근거도 완전히 부적절하거나 환각

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
  "선택": "A",
  "정오": "O",
  "근거점수": 4,
  "평가": "한 줄 평가 (한국어, 50자 이내)"
}}"""


# ─── Gemini 호출 ───────────────────────────────────────────────────────────────
async def grade_one(
    client: genai.Client,
    sem: asyncio.Semaphore,
    model: str,
    row: dict,
    layout_ko: str,
    idx: int,
) -> tuple[int, dict]:
    response_text = str(row.get(f"응답_{layout_ko}", "") or "")
    correct = str(row.get("정답", "")).strip().upper()
    correct_content = str(row.get("정답내용", row.get("해설 (정답 근거)", "")) or "")

    prompt = GRADE_PROMPT.format(
        question=row.get("질문", ""),
        A=row.get("A", ""), B=row.get("B", ""),
        C=row.get("C", ""), D=row.get("D", ""),
        correct=correct,
        correct_content=correct_content,
        response=response_text[:3000],  # 토큰 절약
    )

    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                raw = resp.text.strip()
                # JSON 블록 추출
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    return idx, {
                        f"LLM선택_{layout_ko}":   str(data.get("선택", "?")).upper(),
                        f"LLM정오_{layout_ko}":   str(data.get("정오", "?")),
                        f"LLM근거점수_{layout_ko}": int(data.get("근거점수", 0)),
                        f"LLM평가_{layout_ko}":   str(data.get("평가", "")),
                    }
            except Exception as e:
                if attempt == 2:
                    return idx, {
                        f"LLM선택_{layout_ko}":   "?",
                        f"LLM정오_{layout_ko}":   "?",
                        f"LLM근거점수_{layout_ko}": 0,
                        f"LLM평가_{layout_ko}":   f"[오류] {e}",
                    }
                await asyncio.sleep(2 ** attempt)
    return idx, {}


# ─── 전체 채점 실행 ────────────────────────────────────────────────────────────
async def run_grading(
    df: pd.DataFrame,
    answer_map: dict,
    client: genai.Client,
    model: str,
    concurrency: int,
) -> pd.DataFrame:
    df = df.copy()
    df["정답"] = df["문항ID"].astype(str).str.strip().map(answer_map)

    # 정답 내용 맵 (있으면)
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    for idx, row in df.iterrows():
        for lk in LAYOUTS_KO:
            if f"응답_{lk}" not in df.columns:
                continue
            tasks.append(grade_one(client, sem, model, row.to_dict(), lk, idx))

    print(f"[INFO] 총 {len(tasks)}건 채점 시작 (concurrency={concurrency})")

    # as_completed로 받되, 각 태스크가 자신의 idx를 함께 반환
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="채점 중"):
        row_idx, cols = await coro
        for col, val in cols.items():
            if col not in df.columns:
                df[col] = None
            df.at[row_idx, col] = val

    return df


# ─── 평가 지표 계산 ────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    반환값:
      - "요약": layout × difficulty 별 accuracy/precision/recall/F1
      - "클래스별": layout × 정답클래스(A/B/C/D) 별 precision/recall/F1
      - "혼동행렬_<layout>": layout 별 혼동행렬
    """
    results: dict[str, pd.DataFrame] = {}
    summary_rows = []
    class_rows = []
    confusion_sheets: dict[str, pd.DataFrame] = {}

    for lk in LAYOUTS_KO:
        sel_col = f"LLM선택_{lk}"
        if sel_col not in df.columns:
            continue

        for diff in DIFFICULTIES + ["전체"]:
            sub = df if diff == "전체" else df[df["난이도"] == diff]
            sub = sub.dropna(subset=["정답", sel_col])
            # "?" 제외
            sub = sub[sub[sel_col].isin(CHOICES) & sub["정답"].str.upper().isin(CHOICES)]

            if len(sub) == 0:
                continue

            y_true = sub["정답"].str.upper().tolist()
            y_pred = sub[sel_col].str.upper().tolist()

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=CHOICES, zero_division=0
            )

            summary_rows.append({
                "형식":          lk,
                "난이도":        diff,
                "전체":          len(sub),
                "Accuracy(%)":  round(acc * 100, 1),
                "Precision(%)": round(prec.mean() * 100, 1),
                "Recall(%)":    round(rec.mean() * 100, 1),
                "F1(%)":        round(f1.mean() * 100, 1),
            })

            if diff == "전체":
                for i, cls in enumerate(CHOICES):
                    class_rows.append({
                        "형식":          lk,
                        "클래스":        cls,
                        "지지수":        int(support[i]),
                        "Precision(%)": round(prec[i] * 100, 1),
                        "Recall(%)":    round(rec[i] * 100, 1),
                        "F1(%)":        round(f1[i] * 100, 1),
                    })

                # 혼동행렬
                cm = confusion_matrix(y_true, y_pred, labels=CHOICES)
                cm_df = pd.DataFrame(cm, index=[f"정답:{c}" for c in CHOICES],
                                         columns=[f"예측:{c}" for c in CHOICES])
                confusion_sheets[f"혼동행렬_{lk}"] = cm_df

    results["요약"]    = pd.DataFrame(summary_rows)
    results["클래스별"] = pd.DataFrame(class_rows)
    results.update(confusion_sheets)
    return results


# ─── 엑셀 저장 ────────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, metrics: dict[str, pd.DataFrame], output_path: Path):
    # 컬럼 순서 정리
    fixed = ["난이도", "문항ID", "질문", "A", "B", "C", "D", "정답"]
    layout_cols = []
    for lk in LAYOUTS_KO:
        for prefix in ["선택", "응답", "LLM선택", "LLM정오", "LLM근거점수", "LLM평가"]:
            col = f"{prefix}_{lk}"
            if col in df.columns:
                layout_cols.append(col)
    extra = [c for c in df.columns if c not in fixed + layout_cols]
    df = df[[c for c in fixed + layout_cols + extra if c in df.columns]]

    green_fill  = PatternFill("solid", fgColor="C6EFCE")
    red_fill    = PatternFill("solid", fgColor="FFC7CE")
    yellow_fill = PatternFill("solid", fgColor="FFEB9C")
    bold_font   = Font(bold=True)
    center_align = Alignment(horizontal="center")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # ── 채점결과 시트 ──
        df.to_excel(writer, sheet_name="채점결과", index=False)
        ws = writer.sheets["채점결과"]
        for cell in ws[1]:
            cell.font = bold_font
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                col_name = ws.cell(1, cell.column).value or ""
                if col_name.startswith("LLM정오_"):
                    cell.alignment = center_align
                    if cell.value == "O":
                        cell.fill = green_fill
                    elif cell.value == "X":
                        cell.fill = red_fill
                    elif cell.value == "?":
                        cell.fill = yellow_fill
                elif col_name.startswith("LLM근거점수_"):
                    cell.alignment = center_align
        _auto_width(ws, max_w=50)

        # ── 평가지표 시트들 ──
        for sheet_name, metric_df in metrics.items():
            metric_df.to_excel(writer, sheet_name=sheet_name, index=(sheet_name.startswith("혼동")))
            ws2 = writer.sheets[sheet_name]
            for cell in ws2[1]:
                cell.font = bold_font
            _auto_width(ws2, max_w=20)

            # 퍼센트 컬럼 서식
            pct_cols = {ws2.cell(1, c.column).value: c.column
                        for c in ws2[1] if ws2.cell(1, c.column).value and "%" in str(ws2.cell(1, c.column).value)}
            for row in ws2.iter_rows(min_row=2):
                for cell in row:
                    if ws2.cell(1, cell.column).value in pct_cols.values():
                        pass
                    col_hdr = ws2.cell(1, cell.column).value or ""
                    if "%" in str(col_hdr) and isinstance(cell.value, (int, float)):
                        cell.number_format = "0.0"

    print(f"[저장] {output_path}")


def _auto_width(ws, max_w: int):
    for col in ws.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, max_w)


# ─── 콘솔 요약 출력 ───────────────────────────────────────────────────────────
def print_metrics(metrics: dict[str, pd.DataFrame]):
    summary = metrics.get("요약")
    if summary is None or summary.empty:
        return

    print("\n" + "=" * 70)
    print("LLM 채점 평가 지표 요약")
    print("=" * 70)

    for lk in LAYOUTS_KO:
        sub = summary[summary["형식"] == lk]
        if sub.empty:
            continue
        print(f"\n[{lk}]")
        hdr = f"  {'난이도':<8} {'전체':>5} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}"
        print(hdr)
        print(f"  {'-' * 55}")
        for _, r in sub.iterrows():
            print(
                f"  {r['난이도']:<8} {int(r['전체']):>5} "
                f"{r['Accuracy(%)']:>8.1f}% {r['Precision(%)']:>9.1f}% "
                f"{r['Recall(%)']:>7.1f}% {r['F1(%)']:>6.1f}%"
            )

    class_df = metrics.get("클래스별")
    if class_df is not None and not class_df.empty:
        print("\n[클래스별 Recall / Precision]")
        for lk in LAYOUTS_KO:
            sub = class_df[class_df["형식"] == lk]
            if sub.empty:
                continue
            print(f"  {lk}: ", end="")
            parts = []
            for _, r in sub.iterrows():
                parts.append(f"{r['클래스']}(P={r['Precision(%)']:.0f}% R={r['Recall(%)']:.0f}%)")
            print("  |  ".join(parts))

    print("=" * 70)


# ─── 정답지 자동 탐색 ──────────────────────────────────────────────────────────
def find_answer_file(results_path: Path) -> Path:
    name = results_path.name.lower()
    candidates = (
        ["ANSWER_only_chatgpt.xlsx"]
        if ("chat_gpt" in name or "chatgpt" in name or "google" in name)
        else ["ANSWER_only_claude.xlsx", "ANSWER_only_chatgpt.xlsx"]
    )
    for c in candidates:
        p = results_path.parent / c
        if p.exists():
            return p
    raise FileNotFoundError("정답지를 찾을 수 없습니다. --answers 옵션으로 직접 지정하세요.")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Gemini LLM 기반 채점 스크립트")
    p.add_argument("--results",     required=True, metavar="FILE",  help="답안 결과 파일")
    p.add_argument("--answers",     default=None,  metavar="FILE",  help="정답지 파일 (자동 탐색 가능)")
    p.add_argument("--output",      default=None,  metavar="FILE",  help="출력 파일명")
    p.add_argument("--model",       default=DEFAULT_MODEL,          help=f"Gemini 모델 (기본: {DEFAULT_MODEL})")
    p.add_argument("--concurrency", default=DEFAULT_CONCURRENCY, type=int,
                                                                    help="동시 요청 수 (기본: 5)")
    p.add_argument("--api-key",     default=None,  metavar="KEY",   help="Google API 키 (미지정 시 환경변수 사용)")
    return p.parse_args()


async def main():
    args = parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[오류] 파일 없음: {results_path}")
        sys.exit(1)

    answers_path = Path(args.answers) if args.answers else find_answer_file(results_path)
    if not answers_path.exists():
        print(f"[오류] 정답지 없음: {answers_path}")
        sys.exit(1)

    import os
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[오류] Google API 키가 없습니다. --api-key 또는 GOOGLE_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    print(f"[INFO] 답안지: {results_path}")
    print(f"[INFO] 정답지: {answers_path}")
    print(f"[INFO] 모델:   {args.model}")

    # 데이터 로드
    sheet_names = pd.ExcelFile(results_path).sheet_names
    sheet = "결과" if "결과" in sheet_names else sheet_names[0]
    df_results = pd.read_excel(results_path, sheet_name=sheet)
    df_answers = pd.read_excel(answers_path)

    df_results["문항ID"] = df_results["문항ID"].astype(str).str.strip()
    df_answers["문항ID"] = df_answers["문항ID"].astype(str).str.strip()

    # 정답 맵 (정답 + 정답 내용)
    answer_map = df_answers.set_index("문항ID")["정답"].str.strip().str.upper().to_dict()

    # 정답 내용 맵 (있는 컬럼 우선)
    content_col = next(
        (c for c in ["정답내용", "해설 (정답 근거)", "근거/평가 포인트"] if c in df_answers.columns),
        None,
    )
    content_map = df_answers.set_index("문항ID")[content_col].to_dict() if content_col else {}
    df_results["정답내용"] = df_results["문항ID"].map(content_map).fillna("")

    print(f"[INFO] 답안 {len(df_results)}개, 정답 {len(df_answers)}개")

    # Gemini 클라이언트
    client = genai.Client(api_key=api_key)

    # 채점 실행
    df_graded = await run_grading(
        df_results, answer_map, client, args.model, args.concurrency
    )

    # 평가 지표 계산
    metrics = compute_metrics(df_graded)

    # 저장
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.parent / f"llm_scored_{results_path.name}"

    save_results(df_graded, metrics, output_path)
    print_metrics(metrics)


if __name__ == "__main__":
    asyncio.run(main())
