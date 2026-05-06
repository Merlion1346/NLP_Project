import re
import html
from pathlib import Path


def parse_vtt(vtt_path: str, output_path: str = None) -> str:
    """
    WebVTT 파일을 순수 텍스트로 정제합니다.

    처리 내용:
    - WEBVTT 헤더 및 메타데이터 제거
    - 타임스탬프 줄 제거 (00:00:00.000 --> 형식)
    - 인라인 타임스탬프 태그 제거 (<00:00:01.920>, <c>, </c>)
    - HTML 엔티티 디코딩 (&gt; → > 등)
    - 중복 줄 제거 (VTT의 rolling 방식으로 인한 중복)
    - 공백 줄 및 여백 문자 정리
    - 연속된 중복 문장 제거
    """
    text = Path(vtt_path).read_text(encoding="utf-8")

    # 1. WEBVTT 헤더 및 메타 블록 제거
    text = re.sub(r"^WEBVTT.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Kind:.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Language:.*?\n", "", text, flags=re.MULTILINE)

    # 2. 타임스탬프 줄 제거 (예: 00:00:00.799 --> 00:00:05.190 align:start position:0%)
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> .+", "", text)

    # 3. 인라인 타임스탬프 제거 (예: <00:00:01.920>)
    text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)

    # 4. VTT 태그 제거 (<c>, </c>, <b>, </b> 등)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

    # 5. HTML 엔티티 디코딩 (&gt; &lt; &amp; 등)
    text = html.unescape(text)

    # 6. 줄 단위로 처리하여 중복 제거
    lines = text.splitlines()
    cleaned_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if not line or line == "\ufeff":  # 빈 줄 및 BOM 제거
            continue
        if line in seen:  # 중복 줄 제거
            continue
        seen.add(line)
        cleaned_lines.append(line)

    # 7. 연속된 문장 중 앞 문장이 뒷 문장에 포함되는 경우 제거
    #    (rolling caption 방식: "A" → "A B" → "A B C" 패턴)
    final_lines = []
    for i, line in enumerate(cleaned_lines):
        if i + 1 < len(cleaned_lines) and cleaned_lines[i + 1].startswith(line):
            continue  # 다음 줄이 현재 줄을 포함하면 현재 줄 스킵
        final_lines.append(line)

    result = "\n".join(final_lines)

    # 8. 3개 이상 연속 빈 줄 → 1개로
    result = re.sub(r"\n{3,}", "\n\n", result)

    if output_path:
        Path(output_path).write_text(result, encoding="utf-8")
        print(f"저장 완료: {output_path}")

    return result


if __name__ == "__main__":
    input_file = "[100분토론] '장특공' 운명은？(1147회) - 2026년 4월 28일 밤 11시 20분 [ienQrhL7WeA].ko.vtt"
    output_file = "100분토론_1147회_정제.txt"

    result = parse_vtt(input_file, output_file)

    # 결과 미리보기
    lines = result.splitlines()
    print(f"\n총 {len(lines)}줄 추출됨\n")
    print("=== 앞부분 미리보기 ===")
    print("\n".join(lines[:20]))
    print("\n...\n")
    print("=== 뒷부분 미리보기 ===")
    print("\n".join(lines[-10:]))