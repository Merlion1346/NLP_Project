"""
llama.cpp FastAPI 게이트웨이 사용 예제
게이트웨이를 먼저 실행하세요: uvicorn main:app --reload --port 8000
"""

import httpx
import json

GATEWAY = "http://localhost:8000"

# ─── 1. 헬스 체크 ──────────────────────────────────────────────────────────────
def check_health():
    r = httpx.get(f"{GATEWAY}/health")
    print("Health:", r.json())

# ─── 2. 채팅 (non-stream) ──────────────────────────────────────────────────────
def chat_example():
    payload = {
        "messages": [
            {"role": "system", "content": "당신은 친절한 한국어 AI 어시스턴트입니다."},
            {"role": "user", "content": "파이썬에서 비동기 프로그래밍이란 무엇인가요?"},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    r = httpx.post(f"{GATEWAY}/chat", json=payload, timeout=60)
    data = r.json()
    print("Chat 응답:", data["choices"][0]["message"]["content"])

# ─── 3. 채팅 (streaming) ───────────────────────────────────────────────────────
def chat_stream_example():
    payload = {
        "messages": [{"role": "user", "content": "하늘은 왜 파란가요?"}],
        "stream": True,
        "max_tokens": 256,
    }
    print("Stream 응답: ", end="", flush=True)
    with httpx.stream("POST", f"{GATEWAY}/chat", json=payload, timeout=60) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
    print()

# ─── 4. 텍스트 완성 ─────────────────────────────────────────────────────────────
def completion_example():
    payload = {
        "prompt": "def fibonacci(n):",
        "temperature": 0.2,
        "max_tokens": 200,
        "stop": ["\n\n"],
    }
    r = httpx.post(f"{GATEWAY}/complete", json=payload, timeout=60)
    print("Completion:", r.json()["choices"][0]["text"])

# ─── 5. OpenAI SDK 호환 사용법 ─────────────────────────────────────────────────
def openai_sdk_example():
    """
    pip install openai
    기존 OpenAI 코드에서 base_url 만 바꾸면 됩니다.
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="unsloth/Qwen3.5-9B",
        messages=[{"role": "user", "content": "안녕하세요!"}],
    )
    print("OpenAI SDK:", response.choices[0].message.content)

if __name__ == "__main__":
    check_health()
    chat_example()
    chat_stream_example()
    completion_example()
