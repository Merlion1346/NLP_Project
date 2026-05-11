from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from pathlib import Path
import httpx
import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ─── 설정 ─────────────────────────────────────────────────────────────────────
LLAMA_CPP_BASE_URL = "http://<SERVER_ADDRESS>:30004"
EMBEDDING_BASE_URL = "http://<SERVER_ADDRESS>:30005"
MODEL_NAME         = "unsloth/Qwen3.6.-35B-A3B"
EMBEDDING_MODEL    = "BAAI/bge-m3"
VECTOR_STORE_PATH  = "vectorstore"
TOP_K              = 3   # 검색할 문서 수

# ─── 임베딩 & 벡터스토어 로드 ─────────────────────────────────────────────────
print(f"[RAG] 임베딩 모델: {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")
EMBEDDINGS = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_base=f"{EMBEDDING_BASE_URL}/v1",
    openai_api_key="not-needed",
)

if Path(VECTOR_STORE_PATH).exists():
    print(f"[RAG] 벡터스토어 로드: {VECTOR_STORE_PATH}")
    VECTORSTORE = FAISS.load_local(VECTOR_STORE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    print("[RAG] 벡터스토어 로드 완료 ✅")
else:
    print(f"[RAG] ⚠️  벡터스토어 없음: {VECTOR_STORE_PATH} — RAG 없이 일반 채팅으로 동작합니다.")
    VECTORSTORE = None

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── 스키마 ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=32768, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    use_rag: bool = True
    # ── 실험 조건 ──────────────────────────────────────────────────
    domain: str = "unknown"      # "known" | "unknown"
    difficulty: str = "medium"   # "low" | "medium" | "high"
    layout: str = "free"         # "deductive" | "inductive" | "free"

# ── 실험 조건 → system prompt 생성 ────────────────────────────────────────────
LAYOUT_PROMPTS = {
    "deductive": (
        "반드시 다음 순서로만 답변하라: "
        "1.첫 문장에서 질문에 대한 답변을 명확하게 제시하라 "
        "2.이후 문장에서 결론의 근거와 이유를 설명 "
        "3.마지막 문장에서 결론을 한 번 더 요약. "
        "절대로 첫 문장을 '~때문에', '~에 따르면', '~을 보면' 등 근거 설명으로 시작하지 말 것. "
        "'근거:', '이유:', '배경:' 같은 레이블을 답변 앞부분에 붙이지 말 것."
    ),
    "inductive": (
        "반드시 다음 순서로만 답변하라: "
        "1.참고 문서의 관련 근거를 먼저 나열 "
        "2.근거를 바탕으로 추론 과정 설명 "
        "3.마지막 문장에서만 질문에 대한 답변을 명확하게 제시하라. "
        "절대로 첫 문장에 결론을 쓰지 말 것. "
        "'결론:', '답:', '정답:' 같은 레이블을 답변 앞부분에 붙이지 말 것."
    ),
    "free": "",
}

DIFFICULTY_PROMPTS = {
    "low":    "질문에 직접 명시된 정보를 정확히 인출하여 답하라.",
    "medium": "2~3개의 정보를 결합하여 추론한 뒤 답하라.",
    "high":   "인물의 심리 분석, 가상 상황 시뮬레이션, 다단계 수치 연산 등 고난도 추론을 수행하여 답하라.",
}

def build_experiment_system_prompt(domain: str, difficulty: str, layout: str) -> str:
    """실험 조건(도메인/난이도/답변형식)을 반영한 system prompt를 생성합니다."""
    parts = []

    if domain == "known":
        parts.append("당신은 일반 상식을 활용하여 답변하는 AI 어시스턴트입니다.")
    else:
        parts.append(
            "당신은 주어진 참고 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다. "
            "외부 자료 검색은 금지입니다."
            "참고 문서에 없는 내용은 모른다고 솔직하게 말하세요."
        )

    parts.append(DIFFICULTY_PROMPTS[difficulty])

    layout_instruction = LAYOUT_PROMPTS[layout]
    if layout_instruction:
        parts.append(layout_instruction)

    return " ".join(parts)

# ─── RAG 헬퍼 ─────────────────────────────────────────────────────────────────
def retrieve_context(query: str, k: int = TOP_K) -> str:
    """쿼리와 관련된 문서를 벡터 DB에서 검색해 컨텍스트 문자열로 반환합니다."""
    if VECTORSTORE is None:
        return ""
    docs = VECTORSTORE.similarity_search(query, k=k)
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        parts.append(f"[참고 {i}] {('(' + source + ') ') if source else ''}{doc.page_content.strip()}")
    return "\n\n".join(parts)

def build_rag_messages(messages: List[Message], context: str, layout: str = "free") -> List[dict]:
    result = []
    has_system = messages and messages[0].role == "system"

    rag_instruction = (
        "당신은 주어진 참고 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다. "
        "답변은 반드시 참고 문서의 내용만 활용해주세요, "
        "외부 자료를 통한 검색은 금지입니다. 문서에 없는 내용은 모른다고 솔직하게 말하세요."
    )

    if has_system:
        result.append({"role": "system", "content": messages[0].content + "\n\n" + rag_instruction})
        body = list(messages[1:])
    else:
        result.append({"role": "system", "content": rag_instruction})
        body = list(messages)

    # 레이아웃 지시어
    layout_instruction = {
        "deductive": "답변을 먼저 말하고 이유를 설명해줘.",
        "inductive": "근거를 먼저 설명하고 최종 답변을 마지막에 말해줘.",
        "free":      "",
    }.get(layout, "")

    # 마지막 user 메시지에 컨텍스트 + 레이아웃 지시어 주입
    for i, msg in enumerate(body):
        if i == len(body) - 1 and msg.role == "user" and context:
            parts = ["[참고 문서]", context, "", "[질문]", msg.content]
            if layout_instruction:
                parts += ["", f"[필수 답변 형식 — 반드시 준수할 것] {layout_instruction}"]
            result.append({"role": "user", "content": "\n".join(parts)})
        else:
            result.append({"role": msg.role, "content": msg.content})

    return result

# ─── 스트리밍 헬퍼 ─────────────────────────────────────────────────────────────
async def stream_llama(payload: dict):
    async with httpx.AsyncClient(base_url=LLAMA_CPP_BASE_URL, timeout=120.0) as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise HTTPException(status_code=response.status_code, detail=body.decode())
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"

# ─── UI HTML ───────────────────────────────────────────────────────────────────
CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Chatbot</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0c0c0f;--surface:#13131a;--surface2:#1c1c27;--border:#2a2a3d;
    --accent:#7c6aff;--accent2:#a78bfa;--text:#e8e6f0;--text-muted:#6b6885;
    --user-bg:#1e1b3a;--assistant-bg:#131320;--success:#4ade80;
    --rag-bg:#0d1a0f;--rag-border:#1a3a1f;--rag-text:#6db87a;--radius:14px;
  }
  body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;height:100dvh;display:flex;flex-direction:column;overflow:hidden}

  header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0;gap:12px}
  .header-left{display:flex;align-items:center;gap:12px}
  .logo{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
  .header-title{font-family:'Syne',sans-serif;font-weight:700;font-size:15px;letter-spacing:-0.3px}
  .badge{font-size:10px;color:var(--text-muted);background:var(--surface2);border:1px solid var(--border);padding:2px 7px;border-radius:5px}
  .rag-badge{font-size:10px;color:var(--rag-text);background:var(--rag-bg);border:1px solid var(--rag-border);padding:2px 7px;border-radius:5px}
  .status-dot{width:7px;height:7px;border-radius:50%;background:var(--text-muted);transition:background .3s;flex-shrink:0}
  .status-dot.online{background:var(--success);box-shadow:0 0 5px var(--success)}

  .toolbar{display:flex;gap:12px;padding:9px 24px;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0;flex-wrap:wrap;align-items:center}
  .setting-item{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--text-muted)}
  input[type=range]{-webkit-appearance:none;width:75px;height:3px;background:var(--border);border-radius:2px;outline:none;cursor:pointer}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:var(--accent);cursor:pointer}
  input[type=number]{width:58px;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:2px 5px;border-radius:5px;font-family:'DM Mono',monospace;font-size:11px}
  .val{color:var(--accent2);min-width:26px;text-align:right}

  /* RAG 토글 스위치 */
  .rag-toggle-wrap{display:flex;align-items:center;gap:8px;margin-left:auto}
  .rag-label{font-size:11px;color:var(--text-muted)}
  .toggle{position:relative;width:36px;height:20px;cursor:pointer}
  .toggle input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:var(--border);border-radius:20px;transition:.3s}
  .slider:before{content:'';position:absolute;width:14px;height:14px;left:3px;top:3px;background:white;border-radius:50%;transition:.3s}
  input:checked + .slider{background:var(--accent)}
  input:checked + .slider:before{transform:translateX(16px)}
  .rag-status{font-size:11px;color:var(--rag-text);font-weight:500}

  .sys-toggle{background:none;border:1px solid var(--border);color:var(--text-muted);font-family:'DM Mono',monospace;font-size:11px;padding:3px 9px;border-radius:5px;cursor:pointer;transition:all .2s}
  .sys-toggle:hover{border-color:var(--accent);color:var(--accent2)}
  .sys-area{display:none;padding:9px 24px;border-bottom:1px solid var(--border);background:var(--surface)}
  .sys-area.open{display:block}
  .sys-area textarea{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:9px 11px;border-radius:7px;resize:vertical;min-height:64px;outline:none;transition:border-color .2s}
  .sys-area textarea:focus{border-color:var(--accent)}

  #messages{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px;scroll-behavior:smooth}
  #messages::-webkit-scrollbar{width:4px}
  #messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

  .message{display:flex;gap:11px;animation:fadeIn .22s ease-out;max-width:840px;width:100%}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
  .message.user{align-self:flex-end;flex-direction:row-reverse}
  .message.assistant{align-self:flex-start}
  .avatar{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px}
  .message.user .avatar{background:var(--user-bg);border:1px solid #3d3670}
  .message.assistant .avatar{background:linear-gradient(135deg,var(--accent),var(--accent2))}

  .msg-wrap{display:flex;flex-direction:column;gap:6px;max-width:calc(100% - 42px)}

  /* RAG 컨텍스트 박스 */
  .rag-context{background:var(--rag-bg);border:1px solid var(--rag-border);border-radius:8px;overflow:hidden;font-size:11px}
  .rag-context-hdr{display:flex;align-items:center;gap:5px;padding:5px 10px;color:var(--rag-text);cursor:pointer;user-select:none;background:#0a140c}
  .rag-context-hdr:hover{opacity:.8}
  .rag-context-body{padding:8px 10px;color:#5a9e67;line-height:1.55;white-space:pre-wrap;display:none}
  .rag-context-body.open{display:block}

  .bubble{padding:11px 15px;border-radius:var(--radius);font-size:13.5px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
  .message.user .bubble{background:var(--user-bg);border:1px solid #3d3670;border-top-right-radius:4px;color:#d4cfff}
  .message.assistant .bubble{background:var(--assistant-bg);border:1px solid var(--border);border-top-left-radius:4px;color:var(--text)}

  .empty-state{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--text-muted);pointer-events:none;user-select:none}
  .empty-state .icon{font-size:44px;opacity:.25}
  .empty-state p{font-family:'Syne',sans-serif;font-size:13px;opacity:.45}

  .typing{display:flex;gap:4px;padding:13px 15px}
  .typing span{width:6px;height:6px;border-radius:50%;background:var(--text-muted);animation:bounce 1.2s infinite}
  .typing span:nth-child(2){animation-delay:.2s}
  .typing span:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}

  .input-area{padding:14px 24px 18px;border-top:1px solid var(--border);background:var(--surface);flex-shrink:0}
  .input-row{display:flex;gap:9px;align-items:flex-end;background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:9px 13px;transition:border-color .2s}
  .input-row:focus-within{border-color:var(--accent)}
  #user-input{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:'DM Mono',monospace;font-size:13.5px;line-height:1.5;resize:none;max-height:150px;min-height:22px}
  #user-input::placeholder{color:var(--text-muted)}
  .send-btn{width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s,transform .1s;flex-shrink:0}
  .send-btn:hover{opacity:.85}
  .send-btn:active{transform:scale(.93)}
  .send-btn:disabled{opacity:.3;cursor:not-allowed}
  .send-btn svg{width:15px;height:15px;fill:white}
  .input-footer{display:flex;justify-content:space-between;align-items:center;margin-top:7px;padding:0 1px}
  .hint{font-size:11px;color:var(--text-muted)}
  .clear-btn{background:none;border:none;color:var(--text-muted);font-family:'DM Mono',monospace;font-size:11px;cursor:pointer;padding:2px 5px;border-radius:4px;transition:color .2s}
  .clear-btn:hover{color:#ff6b6b}

  /* ── 실험 조건 패널 ── */
  .exp-panel{padding:16px 24px;border-bottom:1px solid var(--border);background:#0e0e16;flex-shrink:0;display:flex;gap:28px;flex-wrap:wrap;align-items:center}
  .exp-group{display:flex;flex-direction:column;gap:8px}
  .exp-group-label{font-size:12px;color:var(--text-muted);letter-spacing:.5px;text-transform:uppercase}
  .exp-btns{display:flex;gap:8px}
  .exp-btn{background:var(--surface2);border:1px solid var(--border);color:var(--text-muted);font-family:'DM Mono',monospace;font-size:14px;padding:8px 20px;border-radius:8px;cursor:pointer;transition:all .15s}
  .exp-btn:hover{border-color:var(--accent);color:var(--text)}
  .exp-btn.active-known{background:#1a2a1a;border-color:#4ade80;color:#4ade80}
  .exp-btn.active-unknown{background:#1a1a2a;border-color:var(--accent2);color:var(--accent2)}
  .exp-btn.active-low{background:#1a2a1a;border-color:#86efac;color:#86efac}
  .exp-btn.active-medium{background:#2a1f0a;border-color:#fbbf24;color:#fbbf24}
  .exp-btn.active-high{background:#2a0f0f;border-color:#f87171;color:#f87171}
  .exp-btn.active-deductive{background:#1a1635;border-color:var(--accent);color:var(--accent2)}
  .exp-btn.active-inductive{background:#1a1635;border-color:var(--accent);color:var(--accent2)}
  .exp-btn.active-free{background:#1c1c27;border-color:#6b6885;color:#9d9bb5}
  .exp-summary{margin-left:auto;font-size:10px;color:var(--text-muted);background:var(--surface2);border:1px solid var(--border);padding:4px 10px;border-radius:6px;line-height:1.5;max-width:300px}
  .exp-summary span{color:var(--accent2)}</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo">&#129504;</div>
    <div class="header-title">RAG Chatbot</div>
    <div class="badge">Qwen3.5-0.8B</div>
    <div class="rag-badge">&#128269; FAISS RAG</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <div class="status-dot" id="sdot"></div>
    <span style="font-size:11px;color:var(--text-muted)" id="stxt">연결 확인 중...</span>
  </div>
</header>

<div class="toolbar">
  <div class="setting-item">
    <label>Temp</label>
    <input type="range" id="temp" min="0" max="2" step="0.05" value="0.7">
    <span class="val" id="temp-v">0.70</span>
  </div>
  <div class="setting-item">
    <label>Max Tokens</label>
    <input type="number" id="maxtok" value="32768" min="64" max="32768" step="64">
  </div>
  <div class="setting-item">
    <label>Top-p</label>
    <input type="range" id="topp" min="0" max="1" step="0.05" value="0.9">
    <span class="val" id="topp-v">0.90</span>
  </div>
  <button class="sys-toggle" id="sys-btn">&#9881; System</button>
  <div class="rag-toggle-wrap">
    <span class="rag-label">RAG</span>
    <label class="toggle">
      <input type="checkbox" id="rag-toggle" checked>
      <span class="slider"></span>
    </label>
    <span class="rag-status" id="rag-status">ON</span>
  </div>
</div>

<div class="sys-area" id="sys-area">
  <textarea id="sysprompt">당신은 친절하고 유능한 AI 어시스턴트입니다.</textarea>
</div>

<!-- 실험 조건 패널 -->
<div class="exp-panel">
  <div class="exp-group">
    <div class="exp-group-label">지식 도메인</div>
    <div class="exp-btns" id="grp-domain">
      <button class="exp-btn active-known" data-val="known">Known Parametric</button>
      <button class="exp-btn active-unknown" data-val="unknown">Unknown RAG</button>
    </div>
  </div>
  <div class="exp-group">
    <div class="exp-group-label">난이도</div>
    <div class="exp-btns" id="grp-difficulty">
      <button class="exp-btn active-low" data-val="low">Low</button>
      <button class="exp-btn active-medium" data-val="medium">Medium</button>
      <button class="exp-btn active-high" data-val="high">High</button>
    </div>
  </div>
  <div class="exp-group">
    <div class="exp-group-label">답변 형식</div>
    <div class="exp-btns" id="grp-layout">
      <button class="exp-btn active-deductive" data-val="deductive">두괄식</button>
      <button class="exp-btn active-inductive" data-val="inductive">미괄식</button>
      <button class="exp-btn active-free" data-val="free">자유형식</button>
    </div>
  </div>
  <div class="exp-summary" id="exp-summary">
    <span id="exp-sum-domain">Unknown RAG</span> &middot;
    <span id="exp-sum-difficulty">Medium</span> &middot;
    <span id="exp-sum-layout">자유형식</span>
  </div>
</div>

<div id="messages">
  <div class="empty-state" id="empty">
    <div class="icon">&#128269;</div>
    <p>RAG 모드로 대화를 시작하세요</p>
  </div>
</div>

<div class="input-area">
  <div class="input-row">
    <textarea id="user-input" rows="1" placeholder="메시지를 입력하세요..."></textarea>
    <button class="send-btn" id="send-btn">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
  <div class="input-footer">
    <span class="hint">Enter 전송 &middot; Shift+Enter 줄바꿈</span>
    <button class="clear-btn" id="clear-btn">대화 초기화</button>
  </div>
</div>

<script>
var msgs    = document.getElementById('messages');
var inp     = document.getElementById('user-input');
var sBtn    = document.getElementById('send-btn');
var empty   = document.getElementById('empty');
var sdot    = document.getElementById('sdot');
var stxt    = document.getElementById('stxt');
var ragToggle  = document.getElementById('rag-toggle');
var ragStatus  = document.getElementById('rag-status');
var chatHistory = [];
var streaming = false;

// ── 실험 조건 상태 ─────────────────────────────────────────────
var expState = { domain: 'unknown', difficulty: 'medium', layout: 'free' };

var DOMAIN_LABELS    = { known: 'Known Parametric', unknown: 'Unknown RAG' };
var DIFFICULTY_LABELS = { low: 'Low', medium: 'Medium', high: 'High' };
var LAYOUT_LABELS    = { deductive: '두괄식', inductive: '미괄식', free: '자유형식' };

function initExpGroup(groupId, stateKey, defaultVal) {
  var grp = document.getElementById(groupId);
  var btns = grp.querySelectorAll('.exp-btn');
  btns.forEach(function(btn) {
    var isDefault = btn.dataset.val === defaultVal;
    btn.style.opacity = isDefault ? '1' : '0.35';
    btn.addEventListener('click', function() {
      expState[stateKey] = btn.dataset.val;
      btns.forEach(function(b) { b.style.opacity = '0.35'; });
      btn.style.opacity = '1';
      updateExpSummary();
      // Unknown RAG 선택 시 RAG 자동 ON
      if(stateKey === 'domain') {
        ragToggle.checked = (expState.domain === 'unknown');
        ragStatus.textContent = ragToggle.checked ? 'ON' : 'OFF';
        ragStatus.style.color = ragToggle.checked ? 'var(--rag-text)' : 'var(--text-muted)';
      }
    });
  });
}

function updateExpSummary() {
  document.getElementById('exp-sum-domain').textContent     = DOMAIN_LABELS[expState.domain];
  document.getElementById('exp-sum-difficulty').textContent = DIFFICULTY_LABELS[expState.difficulty];
  document.getElementById('exp-sum-layout').textContent     = LAYOUT_LABELS[expState.layout];
}

initExpGroup('grp-domain',     'domain',     'unknown');
initExpGroup('grp-difficulty', 'difficulty', 'medium');
initExpGroup('grp-layout',     'layout',     'free');
updateExpSummary();

// 헬스 체크
fetch('/health').then(function(r){ return r.json(); }).then(function(d){
  if(d.llama_cpp === 'ok'){ sdot.classList.add('online'); stxt.textContent = '연결됨'; }
  else { stxt.textContent = 'llama.cpp 오프라인'; }
}).catch(function(){ stxt.textContent = '연결 실패'; });

// RAG 토글
ragToggle.addEventListener('change', function(){
  ragStatus.textContent = ragToggle.checked ? 'ON' : 'OFF';
  ragStatus.style.color = ragToggle.checked ? 'var(--rag-text)' : 'var(--text-muted)';
});

// 슬라이더
document.getElementById('temp').addEventListener('input', function(e){
  document.getElementById('temp-v').textContent = (+e.target.value).toFixed(2);
});
document.getElementById('topp').addEventListener('input', function(e){
  document.getElementById('topp-v').textContent = (+e.target.value).toFixed(2);
});
document.getElementById('sys-btn').addEventListener('click', function(){
  document.getElementById('sys-area').classList.toggle('open');
});

// textarea 자동 높이
inp.addEventListener('input', function(){
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 150) + 'px';
});

function toggleRagBody(el){
  el.nextElementSibling.classList.toggle('open');
}

function addMsg(role, content, ragContext, expCond){
  empty.style.display = 'none';
  var wrap = document.createElement('div');
  wrap.className = 'message ' + role;
  var av = document.createElement('div');
  av.className = 'avatar';
  av.innerHTML = role === 'user' ? '&#128100;' : '&#129302;';
  var msgWrap = document.createElement('div');
  msgWrap.className = 'msg-wrap';

  // 실험 조건 태그 (user 메시지에만)
  if(role === 'user' && expCond){
    var tag = document.createElement('div');
    tag.style.cssText = 'font-size:10px;color:var(--text-muted);margin-bottom:3px;text-align:right;';
    tag.innerHTML =
      '<span style="background:var(--surface2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;margin-left:4px">' +
      DOMAIN_LABELS[expCond.domain] + '</span>' +
      '<span style="background:var(--surface2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;margin-left:4px">' +
      DIFFICULTY_LABELS[expCond.difficulty] + '</span>' +
      '<span style="background:var(--surface2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;margin-left:4px">' +
      LAYOUT_LABELS[expCond.layout] + '</span>';
    msgWrap.appendChild(tag);
  }

  // RAG 컨텍스트 박스 (assistant에만)
  if(role === 'assistant' && ragContext){
    var ctxBox = document.createElement('div');
    ctxBox.className = 'rag-context';
    ctxBox.innerHTML =
      '<div class="rag-context-hdr" onclick="toggleRagBody(this)">&#128269; 참고 문서 ' +
      '<span style="margin-left:auto;font-size:10px">&#9660; 펼치기</span></div>' +
      '<div class="rag-context-body">' + escHtml(ragContext) + '</div>';
    msgWrap.appendChild(ctxBox);
  }

  var bub = document.createElement('div');
  bub.className = 'bubble';
  if(content) bub.textContent = content;
  msgWrap.appendChild(bub);

  wrap.appendChild(av);
  wrap.appendChild(msgWrap);
  msgs.appendChild(wrap);
  scrollBot();
  return bub;
}

function escHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function scrollBot(){
  requestAnimationFrame(function(){ msgs.scrollTop = msgs.scrollHeight; });
}

async function send(){
  var text = inp.value.trim();
  if(!text || streaming) return;
  inp.value = '';
  inp.style.height = 'auto';
  streaming = true;
  sBtn.disabled = true;

  addMsg('user', text, null, expState);

  // 로딩
  var loadWrap = document.createElement('div');
  loadWrap.className = 'message assistant';
  loadWrap.innerHTML = '<div class="avatar">&#129302;</div><div class="msg-wrap"><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div></div>';
  msgs.appendChild(loadWrap);
  scrollBot();

  var sys = document.getElementById('sysprompt').value.trim();
  var messages = sys ? [{role:'system',content:sys}, {role:'user',content:text}] : [{role:'user',content:text}];

  var payload = {
    messages: messages,
    temperature: +document.getElementById('temp').value,
    max_tokens: +document.getElementById('maxtok').value,
    top_p: +document.getElementById('topp').value,
    stream: true,
    use_rag: ragToggle.checked,
    domain:     expState.domain,
    difficulty: expState.difficulty,
    layout:     expState.layout
  };

  var fullText = '';
  var ragContext = '';

  try {
    var res = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error('서버 오류: ' + res.status);

    // 첫 번째 줄에서 RAG 컨텍스트 추출
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = '';
    var bubble = null;

    while(true){
      var chunk = await reader.read();
      if(chunk.done) break;
      buf += decoder.decode(chunk.value, {stream:true});
      var lines = buf.split('\\n');
      buf = lines.pop();

      for(var i=0;i<lines.length;i++){
        var line = lines[i].trim();
        if(!line || line === 'data: [DONE]' || !line.startsWith('data: ')) continue;
        try {
          var json = JSON.parse(line.slice(6));
          // RAG 컨텍스트 메타 청크
          if(json.rag_context !== undefined){
            ragContext = json.rag_context;
            continue;
          }
          var delta = (json.choices && json.choices[0].delta && json.choices[0].delta.content) || '';
          if(delta){
            if(!bubble){
              loadWrap.remove();
              bubble = addMsg('assistant', '', ragContext);
            }
            fullText += delta;
            bubble.textContent = fullText;
            scrollBot();
          }
        } catch(e){}
      }
    }

    if(!bubble){
      loadWrap.remove();
      bubble = addMsg('assistant', fullText || '(응답 없음)', ragContext);
    }

  } catch(err){
    loadWrap.remove();
    var eb = addMsg('assistant');
    eb.style.color = '#ff6b6b';
    eb.textContent = '오류: ' + err.message;
  } finally {
    streaming = false;
    sBtn.disabled = false;
    inp.focus();
  }
}

inp.addEventListener('keydown', function(e){
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});
sBtn.addEventListener('click', send);
document.getElementById('clear-btn').addEventListener('click', function(){
  chatHistory = [];
  msgs.innerHTML = '';
  msgs.appendChild(empty);
  empty.style.display = '';
});
</script>
</body>
</html>"""


# ─── 라우터 ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=CHAT_UI_HTML)

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{LLAMA_CPP_BASE_URL}/health")
            return {
                "gateway": "ok",
                "llama_cpp": "ok" if r.status_code == 200 else "unreachable",
                "vectorstore": "ok" if VECTORSTORE is not None else "not loaded",
            }
    except Exception as e:
        return JSONResponse(status_code=503, content={"gateway": "ok", "llama_cpp": "unreachable", "error": str(e)})

@app.post("/chat")
async def chat(req: ChatRequest):
    # ── 실험 조건 → system prompt 생성 ───────────────────────────
    exp_system = build_experiment_system_prompt(req.domain, req.difficulty, req.layout)

    # 기존 system 메시지가 있으면 실험 프롬프트를 앞에 prepend, 없으면 삽입
    messages = list(req.messages)
    if messages and messages[0].role == "system":
        merged = exp_system + "\n\n" + messages[0].content if exp_system else messages[0].content
        messages[0] = Message(role="system", content=merged)
    elif exp_system:
        messages.insert(0, Message(role="system", content=exp_system))

    # ── RAG: 마지막 user 메시지로 컨텍스트 검색 ──────────────────
    context = ""
    if req.use_rag and VECTORSTORE is not None:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if last_user:
            context = retrieve_context(last_user)

    # ── 메시지 구성 ───────────────────────────────────────────────
    if context:
        final_messages = build_rag_messages(messages, context, layout=req.layout)
    else:
        layout_instruction = {
            "deductive": "답변을 먼저 말하고 이유를 설명해줘.",
            "inductive": "근거를 먼저 설명하고 최종 답변을 마지막에 말해줘.",
            "free":      "",
        }.get(req.layout, "")
        final_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.role == "user" and layout_instruction:
                final_messages.append({"role": "user", "content": msg.content + f"\n\n[답변 형식] {layout_instruction}"})
            else:
                final_messages.append({"role": msg.role, "content": msg.content})

    payload = {
        "model": MODEL_NAME,
        "messages": final_messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "stream": req.stream,
    }
    if req.stop:
        payload["stop"] = req.stop

    # ── 스트리밍 ─────────────────────────────────────────────────
    if req.stream:
        import json as _json
        async def rag_stream():
            # RAG 컨텍스트를 첫 번째 SSE 청크로 전달
            if context:
                meta = _json.dumps({"rag_context": context}, ensure_ascii=False)
                yield f"data: {meta}\n\n"
            async for chunk in stream_llama(payload):
                yield chunk

        return StreamingResponse(rag_stream(), media_type="text/event-stream")

    # ── 일반 응답 ────────────────────────────────────────────────
    async with httpx.AsyncClient(base_url=LLAMA_CPP_BASE_URL, timeout=120.0) as client:
        try:
            r = await client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            result = r.json()
            if context:
                result["rag_context"] = context
            return result
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"llama.cpp 서버 연결 실패: {e}")

@app.get("/models")
async def list_models():
    return {"object":"list","data":[{"id":MODEL_NAME,"object":"model","owned_by":"llama.cpp"}]}

@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def openai_proxy(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(base_url=LLAMA_CPP_BASE_URL, timeout=120.0) as client:
        try:
            r = await client.request(method=request.method, url=f"/v1/{path}", content=body, headers=headers, params=dict(request.query_params))
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"llama.cpp 서버 연결 실패: {e}")