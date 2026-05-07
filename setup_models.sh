#!/usr/bin/env bash
# Setup llama.cpp with:
#   - unsloth/Qwen3.5-0.8B-GGUF     (chat/generation)   → port 30004
#   - ggml-org/bge-m3-Q8_0-GGUF     (embeddings)        → port 30005
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
MODELS_DIR="$SCRIPT_DIR/models"

QWEN_REPO="unsloth/Qwen3.5-0.8B-GGUF"
BGE_REPO="ggml-org/bge-m3-Q8_0-GGUF"

QWEN_PORT=30004
EMBEDDING_PORT=30005

SERVICE_USER="${SUDO_USER:-$USER}"

# Optional: set HF_TOKEN for gated/private models
# export HF_TOKEN="hf_..."

# ─────────────────────────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in git cmake make pip; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: missing dependencies: ${missing[*]}"
        echo "Install with: sudo apt install -y git cmake build-essential python3-pip"
        exit 1
    fi
}

ensure_hf_cli() {
    if ! command -v hf &>/dev/null; then
        echo ">>> Installing huggingface_hub (provides 'hf' CLI)..."
        pip install -q -U "huggingface_hub[cli]"
    fi
}

build_llama_cpp() {
    if [[ -f "$LLAMA_DIR/build/bin/llama-server" ]]; then
        echo ">>> llama.cpp already built, skipping."
        return
    fi

    if [[ ! -d "$LLAMA_DIR" ]]; then
        echo ">>> Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    echo ">>> Building llama.cpp..."
    cmake -B "$LLAMA_DIR/build" "$LLAMA_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
    echo ">>> Build complete: $LLAMA_DIR/build/bin/"
}

download_model() {
    local repo="$1"
    local local_dir="$MODELS_DIR/$repo"

    if [[ -d "$local_dir" ]] && ls "$local_dir"/*.gguf &>/dev/null 2>&1; then
        echo ">>> Already downloaded: $repo"
        return
    fi

    echo ">>> Downloading $repo..."
    hf download "$repo" --local-dir "$local_dir"
}

# Pick the best GGUF from a directory:
#   prefer Q4_K_M → Q5_K_M → Q8_0 → first found
pick_gguf() {
    local dir="$1"
    local f
    for quant in Q4_K_M Q5_K_M Q8_0; do
        f=$(find "$dir" -maxdepth 1 -iname "*${quant}*.gguf" | sort | head -1)
        [[ -n "$f" ]] && { echo "$f"; return; }
    done
    find "$dir" -maxdepth 1 -name "*.gguf" | sort | head -1
}

install_services() {
    local llama_server="$LLAMA_DIR/build/bin/llama-server"
    local qwen_model="$1"
    local bge_model="$2"

    if [[ ! -x "$llama_server" ]]; then
        echo "ERROR: llama-server not found at $llama_server"
        exit 1
    fi

    echo ">>> Writing systemd service files..."

    sudo tee /etc/systemd/system/llama-server.service > /dev/null <<SERVICE
[Unit]
Description=Qwen3.5-0.8B llama server (port ${QWEN_PORT})
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Restart=on-failure
RestartSec=5
ExecStart=${llama_server} \
    --model ${qwen_model} \
    --host 0.0.0.0 \
    --port ${QWEN_PORT} \
    --ctx-size 32768 \
    --n-predict -1 \
    --log-disable

[Install]
WantedBy=multi-user.target
SERVICE

    sudo tee /etc/systemd/system/llama-embedding.service > /dev/null <<SERVICE
[Unit]
Description=BGE-M3 embedding llama-server (port ${EMBEDDING_PORT})
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Restart=on-failure
RestartSec=5
ExecStart=${llama_server} \
    --model ${bge_model} \
    --host 0.0.0.0 \
    --port ${EMBEDDING_PORT} \
    --ctx-size 512 \
    --embedding \
    --log-disable

[Install]
WantedBy=multi-user.target
SERVICE

    echo ">>> Reloading systemd and enabling services..."
    sudo systemctl daemon-reload
    sudo systemctl enable llama-server.service
    sudo systemctl enable llama-embedding.service

    echo ">>> Starting services..."
    sudo systemctl restart llama-server.service
    sudo systemctl restart llama-embedding.service
}

# ─────────────────────────────────────────────────────────────────────────────
echo "=== llama.cpp model setup ==="

check_deps
ensure_hf_cli

mkdir -p "$MODELS_DIR"

build_llama_cpp

download_model "$QWEN_REPO"
download_model "$BGE_REPO"

QWEN_DIR="$MODELS_DIR/$QWEN_REPO"
BGE_DIR="$MODELS_DIR/$BGE_REPO"

QWEN_MODEL="$(pick_gguf "$QWEN_DIR")"
BGE_MODEL="$(pick_gguf "$BGE_DIR")"

if [[ -z "$QWEN_MODEL" ]]; then
    echo "ERROR: no .gguf file found in $QWEN_DIR"
    exit 1
fi
if [[ -z "$BGE_MODEL" ]]; then
    echo "ERROR: no .gguf file found in $BGE_DIR"
    exit 1
fi

echo ">>> Selected Qwen model  : $QWEN_MODEL"
echo ">>> Selected BGE model   : $BGE_MODEL"

install_services "$QWEN_MODEL" "$BGE_MODEL"

cat <<EOF

=== Setup complete ===
llama.cpp binaries : $LLAMA_DIR/build/bin/
Qwen model         : $QWEN_MODEL
BGE-M3 model       : $BGE_MODEL

Systemd services installed and started:
  llama-server      → http://localhost:${QWEN_PORT}  (chat completions)
  llama-embedding → http://localhost:${EMBEDDING_PORT}  (embeddings)

Manage with:
  sudo systemctl status  llama-server
  sudo systemctl status  llama-embedding
  sudo systemctl restart llama-server
  sudo systemctl restart llama-embedding
  sudo journalctl -u llama-server      -f
  sudo journalctl -u llama-embedding -f
EOF
