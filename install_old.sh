#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  install.sh – idempotente Installation des Video‑Managers + Real‑ESRGAN
# -----------------------------------------------------------------------------
#  • legt (falls nötig) Python 3.11 via apt nach (Ubuntu/Debian)
#  • erzeugt/aktualisiert ein venv mit Python 3.11
#  • installiert Abhängigkeiten (ffmpeg‑python, numpy, torch/torchvision passend)
#  • erkennt CUDA‑GPU → native Real‑ESRGAN | sonst ncnn‑Vulkan‑Binary
#  • skript kann gefahrlos erneut ausgeführt werden (überspringt vorhandenes)
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- Konfiguration -----------------------------------
INSTALL_DIR="$HOME/syscripts/videoManager"
VENV_DIR="$INSTALL_DIR/venv"
PY_VERSION="3.11"
VIDEO_CMD="video"
SCRIPT_PATH="$VENV_DIR/bin/python $INSTALL_DIR/videoManager.py"
ALIAS_LINE="alias $VIDEO_CMD=\"$SCRIPT_PATH\""


# --------------------------- Hilfsfunktionen ---------------------------------
log() { echo -e "\e[1;32m[install]\e[0m $*"; }
warn() { echo -e "\e[1;33m[install]\e[0m $*"; }
err() { echo -e "\e[1;31m[install]\e[0m $*" >&2; exit 1; }

apt_install() {
  sudo apt-get update -qq && sudo apt-get install -y "$@"
}

#python_exec() { "$VENV_DIR/bin/python" - "$@"; }


# --------------------------- CUDA (GPU) Detection ---------------------------
CUDA_DETECTED=false
if command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep -qi "GPU"; then
  CUDA_DETECTED=true
else
  # Fallback: char‑device present → Treiber läuft
  if [ -c /dev/nvidiactl ] || [ -c /dev/nvidia0 ]; then
    CUDA_DETECTED=true
  fi
fi

$CUDA_DETECTED && log "CUDA‑fähige GPU erkannt." || warn "Keine CUDA‑GPU erkannt – CPU‑Pfad wird genutzt."


#======================== Alias Setup ============================
# Check if alias already exists
if grep -qs "$ALIAS_LINE" "$HOME/.bashrc" || grep -qs "$ALIAS_LINE" "$HOME/.bash_aliases"; then
    echo "Alias '$VIDEO_CMD' already exists in .bashrc or .bash_aliases. Skipping alias creation."
else
    echo "Alias '$VIDEO_CMD' not found. Where should it be added?"
    echo "1) ~/.bashrc"
    echo "2) ~/.bash_aliases"
    echo "3) Custom file path"
    echo "4) Do not set alias"
    read -p "Choose [1-4]: " choice

    case $choice in
        1)
            echo "$ALIAS_LINE" >> "$HOME/.bashrc"
            echo "Alias added to ~/.bashrc."
            ;;
        2)
            echo "$ALIAS_LINE" >> "$HOME/.bash_aliases"
            echo "Alias added to ~/.bash_aliases."
            ;;
        3)
            read -p "Enter custom file path: " userfile
            echo "$ALIAS_LINE" >> "$userfile"
            echo "Alias added to $userfile."
            ;;
        4)
            echo "Alias will not be set."
            ;;
        *)
            echo "Invalid input. Alias will not be set."
            ;;
    esac
fi

# --------------------------- Python 3.11 beschaffen --------------------------
if ! command -v "python${PY_VERSION}" &>/dev/null; then
  warn "Python ${PY_VERSION} fehlt – installiere über Deadsnakes PPA …"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  apt_install "python${PY_VERSION}" "python${PY_VERSION}-venv" "python${PY_VERSION}-dev"
fi

# --------------------------- venv erstellen/aktualisieren --------------------
if [ ! -d "$VENV_DIR" ]; then
  log "Erstelle virtuelles Environment in $VENV_DIR (Python ${PY_VERSION}) …"
  "python${PY_VERSION}" -m venv "$VENV_DIR"
else
  log "venv existiert bereits – überspringe Erstellung."
fi

# --------------------------- Basis‑Pakete ------------------------------------
source "$VENV_DIR/bin/activate"
log "pip aktualisieren …"
pip install --upgrade pip >/dev/null

log "Allgemeine Abhängigkeiten installieren …"
pip install -q --upgrade ffmpeg-python numpy

# --------------------------- Torch / Torchvision -----------------------------
TORCH_OK=false
python - <<'PY' && TORCH_OK=true || true
import sys
try:
    import torch, torchvision
    from torchvision.transforms.functional import rgb_to_grayscale  # noqa
    print("torch import OK, version", torch.__version__)
except Exception as e:
    sys.exit(1)
PY

if ! $TORCH_OK; then
  log "Installiere Torch/Torchvision passend zu System …"
  pip uninstall -y torch torchvision >/dev/null 2>&1 || true
  if $CUDA_DETECTED; then
    CUDA_SUFFIX="cu118"
    pip install -q torch==1.13.1+${CUDA_SUFFIX} torchvision==0.14.1+${CUDA_SUFFIX} \
      --index-url https://download.pytorch.org/whl/${CUDA_SUFFIX}
  else
    pip install -q torch==1.13.1+cpu torchvision==0.14.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu
  fi
fi

deactivate

# --------------------------- Real‑ESRGAN Installation ------------------------
log "Installiere (oder aktualisiere) Real‑ESRGAN …"

if $CUDA_DETECTED; then
  RE_DIR="$INSTALL_DIR/Real-ESRGAN"
  if [ ! -d "$RE_DIR" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN "$RE_DIR"
  else
    git -C "$RE_DIR" pull --quiet
  fi

  source "$VENV_DIR/bin/activate"
  pip install -q -r "$RE_DIR/requirements.txt"
  "$VENV_DIR/bin/python" "$RE_DIR/scripts/download_pretrained_models.py" >/dev/null
  deactivate
  log "Real‑ESRGAN GPU‑Version bereit."
else
  NCNN_DIR="$INSTALL_DIR/Real-ESRGAN-ncnn"
  if [ ! -d "$NCNN_DIR" ]; then
    mkdir -p "$NCNN_DIR"
    cd "$NCNN_DIR"
    NCNN_URL=$(curl -sSL https://api.github.com/repos/xinntao/Real-ESRGAN-ncnn-vulkan/releases/latest | \
               grep browser_download_url | grep linux | cut -d '"' -f4 | head -n1)
    log "Lade NCNN‑Binary: ${NCNN_URL##*/} …"
    curl -L -o realesrgan-ncnn.zip "$NCNN_URL"
    unzip -q realesrgan-ncnn.zip && rm realesrgan-ncnn.zip
  else
    log "NCNN‑Binary existiert bereits – überspringe Download."
  fi
  log "Real‑ESRGAN NCNN‑Version bereit."
fi

log "✅ Installation abgeschlossen. Starte das Tool mit 'video'."