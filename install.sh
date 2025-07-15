#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  install.sh – Video‑Manager + Real‑ESRGAN (Conda‑basiert)
#  Unterstützt Linux und macOS; für Windows bitte "Anaconda Prompt" nutzen.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="videoManager"
PYTHON_VERSION="3.11"
TORCH_VER="2.2.1"
VISION_VER="0.17.1"
VIDEO_CMD="video"

# ───────────────────────────── Interaktiver Pfad‑Dialog ─────────────────────
DEFAULT_BASE="$HOME/syscripts/videoManager"
read -erp "Installationsverzeichnis [$DEFAULT_BASE]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_BASE}"
VENV_DIR="$INSTALL_DIR/venv"

ALIAS_LINE="alias $VIDEO_CMD=\"$VENV_DIR/bin/python $INSTALL_DIR/videoManager.py\""


INSTALL_CUDA_TOOLKIT=false
for a in "$@"; do [[ "$a" == "--cuda-toolkit" ]] && INSTALL_CUDA_TOOLKIT=true; done

# optional: CUDA Toolkit (nur falls Flag gesetzt)
if $INSTALL_CUDA_TOOLKIT; then
  log "Installiere NVIDIA CUDA Toolkit 11.8 (deb‑local) …"
  wget -qO /tmp/cuda-repo.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
  sudo mv /tmp/cuda-repo.deb /etc/apt/preferences.d/cuda-repository-pin-600
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y cuda-toolkit-11-8
fi

# ───────────────────────────── Hilfsfunktionen ──────────────────────────────
log()  { echo -e "\e[32m[install]\e[0m $*"; }
warn() { echo -e "\e[33m[install]\e[0m $*"; }
err()  { echo -e "\e[31m[install]\e[0m $*" >&2; exit 1; }


#======================== Alias Setup ============================
if ! grep -qs "$ALIAS_LINE" "$HOME/.bashrc" "$HOME/.bash_aliases" 2>/dev/null; then
  echo -e "\nAlias '$VIDEO_CMD' not found. Where should it be added?"; select d in "$HOME/.bashrc" "$HOME/.bash_aliases" "Custom" "None"; do
    case $REPLY in
      1|2) echo "$ALIAS_LINE" >> "$d"; break ;;
      3) read -rp "Pfad: " p; echo "$ALIAS_LINE" >> "$p"; break ;;
      *) break ;;
    esac; done
fi

# ───────────────────── Miniconda Installation prüfen ─────────────────────
if ! command -v conda &>/dev/null && ! command -v mamba &>/dev/null; then
  log "🔄 Conda nicht gefunden – installiere Miniconda lokal..."
  INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  URL="https://repo.anaconda.com/miniconda/$INSTALLER"
  curl -fsSL "$URL" -o "/tmp/$INSTALLER"
  bash "/tmp/$INSTALLER" -b -p "$HOME/miniconda"
  rm "/tmp/$INSTALLER"
  export PATH="$HOME/miniconda/bin:$PATH"
fi
# Prefer mamba if available
CMD_INSTALL="conda"
if command -v mamba &>/dev/null; then
  CMD_INSTALL="mamba"
fi

# Initialize conda in script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ───────────────────── Create/Activate Env ───────────────────────────
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "🆕 Erstelle Conda-Environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
  $CMD_INSTALL create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

echo "🔁 Activated environment '$ENV_NAME'"

# ───────────────────── GPU-/CPU-Paketwahl ────────────────────────────
GPU=false
if command -v nvidia-smi &>/dev/null; then
  GPU=true
  echo "✅ CUDA-fähige GPU erkannt"
else
  echo "⚠️  Keine CUDA-GPU erkannt – nutze CPU-only"
fi

# ───────────────────── PyTorch & Vision ──────────────────────────────
if [ "$GPU" = true ]; then
  $CMD_INSTALL install -y pytorch="$TORCH_VER" torchvision="$VISION_VER" cudatoolkit=11.8 -c pytorch -c conda-forge
else
  $CMD_INSTALL install -y pytorch="$TORCH_VER" torchvision="$VISION_VER" cpuonly -c pytorch -c conda-forge
fi

# ───────────────────── Sonstige Bibliotheken ─────────────────────────
$CMD_INSTALL install -y \
  ffmpeg pillow networkx sympy jinja2 fsspec filelock requests kiwisolver llvmlite -c conda-forge

# ───────────────────── Real-ESRGAN Setup ─────────────────────────────
RE_DIR="$HOME/real-esrgan"
if [ -d "$RE_DIR" ]; then
  git -C "$RE_DIR" pull --quiet
else
  git clone --quiet https://github.com/xinntao/Real-ESRGAN "$RE_DIR"
fi
python -m pip install -q -r "$RE_DIR/requirements.txt"
python "$RE_DIR/scripts/download_pretrained_models.py" >/dev/null

# ───────────────────── Alias einrichten ──────────────────────────────
CONDA_PREFIX="$(conda info --base)/envs/$ENV_NAME"
ALIAS_LINE="alias $VIDEO_CMD='${CONDA_PREFIX}/bin/python $PWD/videoManager.py'"
if ! grep -qxF "$ALIAS_LINE" ~/.bashrc ~/.zshrc 2>/dev/null; then
  echo "$ALIAS_LINE" >> ~/.bashrc
  echo "Alias '$VIDEO_CMD' in ~/.bashrc hinzugefügt"
fi

echo "🎉 Fertig! Aktiviere mit: conda activate $ENV_NAME"