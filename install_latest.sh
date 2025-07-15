#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  install.sh – Video‑Manager + Real‑ESRGAN (robust, idempotent, OFFLINE)
#  v10 – *Jede* PyTorch‑/Vision‑Variante (GPU **und** CPU) vorher via aria2c
#        herunterladen; optionaler Schalter --cuda-toolkit installiert das
#        systemweite NVIDIA CUDA Toolkit 11.8 (.deb‑local).
# -----------------------------------------------------------------------------
# Aufrufbeispiele
#   ./install.sh               # Standard (GPU nutzt cu118‑Wheels, CPU +cpu)
#   ./install.sh --cuda-toolkit # plus Toolkit‑Pakete (drm, nvidia‑smi etc.)
# -----------------------------------------------------------------------------
set -euo pipefail


# ───────────────────────────── Interaktiver Pfad‑Dialog ─────────────────────
DEFAULT_BASE="$HOME/syscripts/videoManager"
read -erp "Installationsverzeichnis [$DEFAULT_BASE]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_BASE}"
VENV_DIR="$INSTALL_DIR/venv"

DEFAULT_WHEEL_DIR="$INSTALL_DIR/wheels"
read -erp "Wheel‑Download‑Ordner [$DEFAULT_WHEEL_DIR]: " WHEELS_DIR
WHEELS_DIR="${WHEELS_DIR:-$DEFAULT_WHEEL_DIR}"
mkdir -p "$INSTALL_DIR" "$WHEELS_DIR"

# ───────────────────────────── Konfiguration ────────────────────────────────
PY_VERSION="3.11" ; PY_ABI="cp311-cp311"
CUDA_TAG="cu118"   ; TORCH_VER="2.2.1" ; VISION_VER="0.17.1"
VIDEO_CMD="video"
ALIAS_LINE="alias $VIDEO_CMD=\"$VENV_DIR/bin/python $INSTALL_DIR/videoManager.py\""

INSTALL_CUDA_TOOLKIT=false
for a in "$@"; do [[ "$a" == "--cuda-toolkit" ]] && INSTALL_CUDA_TOOLKIT=true; done

# ───────────────────────────── Hilfsfunktionen ──────────────────────────────
log()  { echo -e "\e[32m[install]\e[0m $*"; }
warn() { echo -e "\e[33m[install]\e[0m $*"; }
err()  { echo -e "\e[31m[install]\e[0m $*" >&2; exit 1; }
#aria() { aria2c -x8 -s8 -c -d "$WHEELS_DIR" -o "$2" "$1"; }
aria_dl() { aria2c -x8 -s8 -c -d "$WHEELS_DIR" -o "$2" "$1"; }
get_wheel() {
  local url="$1" file="$2"
  if [ -f "$WHEELS_DIR/$file" ]; then
    log "⏩ Skip – $file bereits vorhanden."
  else
    log "⬇️  Lade $file …"
    aria_dl "$url" "$file"
  fi
}
#get_wheel() { local url="$1" f="$2"; [ -f "$WHEELS_DIR/$f" ] || aria "$url" "$f"; }
install_offline() { install_with_retry --no-index --find-links "$WHEELS_DIR" "$@"; }
install_with_retry() { local m=3 n=1; while (( n<=m )); do pip install -q "$@" && return || warn "pip $* retry $n/$m"; ((n++)); done; err "pip $* failed"; }


install_cuda_pkg() {
  local pkg="$1"
  if [[ -z "$pkg" ]]; then warn "install_cuda_pkg wurde ohne Argument aufgerufen – überspringe."; return 1; fi
  local name="${pkg%%==*}"
  local ver="${pkg##*==}"

  # Zwei Basis-Verzeichnisse: neues Layout + Fallback
  local bases=("https://download.pytorch.org/whl/${CUDA_TAG}" "https://download.pytorch.org/whl")
  local abis=(manylinux2014_x86_64 manylinux1_x86_64)
  local variants=("${name//-/_}" "${name//_/-}")

  local fetched="" url f
  for base in "${bases[@]}"; do
    for abi in "${abis[@]}"; do
      for n in "${variants[@]}"; do
        f="${n}-${ver}-py3-none-${abi}.whl"; url="${base}/${f}"
        if get_wheel "$url" "$f"; then fetched="$f"; break 3; fi
      done; done; done

  if [[ -z "$fetched" ]]; then warn "Wheel $pkg nicht gefunden, übersprungen."; return 1; fi
  install_with_retry --no-index --find-links "$WHEELS_DIR" "$pkg"
}

# ───────────────────────────── CUDA‑Erkennung ───────────────────────────────
CUDA_DETECTED=false
if nvidia-smi -L 2>/dev/null | grep -qi "GPU"; then CUDA_DETECTED=true; fi
$CUDA_DETECTED && log "CUDA‑fähige GPU erkannt." || warn "Keine CUDA‑GPU erkannt – CPU‑Pfad wird gewählt."

# ───────────────────────────── System‑Pakete ────────────────────────────────
cmd() { command -v "$1" >/dev/null; }
cmd aria2c || { sudo apt update -qq && sudo apt install -y aria2; }
cmd curl   || sudo apt install -y curl
cmd unzip  || sudo apt install -y unzip
cmd "python${PY_VERSION}" || { sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update -qq && sudo apt install -y "python${PY_VERSION}" "python${PY_VERSION}-venv"; }

# optional: CUDA Toolkit (nur falls Flag gesetzt)
if $INSTALL_CUDA_TOOLKIT; then
  log "Installiere NVIDIA CUDA Toolkit 11.8 (deb‑local) …"
  wget -qO /tmp/cuda-repo.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
  sudo mv /tmp/cuda-repo.deb /etc/apt/preferences.d/cuda-repository-pin-600
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y cuda-toolkit-11-8
fi

#======================== Alias Setup ============================
if ! grep -qs "$ALIAS_LINE" "$HOME/.bashrc" "$HOME/.bash_aliases" 2>/dev/null; then
  echo -e "\nAlias '$VIDEO_CMD' not found. Where should it be added?"; select d in "$HOME/.bashrc" "$HOME/.bash_aliases" "Custom" "None"; do
    case $REPLY in
      1|2) echo "$ALIAS_LINE" >> "$d"; break ;;
      3) read -rp "Pfad: " p; echo "$ALIAS_LINE" >> "$p"; break ;;
      *) break ;;
    esac; done
fi


# ───────────────────────────── System-Build-Tools ───────────────────────────
# Installiere alles, was meson-python, NumPy und Co. zum Bauen brauchen
# ──────────── System-Build-Tools via Apt ─────────────
if ! dpkg -s ninja-build patchelf python3-setuptools python3-wheel meson python3-pyproject-metadata >/dev/null 2>&1; then
  log "Installiere System-Build-Tools via apt…"
  sudo apt-get update -qq
  sudo apt-get install -y \
    ninja-build \
    patchelf \
    python3-setuptools \
    python3-wheel \
    meson \
    python3-pyproject-metadata
fi

# ──────────── System-Bild-Codecs via Apt ─────────────
if ! dpkg -s libjpeg-dev libpng-dev >/dev/null 2>&1; then
  log "Installiere JPEG/PNG Headers via apt…"
  sudo apt-get update -qq
  sudo apt-get install -y \
    libjpeg-dev \
    libpng-dev
fi


# ───────────────────────────── Virtualenv ───────────────────────────────────
[ -d "$VENV_DIR" ] || "python${PY_VERSION}" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip wheel

# ───────────────────────────── PyTorch / Vision offline ─────────────────────
if $CUDA_DETECTED; then
  # ---- GPU Wheels ----------------------------------------------------------
  TORCH_WHL="torch-${TORCH_VER}+${CUDA_TAG}-${PY_ABI}-linux_x86_64.whl"
  get_wheel "https://download.pytorch.org/whl/${CUDA_TAG}/torch-${TORCH_VER}%2B${CUDA_TAG}-${PY_ABI}-linux_x86_64.whl" "$TORCH_WHL"
  install_with_retry --no-index --no-deps --find-links "$WHEELS_DIR" "$WHEELS_DIR/$TORCH_WHL"

  CU_PKGS=(
    nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6
    nvidia-cuda-cupti-cu11==11.8.87  nvidia-cudnn-cu11==8.7.0.84
    nvidia-cufft-cu11==10.9.0.58     nvidia-curand-cu11==10.3.0.86
    nvidia-cusolver-cu11==11.4.1.48  nvidia-cusparse-cu11==11.7.5.86
    nvidia-nccl-cu11==2.19.3         nvidia-nvtx-cu11==11.8.86 )
  for p in "${CU_PKGS[@]}"; do install_cuda_pkg "$p"; done

  pip install -q numpy
  TV_WHL="torchvision-${VISION_VER}+${CUDA_TAG}-${PY_ABI}-linux_x86_64.whl"
  get_wheel "https://download.pytorch.org/whl/${CUDA_TAG}/torchvision-${VISION_VER}%2B${CUDA_TAG}-${PY_ABI}-linux_x86_64.whl" "$TV_WHL"
  install_with_retry --no-index --no-deps --find-links "$WHEELS_DIR" "$WHEELS_DIR/$TV_WHL"
else
  # ---- CPU Wheels ----------------------------------------------------------
  TORCH_WHL="torch-${TORCH_VER}+cpu-${PY_ABI}-linux_x86_64.whl"
  get_wheel "https://download.pytorch.org/whl/cpu/torch-${TORCH_VER}%2Bcpu-${PY_ABI}-linux_x86_64.whl" "$TORCH_WHL"
  install_with_retry --no-index --no-deps --find-links "$WHEELS_DIR" "$WHEELS_DIR/$TORCH_WHL"

  pip install -q numpy
  TV_WHL="torchvision-${VISION_VER}+cpu-${PY_ABI}-linux_x86_64.whl"
  get_wheel "https://download.pytorch.org/whl/cpu/torchvision-${VISION_VER}%2Bcpu-${PY_ABI}-linux_x86_64.whl" "$TV_WHL"
  install_with_retry --no-index --no-deps --find-links "$WHEELS_DIR" "$WHEELS_DIR/$TV_WHL"
fi


# ───────────────────────────── Pure‑Python Wheels offline ────────────────────
#PY_PKGS=(Cython==0.29.36 meson-python==0.15.0 MarkupSafe==2.1.5 mpmath==1.3.0 future==0.18.3 ffmpeg-python==0.2.0 \
#         filelock==3.14.0 fsspec==2024.3.1 jinja2==3.1.4 networkx==3.3 sympy==1.12 \
#         typing_extensions==4.11.0 numpy==1.26.4 pillow==10.3.0 requests==2.32.2 \
#         kiwisolver==1.4.8 llvmlite==0.44.0)

#PY_PKGS=(Cython==0.29.36 packaging==23.2 pyproject-metadata==0.7.1 meson==1.4.0 meson-python==0.16.0 \\
#         MarkupSafe==2.1.5 mpmath==1.3.0 future==0.18.3 ffmpeg-python==0.2.0 filelock==3.14.0 \\
#         fsspec==2024.3.1 jinja2==3.1.4 networkx==3.3 sympy==1.12 typing_extensions==4.11.0 \\
#         numpy==1.26.4 pillow==10.3.0 requests==2.32.2 kiwisolver==1.4.8 llvmlite==0.44.0)

PY_PKGS=(
  MarkupSafe==2.1.5
  mpmath==1.3.0
  future==0.18.3
  ffmpeg-python==0.2.0
  filelock==3.14.0
  fsspec==2024.3.1
  jinja2==3.1.4
  networkx==3.3
  sympy==1.12
  typing_extensions==4.11.0
  numpy==1.26.4
  pillow==10.3.0
  charset-normalizer==2.1.1
  idna==3.4
  requests==2.32.2
  kiwisolver==1.4.8
  llvmlite==0.44.0
)

for p in "${PY_PKGS[@]}"; do
  pkg="${p%%==*}"; ver="${p##*==}"; dir="${pkg//-/_}"
  uni="${dir}-${ver}-py3-none-any.whl"
  # universelles Wheel
  if ! get_wheel "https://files.pythonhosted.org/packages/py3/${dir:0:1}/$dir/$uni" "$uni"; then
    # try generic linux_x86_64 wheel pattern (numpy etc.)
    arch1="${dir}-${ver}-${PY_ABI}-linux_x86_64.whl"
    arch2="${dir}-${ver}-${PY_ABI}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    if ! get_wheel "https://files.pythonhosted.org/packages/${dir:0:1}/$dir/$arch1" "$arch1"; then
      if ! get_wheel "https://files.pythonhosted.org/packages/${dir:0:1}/$dir/$arch2" "$arch2"; then
        src="${dir}-${ver}.tar.gz"
        get_wheel "https://files.pythonhosted.org/packages/source/${dir:0:1}/$dir/$src" "$src"
      fi
    fi
  fi
  if [[ "$pkg" == "numpy" ]]; then
    # Für NumPy: nur binary wheels nehmen (kein Source-Build)
    pip install --quiet --no-index --only-binary=:all: \
      --find-links "$WHEELS_DIR" "$pkg==$ver" \
      || warn "Binary-Wheel für $pkg==$ver nicht verfügbar"
  else
    install_offline "$p" || warn "$p konnte offline nicht installiert werden"
  fi
done

deactivate

log "✅ Basis‑Installation abgeschlossen – starte das Tool mit '$VIDEO_CMD'."