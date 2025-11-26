#!/usr/bin/env bash
# This file is part of PhotonFrame - VideoKit.
# Copyright (c) 2024–2025 David <…>
# Licensed under the MIT License. See LICENSE in the project root for details.
# -----------------------------------------------------------------------------
#  install.sh - PhotonFrame - VideoKit + Real-ESRGAN/RealCUGAN + (optional) BasicVSR++
#  - Language prompt (en/de) at start
#  - OS/GPU detection -> auto backend selection (torch|ncnn) with CPU/GPU pref
#  - Model downloads chosen per OS/Hardware (PyTorch or NCNN), always via the
#    robust download ladder (resume, retries, fallbacks) to work on bad links
#  - BasicVSR++ (Torch) optional; installs mmengine/mmcv/mmagic + tries weights
#  - At the end: writes ~/.config/PhotonFrameVideoKit/config.ini with detected choices
#  - Safe: does NOT touch system GPU drivers; favors offline/cached installs
# -----------------------------------------------------------------------------
set -Eeuo pipefail
set -o errtrace
trap 'rc=$?; echo -e "\e[31m[install]\e[0m Error at line $LINENO: \"$BASH_COMMAND\" (rc=$rc)"; exit $rc' ERR
[[ "${DEBUG_INSTALL:-0}" = "1" ]] && set -x

export PYTHONWARNINGS="ignore::DeprecationWarning"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
export CONDA_MKL_INTERFACE_LAYER_BACKUP="${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}"

ENV_NAME="PhotonFrameVideoKit"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.*}"   # conda will pick a recent 3.10.x
VIDEO_CMD="video"

GREEN='\e[1;36m'; CEND='\e[0m'

# ───────────────────────────── Language handling ────────────────────────────
VM_LANG="en"
IMPACT_OPT_IN="${IMPACT_OPT_IN:-}"
declare -A INSTALL_I18N_DE=(
  ["Language set to %s."]="Sprache gesetzt auf %s."
  ["Prefer GPU acceleration if available?"]="GPU-Beschleunigung bevorzugen, falls verfuegbar?"
  ["User prefers GPU (if available)."]="Nutzer bevorzugt GPU (falls verfuegbar)."
  ["User prefers CPU."]="Nutzer bevorzugt CPU."
  ["Install AI features (Real-ESRGAN, RealCUGAN, PyTorch, CUDA, ...)?"]="KI-Funktionen installieren (Real-ESRGAN, RealCUGAN, PyTorch, CUDA, ...)?"
  ["Install CodeFormer (face restoration, NON-COMMERCIAL S-Lab License 1.0)?"]="CodeFormer installieren (Gesichtsrestauration, nicht-kommerzielle S-Lab-Lizenz 1.0)?"
  ["CodeFormer requires the NON-COMMERCIAL S-Lab License 1.0. Do not use it commercially."]="CodeFormer unterliegt der nicht-kommerziellen S-Lab-Lizenz 1.0. Keine kommerzielle Nutzung."
  ["CodeFormer installation enabled."]="CodeFormer-Installation aktiviert."
  ["CodeFormer installation skipped (non-commercial license)."]="CodeFormer-Installation ausgelassen (nicht-kommerzielle Lizenz)."
  ["Skipping CodeFormer (non-commercial license)."]="CodeFormer wird aufgrund der nicht-kommerziellen Lizenz ausgelassen."
  ["CodeFormer is not installed - skipping weight download."]="CodeFormer ist nicht installiert - Gewichte werden ausgelassen."
  ["Download Impact font (Impact 2.35) now? This requires accepting the Microsoft Core Fonts EULA."]="Impact-Schrift (Impact 2.35) jetzt herunterladen? Dafuer muss die Microsoft-Core-Fonts-EULA akzeptiert werden."
  ["Impact font download skipped. GIF captions will use fallback fonts."]="Impact-Schrift-Download uebersprungen. GIF-Texte verwenden Ersatzschrift."
  ["cabextract missing - please install it manually to extract Microsoft Core Fonts."]="cabextract fehlt - bitte manuell installieren, um die Microsoft-Core-Fonts zu extrahieren."
  ["No supported package manager for cabextract detected - please install it manually."]="Kein unterstuetzter Paketmanager fuer cabextract erkannt - bitte manuell installieren."
  ["cabextract is required but could not be installed automatically."]="cabextract wird benoetigt, konnte aber nicht automatisch installiert werden."
  ["Homebrew not found – please install cabextract manually (https://www.cabextract.org.uk/)."]="Homebrew nicht gefunden – bitte cabextract manuell installieren (https://www.cabextract.org.uk/)."
  ["Impact installer download failed (%s)."]="Impact-Installer konnte nicht geladen werden (%s)."
  ["cabextract could not extract impact.ttf - please check manually."]="cabextract konnte impact.ttf nicht extrahieren - bitte manuell pruefen."
  ["impact.ttf missing in archive - installation skipped."]="impact.ttf wurde nicht gefunden - Installation uebersprungen."
  ["Impact font already present: %s"]="Impact-Schrift bereits vorhanden: %s"
  ["Impact font installed: %s"]="Impact-Schrift installiert: %s"
  ["cabextract missing - Impact cannot be installed automatically. Please install the font manually."]="cabextract fehlt - Impact kann nicht automatisch installiert werden. Bitte Schrift manuell installieren."
  ["No supported Linux package manager detected – please install ExifTool manually."]="Kein unterstuetzter Linux-Paketmanager erkannt – bitte ExifTool manuell installieren."
  ["Installing ExifTool inside Conda env: %s"]="Installiere ExifTool im Conda-Env: %s"
  ["ExifTool (Conda) OK: %s"]="ExifTool (Conda) OK: %s"
  ["Conda installation of ExifTool failed - trying system package."]="Conda-Installation von ExifTool fehlgeschlagen - Systempaket wird versucht."
  ["Homebrew not found - install Homebrew or use the official ExifTool pkg."]="Homebrew nicht gefunden - bitte Homebrew installieren oder das offizielle ExifTool-Paket verwenden."
["ExifTool already present: %s"]="ExifTool bereits vorhanden: %s"
["ExifTool installed: %s"]="ExifTool installiert: %s"
["ExifTool could not be installed."]="ExifTool konnte nicht installiert werden."
["Windows users: please open an elevated PowerShell and run 'choco install exiftool' or 'scoop install exiftool'."]="Windows: bitte PowerShell mit Administratorrechten oeffnen und 'choco install exiftool' oder 'scoop install exiftool' ausfuehren."
["Hint: your realesrgan-ncnn-vulkan wrapper does not pass a -m path."]="Hinweis: Dein realesrgan-ncnn-vulkan-Wrapper uebergibt keinen -m Pfad."
["You can add it like this:  exec \".../realesrgan-ncnn-vulkan\" -m \"\$MODELS_DIR\" \"\$@\""]="Du kannst ihn so ergaenzen:  exec \".../realesrgan-ncnn-vulkan\" -m \"\$MODELS_DIR\" \"\$@\""
["Environment exports written for face models."]="Umgebungsvariablen fuer Face-Modelle gesetzt."
["GFPGAN weights mirrored to %s (including v1.3 symlink)."]="GFPGAN-Gewichte nach %s kopiert (inkl. v1.3-Symlink)."
["CodeFormer weight mirrored to %s."]="CodeFormer-Gewicht nach %s kopiert."
  ["[impact] The Impact typeface is part of Microsoft's Core fonts for the Web."]="[impact] Die Impact-Schrift gehoert zu Microsofts Core Fonts fuer das Web."
  ["[impact] To comply with the EULA we only download the original impact32.exe from SourceForge after explicit consent."]="[impact] Zur Einhaltung der EULA laden wir impact32.exe nur nach ausdruecklicher Zustimmung von SourceForge."
  ["[impact] License text: https://sourceforge.net/projects/corefonts/files/ (Microsoft Core Fonts EULA)."]="[impact] Lizenztext: https://sourceforge.net/projects/corefonts/files/ (Microsoft-Core-Fonts-EULA)."
  ["Downloading Impact installer -> %s"]="Impact-Installer wird geladen -> %s"
  ["Extracting impact.ttf from impact32.exe via cabextract..."]="impact.ttf wird via cabextract aus impact32.exe extrahiert..."
  ["CodeFormer weight missing."]="CodeFormer-Gewicht fehlt."
  ["GFPGANv1.4 missing."]="GFPGANv1.4 fehlt."
  ["Skipping Impact/Corefonts download because VIDEO_NO_PROPRIETARY_FONTS=1."]="Impact/Corefonts-Download uebersprungen, da VIDEO_NO_PROPRIETARY_FONTS=1."
  ["[y/n] (default %s):"]="[j/n] (Standard %s):"
  ["config.ini copied from src.tar.gz -> %s"]="config.ini aus src.tar.gz uebernommen -> %s"
  ["Existing config.ini kept or none found in archive."]="Bestehende config.ini beibehalten oder keine im Archiv gefunden."
  ["Wrote THIRD_PARTY_LICENSES.md to %s"]="THIRD_PARTY_LICENSES.md nach %s geschrieben"
  ["Third-party license summary updated."]="Drittanbieter-Lizenzuebersicht aktualisiert."
)

loc() {
  local key="$1"
  if [[ "$VM_LANG" == "de" ]]; then
    printf "%s" "${INSTALL_I18N_DE[$key]:-$key}"
  else
    printf "%s" "$key"
  fi
}

loc_printf() {
  local fmt
  fmt="$(loc "$1")"
  shift
  printf "$fmt" "$@"
}

log()  { echo -e "\e[32m[install]\e[0m $(loc "$*")"; }
warn() { echo -e "\e[33m[install]\e[0m $(loc "$*")" >&2; }
err()  { echo -e "\e[31m[install]\e[0m $(loc "$*")" >&2; exit 1; }

# ───────────────────────────── Language prompt (EN/DE) ──────────────────────
ask_language() {
  local ans
  echo -ne "${GREEN}Select language for PhotonFrame - VideoKit / Sprache fuer PhotonFrame - VideoKit waehlen [1] English / [2] Deutsch (default: 1): ${CEND}"
  read -r ans || ans=""
  case "${ans:-1}" in
    2) VM_LANG="de" ;;
    *) VM_LANG="en" ;;
  esac
  log "$(loc_printf "Language set to %s." "${VM_LANG}")"
}

# ───────────────────────────── OS-/GPU detection ────────────────────────────
detect_os_name() {
  local s="$(uname -s | tr '[:upper:]' '[:lower:]')"
  if [[ "$s" == *linux* ]]; then echo "linux"
  elif [[ "$s" == *darwin* ]]; then echo "mac"
  elif [[ "$s" == *mingw* || "$s" == *msys* || "$s" == *cygwin* ]]; then echo "windows"
  else echo "linux"; fi
}

detect_gpu_backend() {
  # echo "<has_gpu> <backend_guess>"
  if command -v nvidia-smi &>/dev/null; then
    echo "true cuda"; return
  fi
  if [[ "$(detect_os_name)" == "mac" ]]; then
    echo "true mps"; return
  fi
  echo "false none"
}

PREFER_GPU=true
INSTALL_CODEFORMER=true
ask_prefer_gpu() {
  if ask_question "Prefer GPU acceleration if available?" "y"; then
    PREFER_GPU=true
    log "User prefers GPU (if available)."
  else
    PREFER_GPU=false
    log "User prefers CPU."
  fi
}

# ───────────────── Impact: Hinweis + frühe Abfrage (EULA) ────────────────────
prompt_impact_opt_in() {
  # bereits entschieden oder via VIDEO_NO_PROPRIETARY_FONTS erzwungen?
  if [[ "${VIDEO_NO_PROPRIETARY_FONTS:-0}" = "1" ]]; then
    log "Skipping Impact/Corefonts download because VIDEO_NO_PROPRIETARY_FONTS=1."
    IMPACT_OPT_IN="no"
    return 0
  fi
  if [[ -n "${IMPACT_OPT_IN:-}" ]]; then
    return 0
  fi

  log "[impact] The Impact typeface is part of Microsoft's Core fonts for the Web."
  log "[impact] To comply with the EULA we only download the original impact32.exe from SourceForge after explicit consent."
  log "[impact] License text: https://sourceforge.net/projects/corefonts/files/ (Microsoft Core Fonts EULA)."

  if ask_question "Download Impact font (Impact 2.35) now? This requires accepting the Microsoft Core Fonts EULA." "n"; then
    IMPACT_OPT_IN="yes"
  else
    IMPACT_OPT_IN="no"
    warn "Impact font download skipped. GIF captions will use fallback fonts."
  fi
}

# ───────────────────── Safety: block system GPU driver installs ─────────────
BLOCKED_APT_REGEX='^(nvidia-|cuda)'
apt_install_safe() {
  local pkgs=()
  for p in "$@"; do
    if [[ "$p" =~ $BLOCKED_APT_REGEX ]]; then
      err "Safety block: refusing apt install '$p' (would change NVIDIA/CUDA drivers)."
    fi
    pkgs+=("$p")
  done
  apt_with_retries install -y "${pkgs[@]}"
}

# ─────────────────────────── Networking / download ──────────────────────────
ask_question() {
  local prompt="$1" default reply display_default
  default="${2:-y}"
  display_default="$default"
  # Sprache: Default-Buchstaben anpassen (y -> j)
  if [[ "${VM_LANG:-en}" == "de" ]]; then
    case "${default,,}" in
      y|j) display_default="j" ;;
      n)   display_default="n" ;;
    esac
  fi
  while true; do
    echo -en "$GREEN$(loc "$prompt") $(loc_printf "[y/n] (default %s):" "$display_default") ${CEND}"
    read -r reply || reply=""
    reply="${reply:-$default}"
    case "$reply" in [YyJj]*) return 0 ;; [Nn]*) return 1 ;; esac
  done
}

is_case_insensitive_fs() {
  # Detect case-insensitive mounts (e.g., Windows/NTFS via WSL drvfs)
  local dir="${1:-.}" probe mountpoint options
  probe="$dir"
  while [[ ! -d "$probe" && "$probe" != "/" ]]; do
    probe="$(dirname "$probe")"
  done
  mountpoint="$(stat -c %m "$probe" 2>/dev/null || stat -f %m "$probe" 2>/dev/null || true)"
  if [[ -n "$mountpoint" ]]; then
    while IFS=' ' read -r _ target _ opts _; do
      if [[ "$target" == "$mountpoint" ]]; then
        options="$opts"
        break
      fi
    done </proc/mounts 2>/dev/null || true
    if [[ "$options" == *"case=off"* || "$options" == *"ignore_case=1"* ]]; then
      return 0
    fi
  fi

  local lower="$probe/.case_test_$$.lower" upper="$probe/.case_test_$$.LOWER"
  rm -f "$lower" "$upper"
  if touch "$lower" 2>/dev/null && touch "$upper" 2>/dev/null; then
    local inode_lower inode_upper
    inode_lower=$(stat -c %i "$lower" 2>/dev/null || stat -f %i "$lower" 2>/dev/null || echo "")
    inode_upper=$(stat -c %i "$upper" 2>/dev/null || stat -f %i "$upper" 2>/dev/null || echo "")
    rm -f "$lower" "$upper"
    [[ -n "$inode_lower" && "$inode_lower" == "$inode_upper" ]] && return 0
    return 1
  fi
  rm -f "$lower" "$upper"
  return 1
}

get_site_packages() {
  python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
}

wait_for_net() {
  local tries="${1:-999999}" delay="${2:-20}"
  local urls=("https://conda.anaconda.org/conda-forge" "https://pypi.org/simple" "https://github.com" "https://download.pytorch.org")
  local ok=0 t=1
  while (( t<=tries )); do
    for u in "${urls[@]}"; do
      if curl -sI --connect-timeout 10 --max-time 20 "$u" >/dev/null; then ok=1; break; fi
    done
    (( ok==1 )) && return 0
    warn "No internet (attempt $t/$tries). Waiting ${delay}s..."
    sleep "$delay"; ((t++))
  done
  return 1
}

# Accept 2xx **und** 3xx, da wir -L / Redirects nutzen
http_ok() {
  local url="$1"
  local code
  code="$(curl -sIL -o /dev/null -w '%{http_code}' -H 'User-Agent: curl/8' "$url" || echo 000)"
  [[ "$code" =~ ^(20[0-9]|30[0-9])$ ]]
}

conda_accept_tos() {
  # Accept Anaconda repo ToS non-interactively (newer conda requires this)
  if conda help tos >/dev/null 2>&1; then
    local channels=(
      "https://repo.anaconda.com/pkgs/main"
      "https://repo.anaconda.com/pkgs/r"
    )
    for ch in "${channels[@]}"; do
      if conda tos accept --yes --override-channels --channel "$ch" >/dev/null 2>&1; then
        log "Accepted Anaconda ToS for $ch."
      else
        warn "Could not auto-accept Anaconda ToS for $ch. If installs fail, run: conda tos accept --override-channels --channel \"$ch\""
      fi
    done
  fi
}


ensure_aria2c() {
  if ! command -v aria2c &>/dev/null; then
    if command -v apt-get &>/dev/null; then
      log "Installing aria2..."
      sudo apt-get update -qq || true
      apt_install_safe aria2 || warn "aria2 install failed"
    elif command -v pacman &>/dev/null; then
      sudo pacman -Sy --noconfirm aria2 || warn "aria2 install failed"
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y aria2 || warn "aria2 install failed"
    elif command -v brew &>/dev/null; then
      brew install aria2 || warn "aria2 install failed"
    else
      warn "No supported package manager for aria2 - please install manually."
    fi
  fi
}

ensure_unzip() {
  if ! command -v unzip &>/dev/null; then
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq || true
      apt_install_safe unzip || true
    elif command -v pacman &>/dev/null; then
      sudo pacman -Sy --noconfirm unzip || true
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y unzip || true
    elif command -v zypper &>/dev/null; then
      sudo zypper --non-interactive in unzip || true
    fi
  fi
}

# ───────────────────────── Impact font (Microsoft Core Fonts) ───────────────
impact_font_target_dir() {
  local os="${1:-$(detect_os_name)}"
  case "$os" in
    mac) echo "$HOME/Library/Application Support/PhotonFrameVideoKit/fonts" ;;
    windows)
      local base="${LOCALAPPDATA:-$HOME/.local/share}"
      echo "$base/PhotonFrameVideoKit/fonts"
      ;;
    *)
      local xdg="${XDG_DATA_HOME:-$HOME/.local/share}"
      echo "$xdg/PhotonFrameVideoKit/fonts"
      ;;
  esac
}

impact_font_path() {
  local os="${1:-$(detect_os_name)}"
  local dir
  dir="$(impact_font_target_dir "$os")"
  printf '%s/impact.ttf\n' "$dir"
}

impact_font_installed() {
  local os="${1:-$(detect_os_name)}"
  local dest
  dest="$(impact_font_path "$os")"
  [[ -f "$dest" ]]
}

ensure_cabextract() {
  if command -v cabextract >/dev/null 2>&1; then
    return 0
  fi
  local os="${1:-$(detect_os_name)}"
  case "$os" in
    linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq || true
        apt_install_safe cabextract || true
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm cabextract || true
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y cabextract || true
      elif command -v zypper >/dev/null 2>&1; then
        sudo zypper --non-interactive in cabextract || true
      else
        warn "No supported package manager for cabextract detected - please install it manually."
        return 1
      fi
      ;;
    mac)
      if command -v brew >/dev/null 2>&1; then
        brew install cabextract || true
      else
        warn "Homebrew not found - please install cabextract manually (https://www.cabextract.org.uk/)."
        return 1
      fi
      ;;
    *)
      warn "cabextract is required but could not be installed automatically."
      return 1
      ;;
  esac
  command -v cabextract >/dev/null 2>&1
}

install_impact_font() {
  local os="${1:-$(detect_os_name)}"
  local dest_dir dest impact_url tmp exe extracted
  dest="$(impact_font_path "$os")"
  dest_dir="$(impact_font_target_dir "$os")"
  impact_url="https://downloads.sourceforge.net/corefonts/impact32.exe"

  if impact_font_installed "$os"; then
    log "$(loc_printf "Impact font already present: %s" "$dest")"
    return 0
  fi

  if [[ "${VIDEO_NO_PROPRIETARY_FONTS:-0}" = "1" ]]; then
    log "Skipping Impact/Corefonts download because VIDEO_NO_PROPRIETARY_FONTS=1."
    return 0
  fi
  if [[ "${IMPACT_OPT_IN:-no}" != "yes" ]]; then
    warn "Impact font download skipped. GIF captions will use fallback fonts."
    return 0
  fi

  if ! ensure_cabextract "$os"; then
    warn "cabextract missing - Impact cannot be installed automatically. Please install the font manually."
    return 0
  fi

  tmp="$(mktemp -d "${TMPDIR:-/tmp}/impact-font.XXXXXX")"
  exe="$tmp/impact32.exe"
  log "$(loc_printf "Downloading Impact installer -> %s" "$exe")"
  if ! download_with_retries "$impact_url" "$exe" 60 10; then
    warn "$(loc_printf "Impact installer download failed (%s)." "$impact_url")"
    rm -rf "$tmp"
    return 0
  fi

  log "Extracting impact.ttf from impact32.exe via cabextract..."
  if ! cabextract -F impact.ttf -d "$tmp" "$exe" >/dev/null 2>&1; then
    warn "cabextract could not extract impact.ttf - please check manually."
    rm -rf "$tmp"
    return 0
  fi
  extracted="$tmp/impact.ttf"
  if [ ! -f "$extracted" ]; then
    warn "impact.ttf missing in archive - installation skipped."
    rm -rf "$tmp"
    return 0
  fi

  mkdir -p "$dest_dir"
  install -m 0644 "$extracted" "$dest"
  rm -rf "$tmp"
  log "$(loc_printf "Impact font installed: %s" "$dest")"
  if command -v fc-cache >/dev/null 2>&1; then
    fc-cache -f "$dest_dir" >/dev/null 2>&1 || true
  fi
}

write_third_party_licenses() {
  local dest="$INSTALL_DIR/THIRD_PARTY_LICENSES.md"
  local cf_status
  if $INSTALL_CODEFORMER; then
    cf_status="Installed because you accepted the S-Lab Non-Commercial License."
  else
    cf_status="Not installed (opt-out or feature disabled during setup)."
  fi
  cat > "$dest" <<EOF
# Third-Party Licenses

This document lists third-party projects and assets that the PhotonFrame - VideoKit installer downloads or references. Please review the upstream licenses before using the software.

## Microsoft Core Fonts - Impact
- Source: https://downloads.sourceforge.net/corefonts/
- License: Microsoft Core Fonts EULA (proprietary, non-redistributable except via original installer).
- Notes: impact32.exe is downloaded only after you accept the license during installation.
- PhotonFrame - VideoKit does not redistribute the Impact font. The installer only downloads the original impact32.exe from the official Core Fonts mirror after you accept the Microsoft EULA.

## Real-ESRGAN (PyTorch)
- Repository: https://github.com/xinntao/Real-ESRGAN
- License: BSD 3-Clause (see upstream LICENSE file).

## Real-ESRGAN NCNN / RealCUGAN NCNN
- Repository: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan and https://github.com/nihui/realcugan-ncnn-vulkan
- License: See upstream repositories for the respective MIT/BSD style licenses.

## BasicSR
- Repository: https://github.com/XPixelGroup/BasicSR
- License: Apache License 2.0.

## facexlib
- Repository: https://github.com/xinntao/facexlib
- License: MIT License.

## GFPGAN
- Repository: https://github.com/TencentARC/GFPGAN
- License: Apache License 2.0.

## CodeFormer (optional)
- Repository: https://github.com/sczhou/CodeFormer
- License: S-Lab Non-Commercial License 1.0 (non-commercial use only).
- Status: $cf_status

## Additional dependencies
- PyTorch, torchvision, NCNN binaries, Conda packages, and other libraries remain under their respective upstream licenses. Please consult the downloaded repositories or package metadata for details.

## Notes
- All these components are cloned from their upstream repositories during installation. Please refer to the LICENSE file in each cloned repository for the full legal terms.
EOF
  log "$(loc_printf "Wrote THIRD_PARTY_LICENSES.md to %s" "$dest")"
  log "Third-party license summary updated."
}

download_with_retries() {
  local url="$1" dest="$2" attempts="${3:-160}" base_sleep="${4:-10}"
  mkdir -p "$(dirname "$dest")"
  local i=1 ok=0 sleep_s="$base_sleep"

  while (( i<=attempts )); do
    wait_for_net 999999 10

    if ! http_ok "$url"; then
      warn "HTTP precheck failed (not 200) for $url (try $i/$attempts)"
    else
      if [[ "$url" == https://github.com/* ]]; then
        if command -v curl >/dev/null; then
          curl -L --fail --retry 80 --retry-delay 5 --retry-all-errors \
               -H 'User-Agent: curl/8' \
               -C - "$url" -o "$dest" && ok=1 || ok=0
        elif command -v aria2c >/dev/null; then
          aria2c --console-log-level=warn --summary-interval=0 \
                 -c -x4 -s4 --min-split-size=5M \
                 --file-allocation=none --auto-file-renaming=false \
                 --user-agent='curl/8' \
                 -o "$(basename "$dest")" -d "$(dirname "$dest")" "$url" && ok=1 || ok=0
        fi
      else
        if command -v aria2c >/dev/null; then
          aria2c --console-log-level=warn --summary-interval=0 \
                 -c -x16 -s16 -k1M --file-allocation=none \
                 --auto-file-renaming=false --max-tries=0 --retry-wait=5 \
                 -o "$(basename "$dest")" -d "$(dirname "$dest")" "$url" && ok=1 || ok=0
        fi
        if [[ $ok -eq 0 ]] && command -v curl >/dev/null; then
          curl -L --fail --retry 200 --retry-delay 5 --retry-all-errors \
               -C - "$url" -o "$dest" && ok=1 || ok=0
        fi
        if [[ $ok -eq 0 ]] && command -v wget >/dev/null; then
          wget -c --tries=0 --timeout=60 "$url" -O "$dest" && ok=1 || ok=0
        fi
      fi
    fi

    if [[ $ok -eq 1 && -s "$dest" ]]; then
      return 0
    fi

    warn "Download error: $url (attempt $i/$attempts). Backoff ${sleep_s}s..."
    sleep "$sleep_s"
    (( sleep_s < 180 )) && sleep_s=$(( sleep_s * 2 ))
    ((i++))
  done
  return 1
}


zip_valid() { local f="$1"; [[ -s "$f" ]] && unzip -tqq "$f" >/dev/null 2>&1; }
tar_valid() { [[ -s "$1" ]] && tar -tf "$1" >/dev/null 2>&1; }
file_min_ok() { local f="$1" min="${2:-1}"; [[ -s "$f" ]] && [[ $(stat -c%s "$f" 2>/dev/null || echo 0) -ge "$min" ]]; }


# ─────────────────────────── ExifTool installer (Conda-first) ───────────────
ensure_exiftool() {
  # Schon vorhanden?
  if command -v exiftool >/dev/null 2>&1; then
    local ex_ver="$(exiftool -ver 2>/dev/null || echo ok)"
    log "$(loc_printf "ExifTool already present: %s" "$ex_ver")"
    return 0
  fi

  local OS_NAME_LOCAL="${1:-$(detect_os_name)}"

  # 1) Prefer installing inside the active Conda env (isolated)
  if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "$(loc_printf "Installing ExifTool inside Conda env: %s" "$CONDA_PREFIX")"
    if conda_retry 16 install -y -c conda-forge exiftool; then
      if command -v exiftool >/dev/null 2>&1; then
        local ex_ver="$(exiftool -ver 2>/dev/null || echo ok)"
        log "$(loc_printf "ExifTool (Conda) OK: %s" "$ex_ver")"
        return 0
      fi
    else
      warn "Conda installation of ExifTool failed - trying system package."
    fi
  fi

  # 2) Fallback: use OS package manager
  case "$OS_NAME_LOCAL" in
    linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq || true
        # Ubuntu/Debian: package name varies by release ('exiftool' or 'libimage-exiftool-perl')
        apt_install_safe exiftool || apt_install_safe libimage-exiftool-perl || true
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm perl-image-exiftool || true
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y perl-Image-ExifTool || true
      elif command -v zypper >/dev/null 2>&1; then
        sudo zypper --non-interactive in exiftool || sudo zypper --non-interactive in perl-Image-ExifTool || true
      else
        warn "No supported Linux package manager detected - please install ExifTool manually."
      fi
      ;;
    mac)
      if command -v brew >/dev/null 2>&1; then
        brew install exiftool || true
      else
        warn "Homebrew not found - install Homebrew or use the official ExifTool pkg."
      fi
      ;;
    windows)
      # This Bash installer does not support Windows directly; show PowerShell hint:
      warn "Windows users: please open an elevated PowerShell and run 'choco install exiftool' or 'scoop install exiftool'."
      ;;
  esac

  if command -v exiftool >/dev/null 2>&1; then
    local ex_ver="$(exiftool -ver 2>/dev/null || echo ok)"
    log "$(loc_printf "ExifTool installed: %s" "$ex_ver")"
  else
    warn "ExifTool could not be installed."
  fi
}


# ───────────────────────────── Pip/Conda helpers ────────────────────────────
apt_with_retries() { local attempts="${APT_RETRIES:-8}" i rc; for ((i=1;i<=attempts;i++)); do if sudo -n apt-get -o Acquire::Retries=8 "$@"; then return 0; fi; rc=$?; warn "apt-get failed (try $i/$attempts) - retrying..."; sleep $(( 5 * i )); done; return "$rc"; }

conda_retry() {
  local attempts="${1:-24}"; shift
  local i rc
  for ((i=1;i<=attempts;i++)); do
    wait_for_net 999999 10
    if "$CMD_INSTALL" "$@" --repodata-fn repodata.json; then return 0; fi
    rc=$?
    warn "Conda failed (repodata.json) - try $i. Cleaning index cache & retry..."
    "$CMD_INSTALL" clean -i -y >/dev/null 2>&1 || true
    sleep $(( 5 * i ))
    if "$CMD_INSTALL" "$@" --repodata-fn current_repodata.json; then return 0; fi
    rc=$?
    "$CMD_INSTALL" clean -i -y >/dev/null 2>&1 || true
    sleep $(( 5 * i ))
  done
  return "$rc"
}

pip_download_and_offline_install_torch() {
  # Args: torch_ver torchvision_ver tag
  local tver="$1" vver="$2" tag="$3"
  local idx_url
  if [[ "$tag" == "cpu" ]]; then
    idx_url="https://download.pytorch.org/whl/cpu"
  else
    idx_url="https://download.pytorch.org/whl/${tag}"
  fi
  local cache="$INSTALL_DIR/.torchcache/${tver}-${vver}-${tag}"
  mkdir -p "$cache"

  python -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
  python -m pip uninstall -y "nvidia-*-cu12" "nvidia-cuda-*cu12" >/dev/null 2>&1 || true

  wait_for_net 999999 10
  log "Preloading Torch wheels (+deps) -> $cache (Index: $idx_url)..."

  local tries=12 i=1 ok=0
  while (( i<=tries )); do
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=900 \
    python -m pip download --progress-bar off \
      --only-binary=:all: --prefer-binary \
      --retries 100 --timeout 90 \
      -i "$idx_url" \
      -d "$cache" \
      "torch==${tver}" "torchvision==${vver}" && ok=1 || ok=0
    if (( ok==1 )); then break; fi
    warn "pip download timeout/error (try $i/$tries) - retrying..."
    sleep $(( 8 * i ))
    ((i++))
  done
  if (( ok==0 )); then warn "pip download failed (tag ${tag})."; return 2; fi
  if ! ls "$cache"/torch-*.whl >/dev/null 2>&1; then warn "No torch wheel cached - abort (tag ${tag})."; return 2; fi

  log "Offline installing from cache..."
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  python -m pip install -q --no-warn-script-location \
    --no-index --find-links "$cache" \
    "torch==${tver}" "torchvision==${vver}" || return 2

  if torch_cuda_check_py; then log "PyTorch import OK (pip/offline, ${tag})."; return 0; else warn "PyTorch import/arch check failed (pip/offline, ${tag})."; return 2; fi
}

venv_pip_install_into_env_from_index() {
  local target="$1"; shift
  local tmp="$INSTALL_DIR/.pipvenv"
  local cache="$INSTALL_DIR/.pipcache"
  mkdir -p "$cache"
  python -m venv "$tmp"
  "$tmp/bin/python" -m ensurepip --upgrade
  wait_for_net
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_PYTHON_VERSION_WARNING=1 \
  "$tmp/bin/pip" download --progress-bar off \
    --retries 999 --timeout 90 --exists-action=w \
    --no-deps --only-binary=:all: --prefer-binary \
    -i https://pypi.org/simple \
    -d "$cache" "$@" || {
      echo "[install] notice: download failed - trying alternate index..."
      wait_for_net
      PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_PYTHON_VERSION_WARNING=1 \
      "$tmp/bin/pip" download --progress-bar off \
        --retries 999 --timeout 90 --exists-action=w \
        --no-deps --only-binary=:all: --prefer-binary \
        -i https://pypi.python.org/simple \
        -d "$cache" "$@" || exit 1
    }
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_PYTHON_VERSION_WARNING=1 \
  "$tmp/bin/pip" install -q \
    --no-deps --no-warn-script-location --no-index --find-links "$cache" \
    --target "$target" "$@"
  rm -rf "$tmp"
}

clone_or_update_repo_with_retries() {
  local repo_url="$1" target_dir="$2" max_attempts=200 delay=15 attempt=1 status=1
  while (( attempt<=max_attempts )); do
    wait_for_net 999999 15
    if [[ -d "$target_dir/.git" ]]; then
      log "Attempt $attempt: updating $target_dir..."
      (
        cd "$target_dir"
        git fetch --all -q
        # Default-Branch sauber ermitteln (origin/HEAD -> 'main' Fallback)
        local def
        def=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||' || echo main)
        git reset -q --hard "origin/$def"
        git pull -q --rebase
      ) && status=0 || status=$?
    else
      log "Attempt $attempt: cloning $repo_url -> $target_dir..."
      rm -rf "$target_dir"
      git clone --depth 1 "$repo_url" "$target_dir" && status=0 || status=$?
    fi
    if [[ $status -eq 0 && -d "$target_dir" ]]; then
      log "Repo ready."
      return 0
    fi
    warn "git failed (attempt $attempt). Waiting ${delay}s..."
    ((attempt++)); sleep "$delay"
  done
  err "Could not clone/update repo."
}


ensure_term_image_stack() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq || true
    apt_install_safe chafa xdg-utils || true
    if command -v snap >/dev/null 2>&1; then
      if ! command -v viu >/dev/null 2>&1; then sudo snap install viu --classic || true; fi
    fi
  elif command -v brew >/dev/null 2>&1; then
    brew install chafa || true
    # xdg-utils ist auf macOS optional; Brew bietet ein Paket, aber wir warnen nur bei Fehlschlag
    brew install xdg-utils || warn "xdg-utils (optional) konnte nicht via Homebrew installiert werden."
  elif command -v pacman >/dev/null 2>&1; then sudo pacman -Sy --noconfirm chafa xdg-utils || true
  elif command -v dnf    >/dev/null 2>&1; then sudo dnf install -y chafa xdg-utils || true
  elif command -v zypper >/dev/null 2>&1; then sudo zypper --non-interactive in chafa xdg-utils || true
  else warn "No known package manager - please install chafa/xdg-utils manually (optional: viu)."
  fi
  if command -v chafa >/dev/null 2>&1; then log "chafa available: $(chafa --version 2>/dev/null | head -n1 || echo ok)"; else warn "chafa not installed - image preview fallbacks limited."; fi
}

ensure_capture_stack() {
  if command -v apt-get >/dev/null; then
    sudo apt-get update -qq || true
    apt_install_safe v4l-utils alsa-utils pulseaudio-utils || true
  elif command -v brew >/dev/null 2>&1; then
    # macOS: v4l/alsa/pulseaudio sind nicht relevant; nur Hinweis ausgeben
    warn "Capture stack (v4l/alsa/pulseaudio) wird auf macOS uebersprungen."
  elif command -v pacman >/dev/null; then sudo pacman -Sy --noconfirm v4l-utils alsa-utils pulseaudio || true
  elif command -v dnf    >/dev/null; then sudo dnf install -y v4l-utils alsa-utils pulseaudio-utils || true
  elif command -v zypper >/dev/null; then sudo zypper --non-interactive in v4l-utils alsa-utils pulseaudio-utils || true
  else warn "Please install v4l-utils/alsa-utils/pulseaudio-utils manually."
  fi
}


ffmpeg_has_vidstab() { "$1" -hide_banner -filters 2>/dev/null | grep -qE 'vidstab(transform|detect)'; }

torch_cuda_check_py() {
  python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    if not ok:
        print("cuda_available:", ok)
        sys.exit(2)
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}"
    try:
        supported = set(getattr(torch.cuda, "get_arch_list")())
    except Exception:
        supported = set()
    print("cuda_available:", ok, "name:", name, "cc:", cc, "supported_arches:", sorted(supported))
    if supported and arch not in supported:
        print("ARCH_MISSING:", arch)
        sys.exit(2)
    sys.exit(0)
except Exception as e:
    print("IMPORT_ERROR:", repr(e))
    sys.exit(2)
PY
}

# GitHub API helpers (for release assets)
gh_api() { local path="$1" out="$2"; local ua='curl/8'; local auth=(); [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN"); curl -fsSL -H 'Accept: application/vnd.github+json' -H "User-Agent: $ua" "${auth[@]}" "https://api.github.com${path}" -o "$out"; }

gh_candidates_assets() {
  local repo="$1" pattern="$2" include_pre="${3:-1}"
  local tmp="$(mktemp)"
  gh_api "/repos/${repo}/releases/latest" "$tmp" && \
  python - "$tmp" "$pattern" "$include_pre" <<'PY'
import json, re, sys
path, pat, allow_pre = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try: data=json.load(open(path,'r',encoding='utf-8'))
except: data={}
rels=[data] if isinstance(data,dict) else []
rx=re.compile(pat,re.I)
for r in rels:
    if not r or r.get("draft"): continue
    if (not allow_pre) and r.get("prerelease"): continue
    for a in r.get("assets",[]):
        n=a.get("name",""); u=a.get("browser_download_url","")
        if rx.search(n) and u: print(u)
PY
  gh_api "/repos/${repo}/releases?per_page=20" "$tmp" && \
  python - "$tmp" "$pattern" "$include_pre" <<'PY'
import json, re, sys
path, pat, allow_pre = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try: data=json.load(open(path,'r',encoding='utf-8'))
except: data=[]
rx=re.compile(pat,re.I)
for r in data:
    if r.get("draft"): continue
    if (not allow_pre) and r.get("prerelease"): continue
    for a in r.get("assets",[]):
        n=a.get("name",""); u=a.get("browser_download_url","")
        if rx.search(n) and u: print(u)
PY
  rm -f "$tmp"
}

gh_fetch_asset() {
  local repo="$1" regex="$2" out="$3" vtype="${4:-file}" min="${5:-1}"; shift 5
  local fallback=("$@")
  local candidates=()
  while IFS= read -r u; do candidates+=("$u"); done < <(gh_candidates_assets "$repo" "$regex" 1 || true)
  if [[ -n "${GH_OVERRIDE_URL:-}" ]]; then candidates=("$GH_OVERRIDE_URL" "${candidates[@]}"); fi
  candidates+=("${fallback[@]}")
  local tmp="$out.__dl__" ok=0
  rm -f "$tmp"
  for url in "${candidates[@]}"; do
    [[ -z "$url" ]] && continue
    if http_ok "$url" && download_with_retries "$url" "$tmp" 160 10; then
      case "$vtype" in
        zip)  zip_valid "$tmp" || { warn "ZIP invalid: $url"; continue; } ;;
        tar)  tar_valid "$tmp" || { warn "TAR invalid: $url"; continue; } ;;
        file) file_min_ok "$tmp" "$min" || { warn "File too small: $url"; continue; } ;;
        *)    file_min_ok "$tmp" "$min" || { warn "Unknown vtype->file-check: $url"; continue; } ;;
      esac
      mv -f "$tmp" "$out"; ok=1; break
    else warn "Download/precheck failed: $url"; fi
  done
  rm -f "$tmp"; [[ $ok -eq 1 ]]
}

download_realesrgan_weights_selected() {
  local WDIR="$INSTALL_DIR/real-esrgan/weights"
  mkdir -p "$WDIR"

  # Only keep the desired four models:
  gh_fetch_asset "xinntao/Real-ESRGAN" "^realesr-general-x4v3\\.pth$"             "$WDIR/realesr-general-x4v3.pth"             "file" 50000000 \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/realesr-general-x4v3.pth"           || warn "realesr-general-x4v3.pth fehlte."

  gh_fetch_asset "xinntao/Real-ESRGAN" "^RealESRGAN_x4plus\\.pth$"                "$WDIR/RealESRGAN_x4plus.pth"                "file" 50000000 \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth"            || warn "RealESRGAN_x4plus.pth fehlte."

  gh_fetch_asset "xinntao/Real-ESRGAN" "^RealESRGAN_x2plus\\.pth$"                "$WDIR/RealESRGAN_x2plus.pth"                "file" 30000000 \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth"            || warn "RealESRGAN_x2plus.pth fehlte."

  gh_fetch_asset "xinntao/Real-ESRGAN" "^RealESRGAN_x4plus_anime_6B\\.pth$"       "$WDIR/RealESRGAN_x4plus_anime_6B.pth"       "file" 30000000 \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth"   || warn "RealESRGAN_x4plus_anime_6B.pth fehlte."
}


# ───────────────── Vulkan loader & NCNN binaries (OS-aware) ─────────────────
install_vulkan_loader() {
  local OS_NAME="${1:-$(detect_os_name)}"
  if [[ "$OS_NAME" == "linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -qq || true
      apt_install_safe libvulkan1 vulkan-tools || true
    elif command -v pacman >/dev/null 2>&1; then sudo pacman -Sy --noconfirm vulkan-icd-loader vulkan-tools || true
    elif command -v dnf    >/dev/null 2>&1; then sudo dnf install -y vulkan-loader vulkan-tools || true
    elif command -v zypper >/dev/null 2>&1; then sudo zypper --non-interactive in libvulkan1 vulkan-tools || true
    else warn "Please install Vulkan loader manually."
    fi
  else
    log "Skipping Vulkan loader install for $OS_NAME (NCNN bundles often include libs)."
  fi
}

_install_ncnn_zip_generic() {
  local repo="$1" outdir="$2" binname="$3" os_pat="$4"

  local dest_dir="$outdir"
  local bin="$dest_dir/$binname"
  mkdir -p "$dest_dir"
  ensure_unzip

  local zip_out="$INSTALL_DIR/.cache/${binname}.zip"
  mkdir -p "$(dirname "$zip_out")"

  # Kandidaten aus GitHub Releases
  local candidates=()
  while IFS= read -r u; do candidates+=("$u"); done < <(
    gh_candidates_assets "$repo" "(${os_pat}).*\\.(zip)$" 1 || true
  )

  # Fallbacks
  if [[ "$repo" == "xinntao/Real-ESRGAN-ncnn-vulkan" ]]; then
    candidates+=(
      "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-ubuntu.zip"
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    )
  elif [[ "$repo" == "nihui/realesrgan-ncnn-vulkan" ]]; then
    candidates+=(
      "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-ubuntu.zip"
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    )
  elif [[ "$repo" == "nihui/realcugan-ncnn-vulkan" ]]; then
    candidates+=(
      "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip"
    )
  fi

  local ok=0
  for url in "${candidates[@]}"; do
    [[ -z "$url" ]] && continue
    log "Trying ${binname} from: $url"
    rm -f "$zip_out"
    if http_ok "$url" && download_with_retries "$url" "$zip_out" 160 10 && zip_valid "$zip_out"; then
      # Entpacken
      # Extract into a temporary directory to inspect structure safely
      local tmp_unpack="$dest_dir/__unpack__"
      rm -rf "$tmp_unpack"
      mkdir -p "$tmp_unpack"
      (cd "$tmp_unpack" && unzip -oq "$zip_out") || true

      # Binary finden (aus dem entpackten Baum)
      local cand
      cand="$(find "$tmp_unpack" -type f -name "$binname" -perm -111 | head -n1 || true)"

      # Models konsolidieren:
      # - Alle "models" **und** "models-*" sammeln
      # - Unterordner **bewahren** (kein Flatten!)
      local models_root="$dest_dir/models"
      mkdir -p "$models_root"

      # 1) Bereits vorhandenes 'models' (falls vorhanden) -> in Root mergen
      while IFS= read -r m; do
        [[ -z "$m" ]] && continue
        rsync -a "$m"/ "$models_root"/ 2>/dev/null || { cp -a "$m"/. "$models_root"/ 2>/dev/null || true; }
      done < <(find "$tmp_unpack" -type d -name models 2>/dev/null || true)

      # 2) Alle 'models-*' -> als Unterordner unter $models_root/<basename>
      while IFS= read -r m; do
        [[ -z "$m" ]] && continue
        local base; base="$(basename "$m")"
        mkdir -p "$models_root/$base"
        rsync -a "$m"/ "$models_root/$base"/ 2>/dev/null || { cp -a "$m"/. "$models_root/$base"/ 2>/dev/null || true; }
      done < <(find "$tmp_unpack" -type d -name 'models-*' 2>/dev/null || true)

      # Binary platzieren
      if [[ -n "$cand" ]]; then
        mkdir -p "$(dirname "$bin")"
        if [[ "$cand" != "$bin" ]]; then
          if ! mv -f "$cand" "$bin" 2>/dev/null; then
            install -m 0755 "$cand" "$bin"
          fi
        else
          log "Binary already at target: $bin"
        fi
        chmod +x "$bin" 2>/dev/null || true
        ok=1
      else
        warn "Binary not found in zip - next candidate..."
      fi

      # Cleanup
      rm -rf "$tmp_unpack"
      [[ $ok -eq 1 ]] && break
    else
      warn "Zip invalid or download failed - next candidate..."
    fi
  done

  if [[ "$ok" -ne 1 ]]; then
    warn "Could not fetch $binname automatically. NCNN fallback remains inactive."
    return 0
  fi

  # Wrapper in der Conda-Env, der stets -m auf den **Root** der Modelle zeigt
  mkdir -p "$CONDA_PREFIX/bin"
  cat > "$CONDA_PREFIX/bin/$binname" <<WRP
#!/usr/bin/env bash
exec "$bin" -m "$dest_dir/models" "\$@"
WRP
  chmod +x "$CONDA_PREFIX/bin/$binname"

  # Mini-Sanity
  if ! "$bin" -h >/dev/null 2>&1; then
    warn "$binname fails to start - check Vulkan libs."
  else
    # Extra check for RealCUGAN: are the expected subdirectories present?
    if [[ "$binname" == "realcugan-ncnn-vulkan" ]]; then
      if ! ls "$dest_dir"/models/models-* >/dev/null 2>&1; then
        warn "RealCUGAN models appear missing (no models-* under $dest_dir/models)."
      fi
    fi
    log "$binname: OK"
  fi
}


install_realesrgan_ncnn_vulkan() {
  local OS_NAME="${1:-$(detect_os_name)}"
  local os_pat="ubuntu|linux"
  [[ "$OS_NAME" == "mac" ]] && os_pat="mac|osx|macos"
  [[ "$OS_NAME" == "windows" ]] && os_pat="windows|win64|win32"
  # **Fix**: richtiges Repo verwenden
  _install_ncnn_zip_generic "xinntao/Real-ESRGAN-ncnn-vulkan" "$INSTALL_DIR/realesrgan-ncnn-vulkan" "realesrgan-ncnn-vulkan" "$os_pat"
}


install_realcugan_ncnn_vulkan() {
  local OS_NAME="${1:-$(detect_os_name)}"
  local os_pat="ubuntu|linux"
  [[ "$OS_NAME" == "mac" ]] && os_pat="mac|osx|macos"
  [[ "$OS_NAME" == "windows" ]] && os_pat="windows|win64|win32"
  _install_ncnn_zip_generic "nihui/realcugan-ncnn-vulkan" "$INSTALL_DIR/realcugan-ncnn-vulkan" "realcugan-ncnn-vulkan" "$os_pat"
}

#
# Real-ESRGAN NCNN - **nur Modelle** laden (ohne Binary, ohne Login)
# - Liest, wenn vorhanden, den -m Pfad aus dem Wrapper in $CONDA_PREFIX/bin
# - Falls back to $CONDA_PREFIX/share/.../models
# - Holt das offizielle v0.2.5.0-Archiv und extrahiert NUR models/
#
install_realesrgan_ncnn_models() {
  log "Installing Real-ESRGAN NCNN models (models/ only)..."

  # prerequisites
  ensure_unzip
  if ! command -v curl >/dev/null 2>&1; then
    warn "curl not found - cannot download Real-ESRGAN NCNN models."
    return 0
  fi
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    warn "CONDA_PREFIX not set - cannot determine target for NCNN models."
    return 0
  fi

  # 1) Zielordner bestimmen
  local WRAP="$CONDA_PREFIX/bin/realesrgan-ncnn-vulkan"
  local MODELS_DIR=""
  if [[ -x "$WRAP" ]]; then
    # Versuche, den -m Pfad (.../models) aus dem Wrapper zu parsen
    MODELS_DIR="$(grep -Eo '\-m[[:space:]]+"([^"]+/models)"' "$WRAP" | sed -E 's/.*"([^"]+)".*/\1/' | head -n1 || true)"
  fi
  # Fallback: env-share
  if [[ -z "$MODELS_DIR" ]]; then
    MODELS_DIR="$CONDA_PREFIX/share/realesrgan-ncnn-vulkan/models"
  fi
  mkdir -p "$MODELS_DIR"
  log "NCNN models target: $MODELS_DIR"

  # 2) Downloadquelle (offizielles Release)
  #    (You can override the URL with NCNN_REALESRGAN_MODELS_URL)
  local URL="${NCNN_REALESRGAN_MODELS_URL:-https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip}"
  local ZIP="$INSTALL_DIR/.cache/realesrgan-ncnn-vulkan-models.zip"
  mkdir -p "$INSTALL_DIR/.cache"
  rm -f "$ZIP"

  # 3) Robuster Download (416-sicher)
  if ! curl -L --fail --retry 50 --retry-delay 5 --retry-all-errors -o "$ZIP" "$URL"; then
    warn "Direct download failed once, retrying with resume..."
    curl -L --fail --retry 50 --retry-delay 5 --retry-all-errors -C - -o "$ZIP" "$URL" || {
      warn "Could not download NCNN models archive."; return 0; }
  fi

  # 4) Nur models/ auspacken
  local UNZ
  UNZ="$(mktemp -d)"
  if ! unzip -q "$ZIP" -d "$UNZ"; then
    warn "Unzip failed for NCNN models archive."; rm -rf "$UNZ"; return 0
  fi
  local SRC_MODELS
  SRC_MODELS="$(find "$UNZ" -type d -name models | head -n1 || true)"
  if [[ -z "$SRC_MODELS" ]]; then
    warn "models/ not found in NCNN archive - skipping."
    rm -rf "$UNZ"
    return 0
  fi

  # Synchronisieren (rsync wenn vorhanden, sonst cp)
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$SRC_MODELS"/ "$MODELS_DIR"/
  else
    rm -rf "$MODELS_DIR"/*
    cp -a "$SRC_MODELS"/. "$MODELS_DIR"/
  fi
  rm -rf "$UNZ"

  # 5) Mini-Sanity
  local missing=0
  for f in realesrgan-x4plus.param realesrgan-x4plus.bin \
           realesrgan-x4plus-anime.param realesrgan-x4plus-anime.bin; do
    [[ -s "$MODELS_DIR/$f" ]] || { warn "Model missing: $f"; missing=$((missing+1)); }
  done
  (( missing==0 )) && log "Real-ESRGAN NCNN models installed." || warn "Some default models are missing - installation completed."

  # Hinweis, falls Wrapper kein -m setzt
  if [[ -x "$WRAP" ]] && ! grep -qE '\-m[[:space:]]+"' "$WRAP"; then
    warn "Hint: your realesrgan-ncnn-vulkan wrapper does not pass a -m path."
    warn "You can add it like this:  exec \".../realesrgan-ncnn-vulkan\" -m \"$MODELS_DIR\" \"\$@\""
  fi
}


# ────────────────────── PyTorch selection & install ─────────────────────────
check_nvidia_driver_ok() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local drv vmaj
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "")"
    if [[ -n "$drv" ]]; then
      vmaj="${drv%%.*}"
      log "Detected NVIDIA driver: $drv (Info: CUDA 12.x usually needs ≥ 535/545)."
      if [[ "$vmaj" =~ ^[0-9]+$ ]] && (( vmaj < 520 )); then
        warn "Driver < 520: newer CUDA wheels (12.x) may not work. Fallback logic will step back."
      fi
    fi
  else
    warn "nvidia-smi not found - CUDA availability will be checked later."
  fi
}

conda_install_torch_combo() {
  local tver="$1" vver="$2" tag="$3"
  log "Conda fallback: torch=${tver}, tv=${vver}, tag=${tag}"
  if [[ "$tag" == "cpu" ]]; then
    conda_retry 20 install -y -c pytorch -c conda-forge \
      "pytorch==${tver}" "torchvision==${vver}" cpuonly || return 2
  else
    local cuda_ver
    case "$tag" in
      cu128) cuda_ver="12.8" ;;
      cu124) cuda_ver="12.4" ;;
      cu121) cuda_ver="12.1" ;;
      cu118) cuda_ver="11.8" ;;
      *)     cuda_ver="" ;;
    esac
    [[ -z "$cuda_ver" ]] && return 2
    conda_retry 20 install -y -c pytorch -c nvidia -c conda-forge \
      "pytorch==${tver}" "torchvision==${vver}" "pytorch-cuda=${cuda_ver}" || return 2
  fi
  if torch_cuda_check_py; then log "PyTorch import OK (conda, ${tag})."; return 0; else warn "PyTorch import failed (conda, ${tag})."; return 2; fi
}

detect_gpu_cc_num() { command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; } ; local s; s="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"; [[ -z "$s" ]] && { echo ""; return; }; echo "$s" | awk -F. '{printf "%d%02d", $1, $2}'; }

pytorch_candidates_for_gpu() {
  local cc_num="${1:-}"
  if [[ -n "$cc_num" && "$cc_num" -ge 1200 ]]; then
    echo "2.8.0 0.23.0 cu128"
    echo "2.7.0 0.22.0 cu128"
    echo "2.8.0 0.23.0 cpu"
  else
    echo "2.4.0 0.19.0 cu124"
    echo "2.3.1 0.18.1 cu121"
    echo "2.1.1 0.16.1 cu118"
    echo "2.3.1 0.18.1 cpu"
  fi
}

install_pytorch_smart() {
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_DEFAULT_TIMEOUT=900

  local GPU=false
  command -v nvidia-smi &>/dev/null && GPU=true
  $GPU && log "CUDA GPU detected - selecting matching Torch builds." || log "No NVIDIA GPU -> CPU wheels."
  check_nvidia_driver_ok

  local GPU_CC_NUM="$(detect_gpu_cc_num || true)"
  local ESRGAN_FORCE_FP32_DEFAULT=0
  if [[ -n "$GPU_CC_NUM" ]]; then
    if (( GPU_CC_NUM < 700 )); then ESRGAN_FORCE_FP32_DEFAULT=1; fi
    log "Compute Capability (×100): ${GPU_CC_NUM}; ESRGAN_FORCE_FP32_DEFAULT=${ESRGAN_FORCE_FP32_DEFAULT}"
  fi

  local CANDIDATE_TORCH=()
  while IFS= read -r line; do CANDIDATE_TORCH+=("$line"); done < <(pytorch_candidates_for_gpu "${GPU_CC_NUM:-}")

  local torch_ok=false tver vver ctag
  for combo in "${CANDIDATE_TORCH[@]}"; do
    set -- $combo; tver="$1"; vver="$2"; ctag="$3"
    if [[ "$GPU" != "true" && "$ctag" != "cpu" ]]; then continue; fi
    if [[ "$ctag" == "cu128" ]]; then
      # Avoid flaky cu128 pip links; prefer conda first
      if conda_install_torch_combo "$tver" "$vver" "$ctag"; then
        torch_ok=true
      else
        warn "Conda cu128 failed - trying pip (may be slow)..."
        if pip_download_and_offline_install_torch "$tver" "$vver" "$ctag"; then torch_ok=true; fi
      fi
    else
      if pip_download_and_offline_install_torch "$tver" "$vver" "$ctag"; then
        torch_ok=true
      else
        warn "pip offline install failed (tag ${ctag}) - trying conda fallback..."
        if conda_install_torch_combo "$tver" "$vver" "$ctag"; then torch_ok=true; fi
      fi
    fi
    if $torch_ok; then
      log "PyTorch installed: torch=$tver, torchvision=$vver, tag=$ctag"
      if [[ -z "${ESRGAN_FORCE_FP32:-}" && "$ESRGAN_FORCE_FP32_DEFAULT" = "1" ]]; then
        export ESRGAN_FORCE_FP32=1
        log "Enabling default FP32 (ESRGAN_FORCE_FP32=1) for this GPU arch."
      fi
      break
    else
      warn "Combo torch=$tver, tv=$vver, tag=$ctag failed - trying next..."
      python -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
      python -m pip uninstall -y "nvidia-*-cu12" "nvidia-cuda-*cu12" >/dev/null 2>&1 || true
      conda_retry 3 remove -y pytorch torchvision pytorch-cuda cpuonly >/dev/null 2>&1 || true
    fi
  done

  # Optional cu128 nightly fallback (last resort)
  if ! $torch_ok && [[ -n "$GPU_CC_NUM" && "$GPU_CC_NUM" -ge 1200 ]]; then
    warn "Stable cu128 wheels not available? - trying nightly cu128."
    wait_for_net 999999 10
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=900 \
    python -m pip install -q --no-warn-script-location --pre \
      --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
      torch torchvision && \
      torch_cuda_check_py && torch_ok=true || true
  fi

  $torch_ok || err "Could not install CUDA or CPU builds for PyTorch/TorchVision."

  set +e
  python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
try:
  if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))
except Exception as e:
  print("cuda query err:", e)
print("torchvision", torchvision.__version__)
PY
  set -e
}

# ───────────────────────── Real-ESRGAN (Python) & weights ───────────────────
install_realesrgan_python_and_models() {
  log "Installing Real-ESRGAN (Python)..."
  RE_DIR="$INSTALL_DIR/real-esrgan"
  clone_or_update_repo_with_retries "https://github.com/xinntao/Real-ESRGAN" "$RE_DIR"

  echo "$RE_DIR" > "$(get_site_packages)/real_esrgan.pth"
  log "Real-ESRGAN package path added via .pth."

  if [ ! -f "$RE_DIR/realesrgan/version.py" ]; then
    log "Generating realesrgan/version.py..."
    python - <<PY
from pathlib import Path
p = Path(r"$RE_DIR") / "realesrgan"
p.mkdir(parents=True, exist_ok=True)
(Path(p/"version.py")).write_text('__version__ = "0.3.0"\\n__gitsha__ = "local"\\n', encoding="utf-8")
PY
  fi

  conda_retry 24 install -y scipy lmdb yapf addict filterpy tqdm pyyaml
  venv_pip_install_into_env_from_index "$(get_site_packages)" \
    "lazy_loader>=0.3" "imageio>=2.27" "tifffile>=2022.8.12" \
    "pywavelets==1.4.1" "scikit-image==0.21.0"

  BASICS_DIR="$INSTALL_DIR/_thirdparty/basicsr"
  FACEX_DIR="$INSTALL_DIR/_thirdparty/facexlib"
  GFPGAN_DIR="$INSTALL_DIR/_thirdparty/GFPGAN"
  CF_DIR="$INSTALL_DIR/_thirdparty/CodeFormer"
  mkdir -p "$INSTALL_DIR/_thirdparty"
  clone_or_update_repo_with_retries "https://github.com/XPixelGroup/BasicSR" "$BASICS_DIR"
  clone_or_update_repo_with_retries "https://github.com/xinntao/facexlib"   "$FACEX_DIR"
  clone_or_update_repo_with_retries "https://github.com/TencentARC/GFPGAN"  "$GFPGAN_DIR"
  if $INSTALL_CODEFORMER; then
    clone_or_update_repo_with_retries "https://github.com/sczhou/CodeFormer"  "$CF_DIR"
  else
    log "Skipping CodeFormer (non-commercial license)."
  fi
  echo "$BASICS_DIR" > "$(get_site_packages)/basicsr_local.pth"
  echo "$FACEX_DIR"  > "$(get_site_packages)/facexlib_local.pth"
  echo "$GFPGAN_DIR" > "$(get_site_packages)/gfpgan_local.pth"
  if $INSTALL_CODEFORMER; then
    echo "$CF_DIR"     > "$(get_site_packages)/codeformer_local.pth"
  else
    rm -f "$(get_site_packages)/codeformer_local.pth" 2>/dev/null || true
  fi

  python - <<PY
from pathlib import Path
pairs = [
  Path(r"$BASICS_DIR")/"basicsr"/"version.py",
  Path(r"$FACEX_DIR")/"facexlib"/"version.py",
  Path(r"$GFPGAN_DIR")/"gfpgan"/"version.py",
]
for f in pairs:
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text('__version__ = "local"\\n__gitsha__ = "local"\\n', encoding="utf-8")
        print("Created:", f)
PY

  # ───────────────────────────── NEW: Face restoration weights & ENV ─────────
  download_face_restoration_models() {
    local gdest="$GFPGAN_DIR/weights"; mkdir -p "$gdest"
    gh_fetch_asset "TencentARC/GFPGAN" "^GFPGANv1\.4\.pth$" "$gdest/GFPGANv1.4.pth" "file" 50000000 \
      "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" || warn "GFPGANv1.4 missing."

    local cdest=
    if $INSTALL_CODEFORMER; then
      cdest="$CF_DIR/weights/CodeFormer"; mkdir -p "$cdest"
      gh_fetch_asset "sczhou/CodeFormer" "^codeformer.*\\.pth$" "$cdest/codeformer.pth" "file" 40000000 || warn "CodeFormer weight missing."
    else
      log "CodeFormer is not installed - skipping weight download."
    fi

    # Spiegeln in Real-ESRGAN Default-Verzeichnisse + Symlinks
    local re_pre="$RE_DIR/experiments/pretrained_models"; mkdir -p "$re_pre"
    if [[ -s "$gdest/GFPGANv1.4.pth" ]]; then
      cp -f "$gdest/GFPGANv1.4.pth" "$re_pre/GFPGANv1.4.pth"
      ( cd "$re_pre" && ln -sf "GFPGANv1.4.pth" "GFPGANv1.3.pth" )
      log "$(loc_printf "GFPGAN weights mirrored to %s (including v1.3 symlink)." "$re_pre")"
    fi
    if $INSTALL_CODEFORMER; then
      local re_cf="$RE_DIR/weights/codeformer"; mkdir -p "$re_cf"
      if [[ -s "$cdest/codeformer.pth" ]]; then
        cp -f "$cdest/codeformer.pth" "$re_cf/codeformer.pth"
        log "$(loc_printf "CodeFormer weight mirrored to %s." "$re_cf")"
      fi
    fi

    # Aktivierungs-Script mit ABSOLUTEN Pfaden
    local actd="$CONDA_PREFIX/etc/conda/activate.d"; mkdir -p "$actd"
    cat > "$actd/99-face-models.sh" <<EOF
# Auto-created by install.sh
export GFPGAN_MODEL_PATH="$GFPGAN_DIR/weights/GFPGANv1.4.pth"
EOF
    if $INSTALL_CODEFORMER; then
      cat >> "$actd/99-face-models.sh" <<EOF
export CODEFORMER_MODEL_PATH="$CF_DIR/weights/CodeFormer/codeformer.pth"
EOF
    fi
    # Deactivate cleanup
    local deact="$CONDA_PREFIX/etc/conda/deactivate.d"; mkdir -p "$deact"
    cat > "$deact/99-face-models.sh" <<'EOF'
unset GFPGAN_MODEL_PATH
EOF
    if $INSTALL_CODEFORMER; then
      cat >> "$deact/99-face-models.sh" <<'EOF'
unset CODEFORMER_MODEL_PATH
EOF
    fi
    log "Environment exports written for face models."
  }
  download_face_restoration_models

  # ───────────────────────────── NEW: BasicSR torchvision-Fix ────────────────
  echo "== basicsr torchvision-Fix =="
  python - <<'PY'
import sys, os, re, shutil
from pathlib import Path

patched_files = []
removed_caches = 0

def patch_file(p: Path):
    if not p.exists():
        return False
    try:
        s = p.read_text(encoding="utf-8")
    except Exception:
        return False
    old = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
    tag = '# vm-compat-patch: torchvision rgb_to_grayscale'
    new = (
f"{tag}\n"
"try:\n"
"    from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
"except Exception:  # torchvision>=0.20 moved API\n"
"    from torchvision.transforms.functional import rgb_to_grayscale\n"
)
    if tag in s:
        return False
    if old in s:
        s = s.replace(old, new)
        p.write_text(s, encoding="utf-8")
        patched_files.append(str(p))
        # __pycache__ clean in module root
        root = p.parent.parent
        for q in root.rglob('__pycache__'):
            try:
                shutil.rmtree(q); globals()['removed_caches'] += 1
            except Exception:
                pass
        return True
    return False

# Candidates: installed basicsr + local clone (if available via env)
cands = []
try:
    import importlib.util
    spec = importlib.util.find_spec("basicsr")
    if spec and spec.submodule_search_locations:
        cands.append(Path(list(spec.submodule_search_locations)[0]))
except Exception:
    pass

for env_key in ("BASICS_DIR",):
    d = os.environ.get(env_key)
    if d:
        p = Path(d)/"basicsr"
        if p.exists(): cands.append(p)

seen = set()
for root in cands:
    root = root.resolve()
    if root in seen: continue
    seen.add(root)
    print(f"[info] basicsr root: {root}")
    degrad = root/"data"/"degradations.py"
    if patch_file(degrad):
        print(f"  [patched] {degrad}")
    else:
        print("  [info] nichts zu patchen (Import schon ok?)")

print(f"[done] Gepatchte Dateien: {len(patched_files)}")
print(f"[info] __pycache__ Ordner entfernt: {removed_caches}\n")

print("== Import-Selbsttest ==")
import importlib
for m in ("basicsr","gfpgan","facexlib"):
    importlib.import_module(m)
    print(f"[OK] import {m}")

print("\n✅ torchvision compatibility fix applied.")
PY

  # Sanfter Guard im Real-ESRGAN CLI (fehlende Bilder)
  RE_INFER="$RE_DIR/inference_realesrgan.py"
  if [[ -f "$RE_INFER" ]] && ! grep -q "failed to read image" "$RE_INFER"; then
    sed -i '/img = cv2.imread/i\
    \ \ \ \ if img is None:\
    \ \ \ \ \ \ print(f"WARN: failed to read image - skip: {imgname}")\
    \ \ \ \ \ \ continue' "$RE_INFER"
    log "Applied benign read-guard patch."
  fi

  set +e
  python - <<'PY'
import importlib, torch, torchvision, cv2, skimage
print("torch", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("OpenCV:", getattr(cv2, "__version__", "n/a"))
print("scikit-image:", getattr(skimage, "__version__", "n/a"))
for m in ("realesrgan","basicsr","facexlib","gfpgan"):
    importlib.import_module(m)
print("Real-ESRGAN import: OK")
PY
  set -e

    log "Real-ESRGAN (Python) installed."
  download_realesrgan_weights_selected
}


# ────────────────────────────── Config writer ───────────────────────────────
write_config_ini() {
  # Writes/merges config with detected choices (non-destructive for existing keys)
  # Args: lang osname gpu_enabled gpu_backend ai_backend torch_device
  local lang="$1" osname="$2" gpu_enabled="$3" gpu_backend="$4" ai_backend="$5" torch_device="$6"
  export lang_override="$VM_LANG" osname_override="$OS_NAME" gpu_enabled="$GPU_ENABLED_CFG" \
       gpu_backend="$GPU_BKND" ai_backend="$AI_BACKEND_PICK" torch_device="$TORCH_DEVICE_PICK" \
       ai_enabled_override="$AI_ENABLED_CFG" codeformer_enabled_override="$CODEFORMER_CFG"

  # Make install base visible to the Python helper
  VM_INSTALL_DIR="$INSTALL_DIR" \
  python - <<'PY'
import os, sys, time, json
from pathlib import Path

# --- Try to use platformdirs; fallback to XDG/macOS/Windows manually -----------
def _platform_dirs(app: str):
    try:
        from platformdirs import PlatformDirs  # type: ignore
        d = PlatformDirs(appname=app, appauthor=False)
        return {
            "config": Path(d.user_config_dir),
            "data":   Path(d.user_data_dir),
            "cache":  Path(d.user_cache_dir),
            "state":  Path(d.user_state_dir),
        }
    except Exception:
        # Fallback: XDG / macOS / Windows minimal mapping
        home = Path.home()
        if sys.platform == "darwin":
            base_cfg  = home / "Library" / "Application Support" / app
            base_data = base_cfg
            base_cache= home / "Library" / "Caches" / app
            base_state= base_cfg / "State"
        elif os.name == "nt":
            appdata   = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
            localapp  = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
            base_cfg  = appdata / app
            base_data = appdata / app / "Data"
            base_cache= localapp / app / "Cache"
            base_state= localapp / app / "State"
        else:
            xcfg   = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
            xdata  = Path(os.environ.get("XDG_DATA_HOME",  home / ".local" / "share"))
            xcache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
            xstate = Path(os.environ.get("XDG_STATE_HOME", home / ".local" / "state"))
            base_cfg  = xcfg  / app
            base_data = xdata / app
            base_cache= xcache/ app
            base_state= xstate/ app
        return {"config": base_cfg, "data": base_data, "cache": base_cache, "state": base_state}

APP = "PhotonFrameVideoKit"
dirs = _platform_dirs(APP)
cfg_dir, data_dir, cache_dir, state_dir = dirs["config"], dirs["data"], dirs["cache"], dirs["state"]
for p in (cfg_dir, data_dir, cache_dir, state_dir):
    p.mkdir(parents=True, exist_ok=True)

cfg_path = cfg_dir / "config.ini"

# --- INI merge/write (preserve existing values) --------------------------------
import configparser
cp = configparser.ConfigParser(interpolation=None)
if cfg_path.exists():
    try:
        cp.read(cfg_path, encoding="utf-8")
    except Exception:
        # If damaged, start fresh but keep a backup
        bak = cfg_path.with_suffix(".bak")
        try:
            cfg_path.replace(bak)
        except Exception:
            pass

def ensure(sec: str, key: str, val):
    if not cp.has_section(sec):
        cp.add_section(sec)
    if not cp.has_option(sec, key):
        cp.set(sec, key, str(val))

def set_always(sec: str, key: str, val):
    if not cp.has_section(sec):
        cp.add_section(sec)
    cp.set(sec, key, str(val))

# Inputs from installer (passed via bash)
lang        = os.environ.get("lang_override",    None) or """{{LANG}}"""
osname      = os.environ.get("osname_override",  None) or """{{OSNAME}}"""
gpu_enabled = os.environ.get("gpu_enabled",      None) or """{{GPU_ENABLED}}"""
gpu_backend = os.environ.get("gpu_backend",      None) or """{{GPU_BACKEND}}"""
ai_backend  = os.environ.get("ai_backend",       None) or """{{AI_BACKEND}}"""
torch_dev   = os.environ.get("torch_device",     None) or """{{TORCH_DEVICE}}"""
ai_enabled  = os.environ.get("ai_enabled_override")
codeformer  = os.environ.get("codeformer_enabled_override")
# NOTE: The placeholders {{...}} are filled below by sed; they’re only placeholders here.

# Fill placeholders from argv env (installer passes real values; we replace via sed)
# (When running directly, they’ll just stay as the literal placeholders.)
now_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

# Meta section
ensure("meta", "config_version", "1")
ensure("meta", "first_run_at", now_iso if not cp.has_section("meta") or not cp.has_option("meta","first_run_at") else cp.get("meta","first_run_at"))
set_always("meta", "last_install_at", now_iso)

# General/hardware/ai (only set if missing to preserve user edits)
ensure("general",  "language", lang)
ensure("general",  "os",       osname)
ensure("hardware", "gpu_enabled", str(gpu_enabled).lower())
ensure("hardware", "gpu_backend", gpu_backend)
ensure("ai",       "backend",     ai_backend)
ensure("ai",       "torch_device", torch_dev)
if ai_enabled is not None:
    set_always("ai", "enabled", str(ai_enabled).lower())
if codeformer is not None:
    set_always("ai", "codeformer_enabled", str(codeformer).lower())

# Paths (defaults; preserve if user already chose different ones)
vm_base = os.environ.get("VM_INSTALL_DIR", "")
set_always("paths", "install_dir", vm_base)  # keep current install base in sync
ensure("paths", "data_dir",   str(data_dir))
ensure("paths", "cache_dir",  str(cache_dir))
ensure("paths", "state_dir",  str(state_dir))
ensure("paths", "default_output_dir", str(data_dir / "outputs"))
ensure("paths", "temp_dir",   str(cache_dir / "tmp"))

# Logging default
ensure("logging", "file", str(state_dir / "PhotonFrameVideoKit.log"))
ensure("logging", "level", "INFO")

# Optional ESRGAN defaults from environment (set if provided & not present)
_esrgan_path = os.environ.get("ESRGAN_DEFAULT_MODEL_PATH")
_esrgan_name = os.environ.get("ESRGAN_DEFAULT_MODEL_NAME")
if _esrgan_path:
    ensure("ai", "esrgan_default_model_path", _esrgan_path)
if _esrgan_name:
    ensure("ai", "esrgan_default_model_name", _esrgan_name)
if "ESRGAN_FORCE_FP32" in os.environ:
    ensure("ai", "esrgan_force_fp32", os.environ.get("ESRGAN_FORCE_FP32","0"))

# Final write (atomic)
tmp = cfg_path.with_suffix(".tmp")
with tmp.open("w", encoding="utf-8") as f:
    cp.write(f)
tmp.replace(cfg_path)

print(f"[config] wrote {cfg_path}")
PY
  # Patch placeholders in the just-run heredoc by re-running the sed substitutions
  # (We can’t modify the heredoc body at runtime; instead we pass values through env above.)
}


# ──────────────────────────────── Main flow ────────────────────────────────
ask_language
prompt_impact_opt_in

# Choose install base
DEFAULT_BASE=$(pwd)
echo -ne "$GREEN Installation directory [$DEFAULT_BASE]: $CEND"
read -r INSTALL_DIR_RAW || INSTALL_DIR_RAW="$DEFAULT_BASE"

sanitize_path_input() {
  local s="$1"
  s="${s//[$'\r\n']}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  s="${s//$'\u200b'/}"; s="${s//$'\u200c'/}"; s="${s//$'\u200d'/}"; s="${s//$'\ufeff'/}"
  printf '%s\n' "$s"
}
expand_path() {
  local p="$1"
  p="${p//[$'\r\n']/}"
  if [[ "$p" == "~" ]]; then printf '%s\n' "$HOME"; return; fi
  if [[ "${p:0:2}" == "~/" ]]; then printf '%s\n' "$HOME/${p:2}"; return; fi
  printf '%s\n' "$p"
}

INSTALL_DIR_RAW="$(sanitize_path_input "$INSTALL_DIR_RAW")"
INSTALL_DIR="${INSTALL_DIR_RAW:-$DEFAULT_BASE}"
INSTALL_DIR="$(expand_path "$INSTALL_DIR")"
if command -v realpath >/dev/null 2>&1; then INSTALL_DIR="$(realpath -m "$INSTALL_DIR")"; fi
INSTALL_DIR="${INSTALL_DIR%/}"
mkdir -p "$INSTALL_DIR"
log "Install dir: raw='${INSTALL_DIR_RAW}' -> resolved='${INSTALL_DIR}'"

# Sudo preflight (keep alive)
if command -v sudo >/dev/null 2>&1 && [ "$EUID" -ne 0 ] && [ "${SKIP_SUDO_PREFLIGHT:-0}" != "1" ]; then
  log "Acquiring sudo once (kept alive during install)..."
  sudo -v || err "sudo required (or set SKIP_SUDO_PREFLIGHT=1 to skip)."
  ( while true; do sleep 60; sudo -n true 2>/dev/null || break; kill -0 "$$" || exit; done ) &
  SUDO_KEEPALIVE_PID=$!
  trap '[[ -n "${SUDO_KEEPALIVE_PID:-}" ]] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT
fi

VENV_DIR="$INSTALL_DIR/venv"
CONDA_DIR_DEFAULT="${CONDA_DIR:-$INSTALL_DIR/miniconda}"
CONDA_DIR="$CONDA_DIR_DEFAULT"
CONDA_PKGS_DIRS_DEFAULT="$INSTALL_DIR/.conda_pkgs"
if is_case_insensitive_fs "$CONDA_DIR"; then
  warn "Case-insensitive filesystem detected at $CONDA_DIR (likely Windows/NTFS under WSL). Conda needs case-sensitive paths."
  CONDA_DIR="$HOME/.photonframe_conda/miniconda"
  : "${CONDA_PKGS_DIRS:=$HOME/.photonframe_conda/pkgs}"
  log "Using case-sensitive Conda dir: $CONDA_DIR (pkgs: $CONDA_PKGS_DIRS)"
else
  : "${CONDA_PKGS_DIRS:=$CONDA_PKGS_DIRS_DEFAULT}"
fi
if is_case_insensitive_fs "$CONDA_PKGS_DIRS"; then
  warn "CONDA_PKGS_DIRS is on a case-insensitive filesystem; switching to $HOME/.photonframe_conda/pkgs."
  CONDA_PKGS_DIRS="$HOME/.photonframe_conda/pkgs"
fi
mkdir -p "$(dirname "$CONDA_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD_INSTALL="${CMD_INSTALL:-conda}"
export PATH="$CONDA_DIR/bin:$PATH"
CONDA_PREFIX="$CONDA_DIR/envs/$ENV_NAME"

# Persistent caches
export CONDA_PKGS_DIRS
export PIP_CACHE_DIR="$INSTALL_DIR/.pipcache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$INSTALL_DIR/.torchcache" "$INSTALL_DIR/.cache"

# Optional payload archive extraction
ARCHIVE="$SCRIPT_DIR/src.tar.gz"
[ -f "$ARCHIVE" ] && log "Extracting src.tar.gz -> $INSTALL_DIR" && tar -xzf "$ARCHIVE" -C "$INSTALL_DIR"

#Create log location and set log
LOG_DIR="$INSTALL_DIR/logs"
mkdir -p "$LOG_DIR"
if [ -f "$INSTALL_DIR/src/definitions.py" ]; then
  sed -i "s|LOG_DIRECTORY|$LOG_DIR|g" "$INSTALL_DIR/src/definitions.py"
else
  warn "definitions.py not found - skipping log dir patch."
fi

# Ask whether to install AI features
INSTALL_AI=true
ask_question "Install AI features (Real-ESRGAN, RealCUGAN, PyTorch, CUDA, ...)?" "y" || INSTALL_AI=false
if $INSTALL_AI; then
  # Nur relevant, wenn KI installiert wird: steuert Auswahl Torch/NCNN im Backend-Pick
  ask_prefer_gpu

  warn "CodeFormer requires the NON-COMMERCIAL S-Lab License 1.0. Do not use it commercially."
  if ask_question "Install CodeFormer (face restoration, NON-COMMERCIAL S-Lab License 1.0)?" "n"; then
    INSTALL_CODEFORMER=true
    log "CodeFormer installation enabled."
  else
    INSTALL_CODEFORMER=false
    log "CodeFormer installation skipped (non-commercial license)."
  fi
else
  INSTALL_CODEFORMER=false
fi

# Conda bootstrap (force local install if missing OR not using our CONDA_DIR)
CURRENT_CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CURRENT_CONDA_BIN" || "$CURRENT_CONDA_BIN" != "$CONDA_DIR/bin/conda" ]]; then
  ensure_aria2c
  wait_for_net 999999 10
  log "Conda missing or not at $CONDA_DIR - installing Miniconda locally..."
  OS_NAME_BOOT="$(detect_os_name)"
  case "$OS_NAME_BOOT" in
    mac)  INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh" ;;            # simple default; arm users may switch manually
    windows) err "Windows is not supported by this bash installer." ;;
    *)    INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
  esac
  URL="https://repo.anaconda.com/miniconda/$INSTALLER"
  download_with_retries "$URL" "/tmp/$INSTALLER" 80 10
  bash "/tmp/$INSTALLER" -b -p "$CONDA_DIR"
  rm -f "/tmp/$INSTALLER"
  export PATH="$CONDA_DIR/bin:$PATH"
  if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    err "Miniconda installation failed at $CONDA_DIR (conda binary missing)."
  fi
fi

# Accept Anaconda ToS early to avoid repeated prompts (needs conda binary in PATH)
conda_accept_tos

# Safe activation
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  err "conda.sh not found after install (expected under $CONDA_DIR)."
fi

# Use mamba if present
if command -v mamba &>/dev/null; then CMD_INSTALL="mamba"; else CMD_INSTALL="conda"; fi
export CONDA_SUBDIR="linux-64"

ROOT_CONDARC="$CONDA_DIR/.condarc"
rm -f "$ROOT_CONDARC"
if [[ "${VIDEO_ALLOW_ANACONDA_DEFAULTS:-0}" == "1" ]]; then
  cat > "$ROOT_CONDARC" <<'YAML'
channels:
  - conda-forge
  - pytorch
  - defaults
channel_priority: strict
add_pip_as_python_dependency: false
default_channels:
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
YAML
else
  cat > "$ROOT_CONDARC" <<'YAML'
channels:
  - conda-forge
  - pytorch
channel_priority: strict
add_pip_as_python_dependency: false
default_channels: []
YAML
  warn "Skipping 'defaults' channel to avoid Anaconda ToS prompts. Set VIDEO_ALLOW_ANACONDA_DEFAULTS=1 to re-enable."
fi
export CONDARC="$ROOT_CONDARC"
export CONDA_HTTP_TIMEOUT=60
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30
export CONDA_REMOTE_READ_TIMEOUT_SECS=60
log "Root condarc set (strict), timeouts tuned."
conda_accept_tos
conda config --env --set channel_priority strict || true
conda clean --index-cache -y || true
rm -f "$CONDA_PREFIX/conda-meta/pinned" 2>/dev/null || true

# Env create/activate
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  conda_retry 24 create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi
conda activate "$ENV_NAME"
log "Activated env '$ENV_NAME' (Python $(python -V | awk '{print $2}'))"
unset PYTHONPATH || true


# Core packages
log "Installing core packages (conda-forge, strict)..."
"$CMD_INSTALL" info >/dev/null || err "Package runner '${CMD_INSTALL}' not functional."

PKGS_BASE_1=( "pip<25" setuptools wheel pillow psutil wcwidth "argcomplete>=3.1,<4" )
PKGS_BASE_2=( "tk>=8.6.13,<8.7" )
conda_retry 24 install -y "${PKGS_BASE_1[@]}"
conda_retry 24 install -y "${PKGS_BASE_2[@]}"

if $INSTALL_AI; then
  # PyTorch + AI stack deps (torch pulls most of these, but we install them only when needed)
  PKGS_AI_BASE=( "numpy<2" "typing-extensions" "packaging" "pyyaml" "fsspec" "filelock" "networkx" "sympy" "llvmlite" "lz4" "requests" )
  conda_retry 24 install -y "${PKGS_AI_BASE[@]}"
fi

# Terminal image preview utils (best effort)
ensure_term_image_stack

# OpenCV headless (pip; cached/offline capable)
SITE_PKGS="$(get_site_packages)"
PIP_BASE_PKGS=( "argcomplete>=3.1,<4" )
if $INSTALL_AI; then
  PIP_BASE_PKGS+=("opencv-python-headless<4.12")
fi
venv_pip_install_into_env_from_index "$SITE_PKGS" "${PIP_BASE_PKGS[@]}"

# ffmpeg via conda (fallback to system)
if ! command -v ffmpeg >/dev/null 2>&1; then
  if ! conda_retry 16 install -y ffmpeg; then
    warn "Conda ffmpeg failed - trying system ffmpeg."
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq || true
      apt_install_safe ffmpeg || true
      if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -qE "vidstab(transform|detect)"; then
        apt_install_safe libvidstab1.1 libvidstab-dev || true
      fi
    fi
  fi
fi
command -v ffmpeg >/dev/null || err "ffmpeg could not be installed."

# Capture stack (best effort)
ensure_capture_stack

# ⬇️ NEU: ExifTool bereitstellen (Conda-first, sonst System)
ensure_exiftool

# ───────────── AI stack (PyTorch + NCNN binaries + Real-ESRGAN + BasicVSR++) ─────────────
AI_BACKEND_PICK="ncnn"
TORCH_DEVICE_PICK="cpu"
GPU_INFO=($(detect_gpu_backend))
HAS_GPU="${GPU_INFO[0]:-false}"
GPU_BKND="${GPU_INFO[1]:-none}"
OS_NAME="$(detect_os_name)"

if $INSTALL_AI; then
  # PyTorch
  install_pytorch_smart

  # Decide torch device
  if python - <<'PY'
import sys
try:
    import torch
    if torch.cuda.is_available(): print("cuda"); sys.exit(0)
    import platform
    if platform.system().lower()=="darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): print("mps"); sys.exit(0)
    print("cpu"); sys.exit(0)
except Exception:
    print("cpu"); sys.exit(0)
PY
  then
    :
  fi > /tmp/_torch_dev_pick
  TORCH_DEVICE_PICK="$(cat /tmp/_torch_dev_pick || echo cpu)"

  # Backend pick logic
  if $PREFER_GPU; then
    if [[ "$TORCH_DEVICE_PICK" == "cuda" || "$TORCH_DEVICE_PICK" == "mps" ]]; then
      AI_BACKEND_PICK="torch"
    else
      # Try NCNN GPU (Vulkan) as GPU-ish fallback
      AI_BACKEND_PICK="ncnn"
    fi
  else
    # User prefers CPU: torch still fine but we default to ncnn for low footprint unless torch is already good
    if [[ "$TORCH_DEVICE_PICK" == "cuda" || "$TORCH_DEVICE_PICK" == "mps" ]]; then
      AI_BACKEND_PICK="torch"
    else
      AI_BACKEND_PICK="torch"  # torch+cpu gives best BasicVSR++ compatibility
    fi
  fi

  log "AI backend choice: $AI_BACKEND_PICK (torch_device=$TORCH_DEVICE_PICK, gpu_backend_guess=$GPU_BKND)"

  # NCNN Vulkan stack (realesrgan/realcugan) available as fallback on all OS
  install_vulkan_loader "$OS_NAME"
  install_realesrgan_ncnn_vulkan "$OS_NAME"
  install_realesrgan_ncnn_models
  install_realcugan_ncnn_vulkan "$OS_NAME"

# NCNN Model-Sanity
if command -v realcugan-ncnn-vulkan >/dev/null 2>&1; then
  RC_WRAP="$(command -v realcugan-ncnn-vulkan)"
  RC_MROOT="$(grep -Eo '\-m[[:space:]]+"([^"]+)"' "$RC_WRAP" | sed -E 's/.*"([^"]+)".*/\1/' | head -n1 || true)"
  if [[ -z "$RC_MROOT" || ! -d "$RC_MROOT" || -z "$(ls -1 "$RC_MROOT"/models-* 2>/dev/null)" ]]; then
    warn "RealCUGAN: keine models-* Verzeichnisse unter '${RC_MROOT:-<unbekannt>}' - bitte nenne mir die Log-Ausgabe."
  else
    log "RealCUGAN models root: $RC_MROOT (ok)"
  fi
fi

  # Real-ESRGAN (PyTorch) + weights + FIX + ENV
  install_realesrgan_python_and_models


  # nach install_realesrgan_python_and_models ...
actd="$CONDA_PREFIX/etc/conda/activate.d"; mkdir -p "$actd"
esrgan_defaults="$actd/98-esrgan-defaults.sh"
cat > "$esrgan_defaults" <<'EOS'
# Autogenerated: ESRGAN defaults (safe if files exist)
VM_BASE="INSTALL_DIR"
if [ -z "${ESRGAN_DEFAULT_MODEL_PATH:-}" ]; then
  for c in \
    "\$VM_BASE/real-esrgan/weights/realesr-general-x4v3.pth:realesr-general-x4v3" \
    "\$VM_BASE/real-esrgan/weights/RealESRGAN_x4plus.pth:RealESRGAN_x4plus" \
    "\$VM_BASE/real-esrgan/weights/RealESRGAN_x4plus_anime_6B.pth:RealESRGAN_x4plus_anime_6B" \
    "\$VM_BASE/real-esrgan/weights/RealESRGAN_x2plus.pth:RealESRGAN_x2plus" \
    "\$VM_BASE/weights/realesr-animevideov3.pth:realesr-animevideov3"
  do
    p="\${c%%:*}"; n="\${c##*:}";
    [ -f "\$p" ] && export ESRGAN_DEFAULT_MODEL_PATH="\$p" ESRGAN_DEFAULT_MODEL_NAME="\$n" && break
  done
fi
EOS

sed -i "s|INSTALL_DIR|$INSTALL_DIR|g" "$esrgan_defaults"

else
  log "AI stack skipped by user."
fi

# ─────────────────────────── Diagnose core versions ─────────────────────────
log "Checking Python/NumPy/OpenCV/ffmpeg..."
python - <<'PY'
import importlib, sys

def safe_ver(name: str) -> tuple[str, bool]:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "n/a"), True
    except Exception:
        return "n/a", False

np_ver, np_ok = safe_ver("numpy")
cv_ver, cv_ok = safe_ver("cv2")

print("Python:", sys.version.split()[0])
print("NumPy :", np_ver, "(missing)" if not np_ok else "")
print("OpenCV:", cv_ver, "(missing)" if not cv_ok else "")
PY
ffmpeg -hide_banner -version | head -n 1

# ─────────────────────────── Write config.ini (persist) ─────────────────────
GPU_ENABLED_CFG="false"
[[ "$GPU_BKND" != "none" ]] && GPU_ENABLED_CFG="true"
AI_ENABLED_CFG="false"
$INSTALL_AI && AI_ENABLED_CFG="true"
CODEFORMER_CFG="false"
$INSTALL_CODEFORMER && CODEFORMER_CFG="true"
write_config_ini "$VM_LANG" "$OS_NAME" "$GPU_ENABLED_CFG" "$GPU_BKND" "$AI_BACKEND_PICK" "$TORCH_DEVICE_PICK"


# ─────────────────────────── Runner & shell alias (aus src.tar.gz) ──────────
RUNNER_SRC="$INSTALL_DIR/video"
if [ -f "$RUNNER_SRC" ]; then
  # Stelle sicher, dass wir eine frische Runner-Vorlage verwenden (Placeholders wiederherstellen)
  if [ -f "$SCRIPT_DIR/video" ]; then
    src_runner="$(realpath -m "$SCRIPT_DIR/video")"
    dest_runner="$(realpath -m "$RUNNER_SRC")"
    if [ "$src_runner" != "$dest_runner" ]; then
      cp "$src_runner" "$dest_runner"
    fi
  fi
  # Platzhalter ersetzen
  sed -i "s|ENV_NAME_PLACEHOLDER|$ENV_NAME|g" "$RUNNER_SRC"
  sed -i "s|APP_PATH_PLACEHOLDER|$INSTALL_DIR|g" "$RUNNER_SRC"
  sed -i "s|CONDA_DIR_PLACEHOLDER|$CONDA_DIR|g" "$RUNNER_SRC"
  chmod +x "$RUNNER_SRC" || true
  mkdir -p "$HOME/.local/bin"
  ln -sf "$RUNNER_SRC" "$HOME/.local/bin/$VIDEO_CMD"
  echo "[install] Runner aktualisiert: $RUNNER_SRC (env=$ENV_NAME) -> ~/.local/bin/$VIDEO_CMD"
else
  warn "Kein Runner 'video' im Installationsverzeichnis gefunden - ich lasse die alte Wrapper-Generierung aus."
fi

# ───────────────── Argcomplete hooks for Bash/Zsh (user scope) ──────────────
setup_argcomplete_hooks() {
  local BIN="$CONDA_PREFIX/bin"
  local REG="$BIN/register-python-argcomplete"
  local ACT="$BIN/activate-global-python-argcomplete"

  if [ ! -x "$REG" ]; then
    warn "register-python-argcomplete not found in $BIN – reinstalling argcomplete."
    conda_retry 4 install -y "argcomplete>=3.1,<4" || true
  fi

  # (Optional) Global hook - not strictly required
  if [ -x "$ACT" ]; then "$ACT" --user >/dev/null 2>&1 || true; fi

  # Bash: idempotent, mit ABSOLUTEM Pfad
  if ! grep -qs '# >>> video (argcomplete)' "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<EOF

# >>> video (argcomplete)
if [ -x "$REG" ]; then
  eval "\$("$REG" video)"
fi
# <<< video (argcomplete)
EOF
    echo "[install] Added argcomplete snippet to ~/.bashrc"
  fi

  # Zsh: bashcompinit + absoluter Pfad
  if ! grep -qs '# >>> video (argcomplete)' "$HOME/.zshrc" 2>/dev/null; then
    cat >> "$HOME/.zshrc" <<EOF

# >>> video (argcomplete)
autoload -U +X bashcompinit && bashcompinit
if [ -x "$REG" ]; then
  eval "\$("$REG" video)"
fi
# <<< video (argcomplete)
EOF
    echo "[install] Added argcomplete snippet to ~/.zshrc"
  fi
}

setup_argcomplete_hooks

install_impact_font "$(detect_os_name)"
write_third_party_licenses

# Copy config.ini from the archive into XDG_CONFIG_HOME if missing
CFG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CFG_DIR="$CFG_HOME/PhotonFrameVideoKit"
CFG_PATH="$CFG_DIR/config.ini"
mkdir -p "$CFG_DIR"

if [ -f "$INSTALL_DIR/config.ini" ] && [ ! -f "$CFG_PATH" ]; then
  cp -n "$INSTALL_DIR/config.ini" "$CFG_PATH"
  log "$(loc_printf "config.ini copied from src.tar.gz -> %s" "$CFG_PATH")"
else
  log "Existing config.ini kept or none found in archive."
fi

# ~/.local/bin auf PATH bringen, falls fehlt
if ! printf '%s' "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
  if ! grep -qs 'export PATH=.*\$HOME/.local/bin' "$HOME/.bashrc" "$HOME/.profile" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo "[install] Added ~/.local/bin to PATH in ~/.bashrc."
  fi
fi

WRAP="$HOME/.local/bin/$VIDEO_CMD"
# Mini-Smoke-Test (kein Abbruch bei Fehler)
if "$WRAP" -h >/dev/null 2>&1; then
  log "Wrapper sanity OK (video -h)."
else
  warn "Wrapper did not print -h; try: VM_TRACE=1 $WRAP -h"
fi

log "🎉 Done."
