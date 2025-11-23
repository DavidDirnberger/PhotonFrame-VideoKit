#!/usr/bin/env bash
# uninstall.sh – removes a PhotonFabric VideoKit install (safe, with prompts)
set -euo pipefail

warn() { echo -e "\e[33m[uninstall]\e[0m $*"; }
err()  { echo -e "\e[31m[uninstall]\e[0m $*" >&2; exit 1; }
log()  { echo -e "\e[32m[uninstall]\e[0m $*"; }

DEFAULT_DIR="$(pwd)"
read -r -p "Install directory to remove [${DEFAULT_DIR}]: " INSTALL_DIR_RAW || INSTALL_DIR_RAW="$DEFAULT_DIR"
INSTALL_DIR="${INSTALL_DIR_RAW:-$DEFAULT_DIR}"
INSTALL_DIR="${INSTALL_DIR%/}"

if [[ -z "$INSTALL_DIR" || "$INSTALL_DIR" == "/" ]]; then
  err "Refusing to operate on an empty path or root."
fi

log "Planned removal target: $INSTALL_DIR"
read -r -p "Proceed with uninstall? [y/N]: " CONFIRM || CONFIRM="n"
case "${CONFIRM,,}" in
  y|yes) ;;
  *) err "Aborted."; ;;
esac

# Remove launcher symlink if it points to this install
WRAP="$HOME/.local/bin/video"
if [[ -L "$WRAP" ]]; then
  TARGET="$(readlink -f "$WRAP" 2>/dev/null || readlink "$WRAP")"
  if [[ "$TARGET" == "$INSTALL_DIR/video" ]]; then
    rm -f "$WRAP"
    log "Removed launcher symlink: $WRAP"
  else
    warn "Launcher points elsewhere ($TARGET) – left untouched."
  fi
fi

# Remove conda env + miniconda under install dir
if [[ -d "$INSTALL_DIR/miniconda" ]]; then
  log "Removing miniconda under $INSTALL_DIR/miniconda"
  rm -rf "$INSTALL_DIR/miniconda"
fi

# Remove cached data under install dir
for d in ".conda_pkgs" ".pipcache" ".torchcache" ".cache" "logs"; do
  if [[ -d "$INSTALL_DIR/$d" ]]; then
    rm -rf "$INSTALL_DIR/$d"
    log "Removed $INSTALL_DIR/$d"
  fi
done

# Remove install dir itself if it looks like our project
if [[ -f "$INSTALL_DIR/src/videoManager.py" || -f "$INSTALL_DIR/videoManager.py" ]]; then
  rm -rf "$INSTALL_DIR"
  log "Removed install directory $INSTALL_DIR"
else
  warn "Install directory does not look like PhotonFabric VideoKit; left untouched."
fi

# Optional: user config
CFG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/PhotonFabricVideoKit"
read -r -p "Remove user config at $CFG_DIR? [y/N]: " CONF_CFG || CONF_CFG="n"
case "${CONF_CFG,,}" in
  y|yes)
    rm -rf "$CFG_DIR"
    log "Removed config directory $CFG_DIR"
    ;;
  *) log "Config kept."; ;;
esac

log "Done."
