#!/bin/bash

INSTALL_DIR="$HOME/syscripts/videoManager"
VENV_DIR="$INSTALL_DIR/venv"
VIDEO_CMD="video"
SCRIPT_PATH="$VENV_DIR/bin/python $INSTALL_DIR/videoManager.py"
ALIAS_LINE="alias $VIDEO_CMD=\"$SCRIPT_PATH\""

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

#======================== System Requirements =====================
# Check for python3 and venv
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install it first."
    exit 1
fi

if ! python3 -m venv --help &> /dev/null; then
    echo "python3-venv is missing. Attempting to install..."
    sudo apt-get update && sudo apt-get install -y python3-venv || {
        echo "Failed to install python3-venv"; exit 1;
    }
fi

#======================== Virtual Environment =====================
# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Activate and install required packages
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install ffmpeg-python
pip install torch
pip install numpy

deactivate

#======================== Real-ESRGAN Installation =====================
echo "\n[Real-ESRGAN] Checking system requirements..."

# Check for NVIDIA GPU with CUDA
if command -v nvidia-smi &> /dev/null && python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA-fähige NVIDIA-GPU erkannt. Installiere native GPU-Version von Real-ESRGAN..."
    git clone https://github.com/xinntao/Real-ESRGAN "$INSTALL_DIR/Real-ESRGAN"
    cd "$INSTALL_DIR/Real-ESRGAN"

    # Aktivieren und Abhängigkeiten installieren
    source "$VENV_DIR/bin/activate"
    pip install -r requirements.txt
    python scripts/download_pretrained_models.py
    deactivate

    echo "Real-ESRGAN GPU-Version wurde erfolgreich installiert."
else
    echo "Keine CUDA-fähige GPU erkannt. Installiere portable NCNN-Version..."
    mkdir -p "$INSTALL_DIR/Real-ESRGAN-ncnn"
    cd "$INSTALL_DIR/Real-ESRGAN-ncnn"

    # Automatischer Download der aktuellen Version (z. B. Linux-x64)
    NCNN_URL="https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/latest/download/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    curl -L -o realesrgan-ncnn.zip "$NCNN_URL"
    unzip realesrgan-ncnn.zip
    rm realesrgan-ncnn.zip

    echo "NCNN-Version wurde erfolgreich installiert."
fi

echo "\nInstallation abgeschlossen."
exit 0