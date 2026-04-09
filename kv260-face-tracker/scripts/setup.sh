#!/bin/bash
# =============================================================================
# setup.sh — Full environment setup for KV260 Face Detection & Tracking
# Run this script on the KV260 board after Ubuntu 22.04 is installed.
# =============================================================================
set -e

echo "================================================================="
echo " KV260 Face Tracker — Environment Setup"
echo "================================================================="

# 1. System update
echo "[1/5] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# 2. Install OpenCV
echo "[2/5] Installing OpenCV..."
sudo apt install -y libopencv-dev

# 3. Clone Vitis-AI v2.5
echo "[3/5] Cloning Vitis-AI v2.5 (shallow clone)..."
cd ~
if [ ! -d "Vitis-AI" ]; then
    git clone https://github.com/Xilinx/Vitis-AI.git \
        --depth 1 --branch v2.5
else
    echo "  Vitis-AI directory already exists — skipping clone."
fi

# 4. Patch build.sh to include opencv_dnn
echo "[4/5] Patching build.sh to link opencv_dnn..."
SAMPLE_DIR=~/Vitis-AI/examples/Vitis-AI-Library/samples/facedetect
cd "$SAMPLE_DIR"

if grep -q "opencv_dnn" build.sh; then
    echo "  build.sh already contains opencv_dnn — skipping patch."
else
    sed -i 's/-lopencv_highgui/-lopencv_highgui -lopencv_dnn/' build.sh
    echo "  Patched successfully."
fi

# 5. Copy source file
echo "[5/5] Copying source file..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/../src/test_video_facedetect_tracked.cpp" "$SAMPLE_DIR/"

echo ""
echo "================================================================="
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Load the DPU bitstream:"
echo "      sudo xmutil unloadapp"
echo "      sudo xmutil loadapp kv260-benchmark-b4096"
echo "      xdputil query   # should show fingerprint 0x101000016010407"
echo ""
echo "   2. Build the project:"
echo "      cd $SAMPLE_DIR"
echo "      sh build.sh"
echo ""
echo "   3. Run the face tracker:"
echo "      ./test_video_facedetect_tracked densebox_320_320 0"
echo "================================================================="
