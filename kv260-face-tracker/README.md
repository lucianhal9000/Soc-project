# Real-Time Face Detection & Tracking on Xilinx Kria KV260

> **Platform:** Xilinx Kria KV260 | **OS:** Ubuntu 22.04 LTS | **AI Toolkit:** Vitis-AI v2.5 | **Vision:** OpenCV 4.x

Detects and tracks human faces in real time using a DPU-accelerated neural network running on the KV260's FPGA fabric. Each face receives a persistent ID, a smoothed bounding box, and a coloured corner-accent overlay.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Requirements](#software-requirements)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Detailed Setup Guide](#detailed-setup-guide)
7. [Building & Running](#building--running)
8. [Tracker Parameters](#tracker-parameters)
9. [Troubleshooting](#troubleshooting)
10. [Command Reference](#command-reference)

---

## How It Works

```
Camera → ARM CPU grabs frame → sends to FPGA (DPU) → AI finds faces
      → ARM CPU runs NMS filter → Tracker assigns IDs → Screen shows boxes
```

| Component | Role |
|---|---|
| **ARM Cortex-A53 (PS)** | Runs Ubuntu, your C++ app, camera I/O, tracker & NMS |
| **FPGA DPU (PL)** | Executes the `densebox_320_320` neural network in parallel hardware |
| **Vitis-AI** | Loads the compiled model onto the DPU at runtime |
| **OpenCV** | Camera capture, drawing, and NMS deduplication |

### Key Algorithms

- **DPU inference** — `densebox_320_320` model detects face bounding boxes in 320×320 crops
- **NMS** (`cv::dnn::NMSBoxes`) — removes duplicate overlapping detections
- **IoU + proximity matching** — pairs each detection to an existing tracked face using a blended score (IoU × 0.7 + distance × 0.3) with velocity prediction
- **EMA smoothing** (`alpha = 0.6`) — reduces box jitter between frames
- **Age gating** (`minAge = 3`) — suppresses flickering IDs from single-frame false detections

---

## Hardware Requirements

| Item | Notes |
|---|---|
| Xilinx Kria KV260 board | Must be the KV260 specifically |
| 12V DC power adapter | Use the one included in the box |
| MicroSD card ≥ 16 GB (Class 10) | Use a quality brand; cheap cards cause mysterious failures |
| MicroSD USB card reader | Any standard reader |
| USB webcam (UVC-compatible) | e.g. Logitech C270 |
| Monitor + DisplayPort cable | KV260 uses DisplayPort, **not** HDMI |
| Ethernet cable | Required during setup; Wi-Fi not natively supported |
| Windows / Mac / Linux PC | To flash the SD card and SSH into the board |

---

## Software Requirements

**On your PC:**
- [Balena Etcher](https://etcher.balena.io) — to flash the Ubuntu image to the SD card
- SSH client — built into Windows 10+, macOS, and Linux terminals

**On the board (installed by `scripts/setup.sh`):**
- Ubuntu 22.04 LTS for KV260
- OpenCV (`libopencv-dev`)
- Vitis-AI v2.5

---

## Quick Start

```bash
# 1. Clone this repo onto the KV260 board
git clone https://github.com/YOUR_USERNAME/kv260-face-tracker.git
cd kv260-face-tracker

# 2. Run the full setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Load the DPU bitstream (required after every reboot)
chmod +x scripts/load_bitstream.sh
./scripts/load_bitstream.sh

# 4. Build
cd ~/Vitis-AI/examples/Vitis-AI-Library/samples/facedetect
sh build.sh

# 5. Run
./test_video_facedetect_tracked densebox_320_320 0
# Press Q to quit
```

---

## Project Structure

```
kv260-face-tracker/
├── src/
│   └── test_video_facedetect_tracked.cpp   # Main C++ source
├── scripts/
│   ├── setup.sh            # Full environment setup (run once)
│   ├── load_bitstream.sh   # Load DPU bitstream (run after every reboot)
│   └── rc.local            # Drop-in /etc/rc.local for auto-load on boot
└── README.md
```

---

## Detailed Setup Guide

### 1. Install Ubuntu 22.04 on the KV260

1. Download the KV260-specific Ubuntu 22.04 image from [ubuntu.com/download/amd-xilinx](https://ubuntu.com/download/amd-xilinx)
   - File name: `ubuntu-22.04.x-preinstalled-server-arm64+kv260.img.xz`
   - ⚠️ The regular Ubuntu desktop image will **not** boot on this board
2. Flash it to your microSD card using Balena Etcher
3. Insert the SD card (gold contacts down) into the KV260's underside slot
4. Connect cables in this order: Ethernet → DisplayPort → USB webcam → 12V power
5. Wait 2–4 minutes for first boot (filesystem expansion)

### 2. First Login

Find your board's IP from your router's device list, then:

```bash
ssh ubuntu@<BOARD_IP>
# default password: ubuntu (you'll be prompted to change it)
```

Update the system:

```bash
sudo apt update && sudo apt upgrade -y && sudo reboot
```

### 3. Understanding PS vs PL (Why the Bitstream Step Exists)

The KV260's Zynq chip has two halves:

| Processing System (PS) | Programmable Logic (PL) |
|---|---|
| ARM Cortex-A53 CPU | FPGA fabric |
| Runs Ubuntu & your code | Contains the DPU accelerator |
| Persistent across reboots | **Loses configuration when power is cut** |

The DPU configuration lives in the PL. It must be reloaded from a **bitstream** file after every boot. This project uses `kv260-benchmark-b4096` — the only bitstream whose DPU fingerprint matches the `densebox_320_320` model.

| Bitstream | DPU Architecture | Fingerprint | Use? |
|---|---|---|---|
| `kv260-smartcam` | DPUCZDX8G_ISA1_B3136 | …406 vs …407 MISMATCH | ❌ Crashes |
| `kv260-benchmark-b4096` | DPUCZDX8G_ISA1_B4096 | `0x101000016010407` | ✅ Correct |

### 4. Auto-Load Bitstream on Boot (Recommended)

```bash
sudo cp scripts/rc.local /etc/rc.local
sudo chmod +x /etc/rc.local
sudo reboot
# after reboot, verify:
xdputil query  # should show 0x101000016010407
```

---

## Building & Running

The source file must be placed inside the Vitis-AI facedetect samples directory. `scripts/setup.sh` does this automatically. To do it manually:

```bash
cp src/test_video_facedetect_tracked.cpp \
   ~/Vitis-AI/examples/Vitis-AI-Library/samples/facedetect/

cd ~/Vitis-AI/examples/Vitis-AI-Library/samples/facedetect
sh build.sh
```

Run the tracker:

```bash
./test_video_facedetect_tracked densebox_320_320 0
```
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/07f11d1f-bf8b-46c1-9da1-324e9ab38c5b" />


| Argument | Meaning |
|---|---|
| `densebox_320_320` | Face-detection model name (loaded onto the DPU) |
| `0` | Camera device index (`/dev/video0`). Try `1` or `2` if camera not found |

A window titled **Face Tracker** opens showing the live camera with coloured bounding boxes. Press **Q** to quit.

---

## Tracker Parameters

Adjust these constants at the top of the `FaceTracker` class in `src/test_video_facedetect_tracked.cpp`:

| Parameter | Default | Increase → | Decrease → |
|---|---|---|---|
| `maxMissing` | 40 frames | Holds IDs longer when face is hidden | Drops IDs quickly when face disappears |
| `minIoU` | 0.25 | Stricter matching, fewer wrong ID links | Looser matching, may mix up nearby faces |
| `maxDist` | 150 px | Matches faster-moving faces over longer distances | Avoids linking to the wrong face across the frame |
| `minAge` | 3 frames | Longer delay before ID label appears (less flicker) | IDs appear almost immediately (may flicker) |

EMA smoothing alpha is set in `smoothRect()`:

| Alpha | Effect |
|---|---|
| 0.9 | Snappy but jittery |
| **0.6** (default) | Balanced |
| 0.3 | Very smooth but lags behind fast movement |

---

## Troubleshooting

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| `fingerprint mismatch …406 vs …407` | Wrong bitstream loaded | `sudo xmutil loadapp kv260-benchmark-b4096` |
| Board hangs after `./test_video…` | No bitstream loaded after reboot | Run `load_bitstream.sh` before launching |
| `undefined reference to NMSBoxes` | `opencv_dnn` not linked in `build.sh` | Run `scripts/setup.sh` (it applies the `sed` patch) |
| `Cannot open camera` | Wrong camera index | Try `0`, `1`, or `2` as the last argument |
| `load Error: -1` on `loadapp` | Slot already occupied | `sudo xmutil unloadapp` first, then `loadapp` |
| Core dumped on startup | Fingerprint mismatch or no bitstream | `xdputil query` to check fingerprint |
| SSH connection refused | Board still booting | Wait 2 more minutes and retry |
| Board won't boot | Bad SD card write or wrong image | Re-flash with Balena Etcher |
| Black screen on monitor | Wrong cable or DisplayPort issue | KV260 uses DisplayPort, not HDMI |

---

## Command Reference

```bash
# Bitstream management
sudo xmutil listapps                        # list available bitstreams
sudo xmutil unloadapp                       # clear current bitstream
sudo xmutil loadapp kv260-benchmark-b4096  # load correct bitstream
xdputil query                               # verify fingerprint (expect 0x101000016010407)

# Build & run
sh build.sh
./test_video_facedetect_tracked densebox_320_320 0

# Patch build.sh (if not using setup.sh)
sed -i 's/-lopencv_highgui/-lopencv_highgui -lopencv_dnn/' build.sh
grep -o 'lopencv[^ ]*' build.sh             # verify the patch

# Useful diagnostics
ip addr show eth0                           # check board IP
find / -name 'densebox_320_320*' -type d 2>/dev/null   # locate model files
cat -n test_video_facedetect_tracked.cpp    # view source with line numbers
```

---


## License

MIT — see [LICENSE](LICENSE) for details.
