#!/bin/bash
# =============================================================================
# load_bitstream.sh — Load the correct DPU bitstream on the KV260
#
# Run manually after every reboot, OR add to /etc/rc.local for auto-load.
# Correct bitstream: kv260-benchmark-b4096 (fingerprint 0x101000016010407)
# =============================================================================

echo "Unloading any currently loaded bitstream..."
sudo xmutil unloadapp

echo "Loading kv260-benchmark-b4096..."
sudo xmutil loadapp kv260-benchmark-b4096

echo "Verifying DPU fingerprint..."
xdputil query

echo ""
echo "Look for fingerprint: 0x101000016010407"
echo "If you see it, the DPU is ready."
