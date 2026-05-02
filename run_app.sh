#!/usr/bin/env bash
# Launch the Streamlit app and print the URL to use from your iPhone.
# Usage: ./run_app.sh
set -e

# Find this Mac's LAN IP (works on macOS)
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")

if [ -z "$LAN_IP" ]; then
  echo "Could not auto-detect LAN IP. Find it manually:"
  echo "  System Settings -> Wi-Fi -> Details -> IP Address"
  echo ""
else
  echo "============================================================"
  echo "  Open this URL on your iPhone (must be on same WiFi):"
  echo ""
  echo "    http://${LAN_IP}:8501"
  echo ""
  echo "  On your Mac:  http://localhost:8501"
  echo "============================================================"
  echo ""
fi

# Stream Streamlit logs
exec streamlit run app/app.py
