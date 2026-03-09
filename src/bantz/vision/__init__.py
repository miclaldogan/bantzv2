"""
Bantz v3 — Vision subsystem (#120)

Remote VLM screenshot analysis — Jetson/Colab fallback when AT-SPI
can't read UI elements (custom-drawn UIs, Electron apps with poor
accessibility).

Modules:
    screenshot.py   — capture full screen or active window, ROI crop
    remote_vlm.py   — REST client for Jetson / Colab / Ollama VLM endpoint
"""
