#!/bin/bash
if ! command -v xvfb-run &> /dev/null
then
    sudo apt-get update && sudo apt-get install -y xvfb
fi
xvfb-run -a python -m pytest tests/tools/test_gui_tool.py
