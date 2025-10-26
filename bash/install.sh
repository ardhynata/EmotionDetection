#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Uncomment for debugging (prints each command before running)
#set -x

# Set virtual environment and install Python dependencies
python -m venv venv
pip install -r requirements.txt

# Update package lists
if [ -f ../go.mod ] || [ -f ../go.sum ]; then
    echo "Go module already initialized — skipping"
else
    cd ..
    go mod init emotiondetection
    go mod tidy
fi