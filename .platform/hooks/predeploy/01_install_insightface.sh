#!/bin/bash
# This hook runs after the application code is deployed but before the app starts.
# It installs insightface (which is NOT in requirements.txt because it needs --no-deps)
# and sets up the lightweight dependency mocks.

set -e

echo "=== [predeploy] Installing insightface and onnx ==="

# Activate the EB-managed virtual environment
source /var/app/venv/*/bin/activate

# Install insightface without its heavy dependencies (sklearn, skimage, scipy)
# Our lightweight_deps.py provides numpy-only replacements
pip install --no-cache-dir --no-deps insightface==2.0 onnx

echo "=== [predeploy] insightface installed successfully ==="

# Verify the model files exist (they should be in the deployment zip)
MODEL_DIR="/var/app/staging/model_cache/models/buffalo_s"
if [ -d "$MODEL_DIR" ]; then
    echo "Model files found in $MODEL_DIR:"
    ls -la "$MODEL_DIR"
else
    echo "WARNING: Model directory not found at $MODEL_DIR"
    echo "The face model will be downloaded at first use (this may take time)"
fi

echo "=== [predeploy] Hook completed ==="
