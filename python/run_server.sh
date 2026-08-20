#!/bin/bash
# Startup script for AI Stock Predictor FastAPI server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting AI Stock Predictor API server..."
echo "📍 API will be available at: http://0.0.0.0:8000"
echo "📖 Swagger API Docs at:     http://0.0.0.0:8000/docs"
echo ""

python3 api.py
