#!/bin/bash

# Voice Detection API - Startup Script
# Downloads models and starts the Flask application

echo "🚀 Starting Voice Detection ML Service..."

# Download models if not present
if [ ! -f "models/resnet_asvspoof.pt" ]; then
    echo "📥 Models not found, downloading..."
    python download_models.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download models!"
        exit 1
    fi
else
    echo "✅ Models already present"
fi

# Start the Flask application
echo "🌐 Starting Flask server on port 5000..."
exec python app.py
