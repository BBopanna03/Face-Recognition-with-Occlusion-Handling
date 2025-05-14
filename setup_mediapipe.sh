#!/bin/bash
# setup_mediapipe.sh - Script to set up the MediaPipe Face Recognition system

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv mediapipe_venv

# Activate virtual environment
echo "Activating virtual environment..."
source mediapipe_venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install opencv-python numpy scikit-learn face-recognition mediapipe

# Create necessary directories
echo "Creating directories..."
mkdir -p data/faces
mkdir -p models

echo "Setup complete!"
echo ""
echo "To activate the environment, run: source mediapipe_venv/bin/activate"
echo ""
echo "Available commands:"
echo "  python mediapipe_face_recognition.py --mode collect --name \"Your Name\"  # Collect face data"
echo "  python mediapipe_face_recognition.py --mode train                      # Train the model"
echo "  python mediapipe_face_recognition.py --mode recognize                  # Run face recognition"
echo ""
echo "Advanced options:"
echo "  --threshold 0.4       # Recognition confidence threshold (lower = more lenient)"
echo "  --skip-frames 2       # Skip N frames for better performance"
echo "  --num-images 20       # Number of images to collect in collection mode"