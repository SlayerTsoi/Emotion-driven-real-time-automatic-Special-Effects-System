# Emotion-driven Real-time Automatic Special Effects System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Introduction

This project implements a **real-time automatic special effects system driven by emotion recognition**. Using a camera to capture facial expressions, the system identifies emotions (e.g., happiness, sadness, surprise) in real time and automatically adds matching visual effects to the video stream. It enhances interactive experiences such as video calls, live streaming, and entertainment applications.

## ✨ Features

- **Real-time face detection**: Utilizes Haar Cascade for fast face localization.
- **Facial landmark detection**: Accurately detects facial key points to improve emotion classification.
- **Emotion classification**: Supports multiple basic emotions (happy, sad, angry, surprised, neutral, etc.).
- **Automatic effects**: Triggers corresponding visual effects based on the detected emotion.
- **Real-time performance**: Optimized algorithms ensure smooth real-time video processing.

## 🏗️ System Architecture
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Camera Input │ ──▶ │ Face Detection │ ──▶ │ Emotion Model │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Effect Output │ ◀── │ Effect Mapping │ ◀── │ Emotion Result │
└─────────────────┘ └─────────────────┘ └─────────────────┘

text

## 📋 Requirements

- Python 3.7 or higher
- A camera supported by OpenCV
- 4GB+ RAM recommended; modern CPU or CUDA-enabled GPU for better performance

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SlayerTsoi/Emotion-driven-real-time-automatic-Special-Effects-System.git
cd Emotion-driven-real-time-automatic-Special-Effects-System
2. Install dependencies
```
```bash
pip install -r requirements.txt
```
requirements.txt typically includes:

opencv-python

numpy

tensorflow / pytorch (depending on your model)

dlib (optional, for facial landmarks)

other necessary libraries

3. Run the project
Full system:

```bash
python run.py
```
Run emotion detection module only (for debugging):

```bash
python run_emotion_detection.py
```
Run landmark detection module only (for debugging):

```bash
python run_landmark_detection.py
```
4. Usage
After launching, the system will automatically open the camera.

Position your face in front of the camera with adequate lighting.

The system will display a bounding box around your face and show the recognized emotion.

Visual effects will be overlaid based on the current emotion.

Press q to quit.

📁 File Structure
text
Emotion-driven-real-time-automatic-Special-Effects-System/
├── haarcascade_files/          # Haar Cascade face detection models
├── images/                     # Project images and resources
├── models/                     # Trained emotion recognition models
├── train/                      # Training code and data
├── run.py                      # Main entry point
├── run_emotion_detection.py    # Standalone emotion detection
├── run_landmark_detection.py   # Standalone landmark detection
├── requirements.txt            # Python dependencies
└── README.txt                  # Original readme file

🔧 Configuration
You can adjust effect styles or sensitivity by modifying the following parameters:

Parameter	Description	Default
EMOTION_THRESHOLD	Confidence threshold for emotion classification	0.6
EFFECT_INTENSITY	Intensity factor for visual effects	0.8
FACE_DETECT_SCALE	Scale factor for face detection	1.1
CAMERA_INDEX	Camera device index	0

🧠 Model Training
If you wish to retrain the emotion recognition model:

Prepare an emotion dataset (e.g., FER2013, CK+).

Place the dataset in train/data/.

Run the training script:

```bash
python train/train_emotion_model.py
```
The trained model will be saved in the models/ directory.

📊 Performance
Environment	Frame Rate (FPS)	Latency (ms)
CPU (Intel i5)	15-20	~50
GPU (NVIDIA GTX)	25-30	~30
Note: Actual performance depends on hardware and effect complexity.

🛠️ Tech Stack
OpenCV: Image processing and face detection

TensorFlow / PyTorch: Deep learning inference

NumPy: Numerical computations

dlib / MediaPipe (optional): Facial landmark detection

📌 Important Notes
Pre-trained model weights may be downloaded on the first run.

Ensure camera permissions are granted.

Poor lighting may reduce emotion recognition accuracy.

For best results, use in a well-lit environment.
