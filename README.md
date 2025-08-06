# 🖼️ Image Caption Generator

A powerful AI-powered web application that generates human-like captions for **images**, **videos**, and **live camera snapshots** using deep learning models. Built with Flask and powered by Hugging Face Transformers, this tool combines Computer Vision and NLP for accurate and multilingual captioning — with audio narration support.

---

## 🎯 Overview

The **Image Caption Generator** is a robust application designed to:

- Automatically generate contextual captions for uploaded images, video frames, and live camera images.
- Support multilingual translation and text-to-speech (TTS) audio output.
- Provide an intuitive UI with drag-and-drop upload, GPU acceleration, and real-time caption display.

---

## ✨ Features

### 🖼️ Image Captioning

- Upload formats: PNG, JPG, JPEG, GIF, BMP, WebP  
- Real-time, high-quality caption generation using ViT + GPT-2  
- Simple drag-and-drop image upload interface  

### 🎥 Video Captioning

- Formats supported: MP4, AVI, MOV, MKV, WebM  
- Smart frame sampling every 2 seconds  
- Batch caption generation with frame-wise audio narration  

### 📸 Live Camera Capture

- Capture real-time snapshots using your webcam  
- Instantly generate captions for captured images  
- Option to download image and caption  

### 🌍 Multi-language Support

- Supports 10+ languages using Google Translate API  
- Available languages: English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese  

### 🔊 Text-to-Speech (TTS)

- Converts generated captions to MP3 audio using Google TTS  
- Language-specific voice output  
- Embedded web audio player  

### 🚀 Advanced Features

- Clean and responsive Bootstrap 5 UI  
- CUDA GPU acceleration for faster captioning  
- Real-time progress tracking with visual feedback  
- Automatic cleanup of uploaded and generated files  
- API health check and basic logging  

---

## 🛠️ Technology Stack

- **Backend:** Flask, Python 3.8+  
- **AI/ML Models:** Hugging Face Transformers (ViT + GPT-2), PyTorch  
- **Computer Vision:** OpenCV, Pillow  
- **NLP Tools:** Google Translate API, Google Text-to-Speech (gTTS)  
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5  
- **Camera Integration:** WebRTC (via JavaScript and Flask route handling)  
- **Deployment:** Docker, Gunicorn, Nginx  

---

## 📋 Prerequisites

- Python 3.8+  
- `pip` package manager  
- At least 4GB RAM (8GB recommended)  
- CUDA-enabled GPU (optional but beneficial)  
- Internet connection (required for translation and TTS)  
- Webcam (for live camera capture feature)

---

## 🚀 Installation and setup

```bash
git clone https://github.com/keerthisaa/ImageCaptionGenerator.git
cd ImageCaptionGenerator

# Create and activate a virtual environment
python -m venv image_caption_env

# Windows
image_caption_env\\Scripts\\activate

# Linux/Mac
source image_caption_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

#Then open http://localhost:5000 in your browser.
