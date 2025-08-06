from flask import Flask, render_template, request, jsonify, send_file
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers import AutoImageProcessor
import torch
from PIL import Image
import os
import uuid
from gtts import gTTS
from googletrans import Translator
import cv2
import numpy as np
import logging
import base64
import io
import time
from functools import wraps
import requests
import tempfile

app = Flask(__name__, static_folder='static')

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
MAX_RETRIES = 3
TIMEOUT = 60
RETRY_DELAY = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model loading with improved error handling
def load_models():
    try:
        logger.info("Loading models...")
        
        # Create model cache directory if it doesn't exist
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir
        ).to(DEVICE)
        
        # Try different feature extractor imports based on transformers version
        try:
            feature_extractor = AutoImageProcessor.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir
            )
        except Exception:
            # Fallback for older versions
            from transformers import AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir
        )
        
        logger.info("Models loaded successfully")
        return model, feature_extractor, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Initialize models
try:
    model, feature_extractor, tokenizer = load_models()
    logger.info(f"Using device: {DEVICE}")
except Exception as e:
    logger.critical(f"Failed to initialize models: {str(e)}")
    # Provide fallback behavior or exit
    model = feature_extractor = tokenizer = None

# Generation parameters
GEN_KWARGS = {
    "max_length": 50,
    "num_beams": 10,
    "early_stopping": True,
    "pad_token_id": 50256  # GPT-2 pad token
}

# Folder setup
UPLOAD_FOLDER = "static/uploads"
AUDIO_FOLDER = "static/audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Translator initialization with error handling
try:
    translator = Translator(service_urls=['translate.google.com'])
except Exception as e:
    logger.warning(f"Translator initialization failed: {e}")
    translator = None

def predict_step(image):
    """Generates captions for an image with error handling."""
    try:
        if model is None or feature_extractor is None or tokenizer is None:
            return "Error: Models not loaded properly"
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process the image
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(DEVICE)
        
        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(pixel_values, **GEN_KWARGS)
        
        preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption: {preds}")
        return preds.strip()
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error generating caption: {str(e)}"

def generate_audio(text, language):
    """Generate audio file from text with improved error handling."""
    try:
        if not text or text.strip() == "":
            return None
            
        audio_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4().hex}.mp3")
        
        # Map some common language codes
        lang_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ja': 'ja',
            'ko': 'ko',
            'zh': 'zh'
        }
        
        lang_code = lang_map.get(language, 'en')
        
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        return None

def translate_text(text, target_language):
    """Translate text with error handling."""
    try:
        if not translator:
            return text
            
        if target_language == 'en':
            return text
            
        result = translator.translate(text, dest=target_language)
        return result.text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text  # Return original text if translation fails

def cleanup_old_files():
    """Clean up old uploaded and audio files."""
    try:
        current_time = time.time()
        
        for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    # Remove files older than 1 hour
                    if file_age > 3600:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login') 
def login(): 
    return render_template('login.html')

@app.route('/icg', methods=["GET", "POST"])
def icg():
    if request.method == "POST":
        if not model:
            return jsonify(error="Model not loaded. Please try again later."), 503
        
        language = request.form.get("language", "en")
        
        try:
            if "image" in request.files:
                return handle_image(request.files["image"], language)
            elif "video" in request.files:
                return handle_video(request.files["video"], language)
            else:
                return jsonify(error="No input provided"), 400
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify(error="Internal server error"), 500
    
    return render_template('icg.html')

def handle_image(image_file, language):
    """Handle image upload and processing."""
    if image_file.filename == "":
        return jsonify(error="No selected file"), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_extension = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
    
    if file_extension not in allowed_extensions:
        return jsonify(error="Invalid file type. Please upload an image."), 400
    
    try:
        # Open and process image
        image = Image.open(image_file.stream)
        
        # Generate caption
        caption = predict_step(image)
        
        if caption.startswith("Error"):
            return jsonify(error=caption), 500
        
        # Translate caption
        translated = translate_text(caption, language)
        
        # Generate audio
        audio_path = generate_audio(translated, language)
        
        response_data = {
            "caption": translated,
            "original_caption": caption if language != 'en' else None
        }
        
        if audio_path:
            response_data["audio_url"] = f"/audio/{os.path.basename(audio_path)}"
            
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return jsonify(error="Failed to process image"), 500

def handle_video(video_file, language):
    """Handle video upload and processing."""
    if video_file.filename == "":
        return jsonify(error="No video selected"), 400
    
    # Validate file type
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    file_extension = video_file.filename.rsplit('.', 1)[1].lower() if '.' in video_file.filename else ''
    
    if file_extension not in allowed_extensions:
        return jsonify(error="Invalid file type. Please upload a video."), 400
    
    video_path = None
    try:
        # Save video file
        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{file_extension}")
        video_file.save(video_path)
        
        captions = []
        audio_urls = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify(error="Could not open video file"), 400
        
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process every 2 seconds of video
        frame_interval = int(frame_rate * 2) if frame_rate > 0 else 60
        processed_frames = 0
        max_frames = 10  # Limit processing to avoid timeouts
        
        logger.info(f"Processing video: {total_frames} frames, {frame_rate} fps")
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_pos % frame_interval != 0:
                continue
                
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Generate caption
            caption = predict_step(pil_image)
            if not caption.startswith("Error"):
                try:
                    translated = translate_text(caption, language)
                    captions.append(translated)
                    
                    # Generate audio for each caption
                    audio_path = generate_audio(translated, language)
                    if audio_path:
                        audio_urls.append(f"/audio/{os.path.basename(audio_path)}")
                    else:
                        audio_urls.append(None)
                        
                except Exception as e:
                    logger.warning(f"Translation/Audio failed: {e}")
                    captions.append(caption)
                    audio_urls.append(None)
            
            processed_frames += 1
            
            # Skip frames to reach next interval
            for _ in range(frame_interval - 1):
                cap.read()
        
        cap.release()
        
        if not captions:
            return jsonify(error="No captions generated from video"), 500
            
        return jsonify(
            captions=captions,
            audio_urls=audio_urls,
            frames_processed=processed_frames
        ), 200
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        return jsonify(error="Video processing error"), 500
    finally:
        # Clean up video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                logger.warning(f"Could not delete video file: {str(e)}")

@app.route('/audio/<filename>')
def audio(filename):
    """Serve audio files."""
    try:
        audio_path = os.path.join(AUDIO_FOLDER, filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        else:
            return jsonify(error="Audio file not found"), 404
    except Exception as e:
        logger.error(f"Audio file error: {str(e)}")
        return jsonify(error="Audio file not found"), 404

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "translator_available": translator is not None
    }
    return jsonify(status)

@app.before_request
def before_request():
    """Run cleanup before each request."""
    cleanup_old_files()

@app.errorhandler(404)
def not_found(error):
    return jsonify(error="Endpoint not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify(error="Internal server error"), 500

if __name__ == "__main__":
    logger.info("Starting Image Caption Generator...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model loaded: {model is not None}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)