from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import traceback
import os
from functools import wraps
from typing import Dict, Any
from config import config
from logger import logger
from audio_utils import audio_processor
from detect import voice_detector

app = Flask(__name__)
CORS(app)

# Configure maximum content length for large audio files
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

# Request counter for logging
request_counter = 0

# API Key for authentication (set via environment variable)
API_KEY = os.getenv('API_KEY', 'sk_test_123456789')

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key') or request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({
                "error": "Missing API key",
                "code": "UNAUTHORIZED"
            }), 401
        if api_key != API_KEY:
            return jsonify({
                "error": "Invalid API key",
                "code": "UNAUTHORIZED"
            }), 401
        return f(*args, **kwargs)
    return decorated_function

def create_error_response(message: str, code: str = "ERROR", status: int = 400) -> tuple:
    """Create standardized error response"""
    return jsonify({
        "error": message,
        "code": code,
        "status": "error"
    }), status

def validate_request_data(data: Dict) -> tuple[bool, str]:
    """Validate incoming request data"""
    if not data:
        return False, "Request body is empty"
    
    if 'language' not in data:
        return False, "Missing required field: language"
    
    if 'audio_base64' not in data:
        return False, "Missing required field: audio_base64"
    
    if data['language'] not in config.LANGUAGE_MAP:
        return False, f"Unsupported language: {data['language']}"
    
    if not isinstance(data['audio_base64'], str):
        return False, "audio_base64 must be a string"
    
    if len(data['audio_base64']) < 100:
        return False, "audio_base64 is too short"
    
    return True, ""

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "voice-detection-ml",
            "timestamp": time.time(),
            "gpu_available": config.USE_GPU,
            "models_loaded": True
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/api/detect', methods=['POST'])
@require_api_key
def api_detect():
    """
    Main API endpoint for submission (with authentication)
    This is the endpoint to share for hackathon evaluation
    """
    # Reuse the analyze_voice logic
    return analyze_voice_internal()

@app.route('/analyze', methods=['POST'])
def analyze_voice():
    """Internal endpoint for voice analysis (no auth required)"""
    return analyze_voice_internal()

def analyze_voice_internal():
    """Main endpoint for voice analysis"""
    global request_counter
    request_counter += 1
    
    start_time = time.time()
    request_id = request.headers.get('X-Request-ID', f'req_{request_counter}')
    
    logger.info(f"[{request_id}] Analysis request received")
    
    try:
        # Parse request data
        data = request.get_json()
        
        # Validate request
        is_valid, error_message = validate_request_data(data)
        if not is_valid:
            logger.warning(f"[{request_id}] Validation failed: {error_message}")
            return create_error_response(error_message, "VALIDATION_ERROR", 400)
        
        language = data['language']
        audio_base64 = data['audio_base64']
        
        logger.info(f"[{request_id}] Processing {language} audio, size: {len(audio_base64)} chars")
        
        # Decode audio
        audio, sr = audio_processor.decode_base64_audio(audio_base64)
        logger.debug(f"[{request_id}] Audio decoded: {audio.shape}, sr={sr}")
        
        # Validate audio
        audio_processor.validate_audio(audio, sr)
        
        # Normalize audio
        audio = audio_processor.normalize_audio(audio)
        
        # Extract features
        features = audio_processor.extract_features(audio, sr)
        logger.debug(f"[{request_id}] Features extracted")
        
        # Detect language
        detected_language, lang_confidence = voice_detector.detect_language(audio, sr, language)
        
        # Detect AI vs Human
        classification, confidence, analysis_details = voice_detector.detect_ai_generated(
            audio, sr, features
        )
        
        # Generate explanation
        explanation = voice_detector._generate_explanation(
            classification,
            analysis_details['ai_probability'],
            analysis_details.get('detection_scores', {})
        )
        
        # Build response (matching TypeScript server expectations - snake_case)
        response = {
            "classification": classification,
            "confidence_score": round(confidence, 4),
            "detected_language": detected_language,
            "language_confidence": round(lang_confidence, 4),
            "features": {
                "pitch_variance": round(analysis_details.get('pitch_variance', 0), 4),
                "spectral_consistency": round(analysis_details.get('spectral_consistency', 0), 4),
                "temporal_patterns": analysis_details.get('temporal_patterns', 'analyzed')
            },
            "explanation": explanation,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(
            f"[{request_id}] Analysis complete: {classification} "
            f"(confidence: {confidence:.4f}, time: {response['processing_time_ms']}ms)"
        )
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return create_error_response(str(e), "VALIDATION_ERROR", 400)
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return create_error_response(
            "Internal server error during analysis",
            "INTERNAL_ERROR",
            500
        )

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        "service": "voice-detection-ml",
        "version": "1.0.0",
        "supported_languages": list(config.LANGUAGE_MAP.keys()),
        "requests_processed": request_counter,
        "uptime": time.time(),
        "device": voice_detector.device
    }), 200

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "code": "INTERNAL_ERROR"
    }), 500

if __name__ == '__main__':
    logger.info(f"Starting Voice Detection ML Service on {config.HOST}:{config.PORT}")
    logger.info(f"GPU Available: {config.USE_GPU}")
    logger.info(f"Supported Languages: {list(config.LANGUAGE_MAP.keys())}")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )