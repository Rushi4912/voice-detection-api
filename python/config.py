import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Server - Railway sets PORT automatically
    PORT = int(os.getenv('PORT', os.getenv('PYTHON_PORT', 5000)))
    HOST = os.getenv('HOST', os.getenv('PYTHON_HOST', '0.0.0.0'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model paths
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    VOICE_CNN_MODEL = os.path.join(MODEL_DIR, 'voice_cnn.onnx')
    
    # SpeechBrain Language ID
    LANGUAGE_ID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
    
    # Language mapping
    LANGUAGE_MAP = {
        'Tamil': 'ta',
        'English': 'en',
        'Hindi': 'hi',
        'Malayalam': 'ml',
        'Telugu': 'te'
    }
    
    # Audio processing
    SAMPLE_RATE = 16000
    MAX_AUDIO_DURATION = 60  # seconds
    MIN_AUDIO_DURATION = 1   # seconds
    
    # Model thresholds
    AI_DETECTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.6
    
    # Performance
    USE_GPU = torch.cuda.is_available() if 'torch' in dir() else False
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    
    # Cache
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'False').lower() == 'true'
    CACHE_TTL = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/python_service.log'

config = Config()