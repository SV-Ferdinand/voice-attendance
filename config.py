import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration
    DATABASE_PATH = 'data/database/attendance.db'
    
    # Upload configuration
    UPLOAD_FOLDER = 'data/voice_samples'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
    
    # Voice recognition configuration
    VOICE_SIMILARITY_THRESHOLD = 0.7
    MIN_AUDIO_DURATION = 1.0  # seconds
    MAX_AUDIO_DURATION = 10.0  # seconds
    SAMPLE_RATE = 16000
    
    # Security configuration
    JWT_EXPIRATION_HOURS = 24
    BCRYPT_ROUNDS = 12
    
    # Admin configuration
    DEFAULT_ADMIN_ID = '000000001'
    DEFAULT_ADMIN_NAME = 'Super Admin'
    DEFAULT_ADMIN_PASSWORD = 'admin123'  # Change in production!
    
    # Speech recognition configuration
    EXPECTED_WORDS = ['HADIR', 'hadir', 'Hadir']
    SPEECH_CONFIDENCE_THRESHOLD = 0.8

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Production database (PostgreSQL example)
    # DATABASE_URL = os.environ.get('DATABASE_URL')

class TestingConfig(Config):
    TESTING = True
    DATABASE_PATH = ':memory:'  # In-memory database for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}