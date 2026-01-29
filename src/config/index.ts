import dotenv from 'dotenv';
import path from 'path';

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

interface Config {
  NODE_ENV: string;
  PORT: number;
  API_BASE_URL: string;
  
  // API Keys
  API_KEYS: string[];
  MASTER_API_KEY: string;
  
  // Rate limiting
  RATE_LIMIT_MAX_REQUESTS: number;
  RATE_LIMIT_WINDOW_MS: number;
  
  // CORS
  ALLOWED_ORIGINS: string[];
  
  // Audio processing
  MAX_AUDIO_SIZE_MB: number;
  SUPPORTED_LANGUAGES: string[];
  SUPPORTED_FORMATS: string[];
  
  // Model configuration
  MODEL_CONFIDENCE_THRESHOLD: number;
  ENSEMBLE_WEIGHTS: {
    acoustic: number;
    deepLearning: number;
    artifactDetection: number;
    languageSpecific: number;
  };
  
  // Logging
  LOG_LEVEL: string;
  LOG_FILE_PATH: string;
  
  // Redis (for caching and rate limiting)
  REDIS_URL?: string;
  REDIS_ENABLED: boolean;
  
  // Performance
  MAX_CONCURRENT_REQUESTS: number;
  REQUEST_TIMEOUT_MS: number;
}

export const config: Config = {
  NODE_ENV: process.env.NODE_ENV || 'development',
  PORT: parseInt(process.env.PORT || '3000', 10),
  API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:3000',
  
  // API Keys - comma-separated in env
  API_KEYS: process.env.API_KEYS?.split(',').map(key => key.trim()) || [],
  MASTER_API_KEY: process.env.MASTER_API_KEY || 'master_key_change_in_production',
  
  // Rate limiting
  RATE_LIMIT_MAX_REQUESTS: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10),
  RATE_LIMIT_WINDOW_MS: 15 * 60 * 1000, // 15 minutes
  
  // CORS
  ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS?.split(',').map(origin => origin.trim()) || ['*'],
  
  // Audio processing
  MAX_AUDIO_SIZE_MB: 50,
  SUPPORTED_LANGUAGES: ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'],
  SUPPORTED_FORMATS: ['MP3', 'audio/mpeg', 'audio/mp3'],
  
  // Model configuration
  MODEL_CONFIDENCE_THRESHOLD: 0.5,
  ENSEMBLE_WEIGHTS: {
    acoustic: 0.25,
    deepLearning: 0.35,
    artifactDetection: 0.25,
    languageSpecific: 0.15,
  },
  
  // Logging
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  LOG_FILE_PATH: process.env.LOG_FILE_PATH || './logs/app.log',
  
  // Redis
  REDIS_URL: process.env.REDIS_URL,
  REDIS_ENABLED: process.env.REDIS_ENABLED === 'true',
  
  // Performance
  MAX_CONCURRENT_REQUESTS: parseInt(process.env.MAX_CONCURRENT_REQUESTS || '10', 10),
  REQUEST_TIMEOUT_MS: parseInt(process.env.REQUEST_TIMEOUT_MS || '30000', 10), // 30 seconds
};

// Validate required config
const requiredEnvVars = ['API_KEYS', 'MASTER_API_KEY'];

const missingVars = requiredEnvVars.filter(varName => {
  const value = process.env[varName];
  return !value || value.trim() === '';
});

if (missingVars.length > 0 && config.NODE_ENV === 'production') {
  throw new Error(
    `Missing required environment variables: ${missingVars.join(', ')}\n` +
    'Please set them in your .env file or environment.'
  );
}

export default config;