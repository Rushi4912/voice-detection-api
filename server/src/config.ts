import dotenv from 'dotenv';

dotenv.config();

interface Config {
  server: {
    port: number;
    nodeEnv: string;
    requestTimeout: number;
  };
  security: {
    apiKey: string;
    allowedOrigins: string[];
  };
  python: {
    serviceUrl: string;
    timeout: number;
  };
  rateLimit: {
    windowMs: number;
    maxRequests: number;
  };
  logging: {
    level: string;
    filePath: string;
  };
  audio: {
    maxSizeMB: number;
    maxSizeBytes: number;
  };
}

const config: Config = {
  server: {
    port: parseInt(process.env.PORT || '8000', 10),
    nodeEnv: process.env.NODE_ENV || 'development',
    requestTimeout: parseInt(process.env.REQUEST_TIMEOUT_MS || '60000', 10),
  },
  security: {
    apiKey: process.env.API_KEY || 'sk_test_123456789',
    allowedOrigins: process.env.ALLOWED_ORIGINS?.split(',') || ['*'],
  },
  python: {
    serviceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:5000',
    timeout: parseInt(process.env.PYTHON_SERVICE_TIMEOUT || '90000', 10), // Increased to 90s for large files
  },
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10),
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10),
  },
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    filePath: process.env.LOG_FILE_PATH || './logs/api.log',
  },
  audio: {
    maxSizeMB: parseInt(process.env.MAX_AUDIO_SIZE_MB || '10', 10),
    maxSizeBytes: parseInt(process.env.MAX_AUDIO_SIZE_MB || '10', 10) * 1024 * 1024,
  },
};

export default config;
