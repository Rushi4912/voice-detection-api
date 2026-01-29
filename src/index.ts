import express, { Request, Response } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import morgan from 'morgan';
import compression from 'compression';
import { config } from './config';
import { apiKeyAuth } from './middleware/auth';
import { errorHandler } from './middleware/errorHandler';
import { voiceDetectionRouter } from './routes/voiceDetection';
import { logger } from './utils/logger';
import { metricsRouter } from './routes/metrics';

const app = express();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
    },
  },
}));

// CORS configuration
app.use(cors({
  origin: config.ALLOWED_ORIGINS,
  credentials: true,
  methods: ['POST', 'GET'],
  allowedHeaders: ['Content-Type', 'X-API-Key', 'X-Request-ID'],
}));

// Compression for responses
app.use(compression());

// Body parser with size limits (50MB for base64 audio)
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging
app.use(morgan('combined', {
  stream: {
    write: (message: string) => logger.info(message.trim())
  }
}));

// Rate limiting - Strict for production
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: config.RATE_LIMIT_MAX_REQUESTS,
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/api/', limiter);

// Health check endpoint (no auth required)
app.get('/health', (req: Request, res: Response) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: config.NODE_ENV,
  });
});

// API info endpoint
app.get('/api/info', (req: Request, res: Response) => {
  res.status(200).json({
    name: 'AI Voice Detection API',
    version: '1.0.0',
    supportedLanguages: ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'],
    supportedFormats: ['MP3'],
    maxAudioSize: '50MB',
    rateLimit: `${config.RATE_LIMIT_MAX_REQUESTS} requests per 15 minutes`,
    documentation: `${config.API_BASE_URL}/docs`,
  });
});

// Protected routes
app.use('/api/detect', apiKeyAuth, voiceDetectionRouter);
app.use('/api/metrics', apiKeyAuth, metricsRouter);

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.path,
    method: req.method,
  });
});

// Global error handler
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM signal received: closing HTTP server');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT signal received: closing HTTP server');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

// Unhandled rejection handler
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

const server = app.listen(config.PORT, () => {
  logger.info(`🚀 Server running on port ${config.PORT}`);
  logger.info(`📊 Environment: ${config.NODE_ENV}`);
  logger.info(`🔒 API Key authentication enabled`);
  logger.info(`⚡ Rate limit: ${config.RATE_LIMIT_MAX_REQUESTS} requests per 15 minutes`);
});

export default app;