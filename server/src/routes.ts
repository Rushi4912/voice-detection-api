import { Router } from 'express';
import { detectVoice, healthCheck } from './voiceController';
import { authenticateAPIKey } from './auth';

const router = Router();

/**
 * @route   POST /api/voice-detection
 * @desc    Detect if voice is AI-generated or human
 * @access  Protected (API Key required)
 */
router.post('/voice-detection', authenticateAPIKey, detectVoice);

/**
 * @route   GET /api/health
 * @desc    Health check endpoint
 * @access  Public
 */
router.get('/health', healthCheck);

/**
 * @route   GET /api/status
 * @desc    API status endpoint
 * @access  Public
 */
router.get('/status', (_req, res) => {
  res.status(200).json({
    status: 'operational',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    supportedLanguages: ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'],
    supportedFormats: ['mp3'],
  });
});

export default router;