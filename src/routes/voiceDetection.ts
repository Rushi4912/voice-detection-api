import { Router } from 'express';
import { detectVoice } from '../controllers/voiceDetectionController';
import { validateDetectionRequest } from '../middleware/validation';
import { asyncHandler } from '../middleware/errorHandler';

const router = Router();

/**
 * POST /api/detect
 * Main endpoint for voice detection
 * 
 * Request body:
 * {
 *   "audio": "base64-encoded-mp3-string",
 *   "language": "English" // Optional
 * }
 * 
 * Response:
 * {
 *   "result": "AI_GENERATED" | "HUMAN",
 *   "confidence": 0.95,
 *   "analysis": { ... },
 *   "requestId": "req_xxx",
 *   "processingTime": 1234
 * }
 */
router.post('/', validateDetectionRequest, asyncHandler(detectVoice));

export const voiceDetectionRouter = router;