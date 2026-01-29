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
 *   "audioBase64": "base64-encoded-audio-string",
 *   "audioFormat": "mp3",  // Supported: mp3, wav, ogg, flac, aac, m4a
 *   "language": "Tamil"    // Optional: Tamil, English, Hindi, Malayalam, Telugu
 * }
 * 
 * Response:
 * {
 *   "status": "success",
 *   "language": "Tamil",
 *   "classification": "AI_GENERATED" | "HUMAN",
 *   "confidenceScore": 0.91,
 *   "explanation": "Unnatural pitch consistency and robotic speech patterns detected",
 *   "analysis": { ... },
 *   "metadata": { ... }
 * }
 */
router.post('/', validateDetectionRequest, asyncHandler(detectVoice));

export const voiceDetectionRouter = router;