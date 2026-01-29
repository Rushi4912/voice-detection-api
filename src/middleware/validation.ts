import { Request, Response, NextFunction } from 'express';
import { ValidationError } from './errorHandler';
import { config } from '../config';
import { logger } from '../utils/logger';

/**
 * Validation schemas
 */
interface DetectionRequestBody {
  audioBase64: string;
  audioFormat: string;
  language?: string;
}

/**
 * Validate detection request
 */
export const validateDetectionRequest = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  try {
    const { audioBase64, audioFormat, language } = req.body as DetectionRequestBody;
    const errors: string[] = [];

    // Validate audioBase64 field
    if (!audioBase64) {
      errors.push('audioBase64 field is required');
    } else if (typeof audioBase64 !== 'string') {
      errors.push('audioBase64 must be a string');
    } else if (audioBase64.trim() === '') {
      errors.push('audioBase64 cannot be empty');
    } else {
      // Validate base64 format
      const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
      const cleanedAudio = audioBase64.replace(/\s/g, '');

      if (!base64Regex.test(cleanedAudio)) {
        errors.push('audioBase64 must be a valid base64-encoded string');
      }

      // Check size (base64 encoding increases size by ~33%)
      const estimatedSizeBytes = (cleanedAudio.length * 3) / 4;
      const maxSizeBytes = config.MAX_AUDIO_SIZE_MB * 1024 * 1024;

      if (estimatedSizeBytes > maxSizeBytes) {
        errors.push(
          `audio size exceeds maximum limit of ${config.MAX_AUDIO_SIZE_MB}MB ` +
          `(estimated size: ${(estimatedSizeBytes / 1024 / 1024).toFixed(2)}MB)`
        );
      }

      // Check minimum size (at least 1KB)
      if (estimatedSizeBytes < 1024) {
        errors.push('audio file is too small (minimum 1KB required)');
      }
    }

    // Validate audioFormat field
    const supportedFormats = ['mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a'];
    if (!audioFormat) {
      errors.push('audioFormat field is required');
    } else if (typeof audioFormat !== 'string') {
      errors.push('audioFormat must be a string');
    } else if (!supportedFormats.includes(audioFormat.toLowerCase())) {
      errors.push(
        `audioFormat must be one of: ${supportedFormats.join(', ')} ` +
        `(received: ${audioFormat})`
      );
    }

    // Validate language field (optional)
    if (language !== undefined) {
      if (typeof language !== 'string') {
        errors.push('language must be a string');
      } else {
        // Case-insensitive check and normalization
        const supportedLanguage = config.SUPPORTED_LANGUAGES.find(
          lang => lang.toLowerCase() === language.toLowerCase()
        );

        if (!supportedLanguage) {
          errors.push(
            `language must be one of: ${config.SUPPORTED_LANGUAGES.join(', ')} ` +
            `(received: ${language})`
          );
        } else {
          // Normalize to proper casing (e.g., "tamil" -> "Tamil")
          req.body.language = supportedLanguage;
        }
      }
    }

    // If there are validation errors, throw ValidationError
    if (errors.length > 0) {
      logger.warn(`Validation failed - ${req.requestId}`, {
        errors,
        bodyKeys: Object.keys(req.body),
      });

      throw new ValidationError('Request validation failed', {
        errors,
        fields: {
          audioBase64: !audioBase64 ? 'missing' : 'invalid',
          audioFormat: !audioFormat ? 'missing' : 'invalid',
          language: language && !config.SUPPORTED_LANGUAGES.includes(language) ? 'invalid' : 'ok',
        },
      });
    }

    // Validation passed
    logger.info(`Validation passed - ${req.requestId}`, {
      audioSize: audioBase64 ? `${((audioBase64.length * 3) / 4 / 1024).toFixed(2)}KB` : 'N/A',
      audioFormat: audioFormat || 'not specified',
      language: language || 'not specified',
    });

    next();
  } catch (error) {
    next(error);
  }
};

/**
 * Sanitize base64 audio string
 * Removes whitespace and data URL prefix if present
 */
export const sanitizeBase64Audio = (audio: string): string => {
  // Remove whitespace
  let cleaned = audio.replace(/\s/g, '');

  // Remove data URL prefix if present (e.g., "data:audio/mp3;base64,")
  const dataUrlRegex = /^data:audio\/[a-zA-Z0-9+-]+;base64,/;
  if (dataUrlRegex.test(cleaned)) {
    cleaned = cleaned.replace(dataUrlRegex, '');
  }

  return cleaned;
};

/**
 * Validate audio buffer after decoding
 */
export const validateAudioBuffer = (buffer: Buffer): void => {
  // Check for MP3 magic numbers
  const mp3MagicNumbers = [
    Buffer.from([0xFF, 0xFB]), // MP3 Frame sync (MPEG-1 Layer 3)
    Buffer.from([0xFF, 0xFA]), // MP3 Frame sync (MPEG-1 Layer 3)
    Buffer.from([0xFF, 0xF3]), // MP3 Frame sync (MPEG-2 Layer 3)
    Buffer.from([0xFF, 0xF2]), // MP3 Frame sync (MPEG-2 Layer 3)
    Buffer.from([0x49, 0x44, 0x33]), // ID3 tag
  ];

  const startsWithValidMagic = mp3MagicNumbers.some(magic =>
    buffer.slice(0, magic.length).equals(magic)
  );

  if (!startsWithValidMagic) {
    throw new ValidationError(
      'Invalid audio format. The file does not appear to be a valid MP3 file.',
      { format: 'Expected MP3, but file signature does not match' }
    );
  }
};