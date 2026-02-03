import Joi from 'joi';
import { SupportedLanguage, AudioFormat, VoiceDetectionRequest } from './types';
import config from './config';

/**
 * Schema for voice detection request validation
 */
export const voiceDetectionSchema = Joi.object({
  language: Joi.string()
    .valid(...Object.values(SupportedLanguage))
    .required()
    .messages({
      'any.required': 'Language is required',
      'any.only': `Language must be one of: ${Object.values(SupportedLanguage).join(', ')}`,
    }),

  audioFormat: Joi.string()
    .valid(AudioFormat.MP3)
    .required()
    .messages({
      'any.required': 'Audio format is required',
      'any.only': 'Audio format must be mp3',
    }),

  audioBase64: Joi.string()
    .required()
    .min(100)
    .custom((value, helpers) => {
      // Validate Base64 format
      const base64Regex = /^[A-Za-z0-9+/]+={0,2}$/;
      if (!base64Regex.test(value)) {
        return helpers.error('string.base64');
      }

      // Calculate approximate decoded size
      const base64Length = value.length;
      const padding = (value.match(/=/g) || []).length;
      const decodedSize = (base64Length * 3) / 4 - padding;

      if (decodedSize > config.audio.maxSizeBytes) {
        return helpers.error('string.maxSize', {
          maxSize: config.audio.maxSizeMB,
        });
      }

      return value;
    })
    .messages({
      'any.required': 'Audio data is required',
      'string.min': 'Audio data is too short',
      'string.base64': 'Audio data must be valid Base64 format',
      'string.maxSize': `Audio file size exceeds maximum of {{#maxSize}}MB`,
    }),
});

/**
 * Normalizes language input to title case (e.g., "english" -> "English")
 */
const normalizeLanguage = (language: string): string => {
  if (!language || typeof language !== 'string') return language;
  return language.charAt(0).toUpperCase() + language.slice(1).toLowerCase();
};

/**
 * Validates request body against schema
 */
export const validateRequest = (
  data: unknown
): { isValid: boolean; errors?: string[]; value?: VoiceDetectionRequest } => {
  // Normalize language input to be case-insensitive
  if (data && typeof data === 'object' && 'language' in data) {
    const requestData = data as any;
    if (requestData.language) {
      requestData.language = normalizeLanguage(requestData.language);
    }
  }

  const { error, value } = voiceDetectionSchema.validate(data, {
    abortEarly: false,
    stripUnknown: true,
  });

  if (error) {
    return {
      isValid: false,
      errors: error.details.map((detail) => detail.message),
    };
  }

  return {
    isValid: true,
    value: value as VoiceDetectionRequest,
  };
};

/**
 * Validates Base64 string format
 */
export const isValidBase64 = (str: string): boolean => {
  try {
    return Buffer.from(str, 'base64').toString('base64') === str;
  } catch {
    return false;
  }
};

/**
 * Calculate size of Base64 encoded data in bytes
 */
export const calculateBase64Size = (base64String: string): number => {
  const padding = (base64String.match(/=/g) || []).length;
  return (base64String.length * 3) / 4 - padding;
};