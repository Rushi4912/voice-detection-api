export enum SupportedLanguage {
  TAMIL = 'Tamil',
  ENGLISH = 'English',
  HINDI = 'Hindi',
  MALAYALAM = 'Malayalam',
  TELUGU = 'Telugu'
}

export enum Classification {
  AI_GENERATED = 'AI_GENERATED',
  HUMAN = 'HUMAN'
}

export enum AudioFormat {
  MP3 = 'mp3'
}

export interface VoiceDetectionRequest {
  language: SupportedLanguage;
  audioFormat: AudioFormat;
  audioBase64: string;
}

export interface VoiceDetectionResponse {
  status: 'success' | 'error';
  language?: SupportedLanguage;
  classification?: Classification;
  confidenceScore?: number;
  explanation?: string;
  message?: string;
}

export interface PythonServiceResponse {
  classification: Classification;
  confidence_score: number;
  detected_language: string;
  features: {
    pitch_variance: number;
    spectral_consistency: number;
    temporal_patterns: string;
  };
  explanation: string;
}

export interface ErrorResponse {
  status: 'error';
  message: string;
  code?: string;
  details?: string;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface RequestMetadata {
  requestId: string;
  timestamp: string;
  ipAddress: string;
  userAgent: string;
}

export class APIError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public code?: string,
    public details?: string
  ) {
    super(message);
    this.name = 'APIError';
    Error.captureStackTrace(this, this.constructor);
  }
}