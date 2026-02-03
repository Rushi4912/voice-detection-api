import { Response } from 'express';
import axios, { AxiosError } from 'axios';
import { AuthenticatedRequest } from './auth';
import {
  VoiceDetectionResponse,
  PythonServiceResponse,
  APIError,
} from './types';
import { validateRequest } from './validation';
import config from './config';
import logger from './logger';

/**
 * Main controller for voice detection endpoint
 */
export const detectVoice = async (
  req: AuthenticatedRequest,
  res: Response
): Promise<void> => {
  const startTime = Date.now();
  const requestId = req.requestId || 'unknown';

  try {
    logger.info('Voice detection request received', {
      requestId,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    });

    // Validate request body
    const validation = validateRequest(req.body);
    if (!validation.isValid) {
      logger.warn('Request validation failed', {
        requestId,
        errors: validation.errors,
      });
      throw new APIError(
        400,
        'Invalid request body',
        'VALIDATION_ERROR',
        validation.errors?.join('; ')
      );
    }

    const { language, audioFormat, audioBase64 } = validation.value!;

    logger.info('Request validated successfully', {
      requestId,
      language,
      audioFormat,
      audioSizeKB: Math.round(audioBase64.length / 1024),
    });

    // Call Python ML service
    const pythonResponse = await callPythonService(
      requestId,
      language,
      audioBase64
    );


    // Build successful response
    const response: VoiceDetectionResponse = {
      status: 'success',
      language: pythonResponse.detected_language as any, // Use actual detected language from ML
      classification: pythonResponse.classification,
      confidenceScore: parseFloat(pythonResponse.confidence_score.toFixed(4)),
      explanation: pythonResponse.explanation,
    };

    const duration = Date.now() - startTime;
    logger.info('Voice detection completed successfully', {
      requestId,
      classification: response.classification,
      confidenceScore: response.confidenceScore,
      durationMs: duration,
    });

    res.status(200).json(response);
  } catch (error) {
    handleError(error, req, res, requestId, startTime);
  }
};

/**
 * Calls the Python ML service for voice analysis
 */
const callPythonService = async (
  requestId: string,
  language: string,
  audioBase64: string
): Promise<PythonServiceResponse> => {
  try {
    logger.debug('Calling Python ML service', {
      requestId,
      serviceUrl: config.python.serviceUrl,
      language,
    });

    const response = await axios.post<PythonServiceResponse>(
      `${config.python.serviceUrl}/analyze`,
      {
        language,
        audio_base64: audioBase64,
      },
      {
        timeout: config.python.timeout,
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': requestId,
        },
        validateStatus: (status) => status < 500,
      }
    );

    if (response.status !== 200) {
      throw new APIError(
        502,
        'Python service returned error',
        'PYTHON_SERVICE_ERROR',
        response.data?.toString()
      );
    }

    logger.debug('Python service responded successfully', {
      requestId,
      classification: response.data.classification,
    });

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;

      if (axiosError.code === 'ECONNREFUSED') {
        logger.error('Python service connection refused', {
          requestId,
          serviceUrl: config.python.serviceUrl,
        });
        throw new APIError(
          503,
          'ML service unavailable',
          'SERVICE_UNAVAILABLE',
          'Unable to connect to Python analysis service'
        );
      }

      if (axiosError.code === 'ETIMEDOUT') {
        logger.error('Python service timeout', {
          requestId,
          timeout: config.python.timeout,
        });
        throw new APIError(
          504,
          'ML service timeout',
          'SERVICE_TIMEOUT',
          'Analysis service took too long to respond'
        );
      }

      logger.error('Python service request failed', {
        requestId,
        error: axiosError.message,
        response: axiosError.response?.data,
      });
      throw new APIError(
        502,
        'Failed to communicate with ML service',
        'PYTHON_SERVICE_ERROR',
        axiosError.message
      );
    }

    throw error;
  }
};

/**
 * Centralized error handler
 */
const handleError = (
  error: unknown,
  _req: AuthenticatedRequest,
  res: Response,
  requestId: string,
  startTime: number
): void => {
  const duration = Date.now() - startTime;

  if (error instanceof APIError) {
    logger.error('API error occurred', {
      requestId,
      code: error.code,
      message: error.message,
      statusCode: error.statusCode,
      durationMs: duration,
    });

    res.status(error.statusCode).json({
      status: 'error',
      message: error.message,
      code: error.code,
      ...(error.details && { details: error.details }),
    });
    return;
  }

  logger.error('Unexpected error occurred', {
    requestId,
    error: error instanceof Error ? error.message : 'Unknown error',
    stack: error instanceof Error ? error.stack : undefined,
    durationMs: duration,
  });

  res.status(500).json({
    status: 'error',
    message: 'Internal server error occurred during voice analysis',
  });
};

/**
 * Health check endpoint
 */
export const healthCheck = async (
  _req: AuthenticatedRequest,
  res: Response
): Promise<void> => {
  try {
    // Check Python service health
    const pythonHealth = await axios.get(
      `${config.python.serviceUrl}/health`,
      { timeout: 5000 }
    );

    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        api: 'operational',
        python: pythonHealth.status === 200 ? 'operational' : 'degraded',
      },
    });
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      services: {
        api: 'operational',
        python: 'unavailable',
      },
    });
  }
};