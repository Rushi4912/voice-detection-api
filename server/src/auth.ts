import { Request, Response, NextFunction } from 'express';
import config from './config';
import logger from './logger';
import { APIError } from './types';

export interface AuthenticatedRequest extends Request {
  apiKey?: string;
  requestId?: string;
}

/**
 * API Key Authentication Middleware
 * Validates x-api-key header against configured API key
 */
export const authenticateAPIKey = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): void => {
  try {
    const apiKey = req.header('x-api-key');

    

    if (!apiKey) {
      logger.warn('Authentication failed: Missing API key', {
        ip: req.ip,
        path: req.path,
      });
      throw new APIError(401, 'API key is required', 'MISSING_API_KEY');
    }

    if (apiKey !== config.security.apiKey) {
      logger.warn('Authentication failed: Invalid API key', {
        ip: req.ip,
        path: req.path,
        providedKey: apiKey.substring(0, 10) + '...',
      });
      throw new APIError(401, 'Invalid API key', 'INVALID_API_KEY');
    }

    req.apiKey = apiKey;
    logger.debug('Authentication successful', {
      ip: req.ip,
      path: req.path,
    });

    next();
  } catch (error) {
    if (error instanceof APIError) {
      res.status(error.statusCode).json({
        status: 'error',
        message: error.message,
        code: error.code,
      });
    } else {
      res.status(500).json({
        status: 'error',
        message: 'Internal authentication error',
      });
    }
  }
};

/**
 * Request ID Middleware
 * Generates unique ID for each request for tracking
 */
export const generateRequestId = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): void => {
  req.requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  res.setHeader('X-Request-ID', req.requestId);
  next();
};