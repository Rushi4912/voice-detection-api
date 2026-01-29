import { Request, Response, NextFunction } from 'express';
import { config } from '../config';
import { logger } from '../utils/logger';
import { AppError } from '../utils/errors';

// Extend Express Request type to include apiKey
declare global {
  namespace Express {
    interface Request {
      apiKey?: string;
      requestId?: string;
    }
  }
}

/**
 * API Key Authentication Middleware
 * Validates X-API-Key header against configured API keys
 */
export const apiKeyAuth = (req: Request, res: Response, next: NextFunction): void => {
  try {
    // Generate request ID for tracing
    req.requestId = generateRequestId();
    
    // Get API key from header
    const apiKey = req.headers['x-api-key'] as string;
    
    if (!apiKey) {
      logger.warn(`Authentication failed: No API key provided - ${req.requestId}`, {
        ip: req.ip,
        path: req.path,
      });
      
      throw new AppError(
        'API key is required. Please provide X-API-Key header.',
        401,
        'MISSING_API_KEY'
      );
    }
    
    // Validate API key
    const validKeys = [...config.API_KEYS, config.MASTER_API_KEY];
    const isValidKey = validKeys.includes(apiKey);
    
    if (!isValidKey) {
      logger.warn(`Authentication failed: Invalid API key - ${req.requestId}`, {
        ip: req.ip,
        path: req.path,
        apiKeyPrefix: apiKey.substring(0, 8) + '...',
      });
      
      throw new AppError(
        'Invalid API key. Please check your credentials.',
        401,
        'INVALID_API_KEY'
      );
    }
    
    // Attach API key to request for later use
    req.apiKey = apiKey;
    
    logger.info(`Authentication successful - ${req.requestId}`, {
      ip: req.ip,
      path: req.path,
      apiKeyPrefix: apiKey.substring(0, 8) + '...',
    });
    
    next();
  } catch (error) {
    next(error);
  }
};

/**
 * Generate unique request ID for tracing
 */
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Optional: Rate limiting per API key
 * Can be used in conjunction with IP-based rate limiting
 */
export const apiKeyRateLimit = new Map<string, { count: number; resetTime: number }>();

export const checkApiKeyRateLimit = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const apiKey = req.apiKey;
  
  if (!apiKey) {
    return next();
  }
  
  const now = Date.now();
  const limit = config.RATE_LIMIT_MAX_REQUESTS;
  const windowMs = config.RATE_LIMIT_WINDOW_MS;
  
  let rateLimitInfo = apiKeyRateLimit.get(apiKey);
  
  // Initialize or reset if window has passed
  if (!rateLimitInfo || now > rateLimitInfo.resetTime) {
    rateLimitInfo = {
      count: 0,
      resetTime: now + windowMs,
    };
    apiKeyRateLimit.set(apiKey, rateLimitInfo);
  }
  
  // Increment request count
  rateLimitInfo.count++;
  
  // Check if limit exceeded
  if (rateLimitInfo.count > limit) {
    const retryAfter = Math.ceil((rateLimitInfo.resetTime - now) / 1000);
    
    logger.warn(`Rate limit exceeded for API key - ${req.requestId}`, {
      apiKeyPrefix: apiKey.substring(0, 8) + '...',
      count: rateLimitInfo.count,
      limit,
    });
    
    res.status(429).json({
      error: 'Rate limit exceeded',
      message: `Too many requests. Please try again in ${retryAfter} seconds.`,
      retryAfter,
      limit,
      requestId: req.requestId,
    });
    
    return;
  }
  
  // Add rate limit headers
  res.setHeader('X-RateLimit-Limit', limit.toString());
  res.setHeader('X-RateLimit-Remaining', (limit - rateLimitInfo.count).toString());
  res.setHeader('X-RateLimit-Reset', rateLimitInfo.resetTime.toString());
  
  next();
};

/**
 * Clean up expired rate limit entries (run periodically)
 */
export const cleanupRateLimitCache = (): void => {
  const now = Date.now();
  
  for (const [apiKey, info] of apiKeyRateLimit.entries()) {
    if (now > info.resetTime) {
      apiKeyRateLimit.delete(apiKey);
    }
  }
};

// Cleanup every 5 minutes
setInterval(cleanupRateLimitCache, 5 * 60 * 1000);