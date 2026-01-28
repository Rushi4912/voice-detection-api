import { Router, Request, Response } from 'express';
import { MetricsCollector } from '../services/metricsCollector';
import { asyncHandler } from '../middleware/errorHandler';

const router = Router();

/**
 * GET /api/metrics
 * Get API usage metrics
 */
router.get('/', asyncHandler(async (req: Request, res: Response) => {
  const stats = MetricsCollector.getStats();
  
  res.status(200).json({
    metrics: stats,
    timestamp: new Date().toISOString(),
  });
}));

export const metricsRouter = router;