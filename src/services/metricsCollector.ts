import { logger } from '../utils/logger';

interface DetectionMetric {
  requestId: string;
  result: string;
  confidence: number;
  processingTime: number;
  language: string;
  timestamp: Date;
}

interface ErrorMetric {
  requestId: string;
  errorType: string;
  errorMessage: string;
  processingTime: number;
  timestamp: Date;
}

/**
 * Metrics Collector
 * Collects and stores API metrics for monitoring
 */
export class MetricsCollector {
  private static detections: DetectionMetric[] = [];
  private static errors: ErrorMetric[] = [];
  private static readonly MAX_STORED_METRICS = 1000;
  
  static recordDetection(metric: Omit<DetectionMetric, 'timestamp'>): void {
    this.detections.push({
      ...metric,
      timestamp: new Date(),
    });
    
    // Keep only recent metrics
    if (this.detections.length > this.MAX_STORED_METRICS) {
      this.detections = this.detections.slice(-this.MAX_STORED_METRICS);
    }
    
    logger.info('Detection metric recorded', metric);
  }
  
  static recordError(metric: Omit<ErrorMetric, 'timestamp'>): void {
    this.errors.push({
      ...metric,
      timestamp: new Date(),
    });
    
    if (this.errors.length > this.MAX_STORED_METRICS) {
      this.errors = this.errors.slice(-this.MAX_STORED_METRICS);
    }
    
    logger.error('Error metric recorded', metric);
  }
  
  static getStats(): any {
    const now = Date.now();
    const last24h = this.detections.filter(d => now - d.timestamp.getTime() < 24 * 60 * 60 * 1000);
    const lastHour = this.detections.filter(d => now - d.timestamp.getTime() < 60 * 60 * 1000);
    
    return {
      total: {
        detections: this.detections.length,
        errors: this.errors.length,
      },
      last24Hours: {
        detections: last24h.length,
        aiGenerated: last24h.filter(d => d.result === 'AI_GENERATED').length,
        human: last24h.filter(d => d.result === 'HUMAN').length,
        avgConfidence: last24h.reduce((sum, d) => sum + d.confidence, 0) / (last24h.length || 1),
        avgProcessingTime: last24h.reduce((sum, d) => sum + d.processingTime, 0) / (last24h.length || 1),
      },
      lastHour: {
        detections: lastHour.length,
        errors: this.errors.filter(e => now - e.timestamp.getTime() < 60 * 60 * 1000).length,
      },
      languages: this.getLanguageStats(last24h),
    };
  }
  
  private static getLanguageStats(detections: DetectionMetric[]): Record<string, number> {
    const stats: Record<string, number> = {};
    for (const detection of detections) {
      stats[detection.language] = (stats[detection.language] || 0) + 1;
    }
    return stats;
  }
}