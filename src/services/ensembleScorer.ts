import { config } from '../config';
import { logger } from '../utils/logger';

export interface EnsembleInput {
  acoustic: { score: number; anomalies: string[] };
  deepLearning: { score: number; confidence: number };
  artifact: { score: number; artifacts: string[] };
  language: { score: number; detectedLanguage: string };
}

export interface EnsembleOutput {
  classification: 'AI_GENERATED' | 'HUMAN';
  confidence: number;
  breakdown: {
    acoustic: number;
    deepLearning: number;
    artifact: number;
    language: number;
  };
  reasoning: string[];
}

/**
 * Ensemble Scorer
 * Combines multiple detection layers into final classification
 */
export class EnsembleScorer {
  
  computeScore(input: EnsembleInput): EnsembleOutput {
    logger.info('Computing ensemble score');
    
    const weights = config.ENSEMBLE_WEIGHTS;
    
    // Weighted average of all scores
    const weightedScore = (
      input.acoustic.score * weights.acoustic +
      input.deepLearning.score * weights.deepLearning +
      input.artifact.score * weights.artifactDetection +
      input.language.score * weights.languageSpecific
    );
    
    // Adjust based on deep learning confidence
    const confidenceAdjustment = input.deepLearning.confidence > 0.8 ? 1.1 : 0.9;
    const adjustedScore = Math.min(1, weightedScore * confidenceAdjustment);
    
    // Classification decision
    const threshold = config.MODEL_CONFIDENCE_THRESHOLD;
    const classification: 'AI_GENERATED' | 'HUMAN' = adjustedScore >= threshold
      ? 'AI_GENERATED'
      : 'HUMAN';
    
    // Calculate final confidence (how sure we are about the classification)
    const confidence = classification === 'AI_GENERATED'
      ? adjustedScore
      : 1 - adjustedScore;
    
    // Build reasoning
    const reasoning: string[] = [];
    
    if (input.acoustic.score > 0.6) {
      reasoning.push(`Acoustic analysis indicates AI patterns (score: ${input.acoustic.score.toFixed(2)})`);
      reasoning.push(...input.acoustic.anomalies);
    }
    
    if (input.deepLearning.score > 0.6) {
      reasoning.push(`Deep learning models detected AI characteristics (score: ${input.deepLearning.score.toFixed(2)})`);
    }
    
    if (input.artifact.score > 0.5) {
      reasoning.push(`Digital artifacts detected (score: ${input.artifact.score.toFixed(2)})`);
      reasoning.push(...input.artifact.artifacts);
    }
    
    if (input.language.score > 0.5) {
      reasoning.push(`Language-specific analysis flagged unnatural patterns (score: ${input.language.score.toFixed(2)})`);
    }
    
    if (classification === 'HUMAN' && adjustedScore < 0.3) {
      reasoning.push('Multiple indicators suggest human speech characteristics');
    }
    
    logger.info('Ensemble score computed', {
      classification,
      confidence,
      weightedScore,
    });
    
    return {
      classification,
      confidence,
      breakdown: {
        acoustic: input.acoustic.score,
        deepLearning: input.deepLearning.score,
        artifact: input.artifact.score,
        language: input.language.score,
      },
      reasoning,
    };
  }
}