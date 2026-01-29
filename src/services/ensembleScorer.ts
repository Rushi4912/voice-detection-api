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
  explanation: string;
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
    const confidenceAdjustment = input.deepLearning.confidence > 0.8 ? 1.1 : 0.95;
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

    if (input.acoustic.score > 0.5) {
      reasoning.push(`Acoustic analysis indicates AI patterns (score: ${input.acoustic.score.toFixed(2)})`);
      reasoning.push(...input.acoustic.anomalies);
    }

    if (input.deepLearning.score > 0.5) {
      reasoning.push(`Deep learning models detected AI characteristics (score: ${input.deepLearning.score.toFixed(2)})`);
    }

    if (input.artifact.score > 0.4) {
      reasoning.push(`Digital artifacts detected (score: ${input.artifact.score.toFixed(2)})`);
      reasoning.push(...input.artifact.artifacts);
    }

    if (input.language.score > 0.4) {
      reasoning.push(`Language-specific analysis flagged unnatural patterns (score: ${input.language.score.toFixed(2)})`);
    }

    if (classification === 'HUMAN' && adjustedScore < 0.3) {
      reasoning.push('Multiple indicators suggest human speech characteristics');
    }

    // Generate human-readable explanation
    const explanation = this.generateExplanation(classification, input, reasoning);

    logger.info('Ensemble score computed', {
      classification,
      confidence,
      weightedScore,
    });

    return {
      classification,
      confidence,
      explanation,
      breakdown: {
        acoustic: input.acoustic.score,
        deepLearning: input.deepLearning.score,
        artifact: input.artifact.score,
        language: input.language.score,
      },
      reasoning,
    };
  }

  /**
   * Generate a human-readable explanation for the classification
   */
  private generateExplanation(
    classification: 'AI_GENERATED' | 'HUMAN',
    input: EnsembleInput,
    _reasoning: string[]
  ): string {
    if (classification === 'AI_GENERATED') {
      const indicators: string[] = [];

      if (input.acoustic.score > 0.5) {
        if (input.acoustic.anomalies.some(a => a.toLowerCase().includes('pitch'))) {
          indicators.push('unnatural pitch consistency');
        }
        if (input.acoustic.anomalies.some(a => a.toLowerCase().includes('jitter'))) {
          indicators.push('abnormal voice jitter');
        }
        if (input.acoustic.anomalies.some(a => a.toLowerCase().includes('shimmer'))) {
          indicators.push('irregular amplitude patterns');
        }
      }

      if (input.artifact.score > 0.4) {
        if (input.artifact.artifacts.some(a => a.toLowerCase().includes('robotic'))) {
          indicators.push('robotic speech patterns');
        }
        if (input.artifact.artifacts.some(a => a.toLowerCase().includes('digital'))) {
          indicators.push('digital processing artifacts');
        }
        if (input.artifact.artifacts.some(a => a.toLowerCase().includes('pause'))) {
          indicators.push('unnatural pause patterns');
        }
      }

      if (input.deepLearning.score > 0.5) {
        indicators.push('AI-generated voice signatures');
      }

      if (indicators.length === 0) {
        indicators.push('synthetic voice characteristics');
      }

      return indicators.slice(0, 2).join(' and ') + ' detected';
    } else {
      const naturalIndicators: string[] = [];

      if (input.acoustic.score < 0.3) {
        naturalIndicators.push('natural pitch variation');
      }
      if (input.artifact.score < 0.3) {
        naturalIndicators.push('no digital artifacts');
      }
      if (input.language.score < 0.3) {
        naturalIndicators.push('natural speech patterns');
      }

      if (naturalIndicators.length === 0) {
        return 'Voice characteristics consistent with human speech';
      }

      return naturalIndicators.slice(0, 2).join(' and ') + ' observed';
    }
  }
}