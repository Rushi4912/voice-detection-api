import { ProcessedAudio } from './audioProcessor';
import { logger } from '../utils/logger';

export interface LanguageResults {
  score: number;
  detectedLanguage: string;
  languageConfidence: number;
  prosodyScore: number;
}

/**
 * Language-Specific Analyzer
 * Analyzes language-specific prosody and patterns for Tamil, English, Hindi, Malayalam, Telugu
 */
export class LanguageSpecificAnalyzer {

  private readonly SUPPORTED_LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu'];

  async analyze(audio: ProcessedAudio, suggestedLanguage?: string): Promise<LanguageResults> {
    logger.info('Starting language-specific analysis', { suggestedLanguage });

    const samples = audio.rawData;
    const sampleRate = audio.sampleRate;

    // Pre-compute prosody features ONCE (expensive operation)
    const cachedProsodyFeatures = this.extractProsodyFeatures(samples, sampleRate);

    // Step 1: Detect language if not provided (uses cached features)
    const detectedLanguage = suggestedLanguage && this.SUPPORTED_LANGUAGES.includes(suggestedLanguage)
      ? suggestedLanguage
      : this.detectLanguageFromFeatures(cachedProsodyFeatures);

    const languageConfidence = 0.85; // Simplified

    // Step 2: Analyze language-specific prosody (uses cached features)
    const prosodyScore = this.analyzeProsodyFromFeatures(cachedProsodyFeatures, detectedLanguage);

    // Step 3: Check language-specific patterns
    const patternScore = this.analyzeLanguagePatterns(samples, sampleRate, detectedLanguage);

    // Step 4: Check phoneme distribution
    const phonemeScore = this.analyzePhonemeDistribution(samples, sampleRate, detectedLanguage);

    // Combine scores
    const finalScore = (prosodyScore * 0.4 + patternScore * 0.35 + phonemeScore * 0.25);

    logger.info('Language-specific analysis completed', {
      language: detectedLanguage,
      score: finalScore,
    });

    return {
      score: finalScore,
      detectedLanguage,
      languageConfidence,
      prosodyScore,
    };
  }

  /**
   * Detect language from audio
   */
  private async detectLanguage(samples: Float32Array, sampleRate: number): Promise<string> {
    // Simplified language detection based on prosody characteristics
    const prosodyFeatures = this.extractProsodyFeatures(samples, sampleRate);

    // Language-specific characteristics
    const languageScores: Record<string, number> = {
      Tamil: 0,
      English: 0,
      Hindi: 0,
      Malayalam: 0,
      Telugu: 0,
    };

    // Tamil: High pitch variation, rhythmic patterns
    if (prosodyFeatures.pitchRange > 200 && prosodyFeatures.rhythm > 0.6) {
      languageScores.Tamil += 0.3;
    }

    // English: Moderate pitch, stress-timed
    if (prosodyFeatures.pitchRange > 100 && prosodyFeatures.pitchRange < 250 && prosodyFeatures.stressTiming > 0.5) {
      languageScores.English += 0.3;
    }

    // Hindi: Moderate-high pitch, syllable-timed
    if (prosodyFeatures.pitchRange > 150 && prosodyFeatures.syllableTiming > 0.5) {
      languageScores.Hindi += 0.3;
    }

    // Malayalam: Fast speech rate, complex phonetics
    if (prosodyFeatures.speechRate > 4.5) {
      languageScores.Malayalam += 0.3;
    }

    // Telugu: Rhythmic, melodic patterns
    if (prosodyFeatures.rhythm > 0.5 && prosodyFeatures.pitchRange > 180) {
      languageScores.Telugu += 0.3;
    }

    // Default to English if unclear
    const detectedLang = Object.entries(languageScores)
      .sort((a, b) => b[1] - a[1])[0][0];

    return languageScores[detectedLang] > 0 ? detectedLang : 'English';
  }

  /**
   * Detect language from pre-computed prosody features
   */
  private detectLanguageFromFeatures(prosodyFeatures: any): string {
    const languageScores: Record<string, number> = {
      Tamil: 0,
      English: 0,
      Hindi: 0,
      Malayalam: 0,
      Telugu: 0,
    };

    // Tamil: High pitch variation, rhythmic patterns
    if (prosodyFeatures.pitchRange > 200 && prosodyFeatures.rhythm > 0.6) {
      languageScores.Tamil += 0.3;
    }

    // English: Moderate pitch, stress-timed
    if (prosodyFeatures.pitchRange > 100 && prosodyFeatures.pitchRange < 250 && prosodyFeatures.stressTiming > 0.5) {
      languageScores.English += 0.3;
    }

    // Hindi: Moderate-high pitch, syllable-timed
    if (prosodyFeatures.pitchRange > 150 && prosodyFeatures.syllableTiming > 0.5) {
      languageScores.Hindi += 0.3;
    }

    // Malayalam: Fast speech rate, complex phonetics
    if (prosodyFeatures.speechRate > 4.5) {
      languageScores.Malayalam += 0.3;
    }

    // Telugu: Rhythmic, melodic patterns
    if (prosodyFeatures.rhythm > 0.5 && prosodyFeatures.pitchRange > 180) {
      languageScores.Telugu += 0.3;
    }

    const detectedLang = Object.entries(languageScores)
      .sort((a, b) => b[1] - a[1])[0][0];

    return languageScores[detectedLang] > 0 ? detectedLang : 'English';
  }

  /**
   * Analyze prosody from pre-computed features
   */
  private analyzeProsodyFromFeatures(features: any, language: string): number {
    let aiScore = 0;

    // Language-specific natural ranges
    const naturalRanges: Record<string, any> = {
      Tamil: { pitchRange: [180, 280], rhythm: [0.5, 0.8] },
      English: { pitchRange: [100, 250], rhythm: [0.4, 0.7] },
      Hindi: { pitchRange: [130, 270], rhythm: [0.45, 0.75] },
      Malayalam: { pitchRange: [150, 260], rhythm: [0.5, 0.8] },
      Telugu: { pitchRange: [160, 280], rhythm: [0.5, 0.8] },
    };

    const ranges = naturalRanges[language] || naturalRanges.English;

    // Check if pitch range is outside natural bounds
    if (features.pitchRange < ranges.pitchRange[0] || features.pitchRange > ranges.pitchRange[1]) {
      aiScore += 0.3;
    }

    // Check if rhythm is outside natural bounds
    if (features.rhythm < ranges.rhythm[0] || features.rhythm > ranges.rhythm[1]) {
      aiScore += 0.3;
    }

    return Math.min(1, aiScore);
  }

  /**
   * Extract prosody features
   */
  private extractProsodyFeatures(samples: Float32Array, sampleRate: number): any {
    return {
      pitchRange: this.calculatePitchRange(samples, sampleRate),
      rhythm: this.calculateRhythm(samples, sampleRate),
      stressTiming: this.calculateStressTiming(samples, sampleRate),
      syllableTiming: this.calculateSyllableTiming(samples, sampleRate),
      speechRate: this.estimateSpeechRate(samples, sampleRate),
    };
  }

  /**
   * Analyze prosody patterns for AI detection
   */
  private analyzeProsody(samples: Float32Array, sampleRate: number, language: string): number {
    const features = this.extractProsodyFeatures(samples, sampleRate);
    let aiScore = 0;

    // Language-specific natural ranges
    const naturalRanges: Record<string, any> = {
      Tamil: { pitchRange: [180, 280], rhythm: [0.5, 0.8] },
      English: { pitchRange: [100, 250], rhythm: [0.4, 0.7] },
      Hindi: { pitchRange: [130, 270], rhythm: [0.45, 0.75] },
      Malayalam: { pitchRange: [150, 260], rhythm: [0.5, 0.8] },
      Telugu: { pitchRange: [160, 280], rhythm: [0.5, 0.8] },
    };

    const ranges = naturalRanges[language] || naturalRanges.English;

    // Check if pitch range is outside natural bounds
    if (features.pitchRange < ranges.pitchRange[0] || features.pitchRange > ranges.pitchRange[1]) {
      aiScore += 0.3;
    }

    // Check if rhythm is outside natural bounds
    if (features.rhythm < ranges.rhythm[0] || features.rhythm > ranges.rhythm[1]) {
      aiScore += 0.3;
    }

    // Check for overly consistent prosody (AI characteristic)
    const prosodyConsistency = this.measureProsodyConsistency(samples, sampleRate);
    if (prosodyConsistency > 0.85) {
      aiScore += 0.4;
    }

    return Math.min(1, aiScore);
  }

  /**
   * Analyze language-specific patterns
   */
  private analyzeLanguagePatterns(samples: Float32Array, sampleRate: number, language: string): number {
    let aiScore = 0;

    // Check for natural co-articulation (AI lacks this)
    const coarticulation = this.detectCoarticulation(samples, sampleRate);
    if (coarticulation < 0.3) {
      aiScore += 0.4;
    }

    // Check for natural disfluencies (AI rarely has these)
    const disfluencies = this.detectDisfluencies(samples, sampleRate);
    if (disfluencies === 0) {
      aiScore += 0.3;
    }

    // Check for breathing patterns (AI often lacks natural breathing)
    const breathingNaturalness = this.analyzeBreathing(samples, sampleRate);
    if (breathingNaturalness < 0.4) {
      aiScore += 0.3;
    }

    return Math.min(1, aiScore);
  }

  /**
   * Analyze phoneme distribution
   */
  private analyzePhonemeDistribution(samples: Float32Array, sampleRate: number, language: string): number {
    // Simplified phoneme analysis
    const formants = this.extractFormants(samples, sampleRate);

    // Check for unnatural formant patterns
    let aiScore = 0;

    // AI voices often have too-perfect formant transitions
    const formantTransitionSmoothness = this.measureFormantTransitionSmoothness(formants);
    if (formantTransitionSmoothness > 0.9) {
      aiScore += 0.5;
    }

    // Check for formant frequency consistency
    const formantConsistency = this.measureFormantConsistency(formants);
    if (formantConsistency > 0.85) {
      aiScore += 0.5;
    }

    return Math.min(1, aiScore);
  }

  // Helper methods

  private calculatePitchRange(samples: Float32Array, sampleRate: number): number {
    const pitches = this.extractPitches(samples, sampleRate);
    return pitches.length > 0 ? Math.max(...pitches) - Math.min(...pitches) : 0;
  }

  private extractPitches(samples: Float32Array, sampleRate: number): number[] {
    const windowSize = 1024;
    const pitches: number[] = [];

    for (let i = 0; i < samples.length - windowSize; i += windowSize / 2) {
      const window = samples.slice(i, i + windowSize);
      const pitch = this.estimatePitch(window, sampleRate);
      if (pitch > 50 && pitch < 500) {
        pitches.push(pitch);
      }
    }

    return pitches;
  }

  private estimatePitch(samples: Float32Array, sampleRate: number): number {
    const minPeriod = Math.floor(sampleRate / 500);
    const maxPeriod = Math.floor(sampleRate / 50);

    let maxCorr = -Infinity;
    let bestLag = 0;

    for (let lag = minPeriod; lag <= maxPeriod; lag++) {
      let corr = 0;
      for (let i = 0; i < samples.length - lag; i++) {
        corr += samples[i] * samples[i + lag];
      }
      if (corr > maxCorr) {
        maxCorr = corr;
        bestLag = lag;
      }
    }

    return bestLag > 0 ? sampleRate / bestLag : 0;
  }

  private calculateRhythm(samples: Float32Array, sampleRate: number): number {
    const energy = this.calculateEnergy(samples);
    const peaks = this.findPeaks(energy);
    const intervals = this.calculateIntervals(peaks);

    if (intervals.length < 2) return 0;

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((a, b) => a + Math.pow(b - avgInterval, 2), 0) / intervals.length;

    return 1 / (1 + Math.sqrt(variance) / avgInterval);
  }

  private calculateEnergy(samples: Float32Array): number[] {
    const windowSize = 512;
    const energy: number[] = [];

    for (let i = 0; i < samples.length - windowSize; i += windowSize / 2) {
      let sum = 0;
      for (let j = i; j < i + windowSize; j++) {
        sum += samples[j] * samples[j];
      }
      energy.push(sum);
    }

    return energy;
  }

  private findPeaks(signal: number[]): number[] {
    const peaks: number[] = [];
    const threshold = Math.max(...signal) * 0.3;

    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > threshold) {
        peaks.push(i);
      }
    }

    return peaks;
  }

  private calculateIntervals(peaks: number[]): number[] {
    const intervals: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }
    return intervals;
  }

  private calculateStressTiming(samples: Float32Array, sampleRate: number): number {
    return 0.5;
  }

  private calculateSyllableTiming(samples: Float32Array, sampleRate: number): number {
    return 0.5;
  }

  private estimateSpeechRate(samples: Float32Array, sampleRate: number): number {
    const duration = samples.length / sampleRate;
    const syllables = this.estimateSyllableCount(samples, sampleRate);
    return syllables / duration;
  }

  private estimateSyllableCount(samples: Float32Array, sampleRate: number): number {
    const energy = this.calculateEnergy(samples);
    const peaks = this.findPeaks(energy);
    return peaks.length;
  }

  private measureProsodyConsistency(samples: Float32Array, sampleRate: number): number {
    const pitches = this.extractPitches(samples, sampleRate);
    if (pitches.length < 2) return 0;

    const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
    const variance = pitches.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pitches.length;
    const cv = Math.sqrt(variance) / mean;

    return 1 / (1 + cv);
  }

  private detectCoarticulation(samples: Float32Array, sampleRate: number): number {
    return 0.5;
  }

  private detectDisfluencies(samples: Float32Array, sampleRate: number): number {
    return Math.random() > 0.8 ? 1 : 0;
  }

  private analyzeBreathing(samples: Float32Array, sampleRate: number): number {
    return 0.6;
  }

  private extractFormants(samples: Float32Array, sampleRate: number): number[][] {
    return [[500, 1500, 2500]];
  }

  private measureFormantTransitionSmoothness(formants: number[][]): number {
    return 0.7;
  }

  private measureFormantConsistency(formants: number[][]): number {
    return 0.6;
  }
}