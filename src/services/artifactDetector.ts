import { ProcessedAudio } from './audioProcessor';
import { logger } from '../utils/logger';

export interface ArtifactResults {
  score: number;
  artifacts: string[];
}

/**
 * Artifact Detector
 * Detects digital artifacts and anomalies specific to AI-generated audio
 */
export class ArtifactDetector {
  
  async detect(audio: ProcessedAudio): Promise<ArtifactResults> {
    logger.info('Starting artifact detection');
    
    const samples = audio.rawData;
    const artifacts: string[] = [];
    let artifactScore = 0;
    
    // 1. Detect robotic/mechanical patterns
    const roboticScore = this.detectRoboticPatterns(samples);
    if (roboticScore > 0.6) {
      artifacts.push('Robotic voice patterns detected');
      artifactScore += 0.25;
    }
    
    // 2. Detect unnatural pauses
    const pauseAnomalies = this.detectPauseAnomalies(samples, audio.sampleRate);
    if (pauseAnomalies > 0.5) {
      artifacts.push('Unnatural pause patterns');
      artifactScore += 0.2;
    }
    
    // 3. Detect clipping artifacts
    const clippingScore = this.detectClipping(samples);
    if (clippingScore > 0.1) {
      artifacts.push('Digital clipping detected');
      artifactScore += 0.15;
    }
    
    // 4. Detect phase inconsistencies
    const phaseIssues = this.detectPhaseInconsistencies(samples);
    if (phaseIssues) {
      artifacts.push('Phase inconsistencies detected');
      artifactScore += 0.2;
    }
    
    // 5. Detect frequency anomalies
    const freqAnomalies = this.detectFrequencyAnomalies(samples, audio.sampleRate);
    if (freqAnomalies > 0.4) {
      artifacts.push('Unusual frequency distribution');
      artifactScore += 0.2;
    }
    
    const finalScore = Math.min(1, artifactScore);
    
    logger.info('Artifact detection completed', {
      score: finalScore,
      artifactsFound: artifacts.length,
    });
    
    return { score: finalScore, artifacts };
  }
  
  private detectRoboticPatterns(samples: Float32Array): number {
    // Check for overly repetitive patterns
    const windowSize = 1024;
    let repetitionScore = 0;
    
    for (let i = 0; i < samples.length - windowSize * 2; i += windowSize) {
      const window1 = samples.slice(i, i + windowSize);
      const window2 = samples.slice(i + windowSize, i + windowSize * 2);
      
      let correlation = 0;
      for (let j = 0; j < windowSize; j++) {
        correlation += window1[j] * window2[j];
      }
      
      if (Math.abs(correlation) > 0.9) {
        repetitionScore += 1;
      }
    }
    
    return repetitionScore / Math.floor((samples.length - windowSize * 2) / windowSize);
  }
  
  private detectPauseAnomalies(samples: Float32Array, sampleRate: number): number {
    const silenceThreshold = 0.01;
    const minPauseDuration = 0.1; // seconds
    const minPauseSamples = Math.floor(minPauseDuration * sampleRate);
    
    let pauseDurations: number[] = [];
    let currentPause = 0;
    
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) < silenceThreshold) {
        currentPause++;
      } else if (currentPause > minPauseSamples) {
        pauseDurations.push(currentPause);
        currentPause = 0;
      } else {
        currentPause = 0;
      }
    }
    
    if (pauseDurations.length < 2) return 0;
    
    // Check if pauses are too uniform (AI characteristic)
    const mean = pauseDurations.reduce((a, b) => a + b, 0) / pauseDurations.length;
    const variance = pauseDurations.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pauseDurations.length;
    const cv = Math.sqrt(variance) / mean; // Coefficient of variation
    
    return cv < 0.2 ? 0.8 : 0.2; // Low variation = likely AI
  }
  
  private detectClipping(samples: Float32Array): number {
    const clippingThreshold = 0.99;
    let clippedSamples = 0;
    
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) > clippingThreshold) {
        clippedSamples++;
      }
    }
    
    return clippedSamples / samples.length;
  }
  
  private detectPhaseInconsistencies(samples: Float32Array): boolean {
    // Check for sudden phase jumps
    const windowSize = 512;
    let phaseJumps = 0;
    
    for (let i = 0; i < samples.length - windowSize * 2; i += windowSize) {
      const window1 = samples.slice(i, i + windowSize);
      const window2 = samples.slice(i + windowSize, i + windowSize * 2);
      
      const phase1 = this.calculatePhase(window1);
      const phase2 = this.calculatePhase(window2);
      
      const phaseDiff = Math.abs(phase2 - phase1);
      if (phaseDiff > Math.PI / 2) {
        phaseJumps++;
      }
    }
    
    return phaseJumps > 5;
  }
  
  private calculatePhase(samples: Float32Array): number {
    let real = 0;
    let imag = 0;
    
    for (let i = 0; i < samples.length; i++) {
      const angle = (2 * Math.PI * i) / samples.length;
      real += samples[i] * Math.cos(angle);
      imag += samples[i] * Math.sin(angle);
    }
    
    return Math.atan2(imag, real);
  }
  
  private detectFrequencyAnomalies(samples: Float32Array, sampleRate: number): number {
    // Simple spectral analysis
    const fft = this.computeFFT(samples);
    const nyquist = sampleRate / 2;
    
    // Check for unusual concentration in specific bands
    const bands = [
      { low: 0, high: 300, expected: 0.15 },
      { low: 300, high: 3400, expected: 0.65 }, // Speech range
      { low: 3400, high: nyquist, expected: 0.20 },
    ];
    
    let anomalyScore = 0;
    
    for (const band of bands) {
      const bandEnergy = this.calculateBandEnergy(fft, band.low, band.high, sampleRate);
      const deviation = Math.abs(bandEnergy - band.expected);
      anomalyScore += deviation;
    }
    
    return anomalyScore / bands.length;
  }
  
  private computeFFT(samples: Float32Array): number[] {
    const n = samples.length;
    const magnitudes: number[] = [];
    
    for (let k = 0; k < n / 2; k++) {
      let real = 0;
      let imag = 0;
      
      for (let t = 0; t < n; t++) {
        const angle = (2 * Math.PI * k * t) / n;
        real += samples[t] * Math.cos(angle);
        imag -= samples[t] * Math.sin(angle);
      }
      
      magnitudes.push(Math.sqrt(real * real + imag * imag));
    }
    
    return magnitudes;
  }
  
  private calculateBandEnergy(
    fft: number[],
    lowFreq: number,
    highFreq: number,
    sampleRate: number
  ): number {
    const binWidth = sampleRate / (2 * fft.length);
    const lowBin = Math.floor(lowFreq / binWidth);
    const highBin = Math.ceil(highFreq / binWidth);
    
    let bandEnergy = 0;
    let totalEnergy = 0;
    
    for (let i = 0; i < fft.length; i++) {
      totalEnergy += fft[i];
      if (i >= lowBin && i <= highBin) {
        bandEnergy += fft[i];
      }
    }
    
    return bandEnergy / totalEnergy;
  }
}