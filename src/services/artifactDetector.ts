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
    if (roboticScore > 0.5) {
      artifacts.push('Robotic voice patterns detected');
      artifactScore += 0.30;
    }

    // 2. Detect unnatural pauses
    const pauseAnomalies = this.detectPauseAnomalies(samples, audio.sampleRate);
    if (pauseAnomalies > 0.4) {
      artifacts.push('Unnatural pause patterns');
      artifactScore += 0.25;
    }

    // 3. Detect clipping artifacts
    const clippingScore = this.detectClipping(samples);
    if (clippingScore > 0.08) {
      artifacts.push('Digital clipping detected');
      artifactScore += 0.20;
    }

    // 4. Detect phase inconsistencies
    const phaseIssues = this.detectPhaseInconsistencies(samples);
    if (phaseIssues) {
      artifacts.push('Phase inconsistencies detected');
      artifactScore += 0.2;
    }

    // 5. Detect frequency anomalies
    const freqAnomalies = this.detectFrequencyAnomalies(samples, audio.sampleRate);
    if (freqAnomalies > 0.35) {
      artifacts.push('Unusual frequency distribution');
      artifactScore += 0.25;
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
    // Use smaller sample size for faster analysis (8192 samples max)
    const maxSamples = 8192;
    const samplesToUse = samples.length > maxSamples
      ? samples.slice(0, maxSamples)
      : samples;

    const n = samplesToUse.length;
    const real = new Float64Array(n);
    const imag = new Float64Array(n);

    // Copy samples to real array
    for (let i = 0; i < n; i++) {
      real[i] = samplesToUse[i];
      imag[i] = 0;
    }

    // Perform in-place Cooley-Tukey FFT
    this.fftInPlace(real, imag);

    // Return magnitude spectrum (first half only)
    const magnitudes: number[] = [];
    for (let k = 0; k < n / 2; k++) {
      magnitudes.push(Math.sqrt(real[k] * real[k] + imag[k] * imag[k]));
    }

    return magnitudes;
  }

  /**
   * In-place Cooley-Tukey FFT (radix-2) - O(n log n)
   */
  private fftInPlace(real: Float64Array, imag: Float64Array): void {
    const n = real.length;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        let temp = real[i];
        real[i] = real[j];
        real[j] = temp;
        temp = imag[i];
        imag[i] = imag[j];
        imag[j] = temp;
      }
      let k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey iterative FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angleStep = -2 * Math.PI / size;

      for (let i = 0; i < n; i += size) {
        for (let k = 0; k < halfSize; k++) {
          const angle = angleStep * k;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);

          const evenIdx = i + k;
          const oddIdx = i + k + halfSize;

          const tReal = cos * real[oddIdx] - sin * imag[oddIdx];
          const tImag = sin * real[oddIdx] + cos * imag[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] = real[evenIdx] + tReal;
          imag[evenIdx] = imag[evenIdx] + tImag;
        }
      }
    }
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