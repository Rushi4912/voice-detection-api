import { ProcessedAudio } from './audioProcessor';
import { logger } from '../utils/logger';

export interface AcousticResults {
  score: number; // 0-1, where 1 = likely AI-generated
  features: {
    mfcc: number[];
    spectralCentroid: number;
    spectralRolloff: number;
    spectralFlatness: number;
    zeroCrossingRate: number;
    rms: number;
    pitchVariation: number;
    jitter: number;
    shimmer: number;
  };
  anomalies: string[];
}

/**
 * Acoustic Analyzer
 * Analyzes acoustic features to detect AI-generated patterns
 */
export class AcousticAnalyzer {

  async analyze(audio: ProcessedAudio): Promise<AcousticResults> {
    logger.info('Starting acoustic analysis');

    const samples = audio.rawData;
    const sampleRate = audio.sampleRate;

    // Pre-compute expensive operations once (caching for reuse)
    const cachedSpectrum = this.calculateAverageSpectrum(samples, sampleRate);
    const cachedPitchContour = this.extractPitchContour(samples, sampleRate);

    // Extract acoustic features (using cached computations)
    const features = {
      mfcc: this.extractMFCC(samples, sampleRate),
      spectralCentroid: this.calculateSpectralCentroidFromSpectrum(cachedSpectrum, sampleRate),
      spectralRolloff: this.calculateSpectralRolloffFromSpectrum(cachedSpectrum, sampleRate),
      spectralFlatness: this.calculateSpectralFlatnessFromSpectrum(cachedSpectrum),
      zeroCrossingRate: this.calculateZeroCrossingRate(samples),
      rms: this.calculateRMS(samples),
      pitchVariation: this.analyzePitchVariationFromContour(cachedPitchContour),
      jitter: this.calculateJitterFromContour(cachedPitchContour),
      shimmer: this.calculateShimmer(samples),
    };

    // Detect anomalies that indicate AI generation
    const anomalies: string[] = [];
    let aiScore = 0;

    // 1. Check for unnatural pitch consistency (AI voices are too consistent)
    if (features.pitchVariation < 0.12) {
      anomalies.push('Unnaturally consistent pitch (typical of AI voice)');
      aiScore += 0.30;
    }

    // 2. Check jitter (frequency perturbation)
    // Human voices: 0.5-1.5%, AI voices: < 0.3% or > 2%
    if (features.jitter < 0.004 || features.jitter > 0.018) {
      anomalies.push('Abnormal jitter levels detected');
      aiScore += 0.25;
    }

    // 3. Check shimmer (amplitude perturbation)
    // Human voices: 3-10%, AI voices: < 2% or very high
    if (features.shimmer < 0.025 || features.shimmer > 0.12) {
      anomalies.push('Abnormal shimmer levels detected');
      aiScore += 0.25;
    }

    // 4. Check spectral characteristics
    // AI voices often have unnatural spectral distribution
    if (features.spectralCentroid > 2800 || features.spectralCentroid < 250) {
      anomalies.push('Unusual spectral characteristics');
      aiScore += 0.20;
    }

    // 5. Check spectral flatness (tonality)
    // AI voices often have unnatural spectral flatness (too flat or too tonal in wrong places)
    if (features.spectralFlatness < 0.01 || features.spectralFlatness > 0.4) {
      anomalies.push('Unnatural spectral flatness detected');
      aiScore += 0.20;
    }

    // 6. Check zero crossing rate consistency
    // AI voices tend to have very consistent ZCR
    const zcrVariation = this.calculateZCRVariation(samples);
    if (zcrVariation < 0.08) {
      anomalies.push('Unnaturally consistent zero crossing rate');
      aiScore += 0.15;
    }

    // 7. Check for digital artifacts in frequency domain (reuse cached spectrum)
    const hasDigitalArtifacts = this.detectDigitalArtifactsFromSpectrum(cachedSpectrum);
    if (hasDigitalArtifacts) {
      anomalies.push('Digital processing artifacts detected');
      aiScore += 0.15;
    }

    // Normalize score to 0-1
    const finalScore = Math.min(1, Math.max(0, aiScore));

    logger.info('Acoustic analysis completed', {
      score: finalScore,
      anomaliesCount: anomalies.length,
    });

    return {
      score: finalScore,
      features,
      anomalies,
    };
  }

  /**
   * Extract Mel-Frequency Cepstral Coefficients (MFCC)
   * Key feature for voice analysis
   */
  private extractMFCC(samples: Float32Array, sampleRate: number): number[] {
    // Simplified MFCC extraction (13 coefficients)
    // In production, use a library like meyda or implement full MFCC
    const frameSize = 512;
    const numCoefficients = 13;
    const mfccs: number[] = [];

    for (let i = 0; i < numCoefficients; i++) {
      let sum = 0;
      const numFrames = Math.floor(samples.length / frameSize);

      for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * frameSize;
        const end = Math.min(start + frameSize, samples.length);
        const frameData = samples.slice(start, end);

        // Simplified DCT calculation
        let frameSum = 0;
        for (let n = 0; n < frameData.length; n++) {
          frameSum += frameData[n] * Math.cos((Math.PI * i * (2 * n + 1)) / (2 * frameSize));
        }
        sum += Math.abs(frameSum);
      }

      mfccs.push(sum / numFrames);
    }

    return mfccs;
  }

  /**
   * Calculate spectral centroid (brightness of sound)
   */
  private calculateSpectralCentroid(samples: Float32Array, sampleRate: number): number {
    const spectrum = this.calculateAverageSpectrum(samples, sampleRate);
    const windowSize = 2048; // Must match calculateAverageSpectrum
    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < spectrum.length; i++) {
      const frequency = (i * sampleRate) / windowSize;
      const magnitude = spectrum[i];
      numerator += frequency * magnitude;
      denominator += magnitude;
    }

    return denominator > 0 ? numerator / denominator : 0;
  }

  /**
   * Calculate spectral rolloff (frequency below which 85% of energy is contained)
   */
  private calculateSpectralRolloff(samples: Float32Array, sampleRate: number): number {
    const spectrum = this.calculateAverageSpectrum(samples, sampleRate);
    const windowSize = 2048;
    const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);
    const threshold = 0.85 * totalEnergy;

    let cumulativeEnergy = 0;
    for (let i = 0; i < spectrum.length; i++) {
      cumulativeEnergy += spectrum[i];
      if (cumulativeEnergy >= threshold) {
        return (i * sampleRate) / windowSize;
      }
    }

    return sampleRate / 2; // Nyquist
  }

  /**
   * Calculate zero crossing rate
   */
  private calculateZeroCrossingRate(samples: Float32Array): number {
    let crossings = 0;
    for (let i = 1; i < samples.length; i++) {
      if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
        crossings++;
      }
    }
    return crossings / samples.length;
  }

  /**
   * Calculate ZCR variation (to detect unnatural consistency)
   */
  private calculateZCRVariation(samples: Float32Array): number {
    const windowSize = 1024;
    const numWindows = Math.floor(samples.length / windowSize);
    const zcrs: number[] = [];

    for (let i = 0; i < numWindows; i++) {
      const start = i * windowSize;
      const window = samples.slice(start, start + windowSize);
      zcrs.push(this.calculateZeroCrossingRate(window));
    }

    // Calculate coefficient of variation
    const mean = zcrs.reduce((sum, val) => sum + val, 0) / zcrs.length;
    const variance = zcrs.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / zcrs.length;
    const stdDev = Math.sqrt(variance);

    return mean > 0 ? stdDev / mean : 0;
  }

  /**
   * Calculate RMS (Root Mean Square) energy
   */
  private calculateRMS(samples: Float32Array): number {
    const sumSquares = samples.reduce((sum, val) => sum + val * val, 0);
    return Math.sqrt(sumSquares / samples.length);
  }

  /**
   * Analyze pitch variation (AI voices lack natural variation)
   */
  private analyzePitchVariation(samples: Float32Array, sampleRate: number): number {
    const pitches = this.extractPitchContour(samples, sampleRate);

    if (pitches.length === 0) return 0;

    const mean = pitches.reduce((sum, val) => sum + val, 0) / pitches.length;
    const variance = pitches.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / pitches.length;
    const stdDev = Math.sqrt(variance);

    // Coefficient of variation
    return mean > 0 ? stdDev / mean : 0;
  }

  /**
   * Extract pitch contour over time
   */
  private extractPitchContour(samples: Float32Array, sampleRate: number): number[] {
    const windowSize = 1024;
    const hopSize = 512;
    const pitches: number[] = [];

    for (let i = 0; i < samples.length - windowSize; i += hopSize) {
      const window = samples.slice(i, i + windowSize);
      const pitch = this.estimatePitch(window, sampleRate);
      if (pitch > 0) {
        pitches.push(pitch);
      }
    }

    return pitches;
  }

  /**
   * Estimate pitch using autocorrelation
   */
  private estimatePitch(samples: Float32Array, sampleRate: number): number {
    const minPeriod = Math.floor(sampleRate / 500); // 500 Hz max
    const maxPeriod = Math.floor(sampleRate / 50);  // 50 Hz min

    let maxCorrelation = -Infinity;
    let bestPeriod = 0;

    for (let lag = minPeriod; lag <= maxPeriod; lag++) {
      let correlation = 0;
      for (let i = 0; i < samples.length - lag; i++) {
        correlation += samples[i] * samples[i + lag];
      }

      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        bestPeriod = lag;
      }
    }

    return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
  }

  /**
   * Calculate jitter (pitch perturbation)
   */
  private calculateJitter(samples: Float32Array, sampleRate: number): number {
    const pitches = this.extractPitchContour(samples, sampleRate);

    if (pitches.length < 2) return 0;

    let sumDifferences = 0;
    for (let i = 1; i < pitches.length; i++) {
      sumDifferences += Math.abs(pitches[i] - pitches[i - 1]);
    }

    const averageDifference = sumDifferences / (pitches.length - 1);
    const averagePitch = pitches.reduce((sum, val) => sum + val, 0) / pitches.length;

    return averagePitch > 0 ? averageDifference / averagePitch : 0;
  }

  /**
   * Calculate shimmer (amplitude perturbation)
   */
  private calculateShimmer(samples: Float32Array): number {
    const windowSize = 512;
    const amplitudes: number[] = [];

    for (let i = 0; i < samples.length - windowSize; i += windowSize / 2) {
      const window = samples.slice(i, i + windowSize);
      const amplitude = this.calculateRMS(window);
      amplitudes.push(amplitude);
    }

    if (amplitudes.length < 2) return 0;

    let sumDifferences = 0;
    for (let i = 1; i < amplitudes.length; i++) {
      sumDifferences += Math.abs(amplitudes[i] - amplitudes[i - 1]);
    }

    const averageDifference = sumDifferences / (amplitudes.length - 1);
    const averageAmplitude = amplitudes.reduce((sum, val) => sum + val, 0) / amplitudes.length;

    return averageAmplitude > 0 ? averageDifference / averageAmplitude : 0;
  }

  /**
   * Detect digital artifacts (typical in AI-generated audio)
   */
  private detectDigitalArtifacts(samples: Float32Array, sampleRate: number): boolean {
    const spectrum = this.calculateAverageSpectrum(samples, sampleRate);

    // Check for unnatural frequency peaks
    // Look for suspicious peaks near Nyquist frequency
    const nyquistBin = Math.floor(spectrum.length * 0.9);
    const nyquistRegion = spectrum.slice(nyquistBin);
    const nyquistEnergy = nyquistRegion.reduce((sum, val) => sum + val, 0);
    const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);

    // If too much energy near Nyquist, likely digital artifact
    const nyquistRatio = totalEnergy > 0 ? nyquistEnergy / totalEnergy : 0;

    return nyquistRatio > 0.05;
  }

  /**
   * Detect digital artifacts using pre-computed spectrum
   */
  private detectDigitalArtifactsFromSpectrum(spectrum: number[]): boolean {
    // Check for unnatural frequency peaks near Nyquist frequency
    const nyquistBin = Math.floor(spectrum.length * 0.9);
    const nyquistRegion = spectrum.slice(nyquistBin);
    const nyquistEnergy = nyquistRegion.reduce((sum, val) => sum + val, 0);
    const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);

    const nyquistRatio = totalEnergy > 0 ? nyquistEnergy / totalEnergy : 0;
    return nyquistRatio > 0.05;
  }

  /**
   * Calculate spectral centroid from pre-computed spectrum
   */
  private calculateSpectralCentroidFromSpectrum(spectrum: number[], sampleRate: number): number {
    const windowSize = 2048;
    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < spectrum.length; i++) {
      const frequency = (i * sampleRate) / windowSize;
      const magnitude = spectrum[i];
      numerator += frequency * magnitude;
      denominator += magnitude;
    }

    return denominator > 0 ? numerator / denominator : 0;
  }

  /**
   * Calculate spectral rolloff from pre-computed spectrum
   */
  private calculateSpectralRolloffFromSpectrum(spectrum: number[], sampleRate: number): number {
    const windowSize = 2048;
    const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);
    const threshold = 0.85 * totalEnergy;

    let cumulativeEnergy = 0;
    for (let i = 0; i < spectrum.length; i++) {
      cumulativeEnergy += spectrum[i];
      if (cumulativeEnergy >= threshold) {
        return (i * sampleRate) / windowSize;
      }
    }

    return sampleRate / 2; // Nyquist
  }

  /**
   * Calculate spectral flatness (tonality coefficient)
   * Ratio of geometric mean to arithmetic mean of power spectrum
   */
  private calculateSpectralFlatnessFromSpectrum(spectrum: number[]): number {
    let sumLog = 0;
    let sum = 0;
    const n = spectrum.length;

    for (let i = 0; i < n; i++) {
      // Add small epsilon to avoid log(0)
      const val = spectrum[i] + 1e-10;
      sumLog += Math.log(val);
      sum += val;
    }

    const geometricMean = Math.exp(sumLog / n);
    const arithmeticMean = sum / n;

    return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
  }

  /**
   * Analyze pitch variation from pre-computed pitch contour
   */
  private analyzePitchVariationFromContour(pitches: number[]): number {
    if (pitches.length === 0) return 0;

    const mean = pitches.reduce((sum, val) => sum + val, 0) / pitches.length;
    const variance = pitches.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / pitches.length;
    const stdDev = Math.sqrt(variance);

    return mean > 0 ? stdDev / mean : 0;
  }

  /**
   * Calculate jitter from pre-computed pitch contour
   */
  private calculateJitterFromContour(pitches: number[]): number {
    if (pitches.length < 2) return 0;

    let sumDifferences = 0;
    for (let i = 1; i < pitches.length; i++) {
      sumDifferences += Math.abs(pitches[i] - pitches[i - 1]);
    }

    const averageDifference = sumDifferences / (pitches.length - 1);
    const averagePitch = pitches.reduce((sum, val) => sum + val, 0) / pitches.length;

    return averagePitch > 0 ? averageDifference / averagePitch : 0;
  }

  /**
   * Calculate average frequency spectrum using windowed FFT
   * optimizing for performance (O(W*K log K) instead of O(N^2))
   */
  private calculateAverageSpectrum(samples: Float32Array, sampleRate: number): number[] {
    const windowSize = 2048; // Power of 2
    const numWindows = 5;    // Analyze 5 representative segments

    if (samples.length < windowSize) {
      return this.computeWindowFFT(samples);
    }

    const spectrum = new Array(windowSize / 2).fill(0);
    const step = Math.floor((samples.length - windowSize) / (numWindows - 1));

    let windowsProcessed = 0;

    for (let i = 0; i < numWindows; ++i) {
      const start = Math.min(i * step, samples.length - windowSize);
      const window = samples.slice(start, start + windowSize);
      const windowSpectrum = this.computeWindowFFT(window);

      for (let j = 0; j < spectrum.length; j++) {
        spectrum[j] += windowSpectrum[j];
      }
      windowsProcessed++;
    }

    // Average
    for (let j = 0; j < spectrum.length; j++) {
      spectrum[j] /= windowsProcessed;
    }

    return spectrum;
  }

  /**
   * Compute FFT for a single window using Cooley-Tukey FFT algorithm
   * O(n log n) complexity instead of O(n²) DFT
   */
  private computeWindowFFT(window: Float32Array): number[] {
    const n = window.length;

    // Apply Hamming window and prepare complex array
    const real = new Float64Array(n);
    const imag = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const windowFunc = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
      real[i] = window[i] * windowFunc;
      imag[i] = 0;
    }

    // Perform in-place Cooley-Tukey FFT
    this.fftInPlace(real, imag);

    // Return magnitude spectrum (first half only - up to Nyquist)
    const magnitudes: number[] = [];
    for (let k = 0; k < n / 2; k++) {
      magnitudes.push(Math.sqrt(real[k] * real[k] + imag[k] * imag[k]));
    }

    return magnitudes;
  }

  /**
   * Cooley-Tukey FFT algorithm (in-place, radix-2)
   * O(n log n) complexity
   */
  private fftInPlace(real: Float64Array, imag: Float64Array): void {
    const n = real.length;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        // Swap real[i] and real[j]
        let temp = real[i];
        real[i] = real[j];
        real[j] = temp;
        // Swap imag[i] and imag[j]
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
}