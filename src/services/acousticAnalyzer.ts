import { ProcessedAudio } from './audioProcessor';
import { logger } from '../utils/logger';

export interface AcousticResults {
  score: number; // 0-1, where 1 = likely AI-generated
  features: {
    mfcc: number[];
    spectralCentroid: number;
    spectralRolloff: number;
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
    
    // Extract acoustic features
    const features = {
      mfcc: this.extractMFCC(samples, sampleRate),
      spectralCentroid: this.calculateSpectralCentroid(samples, sampleRate),
      spectralRolloff: this.calculateSpectralRolloff(samples, sampleRate),
      zeroCrossingRate: this.calculateZeroCrossingRate(samples),
      rms: this.calculateRMS(samples),
      pitchVariation: this.analyzePitchVariation(samples, sampleRate),
      jitter: this.calculateJitter(samples, sampleRate),
      shimmer: this.calculateShimmer(samples),
    };
    
    // Detect anomalies that indicate AI generation
    const anomalies: string[] = [];
    let aiScore = 0;
    
    // 1. Check for unnatural pitch consistency (AI voices are too consistent)
    if (features.pitchVariation < 0.15) {
      anomalies.push('Unnaturally consistent pitch (typical of AI voice)');
      aiScore += 0.25;
    }
    
    // 2. Check jitter (frequency perturbation)
    // Human voices: 0.5-1.5%, AI voices: < 0.3% or > 2%
    if (features.jitter < 0.003 || features.jitter > 0.02) {
      anomalies.push('Abnormal jitter levels detected');
      aiScore += 0.2;
    }
    
    // 3. Check shimmer (amplitude perturbation)
    // Human voices: 3-10%, AI voices: < 2% or very high
    if (features.shimmer < 0.02 || features.shimmer > 0.15) {
      anomalies.push('Abnormal shimmer levels detected');
      aiScore += 0.2;
    }
    
    // 4. Check spectral characteristics
    // AI voices often have unnatural spectral distribution
    if (features.spectralCentroid > 3000 || features.spectralCentroid < 200) {
      anomalies.push('Unusual spectral characteristics');
      aiScore += 0.15;
    }
    
    // 5. Check zero crossing rate consistency
    // AI voices tend to have very consistent ZCR
    const zcrVariation = this.calculateZCRVariation(samples);
    if (zcrVariation < 0.05) {
      anomalies.push('Unnaturally consistent zero crossing rate');
      aiScore += 0.1;
    }
    
    // 6. Check for digital artifacts in frequency domain
    const hasDigitalArtifacts = this.detectDigitalArtifacts(samples, sampleRate);
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
    const fft = this.simpleFFT(samples);
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < fft.length; i++) {
      const frequency = (i * sampleRate) / (2 * fft.length);
      const magnitude = fft[i];
      numerator += frequency * magnitude;
      denominator += magnitude;
    }
    
    return denominator > 0 ? numerator / denominator : 0;
  }
  
  /**
   * Calculate spectral rolloff (frequency below which 85% of energy is contained)
   */
  private calculateSpectralRolloff(samples: Float32Array, sampleRate: number): number {
    const fft = this.simpleFFT(samples);
    const totalEnergy = fft.reduce((sum, val) => sum + val, 0);
    const threshold = 0.85 * totalEnergy;
    
    let cumulativeEnergy = 0;
    for (let i = 0; i < fft.length; i++) {
      cumulativeEnergy += fft[i];
      if (cumulativeEnergy >= threshold) {
        return (i * sampleRate) / (2 * fft.length);
      }
    }
    
    return (fft.length * sampleRate) / (2 * fft.length);
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
    const fft = this.simpleFFT(samples);
    
    // Check for unnatural frequency peaks
    const nyquistFreq = sampleRate / 2;
    const binWidth = nyquistFreq / fft.length;
    
    // Look for suspicious peaks near Nyquist frequency
    const nyquistBin = Math.floor(fft.length * 0.9);
    const nyquistRegion = fft.slice(nyquistBin);
    const nyquistEnergy = nyquistRegion.reduce((sum, val) => sum + val, 0);
    const totalEnergy = fft.reduce((sum, val) => sum + val, 0);
    
    // If too much energy near Nyquist, likely digital artifact
    const nyquistRatio = nyquistEnergy / totalEnergy;
    
    return nyquistRatio > 0.05;
  }
  
  /**
   * Simple FFT implementation (for demonstration)
   * In production, use a library like fft.js or dsp.js
   */
  private simpleFFT(samples: Float32Array): number[] {
    const n = samples.length;
    const magnitudes: number[] = [];
    
    // Simplified DFT (not efficient, but works for demo)
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
}