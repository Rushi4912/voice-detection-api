import { ProcessedAudio } from './audioProcessor';
import { logger } from '../utils/logger';

export interface DeepLearningResults {
  score: number; // 0-1, where 1 = likely AI-generated
  predictions: {
    cnn: number;
    rnn: number;
    ensemble: number;
  };
  confidence: number;
}

/**
 * Deep Learning Detector
 * Uses multiple neural network architectures for detection
 */
export class DeepLearningDetector {

  async detect(audio: ProcessedAudio): Promise<DeepLearningResults> {
    logger.info('Starting deep learning detection');

    // Extract features for ML models
    const features = this.extractDeepFeatures(audio);

    // Run multiple models in parallel
    const [cnnPrediction, rnnPrediction] = await Promise.all([
      this.runCNNModel(features),
      this.runRNNModel(features),
    ]);

    // Ensemble prediction (weighted average)
    const ensemblePrediction = (cnnPrediction * 0.6 + rnnPrediction * 0.4);

    // Calculate confidence (based on agreement between models)
    const modelAgreement = 1 - Math.abs(cnnPrediction - rnnPrediction);
    const confidence = modelAgreement * Math.max(cnnPrediction, rnnPrediction);

    logger.info('Deep learning detection completed', {
      cnn: cnnPrediction,
      rnn: rnnPrediction,
      ensemble: ensemblePrediction,
      confidence,
    });

    return {
      score: ensemblePrediction,
      predictions: {
        cnn: cnnPrediction,
        rnn: rnnPrediction,
        ensemble: ensemblePrediction,
      },
      confidence,
    };
  }

  /**
   * Extract deep learning features
   */
  private extractDeepFeatures(audio: ProcessedAudio): any {
    const samples = audio.rawData;
    const sampleRate = audio.sampleRate;

    // 1. Compute spectrogram ONCE (expensive operation)
    const spectrogram = this.computeSpectrogram(samples, sampleRate);

    // 2. Mel-spectrogram from cached spectrogram (no re-computation)
    const melSpectrogram = this.computeMelSpectrogramFromSpectrogram(spectrogram, sampleRate);

    // 3. Delta features (temporal dynamics)
    const deltaFeatures = this.computeDelta(melSpectrogram);
    const deltaDeltaFeatures = this.computeDelta(deltaFeatures);

    // 4. Chroma features from cached spectrogram (no re-computation)
    const chromaFeatures = this.computeChromaFromSpectrogram(spectrogram, sampleRate);

    return {
      spectrogram,
      melSpectrogram,
      deltaFeatures,
      deltaDeltaFeatures,
      chromaFeatures,
      duration: audio.duration,
      sampleRate: audio.sampleRate,
    };
  }

  /**
   * CNN Model for pattern recognition
   * Analyzes spectrograms for AI artifacts
   */
  private async runCNNModel(features: any): Promise<number> {
    // Simulate CNN model inference
    // In production, this would use TensorFlow.js or ONNX Runtime

    // Analyze spectrogram patterns
    const spectrogramScore = this.analyzeSpectrogramPatterns(features.spectrogram);

    // Analyze mel-spectrogram patterns
    const melScore = this.analyzeMelPatterns(features.melSpectrogram);

    // Analyze temporal consistency
    const temporalScore = this.analyzeTemporalConsistency(features.deltaFeatures);

    // Weighted combination
    const cnnScore = (
      spectrogramScore * 0.4 +
      melScore * 0.4 +
      temporalScore * 0.2
    );

    return Math.min(1, Math.max(0, cnnScore));
  }

  /**
   * RNN Model for temporal pattern analysis
   * Detects unnatural sequences in AI speech
   */
  private async runRNNModel(features: any): Promise<number> {
    // Simulate RNN/LSTM model inference
    // In production, this would use TensorFlow.js or ONNX Runtime

    // Analyze temporal evolution
    const temporalEvolution = this.analyzeTemporalEvolution(features.melSpectrogram);

    // Analyze transition smoothness (AI is often too smooth)
    const transitionScore = this.analyzeTransitionSmoothness(features.deltaFeatures);

    // Analyze harmonic consistency
    const harmonicScore = this.analyzeHarmonicConsistency(features.chromaFeatures);

    // Weighted combination
    const rnnScore = (
      temporalEvolution * 0.4 +
      transitionScore * 0.35 +
      harmonicScore * 0.25
    );

    return Math.min(1, Math.max(0, rnnScore));
  }

  /**
   * Compute spectrogram (Short-Time Fourier Transform)
   */
  private computeSpectrogram(samples: Float32Array, sampleRate: number): number[][] {
    const windowSize = 512;
    const hopSize = 256;
    const spectrogram: number[][] = [];

    for (let i = 0; i < samples.length - windowSize; i += hopSize) {
      const window = this.applyHannWindow(samples.slice(i, i + windowSize));
      const fft = this.fft(window);
      const magnitudes = fft.map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));
      spectrogram.push(magnitudes.slice(0, windowSize / 2));
    }

    return spectrogram;
  }

  /**
   * Compute mel-spectrogram
   */
  private computeMelSpectrogram(samples: Float32Array, sampleRate: number): number[][] {
    const spectrogram = this.computeSpectrogram(samples, sampleRate);
    const melFilters = this.createMelFilterbank(spectrogram[0].length, sampleRate, 40);

    const melSpectrogram: number[][] = [];
    for (const frame of spectrogram) {
      const melFrame: number[] = [];
      for (const filter of melFilters) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) {
          sum += frame[i] * filter[i];
        }
        melFrame.push(Math.log(sum + 1e-10)); // Log-mel
      }
      melSpectrogram.push(melFrame);
    }

    return melSpectrogram;
  }

  /**
   * Compute delta features (first derivative)
   */
  private computeDelta(features: number[][]): number[][] {
    const delta: number[][] = [];

    for (let t = 0; t < features.length; t++) {
      const deltaFrame: number[] = [];

      for (let i = 0; i < features[t].length; i++) {
        let d = 0;

        if (t > 0 && t < features.length - 1) {
          d = (features[t + 1][i] - features[t - 1][i]) / 2;
        } else if (t === 0) {
          d = features[t + 1][i] - features[t][i];
        } else {
          d = features[t][i] - features[t - 1][i];
        }

        deltaFrame.push(d);
      }

      delta.push(deltaFrame);
    }

    return delta;
  }

  /**
   * Compute chroma features (pitch class profile)
   */
  private computeChroma(samples: Float32Array, sampleRate: number): number[][] {
    const spectrogram = this.computeSpectrogram(samples, sampleRate);
    const chroma: number[][] = [];

    for (const frame of spectrogram) {
      const chromaFrame = new Array(12).fill(0);

      for (let i = 0; i < frame.length; i++) {
        const freq = (i * sampleRate) / (2 * frame.length);
        const pitch = 12 * Math.log2(freq / 440) + 69; // MIDI note number
        const pitchClass = Math.round(pitch) % 12;

        if (pitchClass >= 0 && pitchClass < 12) {
          chromaFrame[pitchClass] += frame[i];
        }
      }

      chroma.push(chromaFrame);
    }

    return chroma;
  }

  /**
   * Analyze spectrogram patterns for AI artifacts
   */
  private analyzeSpectrogramPatterns(spectrogram: number[][]): number {
    let aiScore = 0;

    // 1. Check for overly regular patterns (AI characteristic)
    const regularity = this.measureRegularity(spectrogram);
    if (regularity > 0.6) {
      aiScore += 0.35;
    }

    // 2. Check for unnatural frequency concentration
    const frequencyConcentration = this.measureFrequencyConcentration(spectrogram);
    if (frequencyConcentration > 0.5) {
      aiScore += 0.30;
    }

    // 3. Check for phase coherence (AI voices have unnatural coherence)
    const phaseCoherence = this.measurePhaseCoherence(spectrogram);
    if (phaseCoherence > 0.70) {
      aiScore += 0.30;
    }

    return aiScore;
  }

  /**
   * Analyze mel-spectrogram patterns
   */
  private analyzeMelPatterns(melSpectrogram: number[][]): number {
    let aiScore = 0;

    // Check for unnatural energy distribution
    const energyDistribution = this.analyzeEnergyDistribution(melSpectrogram);
    if (energyDistribution < 0.25 || energyDistribution > 0.75) {
      aiScore += 0.45;
    }

    // Check for texture anomalies
    const textureAnomaly = this.detectTextureAnomalies(melSpectrogram);
    aiScore += textureAnomaly * 0.6;

    return aiScore;
  }

  /**
   * Analyze temporal consistency
   */
  private analyzeTemporalConsistency(deltaFeatures: number[][]): number {
    // AI voices have too-consistent temporal evolution
    const variance = this.calculateVariance2D(deltaFeatures);

    // Low variance indicates unnatural consistency
    return variance < 0.05 ? 0.8 : 0.2;
  }

  /**
   * Analyze temporal evolution patterns
   */
  private analyzeTemporalEvolution(melSpectrogram: number[][]): number {
    let aiScore = 0;

    // Check for abrupt transitions (or lack thereof)
    const transitionSmoothness = this.measureTransitionSmoothness(melSpectrogram);

    // AI voices: too smooth or too abrupt
    if (transitionSmoothness < 0.2 || transitionSmoothness > 0.9) {
      aiScore += 0.6;
    }

    return aiScore;
  }

  /**
   * Analyze transition smoothness
   */
  private analyzeTransitionSmoothness(deltaFeatures: number[][]): number {
    let totalChange = 0;
    let smoothTransitions = 0;

    for (let t = 1; t < deltaFeatures.length; t++) {
      let frameChange = 0;
      for (let i = 0; i < deltaFeatures[t].length; i++) {
        frameChange += Math.abs(deltaFeatures[t][i] - deltaFeatures[t - 1][i]);
      }
      totalChange += frameChange;

      if (frameChange < 0.1) {
        smoothTransitions++;
      }
    }

    const smoothnessRatio = smoothTransitions / (deltaFeatures.length - 1);

    // AI: either too smooth or not smooth enough
    return smoothnessRatio > 0.8 || smoothnessRatio < 0.2 ? 0.7 : 0.2;
  }

  /**
   * Analyze harmonic consistency
   */
  private analyzeHarmonicConsistency(chromaFeatures: number[][]): number {
    // AI voices often have too-perfect harmonic structure
    const harmonicVariation = this.calculateVariance2D(chromaFeatures);

    return harmonicVariation < 0.1 ? 0.75 : 0.25;
  }

  // Helper methods

  private measureRegularity(data: number[][]): number {
    // Measure autocorrelation to detect regular patterns
    let sumCorrelation = 0;
    const lag = Math.floor(data.length / 10);

    for (let i = 0; i < data.length - lag; i++) {
      let correlation = 0;
      for (let j = 0; j < data[i].length; j++) {
        correlation += data[i][j] * data[i + lag][j];
      }
      sumCorrelation += Math.abs(correlation);
    }

    return sumCorrelation / (data.length - lag);
  }

  private measureFrequencyConcentration(spectrogram: number[][]): number {
    const avgSpectrum = new Array(spectrogram[0].length).fill(0);

    for (const frame of spectrogram) {
      for (let i = 0; i < frame.length; i++) {
        avgSpectrum[i] += frame[i];
      }
    }

    const totalEnergy = avgSpectrum.reduce((sum, val) => sum + val, 0);
    const maxEnergy = Math.max(...avgSpectrum);

    return maxEnergy / totalEnergy;
  }

  private measurePhaseCoherence(spectrogram: number[][]): number {
    // Simplified phase coherence measure
    let coherence = 0;

    for (let t = 1; t < spectrogram.length; t++) {
      let frameSimilarity = 0;
      for (let i = 0; i < spectrogram[t].length; i++) {
        const diff = Math.abs(spectrogram[t][i] - spectrogram[t - 1][i]);
        frameSimilarity += 1 / (1 + diff);
      }
      coherence += frameSimilarity / spectrogram[t].length;
    }

    return coherence / (spectrogram.length - 1);
  }

  private analyzeEnergyDistribution(melSpectrogram: number[][]): number {
    const totalEnergy = melSpectrogram.flat().reduce((sum, val) => sum + Math.exp(val), 0);
    const lowFreqEnergy = melSpectrogram.map(f => f.slice(0, 10)).flat().reduce((sum, val) => sum + Math.exp(val), 0);

    return lowFreqEnergy / totalEnergy;
  }

  private detectTextureAnomalies(melSpectrogram: number[][]): number {
    // Calculate local binary pattern-like features
    let anomalyScore = 0;

    for (let t = 1; t < melSpectrogram.length - 1; t++) {
      for (let f = 1; f < melSpectrogram[t].length - 1; f++) {
        const center = melSpectrogram[t][f];
        const neighbors = [
          melSpectrogram[t - 1][f], melSpectrogram[t + 1][f],
          melSpectrogram[t][f - 1], melSpectrogram[t][f + 1],
        ];

        const pattern = neighbors.map(n => n > center ? 1 : 0).join('');

        // Check for unnatural patterns
        if (pattern === '1111' || pattern === '0000') {
          anomalyScore += 0.1;
        }
      }
    }

    return Math.min(1, anomalyScore / (melSpectrogram.length * melSpectrogram[0].length));
  }

  private measureTransitionSmoothness(melSpectrogram: number[][]): number {
    let totalDiff = 0;

    for (let t = 1; t < melSpectrogram.length; t++) {
      let frameDiff = 0;
      for (let i = 0; i < melSpectrogram[t].length; i++) {
        frameDiff += Math.abs(melSpectrogram[t][i] - melSpectrogram[t - 1][i]);
      }
      totalDiff += frameDiff;
    }

    const avgDiff = totalDiff / (melSpectrogram.length - 1);
    return 1 / (1 + avgDiff); // Normalize to 0-1
  }

  private calculateVariance2D(data: number[][]): number {
    const flat = data.flat();
    const mean = flat.reduce((sum, val) => sum + val, 0) / flat.length;
    const variance = flat.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flat.length;
    return variance;
  }

  private applyHannWindow(samples: Float32Array): Float32Array {
    const windowed = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const multiplier = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (samples.length - 1)));
      windowed[i] = samples[i] * multiplier;
    }
    return windowed;
  }

  /**
   * Cooley-Tukey FFT algorithm - O(n log n) complexity
   */
  private fft(samples: Float32Array): { real: number; imag: number }[] {
    const n = samples.length;
    const real = new Float64Array(n);
    const imag = new Float64Array(n);

    // Copy samples to real array
    for (let i = 0; i < n; i++) {
      real[i] = samples[i];
      imag[i] = 0;
    }

    // Perform in-place FFT
    this.fftInPlace(real, imag);

    // Return result
    const result: { real: number; imag: number }[] = [];
    for (let k = 0; k < n; k++) {
      result.push({ real: real[k], imag: imag[k] });
    }

    return result;
  }

  /**
   * In-place Cooley-Tukey FFT (radix-2)
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

  /**
   * Compute mel-spectrogram from pre-computed spectrogram (avoids re-computing spectrogram)
   */
  private computeMelSpectrogramFromSpectrogram(spectrogram: number[][], sampleRate: number): number[][] {
    if (spectrogram.length === 0) return [];

    const melFilters = this.createMelFilterbank(spectrogram[0].length, sampleRate, 40);

    const melSpectrogram: number[][] = [];
    for (const frame of spectrogram) {
      const melFrame: number[] = [];
      for (const filter of melFilters) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) {
          sum += frame[i] * filter[i];
        }
        melFrame.push(Math.log(sum + 1e-10)); // Log-mel
      }
      melSpectrogram.push(melFrame);
    }

    return melSpectrogram;
  }

  /**
   * Compute chroma from pre-computed spectrogram (avoids re-computing spectrogram)
   */
  private computeChromaFromSpectrogram(spectrogram: number[][], sampleRate: number): number[][] {
    const chroma: number[][] = [];

    for (const frame of spectrogram) {
      const chromaFrame = new Array(12).fill(0);

      for (let i = 0; i < frame.length; i++) {
        const freq = (i * sampleRate) / (2 * frame.length);
        const pitch = 12 * Math.log2(freq / 440) + 69; // MIDI note number
        const pitchClass = Math.round(pitch) % 12;

        if (pitchClass >= 0 && pitchClass < 12) {
          chromaFrame[pitchClass] += frame[i];
        }
      }

      chroma.push(chromaFrame);
    }

    return chroma;
  }

  private createMelFilterbank(nfft: number, sampleRate: number, nMels: number): number[][] {
    const filters: number[][] = [];
    const melMin = 0;
    const melMax = 2595 * Math.log10(1 + sampleRate / 2 / 700);
    const melPoints = Array.from({ length: nMels + 2 }, (_, i) => melMin + (melMax - melMin) * i / (nMels + 1));
    const freqPoints = melPoints.map(mel => 700 * (Math.pow(10, mel / 2595) - 1));
    const binPoints = freqPoints.map(freq => Math.floor((nfft + 1) * freq / sampleRate));

    for (let i = 1; i <= nMels; i++) {
      const filter = new Array(nfft).fill(0);

      for (let j = binPoints[i - 1]; j < binPoints[i]; j++) {
        filter[j] = (j - binPoints[i - 1]) / (binPoints[i] - binPoints[i - 1]);
      }

      for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
        filter[j] = (binPoints[i + 1] - j) / (binPoints[i + 1] - binPoints[i]);
      }

      filters.push(filter);
    }

    return filters;
  }
}