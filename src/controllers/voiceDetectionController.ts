import { Request, Response } from 'express';
import { logger } from '../utils/logger';
import { sanitizeBase64Audio, validateAudioBuffer } from '../middleware/validation';
import { AudioProcessingError, ModelInferenceError } from '../middleware/errorHandler';
import { AudioProcessor } from '../services/audioProcessor';
import { AcousticAnalyzer } from '../services/acousticAnalyzer';
import { DeepLearningDetector } from '../services/deepLearningDetector';
import { ArtifactDetector } from '../services/artifactDetector';
import { LanguageSpecificAnalyzer } from '../services/languageSpecificAnalyzer';
import { EnsembleScorer } from '../services/ensembleScorer';
import { MetricsCollector } from '../services/metricsCollector';

/**
 * Main voice detection controller
 * Orchestrates the entire detection pipeline
 */
export const detectVoice = async (req: Request, res: Response): Promise<void> => {
  const startTime = Date.now();
  const requestId = req.requestId || 'unknown';
  
  try {
    logger.info(`Starting voice detection - ${requestId}`);
    
    // Extract request data
    const { audio: rawAudio, language } = req.body;
    
    // Step 1: Sanitize and decode base64 audio
    logger.info(`Step 1: Decoding base64 audio - ${requestId}`);
    const cleanedBase64 = sanitizeBase64Audio(rawAudio);
    const audioBuffer = Buffer.from(cleanedBase64, 'base64');
    
    // Validate audio format
    validateAudioBuffer(audioBuffer);
    
    logger.info(`Audio decoded successfully - ${requestId}`, {
      bufferSize: `${(audioBuffer.length / 1024).toFixed(2)}KB`,
    });
    
    // Step 2: Process audio (convert to WAV, extract metadata)
    logger.info(`Step 2: Processing audio - ${requestId}`);
    const audioProcessor = new AudioProcessor();
    const processedAudio = await audioProcessor.process(audioBuffer);
    
    logger.info(`Audio processed - ${requestId}`, {
      duration: `${processedAudio.duration}s`,
      sampleRate: processedAudio.sampleRate,
      channels: processedAudio.channels,
    });
    
    // Step 3: Run parallel detection layers
    logger.info(`Step 3: Running detection layers - ${requestId}`);
    
    const [
      acousticResults,
      deepLearningResults,
      artifactResults,
      languageResults,
    ] = await Promise.all([
      // Layer 1: Acoustic feature analysis
      runAcousticAnalysis(processedAudio, requestId),
      
      // Layer 2: Deep learning classification
      runDeepLearningDetection(processedAudio, requestId),
      
      // Layer 3: AI artifact detection
      runArtifactDetection(processedAudio, requestId),
      
      // Layer 4: Language-specific analysis
      runLanguageSpecificAnalysis(processedAudio, language, requestId),
    ]);
    
    logger.info(`All detection layers completed - ${requestId}`);
    
    // Step 4: Ensemble scoring and final decision
    logger.info(`Step 4: Computing final score - ${requestId}`);
    const ensembleScorer = new EnsembleScorer();
    const finalResult = ensembleScorer.computeScore({
      acoustic: acousticResults,
      deepLearning: deepLearningResults,
      artifact: artifactResults,
      language: languageResults,
    });
    
    // Step 5: Build response
    const processingTime = Date.now() - startTime;
    
    const response = {
      result: finalResult.classification, // "AI_GENERATED" or "HUMAN"
      confidence: parseFloat(finalResult.confidence.toFixed(4)),
      analysis: {
        acoustic: {
          score: parseFloat(acousticResults.score.toFixed(4)),
          features: acousticResults.features,
          anomalies: acousticResults.anomalies,
        },
        deepLearning: {
          score: parseFloat(deepLearningResults.score.toFixed(4)),
          modelPredictions: deepLearningResults.predictions,
        },
        artifact: {
          score: parseFloat(artifactResults.score.toFixed(4)),
          detectedArtifacts: artifactResults.artifacts,
        },
        languageSpecific: {
          score: parseFloat(languageResults.score.toFixed(4)),
          detectedLanguage: languageResults.detectedLanguage,
          languageConfidence: parseFloat(languageResults.languageConfidence.toFixed(4)),
        },
      },
      metadata: {
        audioDuration: parseFloat(processedAudio.duration.toFixed(2)),
        sampleRate: processedAudio.sampleRate,
        fileSize: `${(audioBuffer.length / 1024).toFixed(2)}KB`,
      },
      requestId,
      processingTime: `${processingTime}ms`,
      timestamp: new Date().toISOString(),
    };
    
    // Collect metrics
    MetricsCollector.recordDetection({
      requestId,
      result: finalResult.classification,
      confidence: finalResult.confidence,
      processingTime,
      language: languageResults.detectedLanguage,
    });
    
    logger.info(`Detection completed successfully - ${requestId}`, {
      result: finalResult.classification,
      confidence: finalResult.confidence,
      processingTime: `${processingTime}ms`,
    });
    
    res.status(200).json(response);
    
  } catch (error) {
    const processingTime = Date.now() - startTime;
    
    logger.error(`Detection failed - ${requestId}`, {
      error: error instanceof Error ? error.message : 'Unknown error',
      processingTime: `${processingTime}ms`,
    });
    
    // Collect error metrics
    MetricsCollector.recordError({
      requestId,
      errorType: error instanceof Error ? error.constructor.name : 'UnknownError',
      errorMessage: error instanceof Error ? error.message : 'Unknown error',
      processingTime,
    });
    
    throw error;
  }
};

/**
 * Run acoustic feature analysis
 */
async function runAcousticAnalysis(audio: any, requestId: string): Promise<any> {
  try {
    logger.info(`Running acoustic analysis - ${requestId}`);
    const analyzer = new AcousticAnalyzer();
    const results = await analyzer.analyze(audio);
    logger.info(`Acoustic analysis completed - ${requestId}`, { score: results.score });
    return results;
  } catch (error) {
    logger.error(`Acoustic analysis failed - ${requestId}`, { error });
    throw new AudioProcessingError('Acoustic analysis failed');
  }
}

/**
 * Run deep learning detection
 */
async function runDeepLearningDetection(audio: any, requestId: string): Promise<any> {
  try {
    logger.info(`Running deep learning detection - ${requestId}`);
    const detector = new DeepLearningDetector();
    const results = await detector.detect(audio);
    logger.info(`Deep learning detection completed - ${requestId}`, { score: results.score });
    return results;
  } catch (error) {
    logger.error(`Deep learning detection failed - ${requestId}`, { error });
    throw new ModelInferenceError('Deep learning detection failed');
  }
}

/**
 * Run artifact detection
 */
async function runArtifactDetection(audio: any, requestId: string): Promise<any> {
  try {
    logger.info(`Running artifact detection - ${requestId}`);
    const detector = new ArtifactDetector();
    const results = await detector.detect(audio);
    logger.info(`Artifact detection completed - ${requestId}`, { score: results.score });
    return results;
  } catch (error) {
    logger.error(`Artifact detection failed - ${requestId}`, { error });
    throw new AudioProcessingError('Artifact detection failed');
  }
}

/**
 * Run language-specific analysis
 */
async function runLanguageSpecificAnalysis(
  audio: any,
  language: string | undefined,
  requestId: string
): Promise<any> {
  try {
    logger.info(`Running language-specific analysis - ${requestId}`, { language });
    const analyzer = new LanguageSpecificAnalyzer();
    const results = await analyzer.analyze(audio, language);
    logger.info(`Language-specific analysis completed - ${requestId}`, {
      score: results.score,
      detectedLanguage: results.detectedLanguage,
    });
    return results;
  } catch (error) {
    logger.error(`Language-specific analysis failed - ${requestId}`, { error });
    throw new AudioProcessingError('Language-specific analysis failed');
  }
}