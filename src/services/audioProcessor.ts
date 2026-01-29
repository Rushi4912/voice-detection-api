import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';
import { logger } from '../utils/logger';
import { AudioProcessingError } from '../middleware/errorHandler';

// Use ffmpeg-static to get the bundled ffmpeg binary path
// eslint-disable-next-line @typescript-eslint/no-var-requires
const ffmpegPath = require('ffmpeg-static') as string;
// eslint-disable-next-line @typescript-eslint/no-var-requires
const ffprobePath = require('@ffprobe-installer/ffprobe').path as string;

export interface ProcessedAudio {
  wavBuffer: Buffer;
  duration: number;
  sampleRate: number;
  channels: number;
  bitDepth: number;
  rawData: Float32Array;
  metadata: {
    format: string;
    bitrate?: number;
    encoder?: string;
  };
}

/**
 * Audio Processor
 * Handles audio format conversion, normalization, and feature extraction
 */
export class AudioProcessor {
  private tempDir: string;

  constructor() {
    this.tempDir = os.tmpdir();
  }

  /**
   * Process MP3 audio buffer
   * Converts to WAV and extracts metadata
   */
  async process(mp3Buffer: Buffer): Promise<ProcessedAudio> {
    const tempInputFile = path.join(this.tempDir, `input_${Date.now()}.mp3`);
    const tempOutputFile = path.join(this.tempDir, `output_${Date.now()}.wav`);

    try {
      console.log(`[AudioProcessor] Starting processing. Input size: ${mp3Buffer.length}`);
      // Write MP3 buffer to temporary file
      await fs.writeFile(tempInputFile, mp3Buffer);
      console.log(`[AudioProcessor] Written temp MP3 file: ${tempInputFile}`);

      // Convert MP3 to WAV using ffmpeg
      // Target: 16kHz, mono, 16-bit PCM (standard for speech processing)
      console.log('[AudioProcessor] Starting conversion to WAV...');
      await this.convertToWav(tempInputFile, tempOutputFile);
      console.log(`[AudioProcessor] Conversion complete. Reading output: ${tempOutputFile}`);

      // Read converted WAV file
      const wavBuffer = await fs.readFile(tempOutputFile);
      console.log(`[AudioProcessor] Read WAV buffer. Size: ${wavBuffer.length}`);

      // Parse WAV file and extract audio data
      console.log('[AudioProcessor] Starting WAV parsing...');
      const audioData = this.parseWav(wavBuffer);
      console.log('[AudioProcessor] WAV parsing complete.');

      // Get audio duration
      const duration = this.calculateDuration(audioData);

      // Validate audio duration (minimum 0.5s, maximum 5 minutes)
      if (duration < 0.5) {
        throw new AudioProcessingError('Audio is too short (minimum 0.5 seconds required)');
      }
      if (duration > 300) {
        throw new AudioProcessingError('Audio is too long (maximum 5 minutes allowed)');
      }

      // Extract metadata
      console.log('[AudioProcessor] Extracting metadata...');
      const metadata = await this.extractMetadata(tempInputFile);
      console.log('[AudioProcessor] Metadata extracted.');

      return {
        wavBuffer,
        duration,
        sampleRate: audioData.sampleRate,
        channels: audioData.channels,
        bitDepth: audioData.bitDepth,
        rawData: audioData.samples,
        metadata,
      };

    } catch (error) {
      logger.error('Audio processing failed:', error);
      throw error instanceof AudioProcessingError
        ? error
        : new AudioProcessingError(
          `Failed to process audio: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
    } finally {
      // Clean up temporary files
      await this.cleanup([tempInputFile, tempOutputFile]);
    }
  }

  /**
   * Convert MP3 to WAV using ffmpeg
   */
  private async convertToWav(inputFile: string, outputFile: string): Promise<void> {
    return new Promise((resolve, reject) => {
      logger.info(`Spawning ffmpeg from: ${ffmpegPath}`);
      // ffmpeg command: convert to 16kHz mono WAV
      const ffmpeg = spawn(ffmpegPath, [
        '-i', inputFile,
        '-acodec', 'pcm_s16le',  // 16-bit PCM
        '-ar', '16000',          // 16kHz sample rate
        '-ac', '1',              // Mono
        '-y',                    // Overwrite output file
        outputFile
      ]);

      let stderr = '';

      ffmpeg.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      ffmpeg.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new AudioProcessingError(
            `FFmpeg conversion failed with code ${code}: ${stderr}`
          ));
        }
      });

      ffmpeg.on('error', (error) => {
        reject(new AudioProcessingError(`FFmpeg execution failed: ${error.message}`));
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        ffmpeg.kill();
        reject(new AudioProcessingError('Audio conversion timeout (exceeded 30 seconds)'));
      }, 30000);
    });
  }

  /**
   * Parse WAV file and extract audio samples
   */
  private parseWav(wavBuffer: Buffer): {
    samples: Float32Array;
    sampleRate: number;
    channels: number;
    bitDepth: number;
  } {
    try {
      // Read WAV header
      const riff = wavBuffer.toString('ascii', 0, 4);
      if (riff !== 'RIFF') {
        throw new Error('Invalid WAV file: Missing RIFF header');
      }

      const wave = wavBuffer.toString('ascii', 8, 12);
      if (wave !== 'WAVE') {
        throw new Error('Invalid WAV file: Missing WAVE header');
      }

      // Find 'fmt ' chunk
      let offset = 12;
      while (offset < wavBuffer.length) {
        const chunkId = wavBuffer.toString('ascii', offset, offset + 4);
        const chunkSize = wavBuffer.readUInt32LE(offset + 4);

        if (chunkId === 'fmt ') {
          const audioFormat = wavBuffer.readUInt16LE(offset + 8);
          const channels = wavBuffer.readUInt16LE(offset + 10);
          const sampleRate = wavBuffer.readUInt32LE(offset + 12);
          const bitDepth = wavBuffer.readUInt16LE(offset + 22);

          // Find 'data' chunk
          let dataOffset = offset + 8 + chunkSize;
          while (dataOffset < wavBuffer.length) {
            const dataChunkId = wavBuffer.toString('ascii', dataOffset, dataOffset + 4);
            const dataChunkSize = wavBuffer.readUInt32LE(dataOffset + 4);

            if (dataChunkId === 'data') {
              // Extract audio samples
              const audioData = wavBuffer.slice(dataOffset + 8, dataOffset + 8 + dataChunkSize);

              // Convert to Float32Array (normalized to [-1, 1])
              const samples = new Float32Array(audioData.length / 2);
              for (let i = 0; i < samples.length; i++) {
                const sample = audioData.readInt16LE(i * 2);
                samples[i] = sample / 32768.0; // Normalize to [-1, 1]
              }

              return { samples, sampleRate, channels, bitDepth };
            }

            dataOffset += 8 + dataChunkSize;
          }

          throw new Error('Invalid WAV file: Missing data chunk');
        }

        offset += 8 + chunkSize;
      }

      throw new Error('Invalid WAV file: Missing fmt chunk');

    } catch (error) {
      throw new AudioProcessingError(
        `Failed to parse WAV file: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Calculate audio duration in seconds
   */
  private calculateDuration(audioData: {
    samples: Float32Array;
    sampleRate: number;
    channels: number;
  }): number {
    return audioData.samples.length / audioData.sampleRate / audioData.channels;
  }

  /**
   * Extract metadata from audio file using ffprobe
   */
  private async extractMetadata(inputFile: string): Promise<any> {
    return new Promise((resolve) => {
      const ffprobe = spawn(ffprobePath, [
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        inputFile
      ]);

      let stdout = '';

      ffprobe.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      ffprobe.on('close', (code) => {
        if (code === 0 && stdout) {
          try {
            const metadata = JSON.parse(stdout);
            resolve({
              format: metadata.format?.format_name || 'unknown',
              bitrate: metadata.format?.bit_rate ? parseInt(metadata.format.bit_rate) : undefined,
              encoder: metadata.format?.tags?.encoder,
            });
          } catch {
            resolve({ format: 'unknown' });
          }
        } else {
          resolve({ format: 'unknown' });
        }
      });

      // Timeout and return default
      setTimeout(() => {
        ffprobe.kill();
        resolve({ format: 'unknown' });
      }, 5000);
    });
  }

  /**
   * Clean up temporary files
   */
  private async cleanup(files: string[]): Promise<void> {
    for (const file of files) {
      try {
        await fs.unlink(file);
      } catch {
        // Ignore cleanup errors
      }
    }
  }
}