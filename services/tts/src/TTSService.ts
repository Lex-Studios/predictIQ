/**
 * TTSService — AI text-to-speech for social video narrations.
 *
 * Supports ElevenLabs (primary) and Google Cloud TTS (fallback).
 * Audio jobs are processed asynchronously; output files are stored
 * locally (or an S3-compatible bucket via the configured storage adapter).
 */

import fs from "fs/promises";
import path from "path";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import { createWriteStream } from "fs";
import { trace, context, SpanStatusCode, propagation } from "@opentelemetry/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TTSProvider = "elevenlabs" | "google";

export interface TTSVoice {
  /** Provider-specific voice ID */
  voiceId: string;
  /** BCP-47 language tag, e.g. "en-US", "es-ES" */
  language: string;
  /** Human-readable label */
  label: string;
}

export interface TTSJob {
  id: string;
  text: string;
  voice: TTSVoice;
  provider: TTSProvider;
  status: "pending" | "processing" | "done" | "error";
  outputPath?: string;
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface TTSConfig {
  provider: TTSProvider;
  elevenlabs?: {
    apiKey: string;
    /** Default model, e.g. "eleven_multilingual_v2" */
    modelId?: string;
  };
  google?: {
    /** Path to service-account JSON or inline credentials */
    keyFilename?: string;
    credentials?: object;
  };
  /** Directory where audio files are stored */
  outputDir: string;
  /** Authentication configuration — omit to disable auth */
  auth?: AuthConfig;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

/** API-key auth: caller passes one of the listed keys. */
export interface ApiKeyAuthConfig {
  type: "apikey";
  keys: string[];
}

/** JWT auth: caller passes a Bearer token signed with this secret. */
export interface JwtAuthConfig {
  type: "jwt";
  secret: string;
}

export type AuthConfig = ApiKeyAuthConfig | JwtAuthConfig;

/** Thrown when authentication fails; maps to HTTP 401. */
export class AuthError extends Error {
  readonly statusCode = 401;
  constructor(message = "Unauthorized") {
    super(message);
    this.name = "AuthError";
  }
}

/**
 * Validate a credential string against the configured auth strategy.
 * - API key: `credential` must be one of the configured keys.
 * - JWT: `credential` must be a valid HS256 token signed with the configured secret.
 *
 * Throws `AuthError` on failure; returns void on success.
 */
export function authenticate(credential: string | undefined, auth: AuthConfig): void {
  if (!credential) throw new AuthError("Missing credential");

  if (auth.type === "apikey") {
    if (!auth.keys.includes(credential)) throw new AuthError("Invalid API key");
    return;
  }

  // JWT — minimal HS256 verification without external deps
  const parts = credential.split(".");
  if (parts.length !== 3) throw new AuthError("Malformed JWT");

  const [headerB64, payloadB64, sigB64] = parts;
  const { createHmac } = require("crypto") as typeof import("crypto");
  const expected = createHmac("sha256", auth.secret)
    .update(`${headerB64}.${payloadB64}`)
    .digest("base64url");

  if (expected !== sigB64) throw new AuthError("Invalid JWT signature");

  const payload = JSON.parse(Buffer.from(payloadB64, "base64url").toString());
  if (payload.exp !== undefined && payload.exp < Math.floor(Date.now() / 1000)) {
    throw new AuthError("JWT expired");
  }
}

// ---------------------------------------------------------------------------
// Built-in voice catalogue (extend as needed)
// ---------------------------------------------------------------------------

export const VOICES: Record<string, TTSVoice> = {
  // ElevenLabs
  "el-rachel-en": { voiceId: "21m00Tcm4TlvDq8ikWAM", language: "en-US", label: "Rachel (EN)" },
  "el-adam-en":   { voiceId: "pNInz6obpgDQGcFmaJgB", language: "en-US", label: "Adam (EN)"   },
  "el-bella-en":  { voiceId: "EXAVITQu4vr4xnSDxMaL", language: "en-US", label: "Bella (EN)"  },
  // Google TTS
  "gcp-en-us-f":  { voiceId: "en-US-Neural2-F",      language: "en-US", label: "Google EN-F" },
  "gcp-es-es-f":  { voiceId: "es-ES-Neural2-A",      language: "es-ES", label: "Google ES-F" },
  "gcp-fr-fr-f":  { voiceId: "fr-FR-Neural2-A",      language: "fr-FR", label: "Google FR-F" },
  "gcp-de-de-f":  { voiceId: "de-DE-Neural2-F",      language: "de-DE", label: "Google DE-F" },
};

// ---------------------------------------------------------------------------
// In-memory job store (swap for Redis/DB in production)
// ---------------------------------------------------------------------------

const jobStore = new Map<string, TTSJob>();

function makeId(): string {
  return `tts_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Provider implementations
// ---------------------------------------------------------------------------

async function generateElevenLabs(
  text: string,
  voice: TTSVoice,
  config: NonNullable<TTSConfig["elevenlabs"]>
): Promise<Buffer> {
  const tracer = trace.getTracer("tts-service");
  return tracer.startActiveSpan("elevenlabs.generate", async (span) => {
    try {
      span.setAttribute("tts.provider", "elevenlabs");
      span.setAttribute("tts.voice.id", voice.voiceId);
      span.setAttribute("tts.text.length", text.length);

      const modelId = config.modelId ?? "eleven_multilingual_v2";
      const url = `https://api.elevenlabs.io/v1/text-to-speech/${voice.voiceId}`;

      const res = await fetch(url, {
        method: "POST",
        headers: {
          "xi-api-key": config.apiKey,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: modelId,
          voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        }),
      });

      if (!res.ok) {
        const msg = await res.text().catch(() => res.statusText);
        span.setStatus({ code: SpanStatusCode.ERROR, message: msg });
        throw new Error(`ElevenLabs error ${res.status}: ${msg}`);
      }

      const buffer = Buffer.from(await res.arrayBuffer());
      span.setAttribute("tts.audio.size", buffer.length);
      span.setStatus({ code: SpanStatusCode.OK });
      return buffer;
    } finally {
      span.end();
    }
  });
}

async function generateGoogle(
  text: string,
  voice: TTSVoice,
  config: NonNullable<TTSConfig["google"]>
): Promise<Buffer> {
  const tracer = trace.getTracer("tts-service");
  return tracer.startActiveSpan("google.generate", async (span) => {
    try {
      span.setAttribute("tts.provider", "google");
      span.setAttribute("tts.voice.id", voice.voiceId);
      span.setAttribute("tts.text.length", text.length);

      // Lazy-load @google-cloud/text-to-speech to keep it optional
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { TextToSpeechClient } = require("@google-cloud/text-to-speech") as {
        TextToSpeechClient: new (opts: object) => {
          synthesizeSpeech: (req: object) => Promise<[{ audioContent: Buffer | string }]>;
        };
      };

      const client = new TextToSpeechClient(config);

      const [response] = await client.synthesizeSpeech({
        input: { text },
        voice: { languageCode: voice.language, name: voice.voiceId },
        audioConfig: { audioEncoding: "MP3" },
      });

      const audio = response.audioContent;
      const buffer = Buffer.isBuffer(audio) ? audio : Buffer.from(audio as string, "base64");
      span.setAttribute("tts.audio.size", buffer.length);
      span.setStatus({ code: SpanStatusCode.OK });
      return buffer;
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      throw error;
    } finally {
      span.end();
    }
  });
}

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

async function saveAudio(
  buffer: Buffer,
  outputDir: string,
  jobId: string
): Promise<string> {
  await fs.mkdir(outputDir, { recursive: true });
  const filePath = path.join(outputDir, `${jobId}.mp3`);
  await fs.writeFile(filePath, buffer);
  return filePath;
}

/**
 * Concatenate multiple MP3 buffers into a single file.
 * For production use, prefer ffmpeg via `fluent-ffmpeg` for proper re-encoding.
 * This naive implementation works for CBR MP3 streams of the same bitrate.
 */
export async function mergeAudioFiles(
  inputPaths: string[],
  outputPath: string
): Promise<string> {
  const chunks: Buffer[] = [];
  for (const p of inputPaths) {
    chunks.push(await fs.readFile(p));
  }
  await fs.writeFile(outputPath, Buffer.concat(chunks));
  return outputPath;
}

// ---------------------------------------------------------------------------
// TTSService
// ---------------------------------------------------------------------------

export class TTSService {
  private config: TTSConfig;

  constructor(config: TTSConfig) {
    this.config = config;
  }

  /**
   * Enqueue a TTS job and return its ID immediately.
   * Processing happens asynchronously in the background.
   * @param credential API key or JWT Bearer token (required when auth is configured).
   */
  enqueue(text: string, voice: TTSVoice, provider?: TTSProvider, credential?: string): string {
    if (this.config.auth) authenticate(credential, this.config.auth);
    const id = makeId();
    const job: TTSJob = {
      id,
      text,
      voice,
      provider: provider ?? this.config.provider,
      status: "pending",
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    jobStore.set(id, job);

    // Fire-and-forget — caller polls via getJob()
    this._process(job).catch((err) => {
      const j = jobStore.get(id);
      if (j) {
        j.status = "error";
        j.error = String(err);
        j.updatedAt = new Date();
      }
    });

    return id;
  }

  /** Get current job state */
  getJob(id: string): TTSJob | undefined {
    return jobStore.get(id);
  }

  /** List all jobs (optionally filter by status) */
  listJobs(status?: TTSJob["status"]): TTSJob[] {
    const all = Array.from(jobStore.values());
    return status ? all.filter((j) => j.status === status) : all;
  }

  /**
   * Synchronous generation — awaits completion and returns the output path.
   * Useful for low-latency pipelines where you can afford to wait.
   * @param credential API key or JWT Bearer token (required when auth is configured).
   */
  async generate(text: string, voice: TTSVoice, provider?: TTSProvider, credential?: string): Promise<string> {
    const id = this.enqueue(text, voice, provider, credential);
    return this._waitForJob(id);
  }

  /**
   * Generate narrations for multiple segments and merge them into one file.
   * Returns the path to the merged audio file.
   * @param credential API key or JWT Bearer token (required when auth is configured).
   */
  async generateAndMerge(
    segments: Array<{ text: string; voice: TTSVoice; provider?: TTSProvider }>,
    mergedOutputPath: string,
    credential?: string
  ): Promise<string> {
    if (this.config.auth) authenticate(credential, this.config.auth);
    const paths = await Promise.all(
      segments.map((s) => this.generate(s.text, s.voice, s.provider, credential))
    );
    return mergeAudioFiles(paths, mergedOutputPath);
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  private async _process(job: TTSJob): Promise<void> {
    job.status = "processing";
    job.updatedAt = new Date();

    let buffer: Buffer;

    if (job.provider === "elevenlabs") {
      if (!this.config.elevenlabs) throw new Error("ElevenLabs config missing");
      buffer = await generateElevenLabs(job.text, job.voice, this.config.elevenlabs);
    } else {
      if (!this.config.google) throw new Error("Google TTS config missing");
      buffer = await generateGoogle(job.text, job.voice, this.config.google);
    }

    const outputPath = await saveAudio(buffer, this.config.outputDir, job.id);
    job.outputPath = outputPath;
    job.status = "done";
    job.updatedAt = new Date();
  }

  private _waitForJob(id: string, intervalMs = 200, timeoutMs = 60_000): Promise<string> {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      const tick = setInterval(() => {
        const job = jobStore.get(id);
        if (!job) { clearInterval(tick); return reject(new Error(`Job ${id} not found`)); }
        if (job.status === "done") { clearInterval(tick); return resolve(job.outputPath!); }
        if (job.status === "error") { clearInterval(tick); return reject(new Error(job.error)); }
        if (Date.now() - start > timeoutMs) {
          clearInterval(tick);
          reject(new Error(`Job ${id} timed out after ${timeoutMs}ms`));
        }
      }, intervalMs);
    });
  }
}
