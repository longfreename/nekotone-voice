import { Channel, invoke } from '@tauri-apps/api/core';

export type Accel = 'auto' | 'directml' | 'cpu';
export type ComputeMode = 'auto' | 'both' | 'server-first';

export interface CmdError {
  kind: 'not-implemented' | 'model-missing' | 'not-found' | 'cancelled' | 'invalid' | 'error';
  message: string;
}
export function isCmdError(e: unknown): e is CmdError {
  return !!e && typeof e === 'object' && 'kind' in e && 'message' in e;
}
export function errText(e: unknown): string {
  if (isCmdError(e)) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
export function errKind(e: unknown): CmdError['kind'] {
  return isCmdError(e) ? e.kind : 'error';
}
export function notReady(e: unknown): boolean {
  return errKind(e) === 'not-implemented';
}

export interface Meta {
  version: string;
  dataDir: string;
  modelsDir: string;
  logPath: string;
  supportedExtensions: string[];
}

export interface SettingsData {
  theme: 'dark' | 'light';
  accelerator: Accel;
  voiceLearn: boolean;
  whisperModel: string;
  computeServer: string;
  computeToken: string;
  computeMode: ComputeMode;
  serveEnabled: boolean;
  servePort: number;
  serveToken: string;
  windowWidth: number;
  windowHeight: number;
  voice: unknown;
}

export interface Progress {
  message: string;
  fraction: number | null;
}

export interface ModelFile {
  name: string;
  url: string;
  size_bytes: number;
  sha256: string;
}
export interface ModelInfo {
  id: string;
  purpose: string;
  note: string;
  files: ModelFile[];
  license: string;
}
export interface ModelStatus {
  info: ModelInfo;
  installed: boolean;
  dir: string;
}

export interface ServeStatus {
  running: boolean;
  port: number;
  address: string | null;
  backend: string | null;
  error: string | null;
}

export interface ComputeServerInfo {
  server: string;
  version: string;
  host: string;
  backend: string;
  features: string[];
}

export interface VoiceParamDef {
  name: string;
  min: number;
  max: number;
  default: number;
  unit: string;
  doc: string;
}
export type VoiceBlockSpec = { type: string } & Record<string, number | string>;
export interface VoiceBlockSlot {
  id: string;
  block: VoiceBlockSpec;
}
export interface VoicePreset {
  id: string;
  name: string;
  description: string;
  category: string;
  disguise: boolean;
  builtin: boolean;
  blocks: VoiceBlockSlot[];
}
export interface VoiceBlockType {
  kind: string;
  label: string;
  description: string;
  params: VoiceParamDef[];
  default: VoiceBlockSpec;
}
export interface VoiceFrontEnd {
  hpf_hz: number;
  gate: Record<string, number>;
  deesser_on: boolean;
  deesser: Record<string, number>;
}
export interface VoiceAudioDevice {
  name: string;
  default: boolean;
  virtual_cable: boolean;
  channels: number;
  sample_rate: number;
}
export interface VoiceVirtualCable {
  kind: string;
  output_device: string;
  mic_name: string;
  mic_present: boolean;
}
export interface VoiceVirtualMic {
  platform: string;
  available: boolean;
  kind: 'VbCable' | 'VoiceMeeter' | 'VirtualAudioCable' | 'Loopback' | 'PulseNullSink' | 'None';
  output_device: string | null;
  mic_name: string | null;
  instructions: string;
  candidates: VoiceVirtualCable[];
}
export interface VoiceDevices {
  inputs: VoiceAudioDevice[];
  outputs: VoiceAudioDevice[];
  virtual_mic: VoiceVirtualMic;
}
export interface VoiceConfig {
  input: string | null;
  input_channel: number | null;
  output: string | null;
  virtual_mic: boolean;
  monitor: string | null;
  preset: VoicePreset;
  front_end: VoiceFrontEnd;
  output_gain_db: number;
  monitor_gain_db: number;
  ceiling_db: number;
  block_frames: number;
  sample_rate: number;
  start_muted?: boolean;
  pad_duck_db?: number;
}
export interface VoiceLatency {
  input_ms: number;
  processing_ms: number;
  ring_ms: number;
  output_ms: number;
}
export interface VoiceStyle {
  levelDb: number;
  bodyDb: number;
  brightDb: number;
  hnrDb: number;
  rateSylS: number;
}
export interface VoiceProfile {
  f0Hz: number;
  spreadSt: number;
  tract: number;
  voicedSecs: number;
  style?: VoiceStyle | null;
}
export interface VoiceStatus {
  running: boolean;
  muted: boolean;
  latency_ms: number;
  latency: VoiceLatency;
  input_level_db: number;
  output_level_db: number;
  gate_open: boolean;
  pads_playing?: number;
  gate_threshold_db: number;
  noise_floor_db: number;
  pitch_hz: number;
  out_pitch_hz: number;
  cpu_load: number;
  xruns: number;
  drift_ppm: number;
  sample_rate: number;
  input_device: string;
  output_device: string;
  monitor_device: string | null;
  preset_id: string;
  profile?: VoiceProfile;
  limiter_db?: number;
  lost?: string | null;
}
export type VoiceEvent =
  | { event: 'started'; input: string; output: string; monitor: string | null; sample_rate: number; latency_ms: number }
  | { event: 'level'; input_db: number; output_db: number; gate_open: boolean; pitch_hz: number; out_pitch_hz: number; limiter_db: number }
  | { event: 'xrun'; stream: string; count: number }
  | { event: 'device_lost'; role: string; device: string; message: string }
  | { event: 'recovered'; latency_ms: number }
  | { event: 'error'; message: string }
  | { event: 'stopped' };
export interface VoicePresetList {
  builtin: VoicePreset[];
  user: VoicePreset[];
  problems: string[];
  dir: string;
}
export interface VoiceSample {
  path: string;
  seconds: number;
  peaks: number[];
}
export interface VoicePreview {
  path: string;
  seconds: number;
  peaks: number[];
  cached: boolean;
  sample: 'recorded' | 'synthetic';
  render_ms: number;
}

export interface SpeakVoice {
  id: string;
  name: string;
  gender: 'female' | 'male';
  accent: 'american' | 'british';
  grade: string;
  featured: boolean;
  character: string;
}
export interface SpeakInfo {
  installed: boolean;
  download_bytes: number;
  license: string;
  voices: SpeakVoice[];
  whisper_model: string;
  whisper_installed: boolean;
  running: boolean;
  accents?: { id: string; label: string }[];
}
export interface SpeakConfig {
  input: string | null;
  output: string | null;
  virtual_mic: boolean;
  monitor: string | null;
  voice: string;
  speed: number;
  effect: VoicePreset | null;
  accent: string;
  listen: boolean;
  translate: boolean;
  language: string | null;
  prompt: string | null;
  trim_fillers: boolean;
  mask_profanity: boolean;
  half_duplex: boolean;
  convert?: boolean;
  keep_pauses?: boolean;
  hangover_ms: number;
  auto_pause?: boolean;
  output_gain_db: number;
  monitor_gain_db: number;
  start_muted?: boolean;
}
export interface SpeakStatus {
  running: boolean;
  muted: boolean;
  listening_enabled: boolean;
  hearing: boolean;
  speaking: boolean;
  queued: number;
  input_level_db: number;
  output_level_db: number;
  last_latency_ms: number;
  mean_latency_ms: number;
  utterances: number;
  voice: string;
  speed: number;
  effect_id: string | null;
  sample_rate: number;
  input_device: string | null;
  output_device: string;
  monitor_device: string | null;
  tts_device: string;
  stt_model: string;
  pause_ms: number;
  pauses_heard: number;
}
export type SpeakEvent =
  | { event: 'started'; input: string | null; output: string; monitor: string | null; sample_rate: number }
  | { event: 'listening' }
  | { event: 'transcribing'; speech_secs: number }
  | { event: 'heard'; id: number; text: string; original: string; language: string; stt_ms: number; speech_secs: number }
  | { event: 'dropped'; reason: string }
  | { event: 'queued'; id: number; text: string; source: 'mic' | 'typed' }
  | { event: 'speaking'; id: number; text: string; latency_ms: number | null; synth_ms: number }
  | { event: 'done'; id: number; skipped: boolean }
  | { event: 'level'; input_db: number; output_db: number; listening: boolean; speaking: boolean }
  | { event: 'error'; message: string }
  | { event: 'device_lost'; role: string; message: string }
  | { event: 'device_restored'; role: string; name: string }
  | { event: 'stopped' };
export interface SpeakPreview {
  path: string;
  seconds: number;
  peaks: number[];
  cached: boolean;
  render_ms: number;
}

export interface CloneVoice {
  id: string;
  name: string;
  reference_secs: number;
  created: number;
  source: string;
}
export interface CloneInfo {
  installed: boolean;
  download_bytes: number;
  license: string;
  voice: CloneVoice | null;
  voices: CloneVoice[];
  sample_secs: number;
  loaded_on: string | null;
  gpu_possible: boolean;
  min_secs: number;
}

export const api = {
  meta: () => invoke<Meta>('meta'),
  getSettings: () => invoke<SettingsData>('get_settings'),
  setSettings: (value: unknown) => invoke<void>('set_settings', { value }),
  openPath: (path: string) => invoke<void>('open_path', { path }),
  log: (line: string) => invoke<void>('log_line', { line }),
  voiceModels: () => invoke<ModelStatus[]>('voice_models'),
  voiceModelDownload: (id: string, on?: (p: Progress) => void) => {
    const ch = new Channel<Progress>();
    ch.onmessage = (p) => on?.(p);
    return invoke<string>('voice_model_download', { id, onProgress: ch });
  },
  voiceModelRemove: (id: string) => invoke<void>('voice_model_remove', { id }),
  computeServe: (on: boolean) => invoke<ServeStatus>('compute_serve', { on }),
  computeServeStatus: () => invoke<ServeStatus>('compute_serve_status'),
  computeServerTest: (url: string, token: string | null) => invoke<ComputeServerInfo>('compute_server_test', { url, token }),
  exit: (code: number) => invoke<void>('exit_app', { code }),
};

export const voiceApi = {
  devices: () => invoke<VoiceDevices>('voice_devices'),
  presets: () => invoke<VoicePresetList>('voice_presets'),
  blockTypes: () => invoke<{ blocks: VoiceBlockType[]; front_end: VoiceFrontEnd }>('voice_block_types'),
  start: (config: VoiceConfig, on: (e: VoiceEvent) => void) => {
    const ch = new Channel<VoiceEvent>();
    ch.onmessage = on;
    return invoke<VoiceStatus>('voice_start', { config, onEvent: ch });
  },
  stop: () => invoke<boolean>('voice_stop'),
  status: () => invoke<VoiceStatus | null>('voice_status'),
  setPreset: (preset: VoicePreset) => invoke<void>('voice_set_preset', { preset }),
  setMute: (mute: boolean) => invoke<void>('voice_set_mute', { mute }),
  setOutputGain: (db: number) => invoke<void>('voice_set_output_gain', { db }),
  setMonitorGain: (db: number) => invoke<void>('voice_set_monitor_gain', { db }),
  setFrontEnd: (frontEnd: VoiceFrontEnd) => invoke<void>('voice_set_front_end', { frontEnd }),
  setCeiling: (db: number) => invoke<void>('voice_set_ceiling', { db }),
  sample: () => invoke<VoiceSample | null>('voice_sample'),
  importSample: (path: string) => invoke<VoiceSample>('voice_import_sample', { path }),
  storeSample: (bytes: number[]) => invoke<VoiceSample>('voice_store_sample', { bytes }),
  deleteSample: () => invoke<void>('voice_delete_sample'),
  preview: (preset: VoicePreset, useRecorded: boolean) => invoke<VoicePreview>('voice_preview', { preset, useRecorded }),
  profile: () => invoke<{ saved: VoiceProfile | null; live: VoiceProfile | null }>('voice_profile'),
  calibrate: () => invoke<VoiceProfile>('voice_calibrate'),
  forgetProfile: () => invoke<void>('voice_forget_profile'),
};

export const speakApi = {
  info: () => invoke<SpeakInfo>('speak_info'),
  start: (config: SpeakConfig, on: (e: SpeakEvent) => void) => {
    const ch = new Channel<SpeakEvent>();
    ch.onmessage = on;
    return invoke<SpeakStatus>('speak_start', { config, onEvent: ch });
  },
  stop: () => invoke<boolean>('speak_stop'),
  status: () => invoke<SpeakStatus | null>('speak_status'),
  say: (text: string) => invoke<number | null>('speak_say', { text }),
  skip: () => invoke<void>('speak_skip'),
  clear: () => invoke<void>('speak_clear'),
  setMute: (mute: boolean) => invoke<void>('speak_set_mute', { mute }),
  setListening: (on: boolean) => invoke<void>('speak_set_listening', { on }),
  setVoice: (voice: string, speed: number) => invoke<void>('speak_set_voice', { voice, speed }),
  setEffect: (effect: VoicePreset | null) => invoke<void>('speak_set_effect', { effect }),
  setAccent: (accent: string) => invoke<void>('speak_set_accent', { accent }),
  setTranslate: (translate: boolean, language: string | null) => invoke<void>('speak_set_translate', { translate, language }),
  setOptions: (trimFillers: boolean, maskProfanity: boolean, halfDuplex: boolean) => invoke<void>('speak_set_options', { trimFillers, maskProfanity, halfDuplex }),
  setGains: (outputDb: number, monitorDb: number) => invoke<void>('speak_set_gains', { outputDb, monitorDb }),
  preview: (voice: string, text: string | null = null, speed: number | null = null, effect: VoicePreset | null = null, accent: string | null = null) =>
    invoke<SpeakPreview>('speak_preview', { voice, text, speed, effect, accent }),
};

export const cloneApi = {
  info: () => invoke<CloneInfo>('speak_clone_info'),
  make: (source: string | null, name: string | null, id: string | null = null) => invoke<CloneVoice>('speak_clone_make', { source, name, id }),
  remove: (id: string | null = null) => invoke<void>('speak_clone_delete', { id }),
  rename: (id: string, name: string) => invoke<CloneVoice>('speak_clone_rename', { id, name }),
};
