import type { SettingsData, VoiceFrontEnd } from '../api';

export type VoiceTab = 'changer' | 'speak' | 'clone';

export interface SpeakSettings {
  voice: string;
  speed: number;
  effectId: string | null;
  accent: string;
  listen: boolean;
  translate: boolean;
  convert: boolean;
  keepPauses: boolean;
  cloneConsent: boolean;
  language: string | null;
  trimFillers: boolean;
  maskProfanity: boolean;
  halfDuplex: boolean;
  hangoverMs: number;
  autoPause: boolean;
  outputGainDb: number;
  draft: string;
  showAll: boolean;
}

export interface VoiceSettings {
  tab: VoiceTab;
  presetId: string;
  input: string | null;
  target: 'virtual' | 'device';
  output: string | null;
  monitor: boolean;
  monitorDevice: string | null;
  monitorGainDb: number;
  outputGainDb: number;
  ceilingDb: number;
  gateAuto: boolean;
  gateDb: number;
  hpfHz: number;
  deesser: boolean;
  useRecordedPreview: boolean;
  sampleSeconds: number;
  previewText: string;
  cloneName: string;
  speak: SpeakSettings;
}

export type AppSettings = Omit<SettingsData, 'voice'> & { voice: VoiceSettings };

export const SPEAK_DEFAULTS: SpeakSettings = {
  voice: 'af_heart',
  speed: 1,
  effectId: null,
  accent: 'voice',
  listen: true,
  translate: false,
  convert: false,
  keepPauses: true,
  cloneConsent: false,
  language: null,
  trimFillers: true,
  maskProfanity: false,
  halfDuplex: false,
  hangoverMs: 275,
  autoPause: true,
  outputGainDb: 0,
  draft: '',
  showAll: false,
};

export const VOICE_DEFAULTS: VoiceSettings = {
  tab: 'changer',
  presetId: 'female',
  input: null,
  target: 'virtual',
  output: null,
  monitor: true,
  monitorDevice: null,
  monitorGainDb: -6,
  outputGainDb: 0,
  ceilingDb: -1,
  gateAuto: true,
  gateDb: -48,
  hpfHz: 70,
  deesser: true,
  useRecordedPreview: true,
  sampleSeconds: 8,
  previewText: 'Testing Voicekit. This preview uses the current voice settings.',
  cloneName: 'My voice',
  speak: SPEAK_DEFAULTS,
};

export const APP_DEFAULTS: AppSettings = {
  theme: 'dark',
  accelerator: 'auto',
  voiceLearn: false,
  whisperModel: 'whisper-base',
  computeServer: '',
  computeToken: '',
  computeMode: 'auto',
  serveEnabled: false,
  servePort: 8199,
  serveToken: '',
  windowWidth: 1240,
  windowHeight: 860,
  voice: VOICE_DEFAULTS,
};

export function voiceSettings(raw: unknown): VoiceSettings {
  const r = raw && typeof raw === 'object' ? (raw as Partial<VoiceSettings>) : {};
  const s = { ...VOICE_DEFAULTS, ...r };
  s.tab = s.tab === 'speak' || s.tab === 'clone' ? s.tab : 'changer';
  s.target = s.target === 'device' ? 'device' : 'virtual';
  s.monitorGainDb = clampNum(s.monitorGainDb, -60, 12, -6);
  s.outputGainDb = clampNum(s.outputGainDb, -24, 24, 0);
  s.ceilingDb = clampNum(s.ceilingDb, -24, 0, -1);
  s.gateDb = clampNum(s.gateDb, -60, -6, -48);
  s.hpfHz = clampNum(s.hpfHz, 20, 300, 70);
  s.sampleSeconds = clampNum(s.sampleSeconds, 3, 15, 8);
  s.previewText = typeof s.previewText === 'string' && s.previewText.trim() ? s.previewText : VOICE_DEFAULTS.previewText;
  s.cloneName = typeof s.cloneName === 'string' && s.cloneName.trim() ? s.cloneName : VOICE_DEFAULTS.cloneName;
  s.speak = { ...SPEAK_DEFAULTS, ...(r.speak && typeof r.speak === 'object' ? r.speak : {}) };
  s.speak.speed = clampNum(s.speak.speed, 0.7, 1.35, 1);
  s.speak.hangoverMs = clampNum(s.speak.hangoverMs, 120, 900, 275);
  s.speak.outputGainDb = clampNum(s.speak.outputGainDb, -24, 24, 0);
  s.speak.accent = typeof s.speak.accent === 'string' && s.speak.accent ? s.speak.accent : 'voice';
  s.speak.voice = typeof s.speak.voice === 'string' && s.speak.voice ? s.speak.voice : 'af_heart';
  s.speak.draft = typeof s.speak.draft === 'string' ? s.speak.draft : '';
  return s;
}

export function fillSettings(raw: unknown): AppSettings {
  const r = raw && typeof raw === 'object' ? (raw as Partial<AppSettings>) : {};
  const s: AppSettings = { ...APP_DEFAULTS, ...r, voice: voiceSettings(r.voice) };
  s.theme = s.theme === 'light' ? 'light' : 'dark';
  s.accelerator = s.accelerator === 'cpu' || s.accelerator === 'directml' ? s.accelerator : 'auto';
  s.computeMode = s.computeMode === 'both' || s.computeMode === 'server-first' ? s.computeMode : 'auto';
  s.whisperModel = typeof s.whisperModel === 'string' && s.whisperModel ? s.whisperModel : 'whisper-base';
  s.computeServer = typeof s.computeServer === 'string' ? s.computeServer : '';
  s.computeToken = typeof s.computeToken === 'string' ? s.computeToken : '';
  s.servePort = clampNum(s.servePort, 1024, 65535, 8199);
  s.serveToken = typeof s.serveToken === 'string' ? s.serveToken : '';
  s.windowWidth = clampNum(s.windowWidth, 920, 2200, 1240);
  s.windowHeight = clampNum(s.windowHeight, 620, 1600, 860);
  return s;
}

export function frontEnd(defaults: VoiceFrontEnd | null | undefined, settings: VoiceSettings): VoiceFrontEnd {
  const base: VoiceFrontEnd = defaults ?? { hpf_hz: 70, gate: { threshold_db: -48, auto: 1 }, deesser_on: true, deesser: {} };
  return {
    ...base,
    hpf_hz: settings.hpfHz,
    deesser_on: settings.deesser,
    gate: { ...base.gate, threshold_db: settings.gateDb, auto: settings.gateAuto ? 1 : 0 },
  };
}

export const SPEAK_LANGUAGES: [string, string][] = [
  ['auto', 'Auto detect'],
  ['en', 'English'],
  ['es', 'Spanish'],
  ['fr', 'French'],
  ['de', 'German'],
  ['it', 'Italian'],
  ['pt', 'Portuguese'],
  ['ja', 'Japanese'],
  ['ko', 'Korean'],
  ['zh', 'Chinese'],
];

export function dbToFrac(db: number, floor = -60) {
  if (!Number.isFinite(db)) return 0;
  return Math.max(0, Math.min(1, (db - floor) / -floor));
}

export function fmtDb(db: number, digits = 0) {
  return `${db > 0 ? '+' : ''}${db.toFixed(digits)} dB`;
}

export function fmtHz(hz: number) {
  if (!(hz > 0)) return '—';
  return hz >= 1000 ? `${(hz / 1000).toFixed(2)} kHz` : `${Math.round(hz)} Hz`;
}

export function fmtLatency(ms: number) {
  return `${Math.round(ms)} ms`;
}

export function clampNum(value: unknown, min: number, max: number, fallback: number) {
  const n = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, n));
}
