import './voice.css';
import './voice-speak.css';

import { convertFileSrc } from '@tauri-apps/api/core';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { api, cloneApi, errText, speakApi, voiceApi, type CloneInfo, type CloneVoice, type SpeakEvent, type SpeakInfo, type SpeakStatus, type VoiceDevices, type VoiceFrontEnd, type VoicePreset, type VoiceProfile, type VoiceSample, type VoiceStatus } from '../api';
import { askName, debounce, h, icon, replace, toast } from '../dom';
import { store } from '../store';
import { levelMeter, rangeField, statChip } from './voice-controls';
import { renderClonePanel } from './voice-pads';
import { fillSettings, frontEnd, type AppSettings } from './voice-model';
import { renderSpeakPanel, type SpeakEntry } from './voice-speak';

export class VoiceView {
  readonly el = h('div', { class: 'view voice-view' });
  private devices: VoiceDevices | null = null;
  private presets: VoicePreset[] = [];
  private defaults: VoiceFrontEnd | null = null;
  private voiceStatus: VoiceStatus | null = null;
  private speakInfo: SpeakInfo | null = null;
  private speakStatus: SpeakStatus | null = null;
  private cloneInfo: CloneInfo | null = null;
  private sample: VoiceSample | null = null;
  private profile: { saved: VoiceProfile | null; live: VoiceProfile | null } | null = null;
  private entries: SpeakEntry[] = [];
  private loading = true;
  private speakBusy = false;
  private cloneBusy = false;
  private recordState: { active: boolean; elapsed: number; peakDb: number } | null = null;
  private preview = new Audio();
  private saveLater = debounce(() => {
    void api.setSettings(store.settings).catch((e) => toast(errText(e), 'error'));
  }, 180);

  constructor() {
    this.render();
    window.setInterval(() => void this.pollStatus(), 350);
  }

  summary() {
    if (this.voiceStatus?.running) return `${this.voiceStatus.input_device} → ${this.voiceStatus.output_device}`;
    if (this.speakStatus?.running) return `Speak for me · ${this.speakStatus.queued} queued`;
    return 'Voice changer, cloned voices, and text-to-speech.';
  }

  windowStateText() {
    if (this.voiceStatus?.running) return this.voiceStatus.muted ? 'Live muted' : 'Live';
    if (this.speakStatus?.running) return this.speakStatus.speaking ? 'Speaking' : 'Listening';
    return 'Idle';
  }

  async refreshAll() {
    this.loading = true;
    this.render();
    const [devices, blocks, presets, sample, profile, speakInfo, cloneInfo, voiceStatus, speakStatus] = await Promise.all([
      voiceApi.devices().catch(() => null),
      voiceApi.blockTypes().catch(() => null),
      voiceApi.presets().catch(() => null),
      voiceApi.sample().catch(() => null),
      voiceApi.profile().catch(() => null),
      speakApi.info().catch(() => null),
      cloneApi.info().catch(() => null),
      voiceApi.status().catch(() => null),
      speakApi.status().catch(() => null),
    ]);
    this.devices = devices;
    this.defaults = blocks?.front_end ?? this.defaults ?? { hpf_hz: 70, gate: { threshold_db: -48, auto: 1 }, deesser_on: true, deesser: {} };
    this.presets = [...(presets?.builtin ?? []), ...(presets?.user ?? [])];
    this.sample = sample;
    this.profile = profile;
    this.speakInfo = speakInfo;
    this.cloneInfo = cloneInfo;
    this.voiceStatus = voiceStatus;
    this.speakStatus = speakStatus;
    this.ensureSelections();
    this.loading = false;
    this.render();
  }

  private ensureSelections() {
    const s = this.settings();
    if (this.presets.length && !this.presets.some((preset) => preset.id === s.voice.presetId)) {
      s.voice.presetId = this.presets[0].id;
    }
    if (this.cloneInfo && s.voice.speak.voice.startsWith('clone:') && !this.cloneInfo.voices.some((voice) => voice.id === s.voice.speak.voice)) {
      s.voice.speak.voice = this.speakInfo?.voices[0]?.id ?? 'af_heart';
    }
  }

  private settings(): AppSettings {
    return store.settings;
  }

  private save(render = true) {
    store.emit('settings');
    if (render) this.render();
    this.saveLater();
  }

  private selectedPreset() {
    return this.presets.find((preset) => preset.id === this.settings().voice.presetId) ?? this.presets[0] ?? null;
  }

  private voiceOutput() {
    const s = this.settings().voice;
    if (s.target === 'virtual' && this.devices?.virtual_mic.output_device) {
      return { output: this.devices.virtual_mic.output_device, virtualMic: true };
    }
    return { output: s.output, virtualMic: false };
  }

  private async pollStatus() {
    const [voiceStatus, speakStatus] = await Promise.all([voiceApi.status().catch(() => null), speakApi.status().catch(() => null)]);
    this.voiceStatus = voiceStatus;
    this.speakStatus = speakStatus;
    this.profile = await voiceApi.profile().catch(() => this.profile);
    this.render();
  }

  private async startVoice() {
    const preset = this.selectedPreset();
    if (!preset) {
      toast('No presets are available yet', 'error');
      return;
    }
    const settings = this.settings().voice;
    const route = this.voiceOutput();
    try {
      this.voiceStatus = await voiceApi.start(
        {
          input: settings.input,
          input_channel: null,
          output: route.output,
          virtual_mic: route.virtualMic,
          monitor: settings.monitor ? (settings.monitorDevice ?? '') : null,
          preset,
          front_end: frontEnd(this.defaults, settings),
          output_gain_db: settings.outputGainDb,
          monitor_gain_db: settings.monitorGainDb,
          ceiling_db: settings.ceilingDb,
          block_frames: 240,
          sample_rate: 48_000,
        },
        (event) => this.onVoiceEvent(event),
      );
      toast('Voice changer is live', 'ok');
      this.render();
    } catch (e) {
      toast(errText(e), 'error', 7000);
    }
  }

  private onVoiceEvent(event: { event: string; [key: string]: unknown }) {
    if (event.event === 'device_lost') {
      toast(`Device lost: ${String(event.message ?? '')}`, 'error', 6000);
    }
    if (event.event === 'recovered') {
      toast('Voice devices recovered', 'ok');
    }
    if (event.event === 'error') {
      toast(String(event.message ?? 'Voice error'), 'error', 7000);
    }
  }

  private async stopVoice() {
    await voiceApi.stop().catch((e) => toast(errText(e), 'error'));
    this.voiceStatus = null;
    this.render();
  }

  private async previewPreset() {
    const preset = this.selectedPreset();
    if (!preset) return;
    try {
      const preview = await voiceApi.preview(preset, this.settings().voice.useRecordedPreview);
      await this.playFile(preview.path);
      toast(`Preview rendered in ${Math.round(preview.render_ms)} ms`, 'ok');
    } catch (e) {
      toast(errText(e), 'error', 7000);
    }
  }

  private async playFile(path: string) {
    this.preview.pause();
    this.preview.src = convertFileSrc(path);
    await this.preview.play();
  }

  private async startSpeak() {
    const settings = this.settings().voice;
    const route = this.voiceOutput();
    const effect = this.presets.find((preset) => preset.id === settings.speak.effectId) ?? null;
    this.speakBusy = true;
    this.render();
    try {
      this.entries = [];
      this.speakStatus = await speakApi.start(
        {
          input: settings.speak.listen ? settings.input : null,
          output: route.output,
          virtual_mic: route.virtualMic,
          monitor: settings.monitor ? (settings.monitorDevice ?? '') : null,
          voice: settings.speak.voice,
          speed: settings.speak.speed,
          effect,
          accent: settings.speak.accent,
          listen: settings.speak.listen,
          translate: settings.speak.translate,
          language: settings.speak.language,
          prompt: null,
          trim_fillers: settings.speak.trimFillers,
          mask_profanity: settings.speak.maskProfanity,
          half_duplex: settings.speak.halfDuplex,
          convert: settings.speak.convert,
          keep_pauses: settings.speak.keepPauses,
          hangover_ms: settings.speak.hangoverMs,
          auto_pause: settings.speak.autoPause,
          output_gain_db: settings.speak.outputGainDb,
          monitor_gain_db: settings.monitorGainDb,
        },
        (event) => this.onSpeakEvent(event),
      );
      toast('Speak for me started', 'ok');
    } catch (e) {
      toast(errText(e), 'error', 7000);
    } finally {
      this.speakBusy = false;
      this.render();
    }
  }

  private async stopSpeak() {
    await speakApi.stop().catch((e) => toast(errText(e), 'error'));
    this.speakStatus = null;
    this.render();
  }

  private onSpeakEvent(event: SpeakEvent) {
    switch (event.event) {
      case 'heard':
        this.entries.unshift({ key: `heard-${event.id}`, text: event.text, state: 'heard', detail: `${event.language.toUpperCase()} · ${Math.round(event.speech_secs * 100) / 100}s speech · ${Math.round(event.stt_ms)} ms STT` });
        break;
      case 'queued':
        this.entries.unshift({ key: `queued-${event.id}`, text: event.text, state: 'queued', detail: event.source === 'typed' ? 'Typed by you' : 'Queued from the microphone' });
        break;
      case 'speaking':
        this.entries = this.entries.map((entry) => (entry.key.endsWith(`-${event.id}`) ? { ...entry, state: 'speaking', detail: `Speaking now${event.latency_ms ? ` · ${Math.round(event.latency_ms)} ms` : ''}` } : entry));
        break;
      case 'done':
        this.entries = this.entries.map((entry) => (entry.key.endsWith(`-${event.id}`) ? { ...entry, state: event.skipped ? 'skipped' : 'done', detail: event.skipped ? 'Skipped' : 'Spoken' } : entry));
        break;
      case 'dropped':
        this.entries.unshift({ key: `drop-${Date.now()}`, text: 'Dropped utterance', state: 'info', detail: event.reason });
        break;
      case 'error':
        toast(event.message, 'error', 7000);
        break;
      case 'device_lost':
        toast(`${event.role} device lost: ${event.message}`, 'error', 7000);
        break;
      case 'device_restored':
        toast(`${event.role} device restored: ${event.name}`, 'ok');
        break;
      default:
        break;
    }
    this.entries = this.entries.slice(0, 24);
    this.render();
  }

  private async sayDraft() {
    const draft = this.settings().voice.speak.draft.trim();
    if (!draft) return;
    try {
      await speakApi.say(draft);
      this.settings().voice.speak.draft = '';
      this.save();
    } catch (e) {
      toast(errText(e), 'error');
    }
  }

  private async previewSpeakVoice(voiceId: string) {
    try {
      const settings = this.settings().voice;
      const effect = this.presets.find((preset) => preset.id === settings.speak.effectId) ?? null;
      const preview = await speakApi.preview(voiceId, settings.previewText || null, settings.speak.speed, effect, settings.speak.accent);
      await this.playFile(preview.path);
    } catch (e) {
      toast(errText(e), 'error', 7000);
    }
  }

  private async refreshSampleClone() {
    const [sample, profile, cloneInfo] = await Promise.all([voiceApi.sample().catch(() => null), voiceApi.profile().catch(() => null), cloneApi.info().catch(() => null)]);
    this.sample = sample;
    this.profile = profile;
    this.cloneInfo = cloneInfo;
    this.render();
  }

  private async importSample() {
    const picked = await openDialog({ multiple: false, directory: false, title: 'Import a voice sample', filters: [{ name: 'Audio', extensions: store.meta.supportedExtensions }] }).catch(() => null);
    if (!picked || Array.isArray(picked)) return;
    try {
      this.sample = await voiceApi.importSample(picked);
      toast('Voice sample imported', 'ok');
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error', 7000);
    }
  }

  private async recordSample() {
    const seconds = this.settings().voice.sampleSeconds;
    this.recordState = { active: true, elapsed: 0, peakDb: -60 };
    this.render();
    try {
      const bytes = await captureSample(seconds, (elapsed, peakDb) => {
        this.recordState = { active: true, elapsed, peakDb };
        this.render();
      });
      this.sample = await voiceApi.storeSample(bytes);
      toast('Voice sample recorded', 'ok');
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error', 7000);
    } finally {
      this.recordState = null;
      this.render();
    }
  }

  private async cloneFromSample() {
    this.cloneBusy = true;
    this.render();
    try {
      await cloneApi.make(null, this.settings().voice.cloneName || null);
      toast('Saved your cloned voice', 'ok');
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error', 7000);
    } finally {
      this.cloneBusy = false;
      this.render();
    }
  }

  private async cloneFromFile() {
    const picked = await openDialog({ multiple: false, directory: false, title: 'Choose an audio file to clone from', filters: [{ name: 'Audio', extensions: store.meta.supportedExtensions }] }).catch(() => null);
    if (!picked || Array.isArray(picked)) return;
    this.cloneBusy = true;
    this.render();
    try {
      await cloneApi.make(picked, this.settings().voice.cloneName || null);
      toast('Saved a cloned voice from file', 'ok');
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error', 7000);
    } finally {
      this.cloneBusy = false;
      this.render();
    }
  }

  private async renameClone(voice: CloneVoice) {
    const name = await askName('Rename cloned voice', voice.name);
    if (!name) return;
    try {
      await cloneApi.rename(voice.id, name);
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error');
    }
  }

  private async deleteClone(voice: CloneVoice) {
    try {
      await cloneApi.remove(voice.id);
      if (this.settings().voice.speak.voice === voice.id) {
        this.settings().voice.speak.voice = this.speakInfo?.voices[0]?.id ?? 'af_heart';
        this.save(false);
      }
      toast(`${voice.name} deleted`, 'ok');
      await this.refreshSampleClone();
    } catch (e) {
      toast(errText(e), 'error');
    }
  }

  private renderChanger() {
    const s = this.settings().voice;
    const preset = this.selectedPreset();
    const virtualOutput = this.devices?.virtual_mic.output_device ?? null;
    return h(
      'section',
      { class: 'grid' },
      h(
        'div',
        { class: 'grid two' },
        this.renderCard(
          'Routing & live output',
          'Pick the path your microphone takes when you go live.',
          this.field('Microphone', this.select(this.devices?.inputs ?? [], s.input, (value) => { s.input = value; this.save(); })),
          this.field(
            'Target',
            this.segment(
              [
                ['virtual', 'Virtual mic'],
                ['device', 'Selected device'],
              ],
              s.target,
              (value) => {
                s.target = value as typeof s.target;
                this.save();
              },
            ),
            virtualOutput ? `Virtual output: ${virtualOutput}` : 'No virtual cable detected yet.',
          ),
          this.field('Output device', this.select(this.devices?.outputs ?? [], s.output, (value) => { s.output = value; this.save(); })),
          this.field('Monitor', h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.monitor, onchange: (e: Event) => { s.monitor = (e.target as HTMLInputElement).checked; this.save(); } }), h('span', null, s.monitor ? 'On' : 'Off'))),
          s.monitor ? this.field('Monitor device', this.select(this.devices?.outputs ?? [], s.monitorDevice, (value) => { s.monitorDevice = value; this.save(); }, 'System default output')) : null,
          h('div', { class: 'row wrap' }, this.voiceStatus?.running ? h('button', { class: 'btn primary', type: 'button', onclick: () => void this.stopVoice() }, icon('stop', 13), 'Stop live') : h('button', { class: 'btn primary', type: 'button', disabled: !preset, onclick: () => void this.startVoice() }, icon('play', 13), 'Go live'), h('button', { class: 'btn', type: 'button', disabled: !preset, onclick: () => void this.previewPreset() }, icon('wave', 13), 'Preview preset')),
        ),
        this.renderCard(
          'Live status',
          'Meters, latency, and health from the running engine.',
          levelMeter('Microphone', Math.max(0, Math.min(1, ((this.voiceStatus?.input_level_db ?? -60) + 60) / 60)), 'ok'),
          levelMeter('Output', Math.max(0, Math.min(1, ((this.voiceStatus?.output_level_db ?? -60) + 60) / 60))),
          h(
            'div',
            { class: 'vk-status-grid' },
            statChip('Latency', `${Math.round(this.voiceStatus?.latency_ms ?? 0)} ms`),
            statChip('Pitch', this.voiceStatus?.pitch_hz ? `${this.voiceStatus.pitch_hz.toFixed(0)} Hz` : '—'),
            statChip('CPU', `${Math.round((this.voiceStatus?.cpu_load ?? 0) * 100)}%`),
            statChip('XRuns', String(this.voiceStatus?.xruns ?? 0), (this.voiceStatus?.xruns ?? 0) > 0 ? 'warn' : ''),
          ),
          h(
            'div',
            { class: 'pill-list' },
            this.voiceStatus?.gate_open ? h('span', { class: 'chip ok' }, icon('check', 11), 'Gate open') : h('span', { class: 'chip subtle' }, icon('minus', 11), 'Gate closed'),
            this.voiceStatus?.muted ? h('span', { class: 'chip danger' }, icon('mute', 11), 'Muted') : null,
            this.voiceStatus?.lost ? h('span', { class: 'chip warn' }, icon('alert', 11), this.voiceStatus.lost) : null,
          ),
          this.voiceStatus?.running ? h('button', { class: 'btn', type: 'button', onclick: () => void voiceApi.setMute(!this.voiceStatus?.muted) }, icon(this.voiceStatus?.muted ? 'volume' : 'mute', 12), this.voiceStatus?.muted ? 'Unmute output' : 'Mute output') : h('div', { class: 'vk-settings-note' }, 'Start the changer to populate live latency and meters.'),
        ),
      ),
      h(
        'div',
        { class: 'grid two' },
        this.renderCard(
          'Front-end clean-up',
          'These controls stay outside the preset itself and can update live.',
          rangeField({ label: 'Output gain', value: s.outputGainDb, min: -24, max: 24, step: 1, format: (value) => `${value > 0 ? '+' : ''}${Math.round(value)} dB`, onInput: (value) => { s.outputGainDb = value; if (this.voiceStatus?.running) void voiceApi.setOutputGain(value); }, onChange: (value) => { s.outputGainDb = value; this.save(); } }),
          rangeField({ label: 'Monitor gain', value: s.monitorGainDb, min: -60, max: 12, step: 1, format: (value) => `${value > 0 ? '+' : ''}${Math.round(value)} dB`, onInput: (value) => { s.monitorGainDb = value; if (this.voiceStatus?.running) void voiceApi.setMonitorGain(value); }, onChange: (value) => { s.monitorGainDb = value; this.save(); } }),
          rangeField({ label: 'Limiter ceiling', value: s.ceilingDb, min: -24, max: 0, step: 1, format: (value) => `${Math.round(value)} dB`, onInput: (value) => { s.ceilingDb = value; if (this.voiceStatus?.running) void voiceApi.setCeiling(value); }, onChange: (value) => { s.ceilingDb = value; this.save(); } }),
          rangeField({ label: 'High-pass filter', value: s.hpfHz, min: 20, max: 300, step: 5, format: (value) => `${Math.round(value)} Hz`, onInput: (value) => { s.hpfHz = value; if (this.voiceStatus?.running) void voiceApi.setFrontEnd(frontEnd(this.defaults, s)); }, onChange: (value) => { s.hpfHz = value; this.save(); } }),
          rangeField({ label: 'Gate threshold', value: s.gateDb, min: -60, max: -6, step: 1, format: (value) => `${Math.round(value)} dB`, onInput: (value) => { s.gateDb = value; if (this.voiceStatus?.running) void voiceApi.setFrontEnd(frontEnd(this.defaults, s)); }, onChange: (value) => { s.gateDb = value; this.save(); } }),
          h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.gateAuto, onchange: (e: Event) => { s.gateAuto = (e.target as HTMLInputElement).checked; if (this.voiceStatus?.running) void voiceApi.setFrontEnd(frontEnd(this.defaults, s)); this.save(); } }), h('span', null, 'Auto gate')),
          h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.deesser, onchange: (e: Event) => { s.deesser = (e.target as HTMLInputElement).checked; if (this.voiceStatus?.running) void voiceApi.setFrontEnd(frontEnd(this.defaults, s)); this.save(); } }), h('span', null, 'De-esser')),
        ),
        this.renderCard(
          'Presets',
          'Choose the live voice character. The cloned-voice and Speak tabs can reuse these as effects.',
          h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.useRecordedPreview, onchange: (e: Event) => { s.useRecordedPreview = (e.target as HTMLInputElement).checked; this.save(); } }), h('span', null, 'Preview with my recorded sample when available')),
          this.presets.length
            ? h(
                'div',
                { class: 'vk-preset-grid' },
                ...this.presets.map((item) =>
                  h(
                    'button',
                    {
                      class: `vk-preset-card ${preset?.id === item.id ? 'active' : ''}`,
                      type: 'button',
                      'data-key': item.id,
                      onclick: async () => {
                        s.presetId = item.id;
                        this.save();
                        if (this.voiceStatus?.running) await voiceApi.setPreset(item).catch((e) => toast(errText(e), 'error'));
                      },
                    },
                    h('div', { class: 'row' }, h('strong', null, item.name), item.builtin ? h('span', { class: 'chip subtle' }, item.category) : h('span', { class: 'chip ok' }, 'Saved')),
                    h('div', { class: 'small muted' }, item.description),
                    item.disguise ? h('span', { class: 'chip warn' }, 'Disguise') : null,
                  ),
                ),
              )
            : h('div', { class: 'empty' }, 'No presets loaded yet. Once the core crate is wired up, Voicekit will read them from nekotone-voice-core.'),
        ),
      ),
    );
  }

  render() {
    const settings = fillSettings(store.settings);
    store.settings = settings;
    const s = settings.voice;
    replace(
      this.el,
      h(
        'section',
        { class: 'hero' },
        h('h2', null, 'Voice studio'),
        h('p', null, 'Go live with a preset, build your own saved voice, or let Voicekit speak for you. The shell is simplified, but it keeps Nekotone’s no-framework DOM/morph pattern and upgrade-safe settings defaults.'),
        h(
          'div',
          { class: 'hero-actions' },
          h(
            'div',
            { class: 'vk-tabs' },
            ...([
              ['changer', 'Voice changer'],
              ['clone', 'My voices'],
              ['speak', 'Speak for me'],
            ] as const).map(([tab, label]) =>
              h(
                'button',
                {
                  class: `btn ${s.tab === tab ? 'primary' : ''}`,
                  type: 'button',
                  onclick: () => {
                    s.tab = tab;
                    this.save();
                  },
                },
                label,
              ),
            ),
          ),
          this.loading ? h('span', { class: 'chip warn' }, icon('refresh', 12), 'Loading…') : h('span', { class: 'chip' }, icon('check', 12), this.windowStateText()),
        ),
      ),
      s.tab === 'changer'
        ? this.renderChanger()
        : s.tab === 'clone'
          ? renderClonePanel({
              settings: s,
              sample: this.sample,
              profile: this.profile,
              cloneInfo: this.cloneInfo,
              recordState: this.recordState,
              busy: this.cloneBusy,
              onRecord: () => void this.recordSample(),
              onImportSample: () => void this.importSample(),
              onDeleteSample: () => void voiceApi.deleteSample().then(() => this.refreshSampleClone()).catch((e) => toast(errText(e), 'error')),
              onCalibrate: () => void voiceApi.calibrate().then(() => this.refreshSampleClone()).catch((e) => toast(errText(e), 'error')),
              onForgetCalibration: () => void voiceApi.forgetProfile().then(() => this.refreshSampleClone()).catch((e) => toast(errText(e), 'error')),
              onCloneFromSample: () => void this.cloneFromSample(),
              onCloneFromFile: () => void this.cloneFromFile(),
              onRenameClone: (voice) => void this.renameClone(voice),
              onDeleteClone: (voice) => void this.deleteClone(voice),
              onPreviewClone: (voice) => void this.previewSpeakVoice(voice.id),
              onOpenSettings: () => store.setView('settings'),
              onSetCloneName: (value) => {
                s.cloneName = value;
                this.save(false);
              },
              onSetSampleSeconds: (value) => {
                s.sampleSeconds = value;
                this.save(false);
                this.render();
              },
              onSetPreviewText: (value) => {
                s.previewText = value;
                this.save(false);
              },
              onSetConsent: (value) => {
                s.speak.cloneConsent = value;
                this.save();
              },
            })
          : renderSpeakPanel({
              settings: s.speak,
              info: this.speakInfo,
              cloneInfo: this.cloneInfo,
              status: this.speakStatus,
              entries: this.entries,
              presets: this.presets,
              busy: this.speakBusy,
              onStart: () => void this.startSpeak(),
              onStop: () => void this.stopSpeak(),
              onSay: () => void this.sayDraft(),
              onSkip: () => void speakApi.skip().catch((e) => toast(errText(e), 'error')),
              onClear: () => void speakApi.clear().catch((e) => toast(errText(e), 'error')),
              onOpenSettings: () => store.setView('settings'),
              onPreviewVoice: (voiceId) => void this.previewSpeakVoice(voiceId),
              onSelectVoice: (voiceId) => {
                s.speak.voice = voiceId;
                this.save();
              },
              onDraft: (value) => {
                s.speak.draft = value;
                this.save(false);
                this.render();
              },
              onEffect: (value) => {
                s.speak.effectId = value;
                this.save();
              },
              onAccent: (value) => {
                s.speak.accent = value;
                this.save();
              },
              onToggle: (key, value) => {
                s.speak[key] = value as never;
                this.save();
              },
              onLanguage: (value) => {
                s.speak.language = value;
                this.save();
              },
              onSpeed: (value) => {
                s.speak.speed = value;
                this.save(false);
                this.render();
              },
              onOutputGain: (value) => {
                s.speak.outputGainDb = value;
                this.save(false);
                if (this.speakStatus?.running) void speakApi.setGains(value, s.monitorGainDb);
              },
              onHangover: (value) => {
                s.speak.hangoverMs = value;
                this.save(false);
                this.render();
              },
            }),
    );
  }

  private renderCard(title: string, blurb: string, ...children: (HTMLElement | null)[]) {
    return h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, title)), h('p', { class: 'card-sub' }, blurb), ...children.filter(Boolean) as HTMLElement[]);
  }

  private field(label: string, body: HTMLElement, help = '') {
    return h('label', { class: 'field' }, h('span', { class: 'field-label' }, label), body, help ? h('span', { class: 'field-help' }, help) : null);
  }

  private select(devices: { name: string; default: boolean }[], current: string | null, set: (value: string | null) => void, none = 'System default') {
    return h(
      'select',
      { onchange: (e: Event) => set((e.target as HTMLSelectElement).value || null) },
      h('option', { value: '', selected: !current }, none),
      ...devices.map((device) => h('option', { value: device.name, selected: current === device.name }, `${device.name}${device.default ? ' (default)' : ''}`)),
    );
  }

  private segment(items: [string, string][], current: string, set: (value: string) => void) {
    return h(
      'div',
      { class: 'seg' },
      ...items.map(([value, label]) =>
        h(
          'button',
          {
            class: current === value ? 'active' : '',
            type: 'button',
            onclick: () => set(value),
          },
          label,
        ),
      ),
    );
  }
}

async function captureSample(seconds: number, onLevel: (elapsed: number, peakDb: number) => void) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This WebView does not expose microphone capture yet.');
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const gain = context.createGain();
  gain.gain.value = 0;
  const processor = context.createScriptProcessor(4096, source.channelCount, 1);
  const chunks: Float32Array[] = [];
  let written = 0;
  const targetFrames = Math.floor(seconds * context.sampleRate);
  return await new Promise<number[]>((resolve, reject) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      processor.disconnect();
      source.disconnect();
      gain.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      void context.close();
      const out = new Float32Array(written);
      let at = 0;
      chunks.forEach((chunk) => {
        out.set(chunk, at);
        at += chunk.length;
      });
      resolve(Array.from(encodeWav(out, context.sampleRate)));
    };
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const remaining = targetFrames - written;
      const take = Math.min(remaining, input.length);
      if (take > 0) {
        const part = new Float32Array(take);
        part.set(input.subarray(0, take));
        chunks.push(part);
        written += take;
        let peak = 0;
        for (let i = 0; i < take; i++) peak = Math.max(peak, Math.abs(part[i]));
        onLevel(written / context.sampleRate, 20 * Math.log10(Math.max(peak, 1e-4)));
      }
      if (written >= targetFrames) finish();
    };
    source.connect(processor);
    processor.connect(gain);
    gain.connect(context.destination);
    void context.resume().catch(reject);
    window.setTimeout(finish, seconds * 1000 + 600);
  });
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  }
  return new Uint8Array(buffer);
}
