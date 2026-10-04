import { h, icon } from '../dom';
import type { CloneInfo, CloneVoice, SpeakInfo, SpeakStatus, VoicePreset } from '../api';
import type { SpeakSettings } from './voice-model';
import { SPEAK_LANGUAGES } from './voice-model';
import { levelMeter, rangeField } from './voice-controls';

export interface SpeakEntry {
  key: string;
  text: string;
  state: 'heard' | 'queued' | 'speaking' | 'done' | 'skipped' | 'info';
  detail: string;
}

export interface SpeakPanelModel {
  settings: SpeakSettings;
  info: SpeakInfo | null;
  cloneInfo: CloneInfo | null;
  status: SpeakStatus | null;
  entries: SpeakEntry[];
  presets: VoicePreset[];
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
  onSay: () => void;
  onSkip: () => void;
  onClear: () => void;
  onOpenSettings: () => void;
  onPreviewVoice: (voiceId: string) => void;
  onSelectVoice: (voiceId: string) => void;
  onDraft: (value: string) => void;
  onEffect: (value: string | null) => void;
  onAccent: (value: string) => void;
  onToggle: (key: keyof Pick<SpeakSettings, 'listen' | 'translate' | 'convert' | 'keepPauses' | 'trimFillers' | 'maskProfanity' | 'halfDuplex' | 'autoPause' | 'showAll'>, value: boolean) => void;
  onLanguage: (value: string | null) => void;
  onSpeed: (value: number) => void;
  onOutputGain: (value: number) => void;
  onHangover: (value: number) => void;
}

export function renderSpeakPanel(model: SpeakPanelModel) {
  const voices = [
    ...(model.cloneInfo?.voices ?? []),
    ...((model.info?.voices ?? []).filter((voice) => model.settings.showAll || voice.featured)),
  ];
  const selected = model.settings.voice;
  const hasTts = !!model.info?.installed;
  const hasWhisper = !!model.info?.whisper_installed;
  const usingCloneVoice = selected.startsWith('clone:');
  const canRun = voices.length > 0 && (!model.settings.listen || hasWhisper) && (usingCloneVoice ? !!model.cloneInfo?.installed : hasTts) && (!model.settings.convert || !!model.cloneInfo?.installed);

  return h(
    'section',
    { class: 'grid two' },
    h(
      'article',
      { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', null, 'Speak for me'), model.status?.running ? h('span', { class: 'chip ok' }, icon('check', 11), 'Running') : h('span', { class: 'chip' }, icon('text', 11), 'Ready when models are')),
      h('p', { class: 'card-sub' }, 'Type to speak, or let Voicekit listen to your microphone and re-say each utterance in the selected voice.'),
      h(
        'div',
        { class: 'row wrap' },
        model.status?.running
          ? h('button', { class: 'btn primary', type: 'button', onclick: model.onStop }, icon('stop', 13), 'Stop')
          : h('button', { class: 'btn primary', type: 'button', disabled: !canRun || model.busy, onclick: model.onStart }, icon('play', 13), 'Start'),
        h('button', { class: 'btn', type: 'button', disabled: !model.status?.running, onclick: model.onSkip }, 'Skip current'),
        h('button', { class: 'btn', type: 'button', disabled: !model.status?.running, onclick: model.onClear }, 'Clear queue'),
        !canRun ? h('button', { class: 'btn', type: 'button', onclick: model.onOpenSettings }, icon('gear', 12), 'Open model settings') : null,
      ),
      h(
        'div',
        { class: 'grid' },
        levelMeter('Input', Math.max(0, Math.min(1, ((model.status?.input_level_db ?? -60) + 60) / 60)), 'ok'),
        levelMeter('Output', Math.max(0, Math.min(1, ((model.status?.output_level_db ?? -60) + 60) / 60))),
      ),
      h(
        'div',
        { class: 'pill-list' },
        h('span', { class: 'chip' }, icon('clock', 12), `Latency ${Math.round(model.status?.mean_latency_ms ?? 0)} ms`),
        h('span', { class: 'chip' }, icon('list', 12), `${model.status?.queued ?? 0} queued`),
        h('span', { class: `chip ${model.status?.hearing ? 'ok' : ''}`.trim() }, icon('mic', 12), model.status?.hearing ? 'Hearing speech' : model.settings.listen ? 'Listening armed' : 'Type only'),
      ),
      h(
        'div',
        { class: 'row wrap' },
        h('textarea', { class: 'sp-draft', rows: 4, value: model.settings.draft, placeholder: 'Type what Voicekit should say…', oninput: (e: Event) => model.onDraft((e.target as HTMLTextAreaElement).value) }),
        h('button', { class: 'btn primary', type: 'button', disabled: !model.settings.draft.trim() || !model.status?.running, onclick: model.onSay }, icon('play', 13), 'Say it'),
      ),
      model.entries.length
        ? h(
            'div',
            { class: 'sp-log' },
            ...model.entries.map((entry) =>
              h(
                'div',
                { class: `sp-entry ${entry.state}`, 'data-key': entry.key },
                h('div', { class: 'row' }, h('strong', null, entry.text), h('span', { class: 'chip subtle' }, entry.state)),
                h('div', { class: 'small muted' }, entry.detail),
              ),
            ),
          )
        : h('div', { class: 'empty' }, 'No conversation history yet. Start the engine, then type or talk.'),
    ),
    h(
      'article',
      { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', null, 'Voices & options'), h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: model.settings.showAll, onchange: (e: Event) => model.onToggle('showAll', (e.target as HTMLInputElement).checked) }), h('span', null, 'Show every stock voice'))),
      voices.length
        ? h(
            'div',
            { class: 'sp-voice-grid' },
            ...voices.map((voice) => renderVoiceCard(voice, selected === voiceId(voice), () => model.onSelectVoice(voiceId(voice)), () => model.onPreviewVoice(voiceId(voice)))),
          )
        : h('div', { class: 'empty' }, 'No voices available yet. Download a speech model, or make your own voice in the Clone tab.'),
      rangeField({ label: 'Speed', value: model.settings.speed, min: 0.7, max: 1.35, step: 0.05, format: (v) => `${v.toFixed(2)}×`, onInput: model.onSpeed, onChange: model.onSpeed }),
      rangeField({ label: 'Output gain', value: model.settings.outputGainDb, min: -24, max: 24, step: 1, format: (v) => `${v > 0 ? '+' : ''}${Math.round(v)} dB`, onInput: model.onOutputGain, onChange: model.onOutputGain }),
      rangeField({ label: 'Sentence pause', value: model.settings.hangoverMs, min: 120, max: 900, step: 10, format: (v) => `${Math.round(v)} ms`, onInput: model.onHangover, onChange: model.onHangover, help: 'How long Voicekit waits before treating silence as the end of a sentence.' }),
      h(
        'label',
        { class: 'field' },
        h('span', { class: 'field-label' }, 'Effect'),
        h(
          'select',
          { onchange: (e: Event) => model.onEffect((e.target as HTMLSelectElement).value || null) },
          h('option', { value: '', selected: !model.settings.effectId }, 'Clean voice'),
          ...model.presets.map((preset) => h('option', { value: preset.id, selected: model.settings.effectId === preset.id }, preset.name)),
        ),
      ),
      h(
        'label',
        { class: 'field' },
        h('span', { class: 'field-label' }, 'Accent'),
        h(
          'select',
          { onchange: (e: Event) => model.onAccent((e.target as HTMLSelectElement).value) },
          ...(model.info?.accents ?? [{ id: 'voice', label: 'Voice default' }]).map((accent) => h('option', { value: accent.id, selected: model.settings.accent === accent.id }, accent.label)),
        ),
      ),
      h(
        'label',
        { class: 'field' },
        h('span', { class: 'field-label' }, 'I speak'),
        h(
          'select',
          { onchange: (e: Event) => model.onLanguage((e.target as HTMLSelectElement).value || null) },
          ...SPEAK_LANGUAGES.map(([id, label]) => h('option', { value: id === 'auto' ? '' : id, selected: (model.settings.language ?? '') === (id === 'auto' ? '' : id) }, label)),
        ),
      ),
      h(
        'div',
        { class: 'grid' },
        toggle('Listen to the microphone', model.settings.listen, (value) => model.onToggle('listen', value), hasWhisper ? 'Live speech-to-text is ready.' : 'Needs a Whisper model.'),
        toggle('Translate to English', model.settings.translate, (value) => model.onToggle('translate', value)),
        toggle('Change my voice instead of reading text', model.settings.convert, (value) => model.onToggle('convert', value)),
        toggle('Keep my own pauses', model.settings.keepPauses, (value) => model.onToggle('keepPauses', value)),
        toggle('Auto-learn the pause', model.settings.autoPause, (value) => model.onToggle('autoPause', value)),
        toggle('Trim fillers', model.settings.trimFillers, (value) => model.onToggle('trimFillers', value)),
        toggle('Mask profanity', model.settings.maskProfanity, (value) => model.onToggle('maskProfanity', value)),
        toggle('Half duplex', model.settings.halfDuplex, (value) => model.onToggle('halfDuplex', value)),
      ),
    ),
  );
}

function renderVoiceCard(voice: CloneVoice | SpeakInfo['voices'][number], selected: boolean, choose: () => void, preview: () => void) {
  const clone = 'source' in voice;
  return h(
    'div',
    {
      class: `sp-voice-card ${selected ? 'selected' : ''}`,
      role: 'button',
      tabindex: 0,
      onclick: choose,
      onkeydown: (e: KeyboardEvent) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          choose();
        }
      },
    },
    h('div', { class: 'row' }, h('strong', null, voice.name), clone ? h('span', { class: 'chip ok' }, 'Mine') : h('span', { class: 'chip subtle' }, voice.grade)),
    h('div', { class: 'small muted' }, clone ? voice.source : `${voice.character} · ${voice.accent}`),
    h('span', { class: 'grow' }),
    h('button', { class: 'btn', type: 'button', onclick: (e: Event) => { e.stopPropagation(); preview(); } }, icon('play', 12), 'Preview'),
  );
}

function voiceId(voice: CloneVoice | SpeakInfo['voices'][number]) {
  return 'source' in voice ? voice.id : voice.id;
}

function toggle(label: string, value: boolean, set: (value: boolean) => void, help = '') {
  return h(
    'label',
    { class: 'field' },
    h('span', { class: 'switch' }, h('input', { type: 'checkbox', checked: value, onchange: (e: Event) => set((e.target as HTMLInputElement).checked) }), h('span', null, label)),
    help ? h('span', { class: 'field-help' }, help) : null,
  );
}
