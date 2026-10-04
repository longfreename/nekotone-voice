import { h, icon } from '../dom';
import type { CloneInfo, CloneVoice, VoiceProfile, VoiceSample } from '../api';
import type { VoiceSettings } from './voice-model';
import { levelMeter, waveStrip } from './voice-controls';

export interface ClonePanelModel {
  settings: VoiceSettings;
  sample: VoiceSample | null;
  profile: { saved: VoiceProfile | null; live: VoiceProfile | null } | null;
  cloneInfo: CloneInfo | null;
  recordState: { active: boolean; elapsed: number; peakDb: number } | null;
  busy: boolean;
  onRecord: () => void;
  onImportSample: () => void;
  onDeleteSample: () => void;
  onCalibrate: () => void;
  onForgetCalibration: () => void;
  onCloneFromSample: () => void;
  onCloneFromFile: () => void;
  onRenameClone: (voice: CloneVoice) => void;
  onDeleteClone: (voice: CloneVoice) => void;
  onPreviewClone: (voice: CloneVoice) => void;
  onOpenSettings: () => void;
  onSetCloneName: (value: string) => void;
  onSetSampleSeconds: (value: number) => void;
  onSetPreviewText: (value: string) => void;
  onSetConsent: (value: boolean) => void;
}

export function renderClonePanel(model: ClonePanelModel) {
  const sample = model.sample;
  const cloneInfo = model.cloneInfo;
  const ready = !!cloneInfo?.installed;
  return h(
    'section',
    { class: 'grid two' },
    h(
      'article',
      { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', null, 'Sample & calibration'), sample ? h('span', { class: 'chip ok' }, icon('check', 11), `${sample.seconds.toFixed(1)} s`) : h('span', { class: 'chip' }, icon('wave', 11), 'No sample yet')),
      h('p', { class: 'card-sub' }, 'Record or import a clean reference. Voicekit normalises it, previews from it, and can calibrate adaptive presets from it.'),
      model.recordState?.active ? h('div', { class: 'progress' }, h('div', { class: 'progress-bar' }, h('span', { style: `width:${Math.min(100, (model.recordState.elapsed / Math.max(1, model.settings.sampleSeconds)) * 100)}%` })), levelMeter('Input peak', Math.max(0, Math.min(1, (model.recordState.peakDb + 60) / 60)), 'warn'), h('span', { class: 'small muted' }, `${model.recordState.elapsed.toFixed(1)} / ${model.settings.sampleSeconds} s`)) : null,
      sample ? waveStrip(sample.peaks, 'Reference waveform') : h('div', { class: 'empty' }, 'No reference sample yet.'),
      h(
        'div',
        { class: 'row wrap' },
        h('button', { class: 'btn primary', type: 'button', disabled: !!model.recordState?.active, onclick: model.onRecord }, icon('mic', 13), `Record ${model.settings.sampleSeconds}s`),
        h('button', { class: 'btn', type: 'button', disabled: !!model.recordState?.active, onclick: model.onImportSample }, icon('folder_open', 13), 'Import file…'),
        sample ? h('button', { class: 'btn', type: 'button', onclick: model.onDeleteSample }, icon('trash', 13), 'Delete sample') : null,
      ),
      h(
        'label',
        { class: 'field' },
        h('span', { class: 'field-label row' }, h('span', null, 'Record length'), h('span', { class: 'mono small' }, `${model.settings.sampleSeconds} s`)),
        h('input', {
          type: 'range',
          min: '3',
          max: '15',
          step: '1',
          value: String(model.settings.sampleSeconds),
          oninput: (e: Event) => model.onSetSampleSeconds(parseInt((e.target as HTMLInputElement).value, 10)),
          onchange: (e: Event) => model.onSetSampleSeconds(parseInt((e.target as HTMLInputElement).value, 10)),
        }),
        h('span', { class: 'field-help' }, 'Shorter is faster, longer gives Chatterbox more to learn from.'),
      ),
      h(
        'div',
        { class: 'pill-list' },
        model.profile?.saved ? h('span', { class: 'chip ok' }, icon('speaker', 12), `Saved voice ${model.profile.saved.f0Hz.toFixed(0)} Hz`) : h('span', { class: 'chip' }, icon('speaker', 12), 'No saved calibration'),
        model.profile?.live ? h('span', { class: 'chip' }, icon('mic', 12), `Live ${model.profile.live.f0Hz.toFixed(0)} Hz`) : null,
      ),
      h(
        'div',
        { class: 'row wrap' },
        h('button', { class: 'btn', type: 'button', disabled: !sample, onclick: model.onCalibrate }, 'Calibrate from sample'),
        h('button', { class: 'btn', type: 'button', disabled: !model.profile?.saved, onclick: model.onForgetCalibration }, 'Forget calibration'),
      ),
    ),
    h(
      'article',
      { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', null, 'My cloned voices'), ready ? h('span', { class: 'chip ok' }, icon('db', 11), 'Chatterbox ready') : h('span', { class: 'chip warn' }, icon('alert', 11), 'Model needed')),
      h('p', { class: 'card-sub' }, 'Make one or more saved voice prints from your sample or from a file. These voices appear in “Speak for me”.'),
      !ready
        ? h(
            'div',
            { class: 'empty' },
            h('p', null, 'The cloned-voice model is not downloaded yet.'),
            h('button', { class: 'btn', type: 'button', onclick: model.onOpenSettings }, icon('gear', 12), 'Open model settings'),
          )
        : null,
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Voice name'), h('input', { class: 'text', type: 'text', value: model.settings.cloneName, oninput: (e: Event) => model.onSetCloneName((e.target as HTMLInputElement).value) })),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Preview line'), h('textarea', { rows: 3, value: model.settings.previewText, oninput: (e: Event) => model.onSetPreviewText((e.target as HTMLTextAreaElement).value) })),
      h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: model.settings.speak.cloneConsent, onchange: (e: Event) => model.onSetConsent((e.target as HTMLInputElement).checked) }), h('span', null, 'This is my own voice, or I have permission to clone it.')),
      h(
        'div',
        { class: 'row wrap' },
        h('button', { class: 'btn primary', type: 'button', disabled: !ready || !sample || !model.settings.speak.cloneConsent || model.busy, onclick: model.onCloneFromSample }, icon('sparkles', 13), 'Make from sample'),
        h('button', { class: 'btn', type: 'button', disabled: !ready || !model.settings.speak.cloneConsent || model.busy, onclick: model.onCloneFromFile }, icon('folder_open', 13), 'Make from file…'),
      ),
      cloneInfo?.voices.length
        ? h(
            'div',
            { class: 'vk-clone-list' },
            ...cloneInfo.voices.map((voice) =>
              h(
                'div',
                { class: 'vk-clone-row', 'data-key': voice.id },
                h('div', { class: 'grow' }, h('div', { class: 'row' }, h('strong', null, voice.name), h('span', { class: 'chip subtle' }, `${voice.reference_secs.toFixed(1)} s`)), h('div', { class: 'small muted path' }, voice.source)),
                h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Preview ${voice.name}`, onclick: () => model.onPreviewClone(voice) }, icon('play', 13)),
                h('button', { class: 'icon-btn', type: 'button', 'aria-label': `Rename ${voice.name}`, onclick: () => model.onRenameClone(voice) }, icon('edit', 13)),
                h('button', { class: 'icon-btn danger', type: 'button', 'aria-label': `Delete ${voice.name}`, onclick: () => model.onDeleteClone(voice) }, icon('trash', 13)),
              ),
            ),
          )
        : h('div', { class: 'empty' }, 'No saved voices yet.'),
    ),
  );
}
