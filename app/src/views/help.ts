import { api } from '../api';
import { h, icon, replace } from '../dom';
import { store } from '../store';

export class HelpView {
  readonly el = h('div', { class: 'view help-view' });

  summary() {
    return 'Setup notes, privacy, and quick links.';
  }

  render() {
    replace(
      this.el,
      h(
        'section',
        { class: 'hero' },
        h('h2', null, 'Voicekit is the compact voice fork'),
        h('p', null, 'This shell keeps only the voice changer, voice cloning, text-to-speech, a smaller Settings page, and a lightweight in-app help surface.'),
        h(
          'div',
          { class: 'hero-actions' },
          h('button', { class: 'btn primary', onclick: () => void api.openPath('https://github.com/longfreename/nekotone-voice') }, icon('external', 14), 'Project repository'),
          store.meta.dataDir ? h('button', { class: 'btn', onclick: () => void api.openPath(store.meta.dataDir) }, icon('folder_open', 14), 'Open data folder') : null,
          store.meta.modelsDir ? h('button', { class: 'btn', onclick: () => void api.openPath(store.meta.modelsDir) }, icon('db', 14), 'Open models folder') : null,
        ),
      ),
      h(
        'section',
        { class: 'grid two' },
        h(
          'article',
          { class: 'card' },
          h('h3', null, 'First run'),
          h('ul', null, h('li', null, 'Choose a microphone and a target in Voice Studio.'), h('li', null, 'Download Kokoro for stock TTS voices; add Whisper if you want live listening.'), h('li', null, 'Record or import a clean sample before making your own cloned voice.')),
        ),
        h(
          'article',
          { class: 'card' },
          h('h3', null, 'Privacy'),
          h('p', { class: 'card-sub' }, 'Models run locally by default. Compute-server features stay off until you point Voicekit at another machine or enable “Share this PC”.'),
          h('div', { class: 'pill-list' }, h('span', { class: 'chip ok' }, 'Local by default'), h('span', { class: 'chip' }, 'Optional server mode'), h('span', { class: 'chip' }, 'Settings persist across upgrades')),
        ),
      ),
      h(
        'section',
        { class: 'grid two' },
        h(
          'article',
          { class: 'card compact' },
          h('div', { class: 'card-head' }, h('h3', null, 'What is still intentionally simple?')),
          h('p', { class: 'card-sub' }, 'This fork uses placeholder branding assets, a lighter static Help view, and a trimmed navigation shell while the dedicated Voicekit docs and art catch up.'),
        ),
        h(
          'article',
          { class: 'card compact' },
          h('div', { class: 'card-head' }, h('h3', null, 'Useful paths')),
          h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'Version'), h('span', { class: 'mono small' }, store.meta.version || 'dev')),
          h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'Log file'), h('span', { class: 'mono small path' }, store.meta.logPath || 'Created on first log line')),
        ),
      ),
    );
  }
}
