import { api, errText, voiceApi, type ModelStatus, type Progress, type ServeStatus, type VoiceDevices } from '../api';
import { debounce, h, humanSize, icon, replace, toast } from '../dom';
import { store } from '../store';

export class SettingsView {
  readonly el = h('div', { class: 'view settings-view' });
  private devices: VoiceDevices | null = null;
  private models: ModelStatus[] = [];
  private serverStatus: ServeStatus | null = null;
  private downloads = new Map<string, Progress>();
  private saveLater = debounce(() => {
    void api.setSettings(store.settings).catch((e) => toast(errText(e), 'error'));
  }, 180);

  summary() {
    return 'Devices, models, compute-server, and app defaults.';
  }

  async refresh() {
    const [devices, models, server] = await Promise.all([
      voiceApi.devices().catch(() => null),
      api.voiceModels().catch(() => []),
      api.computeServeStatus().catch(() => null),
    ]);
    this.devices = devices;
    this.models = models;
    this.serverStatus = server;
    this.render();
  }

  render() {
    const s = store.settings;
    const whisperModels = this.models.filter((m) => m.info.id.startsWith('whisper'));
    replace(
      this.el,
      h(
        'section',
        { class: 'page-head' },
        h('div', null, h('h2', null, 'Settings'), h('p', null, 'Keep defaults small: pick the devices, choose where models run, and manage the speech models Voicekit needs.')),
        h('div', { class: 'pill-list' }, h('span', { class: 'chip' }, icon('db', 12), `${this.models.filter((m) => m.installed).length}/${this.models.length || 0} models ready`), h('span', { class: 'chip' }, icon('globe', 12), s.computeServer ? 'Compute server set' : 'Local only')),
      ),
      h(
        'section',
        { class: 'grid two' },
        this.card(
          'Audio defaults',
          'What Voice Studio opens with on the next run.',
          this.field(
            'Theme',
            this.segment(
              [
                ['dark', 'Dark'],
                ['light', 'Light'],
              ],
              s.theme,
              (value) => {
                s.theme = value as 'dark' | 'light';
                this.save();
              },
            ),
          ),
          this.field(
            'Microphone',
            this.select(
              this.devices?.inputs ?? [],
              s.voice.input,
              (value) => {
                s.voice.input = value;
                this.save();
              },
              'System default microphone',
            ),
          ),
          this.field(
            'Output target',
            this.segment(
              [
                ['virtual', 'Virtual mic'],
                ['device', 'Selected device'],
              ],
              s.voice.target,
              (value) => {
                s.voice.target = value as 'virtual' | 'device';
                this.save();
                this.render();
              },
            ),
            this.devices?.virtual_mic.output_device ? `Virtual route: ${this.devices.virtual_mic.output_device}` : 'Choose Selected device when no virtual microphone is available.',
          ),
          this.field(
            'Output device',
            this.select(this.devices?.outputs ?? [], s.voice.output, (value) => {
              s.voice.output = value;
              this.save();
            }),
            s.voice.target === 'virtual' ? 'Used when you switch the target to Selected device.' : '',
          ),
          this.field(
            'Monitor',
            h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.voice.monitor, onchange: (e: Event) => { s.voice.monitor = (e.target as HTMLInputElement).checked; this.save(); this.render(); } }), h('span', null, s.voice.monitor ? 'On' : 'Off')),
          ),
          s.voice.monitor
            ? this.field(
                'Monitor device',
                this.select(this.devices?.outputs ?? [], s.voice.monitorDevice, (value) => {
                  s.voice.monitorDevice = value;
                  this.save();
                }, 'System default output'),
              )
            : null,
          this.field(
            'Whisper model',
            h(
              'select',
              {
                onchange: (e: Event) => {
                  s.whisperModel = (e.target as HTMLSelectElement).value;
                  this.save();
                },
              },
              ...whisperModels.map((m) => h('option', { value: m.info.id, selected: s.whisperModel === m.info.id }, `${m.info.id}${m.installed ? '' : ' · download needed'}`)),
            ),
            'Used when “Speak for me” listens to the microphone.',
          ),
          this.field(
            'Keep learning my voice',
            h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.voiceLearn, onchange: (e: Event) => { s.voiceLearn = (e.target as HTMLInputElement).checked; this.save(); } }), h('span', null, s.voiceLearn ? 'On' : 'Off')),
            'Improves adaptive presets over time on this PC only.',
          ),
        ),
        this.card(
          'Compute & sharing',
          'Optional network features for the cloned-voice backend.',
          this.field(
            'Model engine',
            this.segment(
              [
                ['auto', 'Auto'],
                ['directml', 'DirectML'],
                ['cpu', 'CPU only'],
              ],
              s.accelerator,
              (value) => {
                s.accelerator = value as typeof s.accelerator;
                this.save();
              },
            ),
            'Applies to models loaded from now on.',
          ),
          this.field(
            'Compute server',
            h(
              'div',
              { class: 'row' },
              h('input', {
                class: 'text',
                type: 'text',
                value: s.computeServer,
                placeholder: 'host or host:port',
                onchange: (e: Event) => {
                  s.computeServer = (e.target as HTMLInputElement).value.trim();
                  this.save();
                },
              }),
              h('input', {
                class: 'text tiny',
                type: 'password',
                value: s.computeToken,
                placeholder: 'token',
                onchange: (e: Event) => {
                  s.computeToken = (e.target as HTMLInputElement).value.trim();
                  this.save();
                },
              }),
              h('button', { class: 'btn', type: 'button', onclick: () => void this.testServer() }, 'Test'),
            ),
            this.serverStatus?.running && s.serveEnabled
              ? `Sharing this PC on ${this.serverStatus.address ?? `port ${this.serverStatus.port}`}.`
              : s.computeServer
                ? 'Voicekit tries the remote clone engine first, then this PC if needed.'
                : 'Leave empty to keep everything on this PC.',
          ),
          s.computeServer
            ? this.field(
                'Share the work',
                this.segment(
                  [
                    ['auto', 'Auto'],
                    ['both', 'Both'],
                    ['server-first', 'Server first'],
                  ],
                  s.computeMode,
                  (value) => {
                    s.computeMode = value as typeof s.computeMode;
                    this.save();
                  },
                ),
              )
            : null,
          this.field(
            'Share this PC',
            h(
              'div',
              { class: 'row' },
              h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: s.serveEnabled, onchange: (e: Event) => void this.setServe((e.target as HTMLInputElement).checked) }), h('span', null, s.serveEnabled ? 'On' : 'Off')),
              h('input', {
                class: 'text tiny',
                type: 'number',
                min: '1024',
                max: '65535',
                value: String(s.servePort),
                onchange: (e: Event) => {
                  s.servePort = clamp(parseInt((e.target as HTMLInputElement).value, 10) || 8199, 1024, 65535);
                  this.save();
                },
              }),
              h('input', {
                class: 'text tiny',
                type: 'password',
                value: s.serveToken,
                placeholder: 'token',
                onchange: (e: Event) => {
                  s.serveToken = (e.target as HTMLInputElement).value.trim();
                  this.save();
                },
              }),
            ),
            this.serverStatus?.error ? `Could not share: ${this.serverStatus.error}` : 'Lets another Voicekit or Nekotone install use this PC’s cloned-voice engine while the app stays open.',
          ),
        ),
      ),
      this.card(
        'Speech models',
        'Only the voice, TTS, and speech-to-text models used by this fork.',
        this.models.length
          ? h(
              'div',
              { class: 'grid two' },
              ...this.models.map((model) => {
                const downloading = this.downloads.get(model.info.id);
                const size = model.info.files.reduce((sum, file) => sum + file.size_bytes, 0);
                return h(
                  'article',
                  { class: 'card compact', 'data-key': model.info.id },
                  h('div', { class: 'card-head' }, h('h3', null, model.info.id), model.installed ? h('span', { class: 'chip ok' }, icon('check', 11), 'Ready') : h('span', { class: 'chip' }, icon('download', 11), 'Not installed')),
                  h('p', { class: 'card-sub' }, model.info.purpose),
                  h('div', { class: 'small muted' }, `${humanSize(size)} · ${model.info.note} · ${model.info.license}`),
                  downloading
                    ? h('div', { class: 'progress' }, h('div', { class: 'progress-bar' }, h('span', { style: `width:${((downloading.fraction ?? 0) * 100).toFixed(1)}%` })), h('span', { class: 'small muted' }, downloading.message))
                    : null,
                  h(
                    'div',
                    { class: 'row' },
                    model.installed
                      ? h('button', { class: 'btn', type: 'button', onclick: () => void this.removeModel(model.info.id) }, icon('trash', 12), 'Remove')
                      : h('button', { class: 'btn primary', type: 'button', onclick: () => void this.downloadModel(model.info.id) }, icon('download', 12), `Download (${humanSize(size)})`),
                    model.info.id.startsWith('whisper') && model.installed && store.settings.whisperModel !== model.info.id
                      ? h('button', { class: 'btn', type: 'button', onclick: () => { store.settings.whisperModel = model.info.id; this.save(); this.render(); } }, 'Use for speech')
                      : null,
                  ),
                );
              }),
            )
          : h('div', { class: 'empty' }, 'No model catalogue yet. Once the Rust core is fully extracted, this list will populate from it.'),
      ),
      this.card(
        'Useful paths',
        'Open the folders this shell now owns.',
        this.field('Data folder', h('button', { class: 'btn', type: 'button', onclick: () => void api.openPath(store.meta.dataDir) }, icon('folder_open', 12), store.meta.dataDir || 'Pending')),
        this.field('Models folder', h('button', { class: 'btn', type: 'button', onclick: () => void api.openPath(store.meta.modelsDir) }, icon('db', 12), store.meta.modelsDir || 'Pending')),
        this.field('Log file', h('button', { class: 'btn', type: 'button', onclick: () => void api.openPath(store.meta.logPath) }, icon('file', 12), store.meta.logPath || 'Pending')),
      ),
    );
  }

  private save() {
    store.emit('settings');
    this.saveLater();
  }

  private async testServer() {
    const s = store.settings;
    if (!s.computeServer) {
      toast('Enter a host name first', 'info');
      return;
    }
    try {
      const info = await api.computeServerTest(s.computeServer, s.computeToken || null);
      toast(`${info.host}: ${info.backend.toUpperCase()} · ${info.features.join(', ')}`, 'ok', 5000);
    } catch (e) {
      toast(errText(e), 'error', 6000);
    }
  }

  private async setServe(on: boolean) {
    store.settings.serveEnabled = on;
    this.save();
    try {
      this.serverStatus = await api.computeServe(on);
      this.render();
    } catch (e) {
      toast(errText(e), 'error', 6000);
    }
  }

  private async downloadModel(id: string) {
    this.downloads.set(id, { message: 'Preparing download…', fraction: 0 });
    this.render();
    try {
      await api.voiceModelDownload(id, (p) => {
        this.downloads.set(id, p);
        this.render();
      });
      toast(`${id} is ready`, 'ok');
    } catch (e) {
      toast(errText(e), 'error', 6000);
    } finally {
      this.downloads.delete(id);
      await this.refresh();
    }
  }

  private async removeModel(id: string) {
    try {
      await api.voiceModelRemove(id);
      toast(`${id} removed`, 'ok');
      await this.refresh();
    } catch (e) {
      toast(errText(e), 'error', 6000);
    }
  }

  private card(title: string, blurb: string, ...children: (HTMLElement | null)[]) {
    return h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, title)), h('p', { class: 'card-sub' }, blurb), ...children.filter(Boolean) as HTMLElement[]);
  }

  private field(label: string, body: HTMLElement, help = '') {
    return h('label', { class: 'field inline' }, h('span', { class: 'field-label' }, label), h('span', { class: 'field' }, body, help ? h('span', { class: 'field-help' }, help) : null));
  }

  private select(
    devices: { name: string; default: boolean }[],
    current: string | null,
    set: (value: string | null) => void,
    none = 'Use the system default',
  ) {
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

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
