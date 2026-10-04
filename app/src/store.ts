import type { Meta } from './api';
import { fillSettings, type AppSettings } from './views/voice-model';

export type View = 'voice' | 'settings' | 'help';
export type ChangeKind = 'settings' | 'view' | 'meta';
type Listener = (why: ChangeKind) => void;

const EMPTY_META: Meta = {
  version: '',
  dataDir: '',
  modelsDir: '',
  logPath: '',
  supportedExtensions: [],
};

class Store {
  meta: Meta = EMPTY_META;
  settings: AppSettings = fillSettings({});
  view: View = 'voice';
  private listeners: Listener[] = [];

  on(listener: Listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  emit(why: ChangeKind) {
    this.listeners.forEach((listener) => listener(why));
  }

  setView(view: View) {
    if (this.view === view) return;
    this.view = view;
    this.emit('view');
  }

  setMeta(meta: Meta) {
    this.meta = meta;
    this.emit('meta');
  }

  applySettings(raw: unknown) {
    this.settings = fillSettings(raw);
    this.emit('settings');
  }
}

export const store = new Store();

export function viewLabel(view: View) {
  switch (view) {
    case 'voice':
      return 'Voice studio';
    case 'settings':
      return 'Settings';
    case 'help':
      return 'Help';
  }
}
