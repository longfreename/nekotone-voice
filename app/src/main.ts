import './styles.css';
import { api, errText } from './api';
import { TitleBar } from './chrome';
import { h, icon, installTooltips, morph, toast } from './dom';
import { store, viewLabel, type View } from './store';
import { VoiceView } from './views/voice';
import { SettingsView } from './views/settings';
import { HelpView } from './views/help';

const app = document.getElementById('app');
if (!app) throw new Error('No app host');

const titlebar = new TitleBar();
const content = h('main', { class: 'content' });
const nav = h('nav', { class: 'sidebar-nav' });
const shell = h(
  'div',
  { class: 'app-shell' },
  titlebar.el,
  h(
    'div',
    { class: 'app-body' },
    h(
      'aside',
      { class: 'sidebar' },
      h('div', { class: 'sidebar-head' }, h('span', { class: 'sidebar-badge' }, 'Focused product'), h('h1', null, 'Voicekit'), h('p', null, 'A tighter fork for voice changing, voice cloning, and text-to-speech.')),
      nav,
      h('div', { class: 'sidebar-foot muted small' }, 'Everything stays on your PC unless you point Voicekit at a compute server.'),
    ),
    content,
  ),
);
app.appendChild(shell);

const voiceView = new VoiceView();
const settingsView = new SettingsView();
const helpView = new HelpView();

const views: Record<View, { el: HTMLElement; summary(): string }> = {
  voice: voiceView,
  settings: settingsView,
  help: helpView,
};

function applyTheme() {
  document.documentElement.setAttribute('data-theme', store.settings.theme);
}

function renderNav() {
  const items: { id: View; label: string; icon: Parameters<typeof icon>[0] }[] = [
    { id: 'voice', label: 'Voice studio', icon: 'mic' },
    { id: 'settings', label: 'Settings', icon: 'gear' },
    { id: 'help', label: 'Help', icon: 'help' },
  ];
  morph(
    nav,
    ...items.map((item) =>
      h(
        'button',
        {
          class: `nav-item ${store.view === item.id ? 'active' : ''}`,
          type: 'button',
          'data-key': item.id,
          onclick: () => store.setView(item.id),
        },
        icon(item.icon, 16),
        h('span', null, item.label),
      ),
    ),
  );
}

function showView() {
  content.replaceChildren(views[store.view].el);
  titlebar.set(viewLabel(store.view), views[store.view].summary(), voiceView.windowStateText());
  renderNav();
}

async function boot() {
  installTooltips();
  const [meta, settings] = await Promise.all([api.meta(), api.getSettings()]);
  store.setMeta(meta);
  store.applySettings(settings);
  applyTheme();
  renderNav();
  showView();
  await Promise.all([voiceView.refreshAll(), settingsView.refresh()]);
  helpView.render();
  showView();
  if (JSON.stringify(store.settings) !== JSON.stringify(settings)) {
    void api.setSettings(store.settings).catch((e) => toast(errText(e), 'error'));
  }
  store.on((why) => {
    if (why === 'settings') {
      applyTheme();
      settingsView.render();
      helpView.render();
    }
    if (why === 'view' || why === 'settings' || why === 'meta') {
      showView();
    }
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'F1') {
      e.preventDefault();
      store.setView(store.view === 'help' ? 'voice' : 'help');
    }
    if (e.key === ',' && e.ctrlKey) {
      e.preventDefault();
      store.setView('settings');
    }
    if (e.key === 'Escape' && store.view === 'help') {
      e.preventDefault();
      store.setView('voice');
    }
  });
  window.addEventListener('error', (e) => {
    void api.log(`error: ${e.message} at ${e.filename}:${e.lineno}`);
  });
  window.addEventListener('unhandledrejection', (e) => {
    void api.log(`unhandled rejection: ${String(e.reason)}`);
  });
}

boot().catch((e) => {
  console.error(e);
  toast(errText(e), 'error', 8000);
});
