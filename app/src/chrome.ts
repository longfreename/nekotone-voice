import { getCurrentWindow } from '@tauri-apps/api/window';
import { h, icon, morph } from './dom';

const win = getCurrentWindow();

export class TitleBar {
  readonly el: HTMLElement;
  private metaEl = h('div', { class: 'vk-title-meta' });
  private stateEl = h('span', { class: 'chip subtle' }, 'Idle');
  private maxBtn = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Maximise' }, icon('maximize', 15));

  constructor() {
    const drag = h('div', { class: 'vk-drag' });
    drag.addEventListener('mousedown', (e) => {
      if (e.button === 0) void win.startDragging();
    });
    drag.addEventListener('dblclick', () => void this.toggleMaximize());
    this.maxBtn.addEventListener('click', () => void this.toggleMaximize());
    this.el = h(
      'header',
      { class: 'vk-titlebar' },
      h('div', { class: 'vk-brand' }, h('span', { class: 'vk-brand-mark' }), h('div', { class: 'vk-brand-text' }, h('strong', null, 'Voicekit'), this.metaEl)),
      drag,
      this.stateEl,
      h(
        'div',
        { class: 'vk-window-actions' },
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Minimise', onclick: () => void win.minimize() }, icon('minus', 15)),
        this.maxBtn,
        h('button', { class: 'icon-btn danger', type: 'button', 'aria-label': 'Close', onclick: () => void win.close() }, icon('x', 15)),
      ),
    );
    void this.refreshMax();
  }

  set(title: string, subtitle: string, state: string) {
    morph(this.metaEl, h('div', { class: 'vk-title' }, title), h('div', { class: 'vk-subtitle' }, subtitle));
    this.stateEl.textContent = state;
  }

  private async toggleMaximize() {
    if (await win.isMaximized()) await win.unmaximize();
    else await win.maximize();
    await this.refreshMax();
  }

  private async refreshMax() {
    const isMax = await win.isMaximized().catch(() => false);
    this.maxBtn.replaceChildren(icon(isMax ? 'restore' : 'maximize', 15));
    this.maxBtn.setAttribute('aria-label', isMax ? 'Restore' : 'Maximise');
  }
}
