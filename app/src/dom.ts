// Tiny DOM helpers: no framework, just typed element construction.

type Attrs = Record<string, unknown> & {
  class?: string;
  style?: string;
  onclick?: (e: MouseEvent) => void;
  /** Tooltip text; a `(Key)` suffix is rendered as a key cap. */
  tip?: string;
};
export type Child = Node | string | number | null | undefined | false | Child[];

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: Attrs | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k.startsWith('on') && typeof v === 'function') {
        el.addEventListener(k.slice(2), v as EventListener);
      } else if (k === 'class') {
        el.className = String(v);
      } else if (k === 'style') {
        el.setAttribute('style', String(v));
      } else if (k === 'html') {
        el.innerHTML = String(v);
      } else if (k === 'tip') {
        el.setAttribute('data-tip', String(v));
        el.setAttribute('aria-label', String(v).replace(/\s*\(.*\)$/, ''));
      } else if (k === 'value' || k === 'checked' || k === 'selected' || k === 'disabled' || k === 'indeterminate') {
        (el as unknown as Record<string, unknown>)[k] = v;
      } else if (k in el && typeof v !== 'string') {
        (el as unknown as Record<string, unknown>)[k] = v;
      } else {
        el.setAttribute(k, v === true ? '' : String(v));
      }
    }
  }
  append(el, children);
  return el;
}

export function append(el: Node, children: Child[]) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
  }
}

export function clear(el: Element) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function replace(el: Element, ...children: Child[]) {
  clear(el);
  append(el, children);
}

/**
 * Like `replace`, but keeps the existing elements wherever the new content
 * has the same shape, patching attributes, text and form values in place.
 * Use it for UI that re-renders often (the transport): a button under the
 * mouse stays the same element, so a press that spans a re-render still
 * clicks. The old elements keep their event listeners, so only use it where
 * handlers do not capture state that changes between renders.
 */
export function morph(el: Element, ...children: Child[]) {
  const next = document.createElement('div');
  append(next, children);
  morphChildren(el, next);
}

function morphChildren(a: Node, b: Node) {
  const want = Array.from(b.childNodes);
  for (let i = 0; i < want.length; i++) {
    const cur = a.childNodes[i];
    if (!cur) a.appendChild(want[i]);
    else morphNode(cur, want[i]);
  }
  while (a.childNodes.length > want.length) a.removeChild(a.lastChild!);
}

function morphNode(a: Node, b: Node) {
  // Different element, or a different `data-key` (a control whose handler
  // differs): replace instead of patching, so no element keeps a handler
  // that belongs to something else.
  if (
    a.nodeType !== b.nodeType ||
    a.nodeName !== b.nodeName ||
    (a.nodeType === Node.ELEMENT_NODE && (a as Element).getAttribute('data-key') !== (b as Element).getAttribute('data-key'))
  ) {
    a.parentNode!.replaceChild(b, a);
    return;
  }
  if (a.nodeType !== Node.ELEMENT_NODE) {
    if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue;
    return;
  }
  const ea = a as Element;
  const eb = b as Element;
  for (const { name } of Array.from(ea.attributes)) if (!eb.hasAttribute(name)) ea.removeAttribute(name);
  for (const { name, value } of Array.from(eb.attributes)) if (ea.getAttribute(name) !== value) ea.setAttribute(name, value);
  if (ea instanceof HTMLInputElement && eb instanceof HTMLInputElement && document.activeElement !== ea) {
    if (ea.value !== eb.value) ea.value = eb.value;
    if (ea.checked !== eb.checked) ea.checked = eb.checked;
  }
  morphChildren(ea, eb);
}

/** An inline SVG icon from the set below. */
export function icon(name: IconName, size = 16): SVGSVGElement {
  const wrap = document.createElement('span');
  wrap.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" class="icon" aria-hidden="true">${ICONS[name]}</svg>`;
  return wrap.firstChild as SVGSVGElement;
}

export const ICONS = {
  play: '<path d="M7 4v16l13-8Z" fill="currentColor"/>',
  pause: '<rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor"/><rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>',
  prev: '<path d="M6 5v14"/><path d="M19 5 8 12l11 7Z" fill="currentColor"/>',
  next: '<path d="M18 5v14"/><path d="M5 5l11 7-11 7Z" fill="currentColor"/>',
  volume: '<path d="M4 9v6h4l5 4V5L8 9Z" fill="currentColor"/><path d="M16 9a4 4 0 0 1 0 6"/><path d="M18.5 6.5a8 8 0 0 1 0 11"/>',
  volume_low: '<path d="M4 9v6h4l5 4V5L8 9Z" fill="currentColor"/><path d="M16 9a4 4 0 0 1 0 6"/>',
  mute: '<path d="M4 9v6h4l5 4V5L8 9Z" fill="currentColor"/><path d="m16 9 5 6M21 9l-5 6"/>',
  repeat: '<path d="M17 2l4 4-4 4"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><path d="M7 22l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
  repeat_one: '<path d="M17 2l4 4-4 4"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><path d="M7 22l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/><path d="M11 10l2-1v6"/>',
  shuffle: '<path d="M16 3h5v5"/><path d="M4 20 21 3"/><path d="M21 16v5h-5"/><path d="m15 15 6 6"/><path d="m4 4 5 5"/>',
  loop: '<path d="M3 12h3l3-7 4 14 3-7h5"/>',
  eq: '<path d="M5 20V10M12 20V4M19 20v-8"/><circle cx="5" cy="14" r="1.6" fill="currentColor"/><circle cx="12" cy="9" r="1.6" fill="currentColor"/><circle cx="19" cy="16" r="1.6" fill="currentColor"/>',
  queue: '<path d="M4 6h12M4 12h12M4 18h8"/><path d="M19 12v6"/><path d="m16 15 3 3 3-3"/>',
  lyrics: '<path d="M4 6h16M4 10h10M4 14h16M4 18h8"/>',
  visual: '<path d="M3 12h2l2-5 3 10 3-8 2 5 2-2h4"/>',
  bookmark: '<path d="M6 3h12v18l-6-4-6 4Z"/>',
  bookmark_add: '<path d="M6 3h12v18l-6-4-6 4Z"/><path d="M12 7v6M9 10h6"/>',
  music: '<path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/>',
  library: '<path d="M4 4h4v16H4ZM10 4h4v16h-4Z"/><path d="m16 5 4 1-3 14-4-1Z"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  tools: '<path d="m15 12-8.4 8.4a2 2 0 0 1-2.8-2.8L12.2 9"/><path d="M17.6 14.4 21 11l-5-5-2 1-2-2-3 3 2 2-1 2 3.6 3.6"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
  folder_plus: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M12 11v5M9.5 13.5h5"/>',
  folder_open: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v1"/><path d="M3 19 6 11h16l-3 8Z"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  trash: '<path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
  download: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  save: '<path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z"/><path d="M7 3v5h8"/><rect x="7" y="13" width="10" height="8"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  pin: '<path d="M9 4h6l-1 6 3 3v2H7v-2l3-3Z"/><path d="M12 15v6"/>',
  mini: '<rect x="3" y="4" width="18" height="16" rx="2"/><rect x="11" y="12" width="8" height="6" rx="1" fill="currentColor"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2"/><path d="M9 2h6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>',
  alert: '<path d="M12 3 2 20h20Z"/><path d="M12 10v4M12 17v.01"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16v.01"/>',
  chevron_down: '<path d="m6 9 6 6 6-6"/>',
  chevron_right: '<path d="m9 6 6 6-6 6"/>',
  chevron_left: '<path d="m15 6-6 6 6 6"/>',
  up: '<path d="m6 15 6-6 6 6"/>',
  down: '<path d="m6 9 6 6 6-6"/>',
  drag: '<circle cx="9" cy="6" r="1.4" fill="currentColor"/><circle cx="15" cy="6" r="1.4" fill="currentColor"/><circle cx="9" cy="12" r="1.4" fill="currentColor"/><circle cx="15" cy="12" r="1.4" fill="currentColor"/><circle cx="9" cy="18" r="1.4" fill="currentColor"/><circle cx="15" cy="18" r="1.4" fill="currentColor"/>',
  external: '<path d="M14 4h6v6"/><path d="M20 4 10 14"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/>',
  piano: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 4v10M12 4v10M16 4v10"/><path d="M3 14h18"/>',
  wave: '<path d="M3 12h2l2-6 3 12 3-9 2 6 2-3h4"/>',
  export: '<path d="M12 15V3"/><path d="m7 8 5-5 5 5"/><path d="M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>',
  star: '<path d="m12 3 2.7 5.8 6.3.7-4.7 4.3 1.3 6.2L12 17l-5.6 3 1.3-6.2L3 9.5l6.3-.7Z"/>',
  star_fill: '<path d="m12 3 2.7 5.8 6.3.7-4.7 4.3 1.3 6.2L12 17l-5.6 3 1.3-6.2L3 9.5l6.3-.7Z" fill="currentColor"/>',
  tag: '<path d="M3 12V4h8l10 10-8 8Z"/><circle cx="7.5" cy="8.5" r="1.3" fill="currentColor"/>',
  filter: '<path d="M3 5h18l-7 8v6l-4-2v-4Z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10Z"/>',
  paw: '<ellipse cx="12" cy="15.5" rx="4.2" ry="3.4" fill="currentColor"/><circle cx="6.5" cy="11" r="1.7" fill="currentColor"/><circle cx="9.7" cy="7.2" r="1.8" fill="currentColor"/><circle cx="14.3" cy="7.2" r="1.8" fill="currentColor"/><circle cx="17.5" cy="11" r="1.7" fill="currentColor"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.6 2.6 0 0 1 5 1c0 1.8-2.5 2.2-2.5 4"/><path d="M12 17.5v.01"/>',
  speed: '<path d="M4 15a8 8 0 1 1 16 0"/><path d="m12 15 4-6"/><circle cx="12" cy="15" r="1.5" fill="currentColor"/>',
  fx: '<path d="M10 4H8a2 2 0 0 0-2 2v12M3 10h6"/><path d="m13 10 7 8M20 10l-7 8"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7Z"/>',
  keyboard: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M6 14h.01M18 14h.01M9 14h6"/>',
  device: '<path d="M4 9v6h4l5 4V5L8 9Z"/><path d="M17 8a6 6 0 0 1 0 8"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  sparkles: '<path d="M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8Z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8Z"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  text: '<path d="M4 7V4h16v3"/><path d="M12 4v16"/><path d="M8 20h8"/>',
  more: '<circle cx="5" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="19" cy="12" r="1.6" fill="currentColor"/>',
  maximize: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  restore: '<rect x="4" y="8" width="12" height="12" rx="2"/><path d="M8 8V6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2"/>',
  fullscreen: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  karaoke: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M3 3l18 18"/>',
  transpose: '<path d="M9 18V6l10-2v12"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="16.5" cy="16" r="2.5"/><path d="M3 9l2-2 2 2M5 7v6"/>',
  stems: '<path d="M4 7h16M4 12h10M4 17h13"/><circle cx="19" cy="12" r="2" fill="currentColor"/>',
  split: '<path d="M12 3v7"/><path d="M12 10 5 17v4M12 10l7 7v4"/>',
  speaker: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16Z"/><path d="m13.5 6.5 4 4"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  similar: '<circle cx="8" cy="12" r="5"/><circle cx="16" cy="12" r="5"/>',
  arrow_up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
  arrow_down: '<path d="M12 5v14M6 13l6 6 6-6"/>',
  playlist: '<path d="M4 6h12M4 11h12M4 16h7"/><path d="M16 14v6l5-3Z" fill="currentColor"/>',
  gpu: '<rect x="3" y="7" width="18" height="10" rx="2"/><circle cx="9" cy="12" r="2.2"/><circle cx="15.5" cy="12" r="2.2"/><path d="M6 17v3M18 17v3"/>',
  cat: '<path d="M4 20c0-6 3-9 8-9s8 3 8 9Z"/><path d="M6 12 4 4l5 4M18 12l2-8-5 4"/><circle cx="9.5" cy="15" r="1" fill="currentColor"/><circle cx="14.5" cy="15" r="1" fill="currentColor"/>',
};
export type IconName = keyof typeof ICONS;

export function debounce<A extends unknown[]>(fn: (...a: A) => void, ms: number) {
  let t: number | undefined;
  return (...a: A) => {
    if (t !== undefined) clearTimeout(t);
    t = window.setTimeout(() => {
      t = undefined;
      fn(...a);
    }, ms);
  };
}

export function humanSize(n: number): string {
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let f = n;
  while (f >= 1024 && i < u.length - 1) {
    f /= 1024;
    i++;
  }
  return i === 0 ? `${n} B` : `${f.toFixed(f < 10 ? 1 : 0)} ${u[i]}`;
}

export function basename(p: string): string {
  return p.split(/[\\/]/).filter(Boolean).pop() ?? p;
}

export function stem(p: string): string {
  const b = basename(p);
  const i = b.lastIndexOf('.');
  return i > 0 ? b.slice(0, i) : b;
}

export function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  return i > 0 ? p.slice(0, i) : p;
}

export function extname(p: string): string {
  const b = basename(p);
  const i = b.lastIndexOf('.');
  return i > 0 ? b.slice(i + 1).toLowerCase() : '';
}

export function sleep(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}

/** `m:ss` or `h:mm:ss`; negative or NaN becomes `–:––`. */
export function fmtTime(secs: number | null | undefined, withHours = false): string {
  if (secs === null || secs === undefined || !isFinite(secs) || secs < 0) return withHours ? '–:––:––' : '–:––';
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0 || withHours) return `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
  return `${m}:${String(r).padStart(2, '0')}`;
}

export function fmtTimeMs(secs: number): string {
  const frac = Math.floor((secs % 1) * 100);
  return `${fmtTime(secs)}.${String(frac).padStart(2, '0')}`;
}

export function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

export function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

/** A key cap, e.g. `Ctrl+M` → `<kbd>Ctrl</kbd>+<kbd>M</kbd>`. */
export function keycap(keys: string): HTMLElement {
  const span = h('span', { class: 'keys' });
  keys.split(' / ').forEach((alt, i) => {
    if (i) span.appendChild(document.createTextNode(' / '));
    alt.split('+').forEach((k, j) => {
      if (j) span.appendChild(document.createTextNode('+'));
      span.appendChild(h('kbd', null, prettyKey(k)));
    });
  });
  return span;
}

export function prettyKey(k: string): string {
  const map: Record<string, string> = { ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', Space: 'Space', Escape: 'Esc', ',': ',' };
  return map[k] ?? k;
}

/** Simple toasts in the corner of the panel. */
let toastHost: HTMLElement | null = null;
export function toast(msg: string, kind: 'info' | 'error' | 'ok' = 'info', ms = 3800) {
  if (!toastHost || !toastHost.isConnected) {
    toastHost = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.appendChild(toastHost);
  }
  const t = h('div', { class: `toast ${kind}` }, icon(kind === 'error' ? 'error' : kind === 'ok' ? 'check' : 'info', 15), h('span', null, msg));
  toastHost.appendChild(t);
  requestAnimationFrame(() => t.classList.add('in'));
  const close = () => {
    t.classList.remove('in');
    setTimeout(() => t.remove(), 250);
  };
  t.addEventListener('click', close);
  setTimeout(close, ms);
}

/** A modal question. Resolves with the id of the chosen button. */
export function ask(title: string, body: string | HTMLElement, choices: { id: string; label: string; primary?: boolean; danger?: boolean }[]): Promise<string> {
  return new Promise((resolve) => {
    const back = h('div', { class: 'modal-back', role: 'dialog', 'aria-modal': 'true' });
    const done = (id: string) => {
      back.remove();
      document.removeEventListener('keydown', onKey, true);
      resolve(id);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        done('cancel');
      }
    };
    document.addEventListener('keydown', onKey, true);
    const box = h(
      'div',
      { class: 'modal' },
      h('h3', null, title),
      typeof body === 'string' ? h('p', null, body) : body,
      h(
        'div',
        { class: 'modal-actions' },
        ...choices.map((c) => h('button', { class: `btn ${c.primary ? 'primary' : ''} ${c.danger ? 'danger' : ''}`, onclick: () => done(c.id) }, c.label)),
      ),
    );
    back.appendChild(box);
    back.addEventListener('mousedown', (e) => {
      if (e.target === back) done('cancel');
    });
    document.body.appendChild(back);
    (box.querySelector('button.primary, button') as HTMLElement | null)?.focus();
  });
}

/** Tooltips: one floating element, positioned near whatever has data-tip. */
export function installTooltips() {
  const tipEl = h('div', { class: 'tooltip', role: 'tooltip' });
  document.body.appendChild(tipEl);
  let current: HTMLElement | null = null;
  let timer = 0;
  const hide = () => {
    clearTimeout(timer);
    tipEl.classList.remove('show');
    current = null;
  };
  const show = (el: HTMLElement) => {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    clear(tipEl);
    const m = /^(.*?)\s*\(([^()]+)\)$/.exec(text);
    if (m) append(tipEl, [m[1], ' ', keycap(m[2])]);
    else tipEl.textContent = text;
    tipEl.classList.add('show');
    const r = el.getBoundingClientRect();
    const tr = tipEl.getBoundingClientRect();
    let x = r.left + r.width / 2 - tr.width / 2;
    let y = r.top - tr.height - 8;
    if (y < 4) y = r.bottom + 8;
    x = clamp(x, 4, window.innerWidth - tr.width - 4);
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  };
  document.addEventListener('mouseover', (e) => {
    const el = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-tip]') ?? null;
    if (el === current) return;
    hide();
    if (!el) return;
    current = el;
    timer = window.setTimeout(() => current === el && show(el), 450);
  });
  document.addEventListener('mouseout', (e) => {
    const el = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-tip]') ?? null;
    if (el && el === current && !el.contains(e.relatedTarget as Node | null)) hide();
  });
  document.addEventListener('mousedown', hide, true);
  document.addEventListener('focusin', (e) => {
    const el = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-tip]') ?? null;
    if (el && el.matches(':focus-visible')) {
      current = el;
      show(el);
    }
  });
  document.addEventListener('focusout', hide);
  window.addEventListener('blur', hide);
}

export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** A small modal asking for a name (presets, playlists, speakers). */
export function askName(title: string, initial: string): Promise<string | null> {
  return new Promise((resolve) => {
    const inp = h('input', { type: 'text', value: initial, class: 'text' });
    const back = h('div', { class: 'modal-back', role: 'dialog' });
    const done = (v: string | null) => {
      back.remove();
      resolve(v);
    };
    back.appendChild(
      h(
        'div',
        { class: 'modal' },
        h('h3', null, title),
        inp,
        h('div', { class: 'modal-actions' }, h('button', { class: 'btn', onclick: () => done(null) }, 'Cancel'), h('button', { class: 'btn primary', onclick: () => done(inp.value.trim() || null) }, 'OK')),
      ),
    );
    inp.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') done(inp.value.trim() || null);
      if (e.key === 'Escape') done(null);
    });
    document.body.appendChild(back);
    inp.focus();
    inp.select();
  });
}
