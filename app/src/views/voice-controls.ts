import { h } from '../dom';

export function waveStrip(peaks: number[], label = 'Waveform') {
  const bars = peaks.length ? peaks : Array.from({ length: 48 }, () => 0);
  return h(
    'div',
    { class: 'vk-wave', role: 'img', 'aria-label': label },
    ...bars.map((peak) =>
      h('span', {
        class: 'vk-wave-bar',
        style: `height:${Math.max(8, Math.round(Math.max(0, Math.min(1, peak)) * 100))}%`,
      }),
    ),
  );
}

export function levelMeter(label: string, value: number, accent: 'accent' | 'ok' | 'warn' = 'accent') {
  return h(
    'div',
    { class: `vk-meter ${accent}` },
    h('div', { class: 'vk-meter-label' }, label),
    h('div', { class: 'vk-meter-track' }, h('span', { style: `width:${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%` })),
  );
}

export function statChip(label: string, value: string, tone: '' | 'ok' | 'warn' | 'danger' = '') {
  return h('span', { class: `chip ${tone}`.trim() }, h('strong', null, label), value);
}

export function rangeField(opts: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  help?: string;
  onInput?: (v: number) => void;
  onChange?: (v: number) => void;
}) {
  const readout = h('span', { class: 'mono small' }, opts.format(opts.value));
  const input = h('input', {
    type: 'range',
    min: String(opts.min),
    max: String(opts.max),
    step: String(opts.step),
    value: String(opts.value),
    oninput: () => {
      const next = parseFloat(input.value);
      readout.textContent = opts.format(next);
      opts.onInput?.(next);
    },
    onchange: () => opts.onChange?.(parseFloat(input.value)),
  });
  return h(
    'label',
    { class: 'field' },
    h('span', { class: 'field-label row' }, h('span', null, opts.label), readout),
    input,
    opts.help ? h('span', { class: 'field-help' }, opts.help) : null,
  );
}
