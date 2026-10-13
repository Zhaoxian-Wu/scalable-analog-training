// Native SVG plots shared by the interactive explorer and standalone vector assets.
// All values come from the frozen manuscript data; no raster images are embedded.
export const panelLabels = ['Precision sensitivity', 'Gradient cancellation', 'Accumulation error', 'Transfer fragmentation'];
const palette = { fp32: '#22814e', fp16: '#3568af', fp8_e4m3fn: '#bf443f', mp: '#22814e', rank_update: '#bf443f' };
const ink = '#26364b', muted = '#526277', grid = '#e1e7ef';
const left = 82, right = 610, top = 76, bottom = 320;
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
const number = value => value >= 1000 ? value.toLocaleString('en-US', { maximumFractionDigits: 0 })
  : value !== 0 && (value < .001 || value >= 100) ? value.toExponential(3) : value.toFixed(4);
const text = (x, y, value, attrs = '') => `<text x="${x}" y="${y}" ${attrs}>${escape(value)}</text>`;
const line = (x1, y1, x2, y2, attrs = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join(' ');
const formatLabel = format => format === 'fp8_e4m3fn' ? 'FP8' : format.toUpperCase();

export function gradientChart(data, index) {
  const rows = data[`panel_${String.fromCharCode(97 + index)}`];
  const labels = ['Weight', 'Activation', 'W-grad'];
  const isLog = index === 2;
  // Include the entire observed seed range: the manuscript raster cropped the
  // tallest cancellation whisker and final FP8 point with its fixed axis limits.
  const maximum = index === 0 ? 5 : index === 1 ? Math.ceil(Math.max(...rows.map(r => r.seed_max)) / 500) * 500
    : index === 2 ? 1000 : 1;
  const y = value => bottom - (isLog ? (Math.log10(value) + 6) / 9 : value / maximum) * (bottom - top);
  const ticks = index === 0 ? [0, 1, 2, 3, 4, 5] : index === 1 ? Array.from({ length: 6 }, (_, i) => maximum * i / 5)
    : index === 2 ? [1e-6, 1e-4, 1e-2, 1, 1e2, 1e3] : [0, .2, .4, .6, .8, 1];
  const yLabel = ['Validation loss', 'Mean cancellation factor κ', 'Relative gradient error (log scale)', 'Relative update error'][index];
  const xLabel = ['Quantized signal', 'Projection type', 'Cancellation strength log₁₀ κ', 'MP micro-batch size / direct rank update'][index];
  const id = `gradient-chart-${index}`;
  let content = '';
  const legend = index === 0 ? [['FP16', palette.fp16], ['FP8 (E4M3FN)', palette.fp8_e4m3fn]]
    : index === 2 ? [['FP32', palette.fp32], ['FP16', palette.fp16], ['FP8 (E4M3FN)', palette.fp8_e4m3fn]]
    : index === 3 ? [['MP digital residual', palette.mp], ['Direct rank update', palette.rank_update]] : [];
  legend.forEach(([label, color], i) => {
    const x = left + i * (index === 3 ? 225 : 155);
    content += `<rect x="${x}" y="48" width="13" height="10" rx="2" fill="${color}"/>${text(x + 20, 58, label)}`;
  });
  if (index === 1) content += text(left, 58, 'κ = Σ|gᵦ| / |Σgᵦ| · arithmetic mean');
  ticks.forEach(tick => {
    const label = isLog ? ({ '0.000001': '10⁻⁶', '0.0001': '10⁻⁴', '0.01': '10⁻²', '1': '1', '100': '10²', '1000': '10³' }[String(tick)])
      : tick.toLocaleString('en-US', { maximumFractionDigits: 1 });
    content += line(left, y(tick), right, y(tick), `stroke="${grid}"`);
    content += text(left - 12, y(tick) + 5, label, 'text-anchor="end"');
  });
  content += line(left, top, left, bottom, `stroke="${muted}"`) + line(left, bottom, right, bottom, `stroke="${muted}"`);
  content += text(22, (top + bottom) / 2, yLabel, `transform="rotate(-90 22 ${(top + bottom) / 2})" text-anchor="middle"`);
  content += text((left + right) / 2, 386, xLabel, 'text-anchor="middle"');
  const inspections = [];
  function mark(body, description) {
    const i = inspections.push(description) - 1;
    return `<g class="gradient-mark" tabindex="0" role="img" aria-label="${escape(description)}" data-gradient-point="${i}"><title>${escape(description)}</title>${body}</g>`;
  }
  function bar(row, x, width, value, color, label) {
    const description = `${label}: mean ${number(value)}; seed range ${number(row.seed_min)}–${number(row.seed_max)} (4 seeds).`;
    const cap = 5;
    const body = `<rect x="${x - width / 2}" y="${y(value)}" width="${width}" height="${bottom - y(value)}" rx="2" fill="${color}"/>`
      + `<path d="M${x},${y(row.seed_min)}V${y(row.seed_max)} M${x - cap},${y(row.seed_min)}H${x + cap} M${x - cap},${y(row.seed_max)}H${x + cap}" fill="none" stroke="${ink}" stroke-width="1.5"/>`;
    content += mark(body, description);
    content += text(x, y(row.seed_max) - 9, index === 1 ? Math.round(value).toLocaleString('en-US') : value.toFixed(3), 'text-anchor="middle" font-size="12"');
  }
  if (index === 0) {
    ['weight', 'activation', 'gradient'].forEach((target, i) => {
      const center = left + (i + .5) * (right - left) / 3;
      ['fp16', 'fp8_e4m3fn'].forEach((precision, j) => {
        const row = rows.find(r => r.target === target && r.precision === precision);
        bar(row, center + (j ? 28 : -28), 43, row.validation_loss, palette[precision], `${labels[i]}, ${formatLabel(precision)}`);
      });
      content += text(center, bottom + 25, labels[i], 'text-anchor="middle"');
    });
  } else if (index === 1 || index === 3) {
    const projectionColors = ['#3568af', '#448e9b', '#22814e', '#a58a22', '#bf6653', '#98508d'];
    rows.forEach((row, i) => {
      const center = left + (i + .5) * (right - left) / rows.length;
      const label = index === 1 ? ['Q', 'K', 'V', 'O', 'up', 'down'][i] : row.method === 'mp' ? String(row.micro_batch_size) : 'Rank';
      bar(row, center, 48, index === 1 ? row.mean_kappa : row.relative_error,
        index === 1 ? projectionColors[i] : palette[row.method], index === 1 ? row.kind : row.label);
      content += text(center, bottom + 25, label, 'text-anchor="middle"');
      if (index === 3 && row.method === 'mp') content += text(center, bottom + 45, `${128 / row.micro_batch_size}×`, 'text-anchor="middle" font-size="11"');
    });
    if (index === 3) content += text(left, 411, '× = transfer requests per fixed batch of 128', 'font-size="12"');
  } else {
    const x = value => left + (value - 1) / 3.5 * (right - left);
    [1, 2, 3, 4].forEach(tick => {
      content += text(x(tick), bottom + 25, tick, 'text-anchor="middle"');
    });
    ['fp32', 'fp16', 'fp8_e4m3fn'].forEach(format => {
      const series = rows.filter(r => r.format === format).sort((a, b) => a.log10_kappa_left - b.log10_kappa_left);
      const center = row => x((row.log10_kappa_left + row.log10_kappa_right) / 2);
      const upper = series.map(r => [center(r), y(r.seed_max)]);
      const lower = [...series].reverse().map(r => [center(r), y(r.seed_min)]);
      content += `<path d="${path([...upper, ...lower])}Z" fill="${palette[format]}" fill-opacity=".16"/>`;
      content += `<path d="${path(series.map(r => [center(r), y(r.relative_l2_error)]))}" fill="none" stroke="${palette[format]}" stroke-width="2.5"/>`;
      series.forEach(row => {
        const description = `${formatLabel(format)}, log₁₀ κ bin ${row.log10_kappa_left}–${row.log10_kappa_right}: relative error mean ${number(row.relative_l2_error)}; seed range ${number(row.seed_min)}–${number(row.seed_max)} (4 seeds).`;
        content += mark(`<circle cx="${center(row)}" cy="${y(row.relative_l2_error)}" r="5" fill="${palette[format]}" stroke="white" stroke-width="1.5"/>`, description);
      });
    });
  }
  content += text(right, 411, 'Mean · observed seed min–max', 'text-anchor="end" font-size="12"');
  const description = `${panelLabels[index]}. Four-seed means; whiskers or bands show observed seed min–max, not confidence intervals. ${yLabel} versus ${xLabel}.`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 400" role="group" aria-labelledby="${id}-title ${id}-desc" style="font-family:Inter,Arial,sans-serif;font-size:14px;fill:${muted}"><title id="${id}-title">${escape(panelLabels[index])}</title><desc id="${id}-desc">${escape(description)}</desc><style>.gradient-mark:focus{outline:none}.gradient-mark:focus-visible rect,.gradient-mark:focus-visible circle{stroke:#172a45;stroke-width:3}</style><rect width="640" height="400" fill="white"/><g transform="translate(0 -30)">${content}</g></svg>`;
  return { svg, inspections };
}

export function gradientOverview(data) {
  const panels = panelLabels.map((_, i) => `<g transform="translate(${(i % 2) * 640} ${Math.floor(i / 2) * 400})">${gradientChart(data, i).svg.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '')}</g>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 800" role="img" aria-label="Four gradient-fidelity experiments" style="font-family:Inter,Arial,sans-serif;font-size:14px;fill:${muted}">${panels}</svg>`;
}
