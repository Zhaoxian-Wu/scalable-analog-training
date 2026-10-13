"use strict";

(() => {
  const studies = window.trainingScaleStudies;
  const svg = document.getElementById("scale-plot");
  if (!svg || !studies?.length) return;
  const frame = document.getElementById("scale-chart");
  const tooltip = document.getElementById("scale-tooltip");
  const chooser = document.getElementById("scale-study-select");
  const neighborsBox = document.getElementById("scale-neighbors");
  const namespace = "http://www.w3.org/2000/svg";
  const plot = {left: 82, right: 690, top: 32, bottom: 456};
  const logMap = (value, min, max, start, end) => start + (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min)) * (end - start);
  const x = value => logMap(value, 5, 30_000_000_000, plot.left, plot.right);
  const y = value => logMap(value, 5, 1_000_000_000, plot.bottom, plot.top);
  function element(tag, attrs, text, parent = svg) {
    const node = document.createElementNS(namespace, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  const line = (x1, y1, x2, y2, className, extra = {}) => element("line", {x1, y1, x2, y2, class: className, ...extra});
  const label = (text, px, py, extra = {}) => element("text", {x: px, y: py, class: "scale-axis-label", ...extra}, text);

  const defs = element("defs", {});
  const arrow = element("marker", {id: "scale-arrow", viewBox: "0 0 10 10", refX: 5, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse"}, undefined, defs);
  element("path", {d: "M 1 1 L 9 5 L 1 9", fill: "none", stroke: "#596b82", "stroke-width": 1.5}, undefined, arrow);
  // Use the manuscript's limits; even the smallest hardware examples remain visible.
  for (let decade = 0; decade <= 10; decade++) {
    for (let factor = 1; factor <= 9; factor++) {
      const value = factor * 10 ** decade;
      const className = factor === 1 ? "scale-grid" : "scale-minor-grid";
      if (value >= 5 && value <= 30_000_000_000) line(x(value), plot.top, x(value), plot.bottom, className);
      if (value >= 5 && value <= 1_000_000_000) line(plot.left, y(value), plot.right, y(value), className);
    }
  }
  line(plot.left, plot.top, plot.left, plot.bottom, "scale-axis");
  line(plot.left, plot.bottom, plot.right, plot.bottom, "scale-axis");
  for (const [value, text] of [[1e3, "1k"], [1e6, "1M"], [1e9, "1B"]]) {
    line(x(value), plot.bottom, x(value), plot.bottom + 5, "scale-axis");
    label(text, x(value), plot.bottom + 24, {"text-anchor": "middle"});
    line(plot.left - 5, y(value), plot.left, y(value), "scale-axis");
    label(text, plot.left - 12, y(value) + 5, {"text-anchor": "end"});
  }
  label("Dataset tokens", (plot.left + plot.right) / 2, 515, {"text-anchor": "middle", class: "scale-axis-title"});
  label("Analog-mapped weights", 21, 244, {"text-anchor": "middle", transform: "rotate(-90 21 244)", class: "scale-axis-title"});
  line(plot.left, y(34_000_000), plot.right, y(34_000_000), "scale-prior-guide");
  line(plot.left, y(123_532_032), plot.right, y(123_532_032), "scale-current-guide");
  const arrowAttrs = {"marker-start": "url(#scale-arrow)", "marker-end": "url(#scale-arrow)"};
  line(x(240_000), y(250_000_000), x(2_467_430_400), y(250_000_000), "scale-gap", arrowAttrs);
  label("10,300× dataset-token scale", (x(240_000) + x(2_467_430_400)) / 2, y(250_000_000) - 10, {"text-anchor": "middle", class: "scale-gap-label"});
  line(x(2_000_000), y(34_000_000), x(2_000_000), y(123_532_032), "scale-gap", arrowAttrs);
  label("3.6×", x(2_000_000) + 12, (y(34_000_000) + y(123_532_032)) / 2 + 4, {class: "scale-gap-label"});

  const groups = new Map();
  for (const study of studies) {
    const key = `${study.weights}/${study.tokens}`;
    if (!groups.has(key)) groups.set(key, {x: x(study.tokens), y: y(study.weights), studies: []});
    groups.get(key).studies.push(study);
  }
  const points = [...groups.values()];
  points.forEach((point, index) => {
    const current = point.studies.some(study => study.id === "this-work");
    const names = point.studies.map(study => study.label).join("; ");
    point.node = element("g", {class: `scale-point${current ? " scale-this-work" : ""}`, tabindex: 0, role: "button", "aria-label": `${names}. ${point.studies[0].weights.toLocaleString("en-US")} analog-mapped weights, ${point.studies[0].tokens.toLocaleString("en-US")} dataset tokens. Show study details.`, "data-point": index});
    element("circle", {cx: point.x, cy: point.y, r: 12, class: "scale-point-halo"}, undefined, point.node);
    if (current) element("circle", {cx: point.x, cy: point.y, r: 10, class: "scale-point-ring"}, undefined, point.node);
    element("circle", {cx: point.x, cy: point.y, r: current ? 6.5 : 5.5, class: "scale-point-dot"}, undefined, point.node);
    point.node.addEventListener("focus", () => show(point, false));
    point.node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        show(point, true);
      }
    });
  });
  // Leaders and labels preserve the figure's four representative algorithms.
  for (const [method, reference, tokens, weights, labelY] of [
    ["AGAD", "Rasch et al., 2024", 200_000, 4_300_000, 2_000_000],
    ["Residual Learning", "Wu et al., 2025", 200_000, 235_000, 120_000],
    ["Tiki-Taka", "Gokmen & Haensch, 2020", 240_000, 235_000, 7_200],
    ["Analog MP", "Nandakumar et al., 2020", 240_000, 199_000, 430]
  ]) {
    line(x(tokens) + 6, y(weights), x(7_000_000) - 9, y(labelY), "scale-leader");
    label(method, x(7_000_000), y(labelY) + 4, {class: "scale-method-label"});
    label(reference, x(7_000_000), y(labelY) + 21, {class: "scale-reference-label"});
  }
  label("This work", x(2_467_430_400) - 17, y(123_532_032) + 23, {"text-anchor": "end", class: "scale-method-label"});
  svg.style.display = "";

  let activePoint = null;
  let pinned = false;
  let hideTimer;
  let dismissedPoint = null;
  function hide() {
    clearTimeout(hideTimer);
    dismissedPoint = activePoint;
    activePoint = null;
    pinned = false;
    tooltip.hidden = true;
    points.forEach(point => {
      point.node.classList.remove("is-inspected");
      point.node.removeAttribute("aria-describedby");
    });
  }
  function scheduleHide() {
    clearTimeout(hideTimer);
    if (!pinned) hideTimer = setTimeout(hide, 250);
  }
  function compact(value) {
    return new Intl.NumberFormat("en-US", {notation: "compact", maximumFractionDigits: 2}).format(value);
  }
  function fill(study) {
    document.getElementById("scale-study-label").textContent = study.method && study.method !== "This work" ? `${study.method} · ${study.label}` : study.label;
    document.getElementById("scale-study-title").textContent = study.title;
    const facts = document.getElementById("scale-study-facts");
    facts.replaceChildren();
    for (const [name, value] of [
      ["Analog weights", `${compact(study.weights)} (${study.weights.toLocaleString("en-US")})`],
      ["Dataset tokens", `${compact(study.tokens)} (${study.tokens.toLocaleString("en-US")})`],
      ["Model", study.model], ["Dataset", study.dataset],
      ["Evidence", study.evidence === "H" ? "Physical hardware (H)" : "Simulation (S)"],
      ["Training", study.regime === "T" ? "Transfer learning (T)" : "From scratch (F)"]
    ]) {
      const term = document.createElement("dt"); term.textContent = name;
      const detail = document.createElement("dd"); detail.textContent = value;
      facts.append(term, detail);
    }
    const note = document.getElementById("scale-study-note");
    const notes = [];
    if (study.partialMappingExcluded) notes.push("This paper also reports a larger experiment with partial analog mapping, excluded from this comparison.");
    note.textContent = notes.join(" "); note.hidden = !notes.length;
    chooser.value = study.id;
    points.forEach(point => point.node.classList.toggle("is-inspected", point.studies.includes(study)));
  }
  function position(point) {
    const box = frame.getBoundingClientRect();
    const anchor = new DOMPoint(point.x, point.y).matrixTransform(svg.getScreenCTM());
    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    let left = anchor.x - box.left + 18;
    if (left + width > box.width - 8) left = anchor.x - box.left - width - 18;
    const top = Math.max(8, Math.min(box.height - height - 8, anchor.y - box.top - height / 2));
    tooltip.style.left = `${Math.max(8, Math.min(box.width - width - 8, left))}px`;
    tooltip.style.top = `${top}px`;
  }
  function show(point, pin) {
    clearTimeout(hideTimer);
    dismissedPoint = null;
    pinned = pin;
    activePoint = point;
    const nearby = points.filter(other => Math.hypot(other.x - point.x, other.y - point.y) <= 18).flatMap(other => other.studies);
    // Put the directly inspected point first, keeping all nearby entries reachable.
    nearby.sort((a, b) => Number(point.studies.includes(b)) - Number(point.studies.includes(a)) || a.weights - b.weights || a.label.localeCompare(b.label));
    chooser.replaceChildren();
    for (const study of nearby) {
      const option = document.createElement("option");
      option.value = study.id;
      option.textContent = `${study.label} — ${study.model}, ${study.evidence}/${study.regime}`;
      chooser.append(option);
    }
    neighborsBox.hidden = nearby.length <= 1;
    fill(point.studies[0]);
    tooltip.hidden = false;
    point.node.setAttribute("aria-describedby", "scale-tooltip");
    position(point);
  }
  function nearest(event) {
    const cursor = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    let nearestPoint = null;
    let distance = 15;
    for (const point of points) {
      const candidate = Math.hypot(point.x - cursor.x, point.y - cursor.y);
      if (candidate < distance) { distance = candidate; nearestPoint = point; }
    }
    return nearestPoint;
  }
  svg.addEventListener("pointermove", event => {
    if (event.pointerType !== "mouse" || pinned) return;
    const point = nearest(event);
    if (!point) { dismissedPoint = null; scheduleHide(); return; }
    clearTimeout(hideTimer);
    if (point !== activePoint && point !== dismissedPoint) show(point, false);
  });
  svg.addEventListener("click", event => {
    const point = nearest(event);
    if (point) show(point, true);
    else hide();
  });
  svg.addEventListener("pointerleave", scheduleHide);
  svg.addEventListener("focusout", event => {
    if (!frame.contains(event.relatedTarget)) scheduleHide();
  });
  tooltip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
  tooltip.addEventListener("pointerleave", scheduleHide);
  tooltip.addEventListener("focusin", () => { clearTimeout(hideTimer); pinned = true; });
  chooser.addEventListener("change", () => {
    pinned = true;
    fill(studies.find(study => study.id === chooser.value));
  });
  document.getElementById("scale-tooltip-close").addEventListener("click", () => {
    const focusPoint = activePoint;
    if (focusPoint) focusPoint.node.focus();
    hide();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !tooltip.hidden) {
      const focusPoint = activePoint;
      if (tooltip.contains(document.activeElement) && focusPoint) focusPoint.node.focus();
      hide();
      event.preventDefault();
    }
  });
  document.addEventListener("click", event => { if (!frame.contains(event.target)) hide(); });
  document.addEventListener("focusin", event => { if (!frame.contains(event.target)) hide(); });
  window.addEventListener("resize", () => { if (activePoint) position(activePoint); });
  frame.querySelector(".scale-plot-scroll").addEventListener("scroll", () => { if (activePoint) position(activePoint); });
})();
