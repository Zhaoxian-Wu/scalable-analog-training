"use strict";

const steps = [
  {domain: "Analog matrix execution", equation: "y = Wx", latex: String.raw`y=Wx`, description: "Inputs pass through DACs into the conductance array. The crossbar computes the matrix–vector product, and ADCs return a digital output. Mapping and converter range alignment preserve signal fidelity."},
  {domain: "Analog matrix execution", equation: "δx = Wᵀδy", latex: String.raw`\delta_x=W^{\mathsf T}\delta_y`, description: "The transpose matrix pass propagates the error signal through the same array. Converter ranges also need to match the backward signals; their distributions can differ from those in the forward pass."},
  {domain: "Digital gradient handling", equation: "g = Σᵦ δyᵦ xᵦᵀ", latex: String.raw`G=\sum_{b=1}^B\delta_{b}x_b^{\mathsf T}`, description: "Digital circuitry forms and accumulates weight gradients across the batch. The logical-space optimizer computes the update, keeping its moments and preconditioners digital. The update is added to a residual buffer."},
  {domain: "Digital control · analog programming", equation: "Accumulate → threshold → pulse", latex: String.raw`\text{Accumulate}\to\text{threshold}\to\text{pulse}`, description: "The controller converts accumulated updates above device granularity into open-loop programming pulses. The array receives the consolidated update; the digital buffer retains the untransferred remainder for a later step."}
];
const buttons = [...document.querySelectorAll(".step-button")];
buttons.forEach((button) => {
  button.addEventListener("click", () => {
    const step = steps[Number(button.dataset.step)];
    buttons.forEach((other) => {
      const selected = other === button;
      other.classList.toggle("selected", selected);
      other.setAttribute("aria-pressed", String(selected));
    });
    document.getElementById("step-domain").textContent = step.domain;
    const equation = document.getElementById("step-equation");
    equation.textContent = step.equation;
    equation.dataset.math = step.latex;
    document.dispatchEvent(new CustomEvent('math-content-changed', { detail: { element: equation } }));
    document.getElementById("step-description").textContent = step.description;
  });
});

const links = [...document.querySelectorAll(".project-toc-nav a")];
const sections = links.map((link) => document.querySelector(link.getAttribute("href")));
function updateContents() {
  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= 125) current = section;
  }
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 8) current = sections[sections.length - 1];
  for (const link of links) {
    const active = link.hash === `#${current.id}`;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
let scheduled = false;
window.addEventListener("scroll", () => {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(() => { updateContents(); scheduled = false; });
  }
}, {passive: true});
window.addEventListener("resize", updateContents);
updateContents();

const copyBibtex = document.getElementById("copy-bibtex");
const citation = document.getElementById("bibtex-citation");
const copyStatus = document.getElementById("bibtex-copy-status");
copyBibtex.hidden = false;
copyBibtex.addEventListener("click", async () => {
  const text = citation.textContent.trim();
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    // Allow copying when opening the page locally or clipboard access is denied.
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    try { copied = document.execCommand("copy"); } catch { /* Keep text selected. */ }
    if (copied) selection.removeAllRanges();
  }
  copyBibtex.textContent = copied ? "Copied!" : "Copy";
  copyStatus.textContent = copied ? "BibTeX citation copied." : "Citation selected. Press Ctrl+C or Command+C to copy.";
});
