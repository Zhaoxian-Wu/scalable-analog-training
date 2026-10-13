# Slide Editing

Read this guide before editing the Slidev deck in `slides/`.

## Punctuation convention

**Omit sentence-ending periods from all text displayed inside a slide.** This
applies to titles, subtitles, paragraphs, bullets, callouts, captions, table
text, footnotes, references, and interactive descriptions or tooltips.
Use the same convention for full sentences and short phrases.

- Remove the final `.` from each displayed sentence or phrase
- Separate multiple thoughts with line breaks, bullets, semicolons, or ` · `
- Keep question marks when the text asks a question
- Preserve meaningful periods in decimals (`123.6M`, `3.3950`), names and
  abbreviations (`T. Patrick Xiao`, `et al.`), URLs, filenames, code, and math
- Prefer a full month name such as `November` in footnotes to avoid ambiguous
  abbreviation punctuation
- Preserve reference details and scientific caveats when changing punctuation

| Before | After |
| --- | --- |
| `Power demand keeps growing.` | `Power demand keeps growing` |
| `Analog passes. Digital gradients.` | `Analog passes · Digital gradients` |
| `One seed per condition. Buttons reveal measured stages.` | `One seed per condition · Buttons reveal measured stages` |
| `Stable training through 123.6M parameters.` | `Stable training through 123.6M parameters` |

This is a convention for projected slide text. Presenter notes in HTML comments,
repository documentation, source-code comments, and the manuscript can use
normal sentence punctuation. Keep source code shown as an example intact.

## Where to apply the rule

Edit visible prose in `slides/slides.md` and `slides/components/*.vue`, including
strings selected by buttons, dropdowns, hover, or focus. Check displayed text in
figure assets and data-backed descriptions when editing or adding them. Avoid a
global replacement of `.`: it can corrupt numeric values, links, or program code.

## Preserve the deck layout

- Keep content-slide titles red and fixed at the top
- Center only the main content between the title area and footer
- Keep footnotes and references at bottom left and page numbers at bottom right
- Preserve the cover's separate alignment: its title, subtitle, author details,
  and logos center together
- Keep the PowerPoint-equivalent 32 pt content titles and 24 pt standard text
  defined by the CSS variables; retain intentional emphasis and smaller annotations
- Omit source declarations for our own paper; retain external attribution

## Verify an edit

Review rendered slides and interactive states for sentence-ending periods,
while allowing the meaningful periods listed above. Confirm that punctuation
changes preserve readability and do not cause title wrapping or footer overlap.
Build from `slides/` with `npm run build` in the `analog-slides` environment, as
described in `slides/README.md`.
