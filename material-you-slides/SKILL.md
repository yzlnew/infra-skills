---
name: material-you-slides
description: Create presentation slides using Material You (Material Design 3) style. Generates 1280x720 HTML slides with M3 color tokens, Roboto typography, rounded cards, flow diagrams, metric cards, code blocks, and structured layouts. Use when the user asks to create slides, presentations, or decks and wants a clean, modern Material Design 3 aesthetic.
---

# Material You Slides Skill

Create presentation slide decks as single-file HTML using the Material You (Material Design 3) design language. The output is a self-contained `.html` file optimized for 1280x720 presentation dimensions, printable via `@page` rules.

## When to Use

Use this skill when the user asks to create slides, presentations, or decks and wants (or would benefit from) a clean, modern Material Design 3 look. This is the default slide style unless the user explicitly requests a different theme.

## Design Principles

1. **M3 Color Tokens** - Use CSS custom properties following Material Design 3 naming (`--md-sys-color-*`).
2. **Roboto Typography** - Import from Google Fonts. Use weights 300 (light/subtitle), 400 (body), 500 (medium), 600 (semibold headings), 700 (bold), 800-900 (display/hero).
3. **Rounded Shapes** - Four tiers of corner radius: small (8px), medium (12px), large (16px), extra-large (28px).
4. **Surface Hierarchy** - Use `surface`, `surface-container-low`, `surface-container`, `surface-container-high` for layered depth without drop shadows.
5. **Container Colors** - Cards use `*-container` / `on-*-container` pairs for accessible contrast.
6. **No Drop Shadows** - Rely on surface tones and subtle borders for elevation.
7. **Generous Whitespace** - Slide padding 24px 48px. Card padding 24px. Gap 16-32px between elements.

## CSS Foundation

Every generated deck MUST include this CSS foundation in a `<style>` block. You may add component-specific styles but must not remove or override these base tokens.

```css
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  /* M3 Color Palette - Blue Theme */
  --md-sys-color-primary: #005AC1;
  --md-sys-color-on-primary: #FFFFFF;
  --md-sys-color-primary-container: #D8E2FF;
  --md-sys-color-on-primary-container: #001A41;

  --md-sys-color-secondary: #575E71;
  --md-sys-color-on-secondary: #FFFFFF;
  --md-sys-color-secondary-container: #DBE2F9;
  --md-sys-color-on-secondary-container: #141B2C;

  --md-sys-color-tertiary: #715573;
  --md-sys-color-on-tertiary: #FFFFFF;
  --md-sys-color-tertiary-container: #FBD7FC;
  --md-sys-color-on-tertiary-container: #29132D;

  --md-sys-color-error: #BA1A1A;
  --md-sys-color-error-container: #FFDAD6;

  --md-sys-color-surface: #FEF7FF;
  --md-sys-color-on-surface: #1D1B20;
  --md-sys-color-surface-container-low: #F7F2FA;
  --md-sys-color-surface-container: #F3EDF7;
  --md-sys-color-surface-container-high: #ECE6F0;

  --md-sys-color-outline: #74777F;
  --md-sys-color-outline-variant: #C4C6CF;

  /* Custom Accents */
  --accent-blue: #005AC1;
  --accent-teal: #006D5B;
  --accent-gold: #735C00;

  /* Shape */
  --md-sys-shape-corner-extra-large: 28px;
  --md-sys-shape-corner-large: 16px;
  --md-sys-shape-corner-medium: 12px;
  --md-sys-shape-corner-small: 8px;

  /* Typography */
  --font-heading: 'Roboto', -apple-system, sans-serif;
  --font-body: 'Roboto', -apple-system, sans-serif;
}

body {
  font-family: var(--font-body);
  background: #fff;
  color: var(--md-sys-color-on-surface);
  -webkit-font-smoothing: antialiased;
  font-size: 24px;
}
```

## Slide Types

### 1. Title Slide (`.slide-title`)

Full-bleed primary-color background with radial gradient accents. Used for the opening and closing slides.

```html
<div class="slide slide-title">
  <div class="brand-line"></div>
  <h1>Presentation Title</h1>
  <div class="subtitle">A concise tagline or description</div>
  <div class="meta-info">
    <div class="author">Author / Team Name</div>
  </div>
</div>
```

**CSS:**
```css
.slide-title {
  background-color: var(--md-sys-color-primary);
  color: var(--md-sys-color-on-primary);
  justify-content: center;
  align-items: flex-start;
  padding-left: 100px;
  background-image: radial-gradient(circle at 90% 10%, rgba(255,255,255,0.1) 0%, transparent 40%),
                    radial-gradient(circle at 80% 90%, rgba(255,255,255,0.05) 0%, transparent 30%);
}
.slide-title .brand-line {
  width: 80px; height: 10px;
  background: var(--md-sys-color-tertiary-container);
  border-radius: 5px; margin-bottom: 40px;
}
.slide-title h1 {
  font-family: var(--font-heading);
  font-size: 96px; font-weight: 800;
  letter-spacing: -2px; line-height: 1.1; margin-bottom: 24px;
}
.slide-title .subtitle {
  font-size: 40px; font-weight: 300;
  opacity: 0.9; margin-bottom: 80px; max-width: 900px;
}
.slide-title .meta-info { display: flex; gap: 40px; align-items: center; }
.slide-title .author {
  font-size: 24px; font-weight: 500;
  background: rgba(255,255,255,0.15);
  padding: 16px 32px; border-radius: 50px;
}
```

### 2. Section Divider (`.slide-section`)

Light surface background with a left-edge primary color bar and a large translucent section number.

```html
<div class="slide slide-section">
  <div class="section-num">01</div>
  <h2>Section Title</h2>
  <div class="section-sub">Subtitle or English tagline</div>
</div>
```

**CSS:**
```css
.slide-section {
  background: var(--md-sys-color-surface-container-low);
  color: var(--md-sys-color-on-surface);
  justify-content: center;
  position: relative;
}
.slide-section::before {
  content: "";
  position: absolute;
  top: 0; left: 0; bottom: 0; width: 24px;
  background: var(--md-sys-color-primary);
}
.slide-section .section-num {
  font-size: 180px; font-weight: 900;
  color: var(--md-sys-color-primary-container);
  position: absolute; top: -30px; right: 60px; opacity: 0.5;
}
.slide-section h2 { font-size: 72px; font-weight: 700; line-height: 1.2; z-index: 1; position: relative; }
.slide-section .section-sub {
  font-size: 32px; color: var(--md-sys-color-secondary);
  margin-top: 32px; font-weight: 400; z-index: 1;
}
```

### 3. Content Slide (`.slide-content`)

The workhorse slide with a header (tag + title) and a flexible body area.

```html
<div class="slide slide-content">
  <div class="header">
    <div class="header-top">
      <div class="header-tag">TAG</div>
    </div>
    <h3>Slide Title</h3>
  </div>
  <div class="body">
    <!-- Content goes here -->
  </div>
  <div class="page-num">3</div>
</div>
```

**CSS:**
```css
.slide {
  width: 1280px; height: 720px;
  page-break-after: always; position: relative; overflow: hidden;
  padding: 24px 48px; display: flex; flex-direction: column;
  background-color: var(--md-sys-color-surface);
}
.slide:last-child { page-break-after: avoid; }

.slide-content .header {
  display: flex; flex-direction: column; align-items: flex-start;
  margin-bottom: 20px; gap: 12px;
  border-bottom: 1px solid var(--md-sys-color-outline-variant);
  padding-bottom: 16px; flex-shrink: 0;
}
.slide-content .header-top {
  display: flex; align-items: center; width: 100%; justify-content: space-between;
}
.slide-content .header-tag {
  background: var(--md-sys-color-secondary-container);
  color: var(--md-sys-color-on-secondary-container);
  font-size: 16px; font-weight: 600;
  padding: 6px 20px; border-radius: 100px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.slide-content .header h3 {
  font-size: 42px; font-weight: 600;
  color: var(--md-sys-color-on-surface); letter-spacing: -0.5px; line-height: 1.2;
}
.slide-content .body { flex: 1; display: flex; flex-direction: column; gap: 16px; min-height: 0; }
.page-num {
  position: absolute; bottom: 24px; right: 40px;
  font-size: 16px; color: var(--md-sys-color-outline); font-variant-numeric: tabular-nums;
}
```

## Component Library

### Cards

Use M3 container color pairs. Variants: `primary`, `secondary`, `tertiary`, `outlined`.

```html
<div class="card">Default surface card</div>
<div class="card primary">Primary container card</div>
<div class="card secondary">Secondary container card</div>
<div class="card tertiary">Tertiary container card</div>
<div class="card outlined">Outlined card (transparent bg, border)</div>
```

```css
.card {
  background: var(--md-sys-color-surface-container);
  border-radius: var(--md-sys-shape-corner-extra-large);
  padding: 24px; display: flex; flex-direction: column; gap: 12px;
  box-shadow: none; position: relative;
}
.card.primary { background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
.card.secondary { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.card.tertiary { background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
.card.outlined { background: transparent; border: 1px solid var(--md-sys-color-outline-variant); }
.card-title { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.card-text { font-size: 22px; line-height: 1.5; opacity: 0.9; }
```

### Layouts

```css
.columns { display: flex; gap: 32px; flex: 1; min-height: 0; }
.column { flex: 1; display: flex; flex-direction: column; gap: 16px; min-height: 0; }
.col-60 { flex: 3; }
.col-40 { flex: 2; }
```

### Lists

Styled with primary-colored bullet dots, no default list-style.

```css
ul { list-style: none; padding: 0; }
ul li {
  font-size: 24px; line-height: 1.5;
  padding: 6px 0 6px 32px; position: relative;
  color: var(--md-sys-color-on-surface);
}
ul li::before {
  content: "\2022"; position: absolute; left: 8px;
  color: var(--md-sys-color-primary); font-weight: 900; font-size: 24px;
}
```

### Tables

Rounded-corner tables with surface-container header background.

```css
table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: var(--md-sys-shape-corner-large); overflow: hidden;
}
table th {
  background: var(--md-sys-color-surface-container);
  color: var(--md-sys-color-on-surface);
  padding: 12px 20px; text-align: left; font-weight: 700; font-size: 20px;
}
table td {
  padding: 12px 20px;
  border-top: 1px solid var(--md-sys-color-outline-variant);
  color: var(--md-sys-color-on-surface); font-size: 20px;
}
```

### Tags / Chips

```html
<span class="tag">Default</span>
<span class="tag blue">Blue</span>
<span class="tag green">Green</span>
<span class="tag red">Red</span>
```

```css
.tag {
  display: inline-flex; align-items: center;
  background: var(--md-sys-color-surface-container-high);
  color: var(--md-sys-color-on-surface);
  font-size: 16px; font-weight: 500;
  padding: 6px 14px; border-radius: 8px; margin-right: 8px;
}
.tag.blue { background: #D8E2FF; color: #001A41; }
.tag.green { background: #C4EED0; color: #072111; }
.tag.red { background: #FFDAD6; color: #410002; }
```

### Flow Diagrams

Horizontal process flows with bordered boxes and arrow connectors.

```html
<div class="flow-row">
  <div class="flow-box dark">Step 1</div>
  <div class="flow-arrow">&rarr;</div>
  <div class="flow-box">Step 2</div>
  <div class="flow-arrow">&rarr;</div>
  <div class="flow-box dark" style="background:var(--accent-teal);">Step 3</div>
</div>
```

```css
.flow-row { display: flex; align-items: center; gap: 12px; justify-content: center; margin: 16px 0; }
.flow-box {
  background: var(--md-sys-color-surface);
  border: 2px solid var(--md-sys-color-primary);
  border-radius: 16px; padding: 12px 20px;
  text-align: center; min-width: 120px; font-weight: 600; font-size: 18px;
}
.flow-box.dark { background: var(--md-sys-color-primary); color: var(--md-sys-color-on-primary); border: none; }
.flow-arrow { color: var(--md-sys-color-outline); font-size: 24px; }
```

### Metric Cards

Large numeric KPI displays in a row.

```html
<div class="metric-row">
  <div class="metric-card">
    <div class="val">42%</div>
    <div class="label">Description</div>
  </div>
</div>
```

```css
.metric-row { display: flex; gap: 24px; }
.metric-card {
  flex: 1; background: var(--md-sys-color-surface-container);
  border-radius: 20px; padding: 24px; text-align: center;
}
.metric-card .val { font-size: 56px; font-weight: 800; color: var(--md-sys-color-primary); line-height: 1; margin-bottom: 8px; }
.metric-card .label { font-size: 20px; color: var(--md-sys-color-secondary); }
```

### Code Blocks

Dark-themed code display with syntax highlight classes.

```css
.code-block {
  background: #1E1E1E; color: #D4D4D4;
  border-radius: var(--md-sys-shape-corner-large);
  padding: 20px;
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 16px; line-height: 1.6;
  overflow: hidden; white-space: pre-wrap;
}
.code-block .kw { color: #569CD6; }   /* keywords */
.code-block .str { color: #CE9178; }  /* strings */
.code-block .cmt { color: #6A9955; font-style: italic; } /* comments */
```

## Typical Deck Structure

A well-structured deck follows this pattern:

1. **Title Slide** (`.slide-title`) - Project name, tagline, author
2. **Agenda/Overview Slide** (`.slide-content`) - Grid of outlined cards with colored left borders for each section
3. **Section Divider** (`.slide-section`) - For each major part (01, 02, 03...)
4. **Content Slides** (`.slide-content`) - The actual content using cards, columns, tables, flow diagrams, code blocks, metrics
5. **Closing Slide** (`.slide-title`) - "Thanks!" with links and credits

## Color Customization

The default theme is Blue. To create a different color variant, replace the `--md-sys-color-primary` family with a new hue while keeping the M3 tonal structure:

- Generate tonal palette from a seed color
- Keep surface/outline tokens unchanged for consistency
- Update `--accent-*` custom properties to complement the new primary

## Content Density Guidelines

- **Max 3-4 key points** per content slide
- **Use columns** (`.columns` > `.column`) for side-by-side comparisons
- **Use tables** for structured comparisons (max 5-6 rows visible)
- **Use flow diagrams** for process explanations
- **Use metric cards** for KPI/result highlights (max 3-4 per row)
- **Use code blocks** sparingly, only for key snippets
- **Font sizes**: Body text 22-24px, card text 16-22px, table 20px, code 16px
- **Agenda slide**: Use a 2x2 grid of outlined cards with colored left borders (`border-left: 6px solid`)

## Output Format

Generate a single self-contained `.html` file. All CSS is inlined in a `<style>` tag. No external dependencies except the Google Fonts import. The file should be directly openable in a browser for presentation.
