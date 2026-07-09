/* ============================================================================
 * OpenAI "dotcom" bar-chart renderer — zero dependencies, pure SVG.
 *
 *   renderOpenAIBarChart(targetSvgOrSelector, spec)
 *
 * Draws a grouped (or single-series) bar chart in the visual style OpenAI uses
 * on its blog / research posts / system cards: monochrome bars with a darker
 * same-hue stroke, rounded corners, a black y-axis with outward ticks and NO
 * gridlines, circle legend markers left-aligned to the y-axis, value labels
 * above each bar, angled category labels, and the real OpenAI Sans typeface.
 *
 * The script injects its own <style> (font-face + classes) once, so a single
 * <script src> + a spec object is all a page needs.
 *
 * See SKILL.md for the spec shape and references/design-spec.md for the tokens.
 * ==========================================================================*/
(function (global) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  // Default categorical order. Each entry is {fill, stroke}: fill is a mid/light
  // shade, stroke is a darker step of the SAME hue. Pink first because that is
  // OpenAI's most common editorial accent. Full palette in design-spec.md.
  var PALETTE = [
    { fill: "#bd569b", stroke: "#8a3a6f" }, // pink   (dark)
    { fill: "#fcdad6", stroke: "#bd569b" }, // pink   (light)
    { fill: "#5477c4", stroke: "#2e4780" }, // blue   (dark)
    { fill: "#cedffe", stroke: "#5477c4" }, // blue   (light)
    { fill: "#71b436", stroke: "#386411" }, // green
    { fill: "#ffbda1", stroke: "#cc6f47" }, // orange (light)
    { fill: "#767881", stroke: "#47484f" }, // gray
  ];

  var STYLE_ID = "openai-chart-style";
  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var cdn = "https://cdn.openai.com/common/fonts/openai-sans/";
    var face = function (w, file) {
      return '@font-face{font-family:"OpenAI Sans";font-weight:' + w +
        ';font-style:normal;font-display:swap;src:url("' + cdn + file +
        '") format("woff2");}';
    };
    var css =
      face(400, "OpenAISans-Regular.woff2") +
      face(500, "OpenAISans-Medium.woff2") +
      face(600, "OpenAISans-Semibold.woff2") +
      face(700, "OpenAISans-Bold.woff2") +
      '.oai-chart{font-family:"OpenAI Sans","Helvetica Neue",Arial,system-ui,sans-serif;}' +
      '.oai-chart .oai-title{font-weight:700;fill:#0d0d0d;}' +
      '.oai-chart .oai-legend{font-weight:400;fill:#0d0d0d;}' +
      '.oai-chart .oai-axis{stroke:#0d0d0d;stroke-width:1.5;}' +
      '.oai-chart .oai-axis-lbl,.oai-chart .oai-axis-title,' +
      '.oai-chart .oai-val,.oai-chart .oai-cat{fill:#0d0d0d;}';
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = css;
    document.head.appendChild(s);
  }

  function el(parent, tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    parent.appendChild(e);
    return e;
  }

  // Smallest "nice" number >= x (1/2/5 * 10^n) — used to pick the axis maximum.
  function niceCeil(x) {
    if (!(x > 0)) return 1;
    var mag = Math.pow(10, Math.floor(Math.log(x) / Math.LN10));
    var n = x / mag;
    var step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
    return step * mag;
  }

  function makeFormatter(mode) {
    if (typeof mode === "function") return mode;
    if (mode === "number") return function (v) { return String(v); };
    // "percent" (default): values are fractions, 0.144 -> "14.4%"
    return function (v) { return (Math.round(v * 1000) / 10) + "%"; };
  }

  function render(target, spec) {
    injectStyle();
    var svg = typeof target === "string" ? document.querySelector(target) : target;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var cls = (svg.getAttribute("class") || "").replace(/\boai-chart\b/g, "").trim();
    svg.setAttribute("class", (cls + " oai-chart").trim());

    var fmt = makeFormatter(spec.valueFormat || "percent");
    var cats = spec.categories;
    var series = spec.series;
    var palette = spec.palette || PALETTE;
    series.forEach(function (s, i) {
      var p = palette[i % palette.length];
      if (!s.fill) s.fill = p.fill;
      if (!s.stroke) s.stroke = p.stroke;
    });

    // ---- y domain & ticks ----
    var allVals = [];
    series.forEach(function (s) { allVals = allVals.concat(s.values); });
    var yMax = spec.yMax != null ? spec.yMax : niceCeil(Math.max.apply(null, allVals));
    var ticks = spec.yTicks;
    if (!ticks) {
      var step = spec.yTickStep != null ? spec.yTickStep : yMax / 4;
      ticks = [];
      var count = Math.round(yMax / step);
      for (var t = 0; t <= count; t++) ticks.push(+(t * step).toFixed(10));
    }

    // ---- layout ----
    var fs = 13.5;                       // axis / value / category labels
    var W = spec.width || Math.max(520, 74 + 20 + cats.length * 132);
    var hasTitle = !!spec.title;
    var showLegend = spec.legend !== false && series.some(function (s) { return s.name; });
    var padTop = 16;
    var titleY = padTop + 12;
    var legendTop = hasTitle ? titleY + 20 : padTop;
    var legendCy = legendTop + 8;
    var plotTop = showLegend ? legendTop + 34 : legendTop + 6;
    var bottom = spec.bottomMargin != null ? spec.bottomMargin : 96;
    var H = spec.height || plotTop + 260 + bottom;
    var m = { t: plotTop, r: 20, b: bottom, l: 74 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;

    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    if (spec.ariaLabel) svg.setAttribute("aria-label", spec.ariaLabel);

    function y(v) { return m.t + ih - (v / yMax) * ih; }

    // ---- title ----
    if (hasTitle) el(svg, "text", { x: 4, y: titleY, class: "oai-title", "font-size": 15 }, spec.title);

    // ---- legend (inside the svg, left edge aligned to the y-axis line) ----
    if (showLegend) {
      var lx = m.l;
      series.forEach(function (s) {
        el(svg, "circle", { cx: lx + 7.5, cy: legendCy, r: 6, fill: s.fill, stroke: s.stroke, "stroke-width": 1.5 });
        var t = el(svg, "text", { x: lx + 22, y: legendCy + 5, class: "oai-legend", "font-size": 15 }, s.name);
        var w;
        try { w = t.getComputedTextLength(); } catch (e) { w = (s.name || "").length * 8; }
        lx += 22 + w + 32;
      });
    }

    // ---- y-axis: line + outward ticks + right-aligned labels (no gridlines) ----
    el(svg, "line", { x1: m.l, y1: y(yMax), x2: m.l, y2: y(0), class: "oai-axis" });
    ticks.forEach(function (tv) {
      el(svg, "line", { x1: m.l - 6, y1: y(tv), x2: m.l, y2: y(tv), class: "oai-axis" });
      el(svg, "text", { x: m.l - 12, y: y(tv) + 5, "text-anchor": "end", class: "oai-axis-lbl", "font-size": fs }, fmt(tv));
    });
    if (spec.yAxisTitle) {
      var yt = el(svg, "text", { class: "oai-axis-title", "text-anchor": "middle", "font-size": 14 }, spec.yAxisTitle);
      yt.setAttribute("transform", "translate(20," + (m.t + ih / 2) + ") rotate(-90)");
    }

    // ---- bars + value labels + category labels ----
    var n = cats.length, groupW = iw / n, pad = 0.34, gw = groupW * (1 - pad);
    var k = series.length, innerGap = k > 1 ? 4 : 0, barW = (gw - innerGap * (k - 1)) / k;

    cats.forEach(function (cat, i) {
      var gx = m.l + i * groupW + (groupW * pad) / 2;
      series.forEach(function (s, j) {
        var val = s.values[i];
        var x = gx + j * (barW + innerGap);
        var bh = (val / yMax) * ih;
        el(svg, "rect", { x: x, y: y(val), width: barW, height: bh, rx: 4, ry: 4, fill: s.fill, stroke: s.stroke, "stroke-width": 1.5 });
        el(svg, "text", { x: x + barW / 2, y: y(val) - 8, "text-anchor": "middle", class: "oai-val", "font-size": fs }, fmt(val));
      });
      var cx = gx + gw / 2;
      var g = document.createElementNS(NS, "text");
      g.setAttribute("transform", "translate(" + cx + "," + (y(0) + 16) + ") rotate(-45)");
      g.setAttribute("text-anchor", "end");
      g.setAttribute("class", "oai-cat");
      g.setAttribute("font-size", 13);
      String(cat).split("\n").forEach(function (ln, li) {
        var ts = document.createElementNS(NS, "tspan");
        ts.setAttribute("x", 0);
        ts.setAttribute("dy", li === 0 ? 0 : 15);
        ts.textContent = ln;
        g.appendChild(ts);
      });
      svg.appendChild(g);
    });

    return svg;
  }

  global.renderOpenAIBarChart = render;
})(typeof window !== "undefined" ? window : this);
