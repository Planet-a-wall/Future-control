// Build every chapters/chNN.html fragment into a print-ready A4 PDF.
// Math: $...$ (inline) and $$...$$ (display) rendered server-side with KaTeX.
// Figures: <fig src="name" cap="..."/> inlines figures/name.svg (vector).
// Usage: node build.js [chNN ...]
const fs = require("fs");
const path = require("path");
const katex = require("katex");
const { chromium } = require("playwright");

const ROOT = __dirname;
const NM = path.join(ROOT, "node_modules");
const OUT = path.join(ROOT, "pdf");
const TMP = path.join(ROOT, ".build");
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(TMP, { recursive: true });

const macros = { "\\E": "\\mathbb{E}", "\\Var": "\\operatorname{Var}", "\\R": "\\mathbb{R}",
  "\\d": "\\mathrm{d}", "\\Pr": "\\mathbb{P}" };

function renderMath(src) {
  const dec = m => m.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) =>
    `<div class="eq">${katex.renderToString(dec(m.trim()), { displayMode: true, throwOnError: true, macros, trust: true })}</div>`);
  return src.replace(/\$([^$\n]+?)\$/g, (_, m) =>
    katex.renderToString(dec(m), { displayMode: false, throwOnError: true, macros }));
}

function inlineFigures(src, chap) {
  let k = 0;
  return src.replace(/<fig src="([^"]+)" cap="([^"]*)"(?: w="([^"]+)")?\s*\/>/g, (_, name, cap, w) => {
    k += 1;
    let svg = fs.readFileSync(path.join(ROOT, "figures", name + ".svg"), "utf8");
    svg = svg.replace(/<\?xml[^>]*>/, "").replace(/<!DOCTYPE[^>]*>/, "")
      .replace(/<svg([^>]*?)width="[^"]*"([^>]*?)height="[^"]*"/, "<svg$1$2");
    return `<figure style="width:${w || "100%"}">${svg}<figcaption><b>รูปที่ ${chap}.${k}</b> ${cap}</figcaption></figure>`;
  });
}

const css = `
@page { size: A4; margin: 22mm 21mm 24mm 21mm; }
:root { --ink:#1B1F27; --muted:#5B6472; --rule:#D7DCE3; --accent:#1F4E8C; --accent-soft:#EEF3FA;
        --thm:#F5F7FB; --warn:#FFF7EB; --warn-ink:#8A4B08; --ok:#ECF7F0; --ok-ink:#1E6B3B; --bad:#FDEEEE; --bad-ink:#9B2C2C; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Sarabun", "Loma", sans-serif; font-size: 11.2pt; line-height: 1.72; color: var(--ink);
       margin: 0; background: #fff; font-feature-settings: "kern"; }
h1,h2,h3,h4 { font-family: "Noto Serif Thai", "Sarabun", serif; color: var(--ink); line-height: 1.4; break-after: avoid; }
h2 { font-size: 15pt; margin: 18pt 0 6pt; padding: 6pt 0 3pt; line-height: 1.6; border-bottom: 1.2pt solid var(--accent); }
h3 { font-size: 12.4pt; margin: 12pt 0 4pt; padding-top: 4pt; line-height: 1.6; color: var(--accent); }
h4 { font-size: 11.4pt; margin: 9pt 0 2pt; padding-top: 3pt; line-height: 1.6; }
p { margin: 0 0 7pt; text-align: left; orphans: 3; widows: 3; }
.cover { border-top: 5pt solid var(--accent); padding-top: 10pt; margin-bottom: 14pt; }
.cover .series { font-size: 9pt; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.cover .num { font-family: "Noto Serif Thai", serif; font-size: 13pt; color: var(--accent); margin-top: 6pt; }
.cover h1 { font-size: 22pt; margin: 2pt 0 2pt; }
.cover .en { font-size: 11.5pt; color: var(--muted); font-style: italic; }
.cover .meta { font-size: 9pt; color: var(--muted); margin-top: 8pt; }
.summary { border: 1pt solid var(--rule); border-radius: 4pt; margin: 6pt 0 14pt; break-inside: avoid; }
.summary .hd { background: var(--accent); color: #fff; font-weight: 700; padding: 4pt 10pt; border-radius: 3pt 3pt 0 0; font-size: 10.5pt; }
.summary .row { display: grid; grid-template-columns: 30mm 1fr; border-top: 1pt solid var(--rule); }
.summary .row:first-of-type { border-top: 0; }
.summary .k { background: var(--accent-soft); padding: 6pt 8pt; font-weight: 700; font-size: 10pt; color: var(--accent); }
.summary .v { padding: 6pt 10pt; font-size: 10.4pt; }
.summary .v p { margin: 0 0 3pt; } .summary ul { margin: 0; padding-left: 14pt; }
.box { border-left: 3pt solid var(--accent); background: var(--thm); padding: 6pt 11pt 4pt; margin: 9pt 0 10pt; break-inside: avoid; border-radius: 0 3pt 3pt 0; }
.box > .t { font-weight: 700; color: var(--accent); margin-right: 4pt; }
.box.def { border-color: #2F6FB2; } .box.ax { border-color: #7A4FB5; background: #F6F3FB; } .box.ax > .t { color: #5B3A8C; }
.box.thm { border-color: #1F4E8C; } .box.hyp { border-color: #D9822B; background: var(--warn); } .box.hyp > .t { color: var(--warn-ink); }
.box.note { border-color: #6B7280; background: #F7F7F8; } .box.note > .t { color: #374151; }
.box.res { border-color: #3A9A5B; background: var(--ok); } .box.res > .t { color: var(--ok-ink); }
.box.fail { border-color: #C23B3B; background: var(--bad); } .box.fail > .t { color: var(--bad-ink); }
.proof { margin: -2pt 0 10pt; padding: 0 11pt; font-size: 10.6pt; }
.proof::before { content: "พิสูจน์. "; font-weight: 700; font-style: italic; }
.proof::after { content: "∎"; float: right; }
.eq { margin: 6pt 0 8pt; overflow-x: visible; }
.katex { font-size: 1.04em; } .katex-display { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.9pt; break-inside: avoid; }
caption { caption-side: top; text-align: left; font-size: 9.8pt; color: var(--muted); padding-bottom: 3pt; }
th { background: var(--accent-soft); color: var(--accent); font-weight: 700; text-align: left; }
th, td { border-bottom: .8pt solid var(--rule); padding: 3.2pt 6pt; vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.ok td { background: #F3FAF5; } tr.bad td { background: #FDF3F3; }
figure { margin: 8pt auto 12pt; break-inside: avoid; } figure svg { width: 100%; height: auto; display: block; }
figcaption { font-size: 9.7pt; color: var(--muted); margin-top: 3pt; line-height: 1.55; }
ul, ol { margin: 2pt 0 8pt; padding-left: 18pt; } li { margin-bottom: 2.5pt; }
.refs { font-size: 9.6pt; line-height: 1.55; } .refs li { margin-bottom: 3pt; }
.tag { display: inline-block; font-size: 8.6pt; font-weight: 700; padding: 0 5pt; border-radius: 8pt; line-height: 1.6; }
.tag.g { background: var(--ok); color: var(--ok-ink); } .tag.y { background: var(--warn); color: var(--warn-ink); }
.tag.r { background: var(--bad); color: var(--bad-ink); } .tag.b { background: var(--accent-soft); color: var(--accent); }
.small { font-size: 9.8pt; color: var(--muted); }
.pb { break-before: page; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #F3F4F6; padding: 0 3pt; border-radius: 2pt; }
`;

function fontLinks() {
  const f = [];
  for (const w of ["400", "400-italic", "600", "700", "700-italic"]) f.push(`@fontsource/sarabun/${w}.css`);
  for (const w of ["500", "600", "700"]) f.push(`@fontsource/noto-serif-thai/${w}.css`);
  f.push("katex/dist/katex.min.css");
  return f.map(p => `<link rel="stylesheet" href="file://${path.join(NM, p)}">`).join("\n");
}

async function build(file, browser) {
  let src = fs.readFileSync(path.join(ROOT, "chapters", file), "utf8");
  const meta = JSON.parse(src.match(/<!--meta\s*([\s\S]*?)-->/)[1]);
  src = src.replace(/<!--meta[\s\S]*?-->/, "");
  src = inlineFigures(src, meta.num);
  src = renderMath(src);
  const html = `<!doctype html><html lang="th"><head><meta charset="utf-8">
<title>CIOT บทที่ ${meta.num}: ${meta.title}</title>${fontLinks()}<style>${css}</style></head><body>
<div class="cover"><div class="series">Cumulative Intentional Order Theory (CIOT) · Research Proposal</div>
<div class="num">บทที่ ${meta.num}</div><h1>${meta.title}</h1><div class="en">${meta.en}</div>
<div class="meta">ฉบับร่างเพื่อเสนอการวิจัย · v2.0 · พฤศจิกายน 2569 · ผลเชิงตัวเลขทั้งหมดทำซ้ำได้ด้วย <code>paper/analysis.py</code></div></div>
${src}</body></html>`;
  const htmlPath = path.join(TMP, file);
  fs.writeFileSync(htmlPath, html);
  const page = await browser.newPage();
  await page.goto("file://" + htmlPath, { waitUntil: "load" });
  await page.evaluate(async () => { await document.fonts.ready; });
  const pdfName = file.replace(".html", "") + "_" + meta.slug + ".pdf";
  await page.pdf({
    path: path.join(OUT, pdfName), format: "A4", printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: `<div style="font-size:7.5pt;color:#8A93A3;width:100%;padding:0 21mm;font-family:Loma,sans-serif;">
      <span>CIOT · Research Proposal</span><span style="float:right">Chapter ${meta.num} — ${meta.en}</span></div>`,
    footerTemplate: `<div style="font-size:8pt;color:#8A93A3;width:100%;text-align:center;font-family:Loma,sans-serif;">
      <span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
  });
  await page.close();
  console.log("built", pdfName);
}

(async () => {
  const only = process.argv.slice(2);
  const files = fs.readdirSync(path.join(ROOT, "chapters")).filter(f => /^ch\d+\.html$/.test(f))
    .filter(f => !only.length || only.includes(f.replace(".html", ""))).sort();
  const browser = await chromium.launch();
  for (const f of files) await build(f, browser);
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
