# CIOT research proposal (one PDF per chapter)

- `pdf/` — the seven chapters, A4, vector figures
- `chapters/chNN.html` — chapter sources (Thai; math in `$...$` / `$$...$$`)
- `analysis.py` — every number and figure in the chapters (writes `figures/*.svg`, `results.json`)
- `build.js` — renders chapters to PDF with KaTeX + Chromium (Playwright)

Rebuild:

```bash
pip install numpy scipy matplotlib
npm install
python3 analysis.py
node build.js            # or: node build.js ch04
```
