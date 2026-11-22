# CIOT research proposal (one PDF per chapter)

- `pdf/` — chapters 1–7 plus supplements 8 (mental states, emotion, brainwaves) and 9 (profiles × routines); A4, vector figures
- `chapters/chNN.html` — chapter sources (Thai; math in `$...$` / `$$...$$`)
- `analysis.py` — every number and figure in chapters 1–7 (writes `figures/*.svg`, `results.json`)
- `analysis_supp.py` — profile × routine simulations for chapters 8–9 (writes `figures/f8_*`, `f9_*`, `results_supp.json`)
- `build.js` — renders chapters to PDF with KaTeX + Chromium (Playwright)

Rebuild:

```bash
pip install numpy scipy matplotlib
npm install
python3 analysis.py
python3 analysis_supp.py
node build.js            # or: node build.js ch04
```
