"""
Extended ignition condition (U2+): planning steps, mental state, interruptions, environment.
  k = M * B,  M = cos^{2n}(theta_eff/n) * cos^{2D}(phi),  B = b * r
  ignition  <=>  k >= 1  <=>  n >= theta_eff^2 / ln(B * cos^{2D}(phi))     (Zeno approximation)
  'no-ignition zone': B * cos^{2D}(phi) <= 1  -> no amount of planning ignites; change state/environment.
State shifts (illustrative magnitudes, directions from Ch. 8):
  theta_eff = theta + 8deg*anxiety + 6deg*sleep_debt_h - 5deg*calm_positive_mood
  r (follow-through retention) = 0.9 - 0.05*sleep_debt_h - 0.03*fatigue_level
Run: python3 ignition.py -> ignition_results.json
"""
import json
import numpy as np

def M(theta, n, D=0, phi=0.0):
    return np.cos(theta / n) ** (2 * n) * np.cos(phi) ** (2 * D)

def n_min(theta, B, D=0, phi=0.0, nmax=200):
    for n in range(1, nmax + 1):
        if M(theta, n, D, phi) * B >= 1:
            return n
    return None                                  # no-ignition zone

OUT = {}
tab = {}
for deg in (30, 45, 60, 75, 90):
    tab[deg] = {B: n_min(np.radians(deg), B) for B in (1.1, 1.5, 2.0, 4.0)}
OUT["base_n_min"] = tab
print("n_min (steps+1) by dream-action angle x environment multiplier B")
for deg, row in tab.items():
    print(f"  {deg:2d} deg:", row)

OUT["interruptions"] = {}
for D in (0, 5, 20, 60):
    eff = 2.0 * np.cos(np.radians(10)) ** (2 * D)
    OUT["interruptions"][D] = dict(effective_B=float(eff), n_min_60deg=n_min(np.radians(60), 2.0, D, np.radians(10)))
    print(f"  interruptions D={D:2d} (10 deg each): effective B={eff:.3f}, n_min(60 deg)={OUT['interruptions'][D]['n_min_60deg']}")

profiles = {
    "office worker, phone always on":        (60, 1.6, 0.5, 1.0, 0.0, 0.5, 30),
    "student before exams":                  (55, 1.4, 1.0, 1.5, 0.0, 0.4, 25),
    "content creator (high reach)":          (50, 3.0, 0.5, 1.0, 0.3, 0.6, 40),
    "early-stage founder":                   (70, 2.5, 0.8, 1.5, 0.2, 0.8, 20),
    "gig worker, two jobs":                  (65, 1.3, 0.8, 2.0, 0.0, 1.0, 15),
    "dreamer (vision only)":                 (88, 1.5, 0.0, 0.5, 1.0, 0.2, 20),
}
interventions = {
    "none": dict(),
    "sleep 7.5h": dict(sleep=0.0),
    "phone-free focus blocks": dict(D_mult=0.2),
    "calm practice": dict(anx_mult=0.5, calm_add=0.3),
    "better environment (b +50%)": dict(b_mult=1.5),
    "all four": dict(sleep=0.0, D_mult=0.2, anx_mult=0.5, calm_add=0.3, b_mult=1.5),
}
OUT["profiles"] = {}
for pn, (th, b, anx, debt, calm, fat, D) in profiles.items():
    OUT["profiles"][pn] = {}
    for iname, iv in interventions.items():
        a = anx * iv.get("anx_mult", 1); d = iv.get("sleep", debt); c = min(1, calm + iv.get("calm_add", 0))
        th_eff = np.radians(min(89.9, th + 8 * a + 6 * d - 5 * c))
        r = 0.9 - 0.05 * d - 0.03 * fat
        B = b * iv.get("b_mult", 1) * r
        DD = int(round(D * iv.get("D_mult", 1)))
        nm = n_min(th_eff, B, DD, np.radians(10))
        k4 = M(th_eff, 4, DD, np.radians(10)) * B
        OUT["profiles"][pn][iname] = dict(theta_eff_deg=float(np.degrees(th_eff)), B=float(B), D=DD,
                                          zone=bool(B * np.cos(np.radians(10)) ** (2 * DD) <= 1),
                                          n_min=nm, k_with_3_step_plan=float(k4))
    print(f"\n{pn}")
    for iname, v in OUT["profiles"][pn].items():
        print(f"   {iname:28s} theta={v['theta_eff_deg']:5.1f} B={v['B']:.2f} D={v['D']:2d} "
              f"{'NO-IGNITION ZONE' if v['zone'] else 'n_min=' + str(v['n_min']):18s} k(3-step)={v['k_with_3_step_plan']:.2f}")
json.dump(OUT, open("ignition_results.json", "w"), indent=1, default=str)
