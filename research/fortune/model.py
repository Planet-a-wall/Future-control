"""
Fortune dynamics (CIOT v3) - agent-based model, monthly steps.

Each agent has wealth/status capital K (units: one median annual income),
traits drawn at birth, and mental state variables that evolve:

  v      vision / information quality about the world (0-1)
  q      drive = phi * q_intr + (1-phi) * q_extr(K / goal)
  phi    intrinsic share of motivation ("enjoyment of the work itself")
  c      confidence multiplier on perceived edge (1 = calibrated)
  iota   integrity (0-1)
  omega  giving rate -> reputation capital R ("merit" built in this life)

Investment rule (fractional Kelly with perceived edge):
  mu    = mu_max * q * v  (+ merit opportunity premium)
  f     = kelly_frac * c * mu / sigma^2        (capped)
  K'    = K + K*(r + f*(X - r)) + income - consumption - giving

Feature flags switch mechanisms on/off so each study can add one at a time:
  ratchet    consumption habit (lifestyle inflation is sticky downward)
  goal_drive extrinsic drive collapses once K passes the personal goal
  hubris     self-attribution: wins raise c more than losses lower it;
             status weakens correction
  loss_risk  drawdown from peak -> risk seeking (prospect theory)
  fraud      drawdown + low integrity -> fraud -> detection -> jail
  merit      reputation capital: opportunity premium + rescue near ruin
"""
import numpy as np
from scipy.special import erf as _erf

DEFAULT = dict(
    T=480, r=0.0033, sigma=0.10, mu_max=0.03, kelly_frac=0.5, f_cap=2.5,
    income_base=0.5, save_wealth_rate=0.003, cons_income=0.85, habit_decay=0.004,
    beta=0.04, beta_neg=0.03, gamma=0.02, psi=0.5, K_med=2.0, ego_gain=3.0, c_cap=6.0,
    goal_med=30.0, goal_sd=0.8, q_extr_free=0.7,
    lam_loss=1.5, fraud_pi=0.01, fraud_det=0.02,
    y_R=0.15, R_half=5.0, fb_gain=2.0, rho_R=0.01, rescue_p=0.6,
    great=100.0, cap_tax=0.0, ret_prem=0.0, sat_k=3.0, hedon_gain=2.0,
    # --- mood module (multi-timescale mental state)
    mood_tau=6.0, mood_vol=0.5, mood_eta=0.15, mood_pov=0.10, a_q=0.3, a_c=0.3, a_v=0.1,
    spree=0.5, a_f=1.0, day_sd=0.6, days=22, theta_hi=2.0, theta_lo=-2.0, imp_act=0.3, imp_mult=1.0,
    dep_th=-1.2, elev_th=1.5, bipolar_share=0.03, inertia=0.0,
    # --- sleep and fatigue module
    sleep_mean=7.3, sleep_sd=0.8, k_work=1.5, k_elev=0.7, k_stress=1.0, sleep_need=7.0,
    rho_fat=0.15, omega_fat=0.6, debt_rec=0.25, debt_mood=0.04, debt_day=0.15, debt_imp=0.2, debt_v=0.04,
    protect_floor=7.5, breaks_gain=1.3,
)
ALL_OFF = dict(ratchet=False, goal_drive=False, hubris=False, loss_risk=False,
               fraud=False, merit=False, merit_feedback=False,
               mood=False, precommit=False, regulate=False, delay=False,
               sleep=False, sleep_protect=False, breaks=False)


def sleep_traits(N, rng, P=None, **over):
    """Habitual sleep duration (hours/night)."""
    P = {**DEFAULT, **(P or {})}
    t = dict(S_base=np.clip(rng.normal(P["sleep_mean"], P["sleep_sd"], N), 4.5, 9.5))
    for k, val in over.items():
        t[k] = np.broadcast_to(np.asarray(val, float), (N,)).copy()
    return t


def mood_traits(N, rng, P=None, **over):
    """Individual mood dynamics: set point, slow volatility and inertia, daily volatility, reactivity."""
    P = {**DEFAULT, **(P or {})}
    t = dict(m_set=rng.normal(0, 0.3, N),
             m_vol=P["mood_vol"] * np.exp(rng.normal(0, 0.35, N)),
             m_tau=P["mood_tau"] * np.exp(rng.normal(0, 0.3, N)),
             d_sd=P["day_sd"] * np.exp(rng.normal(0, 0.3, N)),
             m_eta=P["mood_eta"] * np.exp(rng.normal(0, 0.4, N)))
    bip = rng.random(N) < P["bipolar_share"]
    t["m_vol"][bip] = 1.1; t["m_tau"][bip] = 4.0; t["bipolar"] = bip.astype(float)
    for k, val in over.items():
        t[k] = np.broadcast_to(np.asarray(val, float), (N,)).copy()
    return t


def traits(N, rng, **over):
    t = dict(
        v=rng.beta(2, 3, N), q_intr=rng.beta(2, 2, N), phi=rng.beta(2, 2, N),
        iota=rng.beta(4, 2, N), omega=rng.beta(1.2, 12, N), ego=rng.beta(1.5, 4, N),
        c0=np.exp(rng.normal(0, 0.2, N)), goal=None,
    )
    t["goal"] = DEFAULT["goal_med"] * np.exp(rng.normal(0, DEFAULT["goal_sd"], N))
    for k, val in over.items():
        t[k] = np.broadcast_to(np.asarray(val, float), (N,)).copy() if np.ndim(val) == 0 else np.asarray(val, float)
    return t


def simulate(K0, tr, flags, seed=0, R0=0.0, P=None, windfall=None, record=False):
    P = {**DEFAULT, **(P or {})}
    fl = {**ALL_OFF, **flags}
    rng = np.random.default_rng(seed)
    N = len(K0)
    K = np.asarray(K0, float).copy()
    R = np.broadcast_to(np.asarray(R0, float), (N,)).copy()
    c = tr["c0"].copy()
    peak = K.copy()
    income = (P["income_base"] + tr["v"]) / 12.0
    H = np.zeros(N)
    alive = np.ones(N, bool)          # not in jail
    exposed = np.zeros(N, bool)
    n_bankrupt = np.zeros(N, int); t_first_bankrupt = np.full(N, -1)
    jailed_t = np.full(N, -1)
    ever_great = np.zeros(N, bool); t_great = np.full(N, -1)
    rescued = np.zeros(N, int); frauds = np.zeros(N, int)
    peak_c = c.copy(); max_lev = np.zeros(N)
    q_at_goal = np.full(N, np.nan); q_after_goal = np.full(N, np.nan); t_goal = np.full(N, -1)
    lev_hist, ret_hist = [], []
    traj = [] if record else None
    use_mood = fl["mood"]
    if use_mood:
        rng_m = np.random.default_rng(seed + 999)          # separate stream: common random returns
        m_vol = tr["m_vol"] * (0.7 if fl["regulate"] else 1.0)
        d_sd = tr["d_sd"] * (0.7 if fl["regulate"] else 1.0)
        m_eta = tr["m_eta"] * (0.5 if fl["regulate"] else 1.0)
        a_ar = np.exp(-1.0 / tr["m_tau"])
        m = tr["m_set"] + m_vol * rng_m.standard_normal(N)
        in_dep = np.zeros(N, bool); run = np.zeros(N, int); dep_durs = []
        n_dep = np.zeros(N, int); months_dep = np.zeros(N, int); months_elev = np.zeros(N, int)
        ever_elev = np.zeros(N, bool); imp_hi = np.zeros(N, int); imp_lo = np.zeros(N, int)
        m_sum = np.zeros(N); prev_ret = np.zeros(N)
    use_sleep = fl["sleep"]
    if use_sleep:
        fat = np.zeros(N); debt = np.zeros(N); debt_sum = np.zeros(N); fat_sum = np.zeros(N)
        short_months = np.zeros(N, int)
        f_ref = P["rho_fat"] * 0.5 / (P["rho_fat"] * 0.5 + P["omega_fat"])
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    for t in range(P["T"]):
        if windfall is not None and t == 0:
            K = K + windfall
            peak = np.maximum(peak, K)
        # --- drive
        if fl["goal_drive"]:
            q_extr = 1.0 / (1.0 + np.exp(-P["sat_k"] * (1.0 - np.maximum(K, 0) / tr["goal"])))
        else:
            q_extr = np.full(N, P["q_extr_free"])
        q = tr["phi"] * tr["q_intr"] + (1 - tr["phi"]) * q_extr
        # record drive around goal attainment
        hit = (t_goal < 0) & (K >= tr["goal"])
        t_goal[hit] = t; q_at_goal[hit] = q[hit]
        later = (t_goal >= 0) & (t == t_goal + 60)
        q_after_goal[later] = q[later]
        # --- mood: slow AR(1) with outcome feedback and poverty stress; daily extremes
        if use_mood:
            push = m_eta * np.tanh(prev_ret / 0.05) - P["mood_pov"] * (K < 0.3)
            if use_sleep:
                push = push - P["debt_mood"] * debt
            # emotional inertia: dynamics slow down inside the low-mood region (Kuppens et al. 2010)
            a_eff = np.where(m < P["dep_th"] + 0.5, np.exp(-1.0 / (tr["m_tau"] * (1 + P["inertia"]))), a_ar)
            m = tr["m_set"] + a_eff * (m - tr["m_set"]) + push + m_vol * np.sqrt(1 - a_eff ** 2) * rng_m.standard_normal(N)
            m = np.clip(m, -4, 4)
            d_sd_t = d_sd * (1 + P["debt_day"] * debt) if use_sleep else d_sd
            zhi = (P["theta_hi"] - m) / (d_sd_t * np.sqrt(2)); zlo = (m - P["theta_lo"]) / (d_sd_t * np.sqrt(2))
            ph = 0.5 * (1 - _erf(zhi)); pl = 0.5 * (1 - _erf(zlo))
            any_hi = rng_m.random(N) < 1 - (1 - ph) ** P["days"]
            any_lo = rng_m.random(N) < 1 - (1 - pl) ** P["days"]
            act = 0.0 if fl["delay"] else P["imp_act"] * ((1 + P["debt_imp"] * debt) if use_sleep else 1.0)
            hi_act = alive & any_hi & (rng_m.random(N) < act)
            lo_act = alive & any_lo & (rng_m.random(N) < act)
            imp_hi += hi_act; imp_lo += lo_act
            q = np.clip(q * (1 + P["a_q"] * m), 0.0, 1.0)
            # episodes
            dep_now = m < P["dep_th"]; elev_now = m > P["elev_th"]
            onset = dep_now & ~in_dep; n_dep += onset
            ended = in_dep & ~dep_now
            if ended.any():
                dep_durs.extend(run[ended].tolist())
            run = np.where(dep_now, run + 1, 0); in_dep = dep_now
            months_dep += dep_now; months_elev += elev_now; ever_elev |= elev_now; m_sum += m
        # --- sleep and fatigue
        if use_sleep:
            dd_s = np.where(peak > 5.0, 1.0 - np.maximum(K, 0) / peak, 0.0)
            S_t = tr["S_base"] - P["k_work"] * np.maximum(0, q - 0.6) - P["k_stress"] * np.clip(dd_s, 0, 1)
            if use_mood:
                S_t = S_t - P["k_elev"] * np.maximum(0, m - 1.0)
            if fl["sleep_protect"]:
                S_t = np.maximum(S_t, P["protect_floor"])
            debt = 0.6 * debt + 0.4 * np.maximum(0, P["sleep_need"] - S_t)
            om = P["omega_fat"] * np.clip(1 - P["debt_rec"] * debt, 0.2, 1.0) * (P["breaks_gain"] if fl["breaks"] else 1.0)
            fat = np.clip(fat + P["rho_fat"] * q * (1 - fat) - om * fat, 0, 0.95)
            q = np.clip(q * (1 - fat) / (1 - f_ref), 0, 1.0)
            debt_sum += debt; fat_sum += fat; short_months += S_t < 7.0
        # --- edge and leverage
        v_eff = tr["v"] * (np.clip(1 - P["a_v"] * m ** 2, 0.3, 1.0) if use_mood else 1.0)
        if use_sleep:
            v_eff = v_eff * np.clip(1 - P["debt_v"] * debt, 0.5, 1.0)
        mu = P["mu_max"] * q * v_eff
        c_dec = c * (np.exp(P["a_c"] * m) if (use_mood and not fl["precommit"]) else 1.0)
        f = P["kelly_frac"] * c_dec * mu / P["sigma"] ** 2
        dd = np.where(peak > 5.0, 1.0 - np.maximum(K, 0) / peak, 0.0)
        if fl["loss_risk"]:
            f = f * (1.0 + P["lam_loss"] * dd * (1 - tr["phi"]))
        if use_mood and not fl["precommit"]:
            f = np.where(hi_act, f * (1 + P["imp_mult"]), f)
            f = np.where(lo_act, 0.0, f)
        f = np.clip(f, 0, P["f_cap"])
        f = np.where(K > 0, f, 0.0)
        # --- returns
        z = rng.standard_normal(N)
        X = P["r"] + mu + P["sigma"] * z
        invest = K * (P["r"] + f * (X - P["r"]))
        if P["ret_prem"] > 0:        # returns rise with wealth (Fagereng et al. 2020): access, not skill
            invest = invest + K * P["ret_prem"] * np.clip(np.log1p(np.maximum(K, 0) / 10) / np.log(101), 0, 1)
        if P["cap_tax"] > 0:          # tax on positive capital income
            invest = np.where(invest > 0, invest * (1 - P["cap_tax"]), invest)
        net_income = income + (P["y_R"] * R / (R + P["R_half"]) / 12.0 if fl["merit"] else 0.0)
        cons = P["cons_income"] * income + P["save_wealth_rate"] * np.maximum(K, 0) * (1 + P["hedon_gain"] * tr.get("hedon", 0.0))
        if fl["ratchet"]:
            # habit: spending can rise at once but falls back only at rate habit_decay
            H = np.maximum(cons, H * (1 - P["habit_decay"]))
            cons = H
        if use_mood:
            cons = cons + P["spree"] * np.maximum(m - 1.0, 0) * income + np.where(hi_act, P["spree"] * income, 0.0)
        give = tr["omega"] * (income + P["save_wealth_rate"] * np.maximum(K, 0)) if fl["merit"] else 0.0
        if use_mood:
            prev_ret = np.where(K > 0, invest / np.maximum(K, 1e-9), 0.0)
        K = np.where(alive, K + invest + net_income - cons - give, 0.0)
        if fl["merit"]:
            R = np.where(alive, R + give - P["rho_R"] * R, 0.0)
        # --- confidence (self-attribution)
        if fl["hubris"]:
            status = np.maximum(0, np.log(np.maximum(K, 1e-9) / P["K_med"]))
            s_sat = 1.0 - np.exp(-status / 3.0)          # power effect saturates
            b_eff = P["beta"] * (1 + P["ego_gain"] * tr["ego"]) * (1 + P["psi"] * s_sat)
            g_eff = P["gamma"] * (0.5 + tr["iota"]) / (1 + P["psi"] * s_sat)
            if fl.get("merit_feedback"):
                # hypothesis H-merit: reputation buys honest feedback that corrects confidence
                g_eff = g_eff * (1 + P["fb_gain"] * R / (R + P["R_half"]))
            sig = np.where(f > 0, np.where(z > 0, b_eff * z, P["beta_neg"] * z), 0.0)
            c = np.clip(c + sig + g_eff * (1 - c), 0.3, P["c_cap"])
        # --- fraud and jail
        if fl["fraud"]:
            pf = P["fraud_pi"] * (1 - tr["iota"]) ** 2 * (1 - tr["phi"]) * np.maximum(0, dd - 0.3)
            if use_mood:
                pf = pf * (1 + P["a_f"] * np.maximum(-m, 0))
            new = alive & ~exposed & (peak > 5.0) & (rng.random(N) < pf)
            K = np.where(new, K + 0.3 * (peak - K), K)
            exposed |= new; frauds += new
            det = alive & exposed & (rng.random(N) < P["fraud_det"])
            jailed_t[det] = t; alive &= ~det
            K = np.where(det, 0.0, K); R = np.where(det, 0.0, R)
        # --- ruin, rescue
        ruin = alive & (K < 0)
        if fl["merit"]:
            res = ruin & (rng.random(N) < P["rescue_p"] * R / (R + P["R_half"]))
            K = np.where(res, 0.5, K); R = np.where(res, R * 0.5, R); rescued += res
            ruin &= ~res
        first = ruin & (t_first_bankrupt < 0)
        t_first_bankrupt[first] = t
        n_bankrupt += ruin
        K = np.where(ruin, 0.0, K)
        if not fl.get("repeat"):      # learning after ruin; 'repeat' = repetition compulsion (Freud)
            c = np.where(ruin, 1.0, c)
        H = np.where(ruin, P["cons_income"] * income, H)
        peak = np.where(ruin, 0.0, np.maximum(peak, K))
        g_now = alive & (K >= P["great"]) & ~ever_great
        t_great[g_now] = t; ever_great |= g_now
        peak_c = np.maximum(peak_c, c); max_lev = np.maximum(max_lev, f)
        if t % 12 == 0:
            lev_hist.append(f.copy()); ret_hist.append(K.copy())
        if record:
            traj.append(K.copy())
    mood_out = {}
    if use_mood:
        if in_dep.any():
            dep_durs.extend(run[in_dep].tolist())
        mood_out = dict(n_dep=n_dep, months_dep=months_dep, months_elev=months_elev, ever_elev=ever_elev,
                        dep_durs=np.array(dep_durs), imp_hi=imp_hi, imp_lo=imp_lo, m_mean=m_sum / P["T"])
    if use_sleep:
        mood_out.update(debt_mean=debt_sum / P["T"], fat_mean=fat_sum / P["T"], short_share=short_months / P["T"])
    return dict(**mood_out, K=K, peak=peak, alive=alive, jailed_t=jailed_t, n_bankrupt=n_bankrupt,
                t_first_bankrupt=t_first_bankrupt, ever_great=ever_great, t_great=t_great,
                rescued=rescued, frauds=frauds, peak_c=peak_c, max_lev=max_lev, c=c, R=R,
                q_at_goal=q_at_goal, q_after_goal=q_after_goal, t_goal=t_goal,
                lev=np.array(lev_hist), wealth_yr=np.array(ret_hist),
                traj=np.array(traj) if record else None)
