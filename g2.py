
import numpy as np
EPS = 1e-9   # tiny constant to avoid divide-by-zero in every normalisation


# =============================================================================
# ENGINE 1 - GPTCore : adaptive online learner (the high-score engine)
# =============================================================================
class GPTCore:
    def __init__(self, nInst=51):
        # ---- market / position limits ----
        self.nInst = nInst
        self.dlrLimit = np.full(nInst, 10_000.0); self.dlrLimit[0] = 100_000.0  # ALGO cap = 10x

        # ---- lead-lag signal (single long rolling window) ----
        self.LL_W, self.LL_HL, self.CORR_THRESH = 240, 120.0, 0.02
        #   LL_W=240    : days of history used to estimate the lead-lag matrix
        #   LL_HL=120   : EWMA half-life (recent days weighted more heavily)
        #   CORR_THRESH : zero any matrix entry weaker than this (noise filter)

        # ---- factor lead-lag (short) ----
        self.FACTOR_W, self.FACTOR_K = 120, 30      # window, # principal factors

        # ---- residual-ridge lead-lag (BIGGEST single edge) ----
        self.NEW_RIDGE_WINDOW, self.NEW_RIDGE_HL = 240, 120.0
        self.NEW_RIDGE_LAMBDA, self.RESID_K = 10.0, 3
        #   NEW_RIDGE_LAMBDA : ridge regularisation strength (higher = smoother)
        #   RESID_K=3        : # common factors removed before regressing

        # ---- factor lead-lag (long) ----
        self.FACTOR_LONG_W, self.FACTOR_LONG_K, self.FACTOR_LONG_RIDGE = 240, 30, 10.0

        # ---- how the 4 "base" forecasts are mixed (online-adapted around this prior) ----
        self.BLEND4 = np.array((0.45, 0.15, 0.15, 0.25))   # [lead-lag, factor, ridge, factor-long]
        self.ONLINE4_ETA, self.ONLINE4_BLEND = 0.20, 0.35  # adaptation speed / how far from prior
        self.ONLINE4_MIN, self.ONLINE4_FLOOR, self.ONLINE4_CAP = 12, 0.05, 0.60

        # ---- RLS: recursive least-squares adaptive lead-lag matrix (KEY component) ----
        self.RLS_WEIGHT = 0.15                               # its weight in the final blend
        self.RLS_LAM_SLOW, self.RLS_LAM_FAST = 0.995, 0.970  # slow=stable / fast=adaptive forgetting
        self.RLS_RIDGE = 10.0                               # initial regularisation of the matrix
        self.RLS_FAST_BLEND = 0.25                          # base mix of fast vs slow learner
        self.RLS_INIT = 240                                 # days used to warm-start the matrix
        self.RLS_COMP_ETA, self.RLS_COMP_FAST_HL, self.RLS_COMP_SLOW_HL = 2.0, 12.0, 70.0
        self.RLS_COMP_MIN, self.RLS_COMP_MAX = 0.20, 0.45   # bounds on the adaptive fast/slow mix

        # ---- FRLS: reduced-rank (factor-space) version of the RLS learner ----
        self.FRLS_WEIGHT = 0.02                             # small weight in final blend
        self.FRLS_KS = (12, 16)                             # two factor counts, blended
        self.FRLS_KBLEND = np.array((0.3, 0.7))
        self.FRLS_LAM_SLOW, self.FRLS_LAM_FAST = 0.995, 0.970
        self.FRLS_RIDGE, self.FRLS_FAST_BLEND = 8.0, 0.35

        # ---- sizing / risk ----
        self.VOL_HL = 40.0            # half-life for per-instrument volatility estimate
        self.N_ACTIVE = 26.0          # ~how many names get meaningful size (rank centre)
        self.RANK_SOFT = 2.0          # softness of the rank cutoff (higher = smoother, less churn)
        self.INVVOL_POWER = 0.0       # 0 = volatility-normalisation OFF (tested: hurt score here)
        self.INVVOL_CAP = 2.5
        self.ALGO_MULT = 1.0          # 1.0 = no special ALGO boost (tested: neutral)
        self.NEUTRALIZE = 1.0         # 1.0 = fully strip net $ exposure (market-neutral)
        self.SMOOTH = 0.05            # light EMA on positions (day-to-day stability)
        self.NO_TRADE = 0.01          # no-trade band: ignore tiny position changes (saves commission)
        self.BOOTSTRAP_DAYS = 60      # on first call, replay this many days to warm the online state
        self._reset_state()

    def _reset_state(s):
        # Initialise every piece of persistent (day-to-day) state to empty/zero.
        n = s.nInst
        s._prevPos = np.zeros(n)
        s._rlsBSlow = None; s._rlsPSlow = None; s._rlsBFast = None; s._rlsPFast = None; s._rlsLast = 0
        s._currentRLSSlow = None; s._currentRLSFast = None
        s._rlsCompFM = np.zeros(2); s._rlsCompFV = np.ones(2); s._rlsCompSM = np.zeros(2); s._rlsCompSV = np.ones(2); s._rlsCompCount = 0
        s._prevRLSSlowT = None; s._olderRLSSlowT = None; s._prevRLSFastT = None; s._olderRLSFastT = None
        s._frls = {}; s._currentFRLS = None
        s._frlsQFM = None; s._frlsQFV = None; s._frlsQSM = None; s._frlsQSV = None; s._frlsQCount = 0
        s._prevFRLST = None; s._olderFRLST = None
        s._o4FM = np.zeros(4); s._o4FV = np.full(4, 1e-4); s._o4SM = np.zeros(4); s._o4SV = np.full(4, 1e-4); s._o4Count = 0
        s._prev4Sig = None; s._prev4T = None; s._older4T = None
        s._bootstrapped = False

    # ---- generic helpers ----
    @staticmethod
    def _zc(v):
        # Cross-sectional z-score: centre & scale a vector to mean 0, std 1.
        sd = v.std(); return (v - v.mean()) / (sd + EPS) if sd > EPS else v * 0.0

    @staticmethod
    def _ewma_std(rets, hl):
        # Exponentially-weighted volatility per instrument (recent days weighted more).
        T = rets.shape[1]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(T)[::-1]; w /= w.sum()
        mu = (rets * w).sum(1, keepdims=True); var = ((rets - mu) ** 2 * w).sum(1); return np.sqrt(var) + EPS

    def _target_dollars(s, sig):
        # Turn a raw forecast into the dollar book it *would* produce on its own
        # (used only to score each sub-model's realised quality for adaptation).
        z = s._zc(sig); order = np.argsort(-np.abs(z)); ranks = np.empty(s.nInst, int); ranks[order] = np.arange(s.nInst)
        rm = 1.0 / (1.0 + np.exp((ranks - s.N_ACTIVE) / s.RANK_SOFT)); return np.sign(z) * rm * s.dlrLimit

    @staticmethod
    def _payoff(prevT, realized, olderT):
        # Realised commission-free return of a target book against the day's returns.
        if prevT is None: return 0.0
        return float(np.sum(prevT * realized)) / (float(np.sum(np.abs(prevT))) + EPS)

    # ---- signal families ----
    def _leadlag(s, rets):
        # Cross-asset lead-lag: EWMA lag-1 cross-correlation matrix (diagonal removed,
        # weak entries thresholded), applied to today's returns to forecast tomorrow.
        w, hl = s.LL_W, s.LL_HL; T = rets.shape[1]
        if T < w + 1: return None
        r = rets[:, -w:]; Tw = r.shape[1]; lam = 0.5 ** (1.0 / hl); ww = lam ** np.arange(Tw - 1)[::-1]; ww /= ww.sum()
        X, Y = r[:, :-1], r[:, 1:]
        xmu = (X * ww).sum(1, keepdims=True); xsd = np.sqrt(((X - xmu) ** 2 * ww).sum(1)) + EPS
        ymu = (Y * ww).sum(1, keepdims=True); ysd = np.sqrt(((Y - ymu) ** 2 * ww).sum(1)) + EPS
        Xs = (X - xmu) / xsd[:, None]; Ys = (Y - ymu) / ysd[:, None]
        LL = ((Xs * ww) @ Ys.T).T; np.fill_diagonal(LL, 0.0)
        if s.CORR_THRESH > 0: LL[np.abs(LL) < s.CORR_THRESH] = 0.0
        return LL @ ((r[:, -1] - xmu[:, 0]) / xsd)

    def _resid_ridge(s, rets):
        # Multi-output RIDGE regression lead-lag AFTER removing the top common
        # factors - isolates idiosyncratic predictability. (Biggest single edge.)
        W = s.NEW_RIDGE_WINDOW
        if rets.shape[1] < W: return np.zeros(s.nInst)
        r = rets[:, -W:]; r0 = r - r.mean(1, keepdims=True)
        if s.RESID_K > 0:
            _, vec = np.linalg.eigh(np.cov(r0)); V = vec[:, -min(s.RESID_K, vec.shape[1] - 1):]; r0 = r0 - V @ (V.T @ r0)
        X, Y = r0[:, :-1].T, r0[:, 1:].T
        w = 0.5 ** (np.arange(X.shape[0])[::-1] / s.NEW_RIDGE_HL); w /= w.sum()
        xmu = (X * w[:, None]).sum(0); ymu = (Y * w[:, None]).sum(0)
        xsd = np.sqrt(((X - xmu) ** 2 * w[:, None]).sum(0)) + EPS; ysd = np.sqrt(((Y - ymu) ** 2 * w[:, None]).sum(0)) + EPS
        Xs = (X - xmu) / xsd; Ys = (Y - ymu) / ysd; sw = np.sqrt(w)[:, None]; Xw, Yw = Xs * sw, Ys * sw
        coef = np.linalg.solve(Xw.T @ Xw + s.NEW_RIDGE_LAMBDA * np.eye(Xw.shape[1]), Xw.T @ Yw); np.fill_diagonal(coef, 0.0)
        return ((r0[:, -1] - xmu) / xsd) @ coef

    def _factor(s, rets, window, k, ridge):
        # Factor lead-lag: regress each asset's return on LAGGED principal-factor
        # returns (common-factor propagation), ridge-regularised.
        if rets.shape[1] < window + 1: return np.zeros(s.nInst)
        r = rets[:, -window:]; r0 = r - r.mean(1, keepdims=True)
        _, vec = np.linalg.eigh(np.cov(r0)); V = vec[:, -min(k, vec.shape[1]):]; F = V.T @ r0
        Ft, Yt = F[:, :-1].T, r0[:, 1:].T; Fmu, Fsd = Ft.mean(0), Ft.std(0) + EPS; Z = (Ft - Fmu) / Fsd
        B = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ Yt); return ((F[:, -1] - Fmu) / Fsd) @ B

    # ---- RLS adaptive learner ----
    @staticmethod
    def _rls_step(B, P, x, y, forget):
        # One recursive-least-squares update of coefficient matrix B (and its
        # covariance P) given input x -> target y, with an exponential forgetting
        # factor. This is the Kalman-style online update that lets the model TRACK
        # a drifting relationship instead of re-fitting a frozen one.
        px = P @ x; den = forget + float(x @ px); gain = px / (den + EPS); err = y - x @ B
        B = B + gain[:, None] * err[None, :]; P = (P - np.outer(gain, x @ P)) / forget; P = 0.5 * (P + P.T); return B, P

    def _rls(s, rets):
        # Maintain TWO live 51x51 prediction matrices (slow + fast forgetting),
        # updated one new day at a time, and blend their forecasts for today.
        T = rets.shape[1]
        if T < 65: return np.zeros(s.nInst)
        z = np.apply_along_axis(s._zc, 0, rets)
        if s._rlsBSlow is None:                    # first call: warm-start over RLS_INIT days
            s._rlsBSlow = np.zeros((s.nInst, s.nInst)); s._rlsBFast = np.zeros((s.nInst, s.nInst))
            s._rlsPSlow = np.eye(s.nInst) / s.RLS_RIDGE; s._rlsPFast = np.eye(s.nInst) / s.RLS_RIDGE
            for t in range(max(0, T - 1 - s.RLS_INIT), T - 1):
                x, y = z[:, t], z[:, t + 1]
                s._rlsBSlow, s._rlsPSlow = s._rls_step(s._rlsBSlow, s._rlsPSlow, x, y, s.RLS_LAM_SLOW)
                s._rlsBFast, s._rlsPFast = s._rls_step(s._rlsBFast, s._rlsPFast, x, y, s.RLS_LAM_FAST)
            s._rlsLast = T
        elif T > s._rlsLast:                        # later calls: update with the newly-seen day(s)
            for yi in range(s._rlsLast, T):
                xi = yi - 1
                if xi >= 0:
                    x, y = z[:, xi], z[:, yi]
                    s._rlsBSlow, s._rlsPSlow = s._rls_step(s._rlsBSlow, s._rlsPSlow, x, y, s.RLS_LAM_SLOW)
                    s._rlsBFast, s._rlsPFast = s._rls_step(s._rlsBFast, s._rlsPFast, x, y, s.RLS_LAM_FAST)
            s._rlsLast = T
        np.fill_diagonal(s._rlsBSlow, 0.0); np.fill_diagonal(s._rlsBFast, 0.0)
        x = z[:, -1]; slow = s._zc(x @ s._rlsBSlow); fast = s._zc(x @ s._rlsBFast)
        s._currentRLSSlow = slow.copy(); s._currentRLSFast = fast.copy()
        fw = s._rls_comp_fast_weight(); return (1 - fw) * slow + fw * fast

    def _rls_comp_fast_weight(s):
        # Adaptively decide how much to trust the FAST vs SLOW RLS learner, based
        # on which has been converting to PnL recently (bounded so neither dominates).
        prior = float(s.RLS_FAST_BLEND)
        if s.RLS_COMP_ETA <= 0 or s._rlsCompCount < 10: return prior
        qf = s._rlsCompFM / np.sqrt(s._rlsCompFV + 1e-5); qs = s._rlsCompSM / np.sqrt(s._rlsCompSV + 1e-5); q = 0.6 * qf + 0.4 * qs
        diff = float(np.clip(q[1] - q[0], -2, 2)); shrink = s._rlsCompCount / (s._rlsCompCount + 30.0)
        p = np.clip(prior, 1e-4, 1 - 1e-4); logit = np.log(p / (1 - p)) + s.RLS_COMP_ETA * shrink * diff
        return float(np.clip(1 / (1 + np.exp(-logit)), s.RLS_COMP_MIN, s.RLS_COMP_MAX))

    def _rls_comp_update(s, realized):
        # Track realised quality of the slow vs fast RLS books (feeds the weight above).
        if s._prevRLSSlowT is None or s._prevRLSFastT is None: return
        obs = np.clip(np.array([s._payoff(s._prevRLSSlowT, realized, s._olderRLSSlowT),
                                s._payoff(s._prevRLSFastT, realized, s._olderRLSFastT)]), -0.05, 0.05)
        for mn, vn, hl in (('_rlsCompFM', '_rlsCompFV', s.RLS_COMP_FAST_HL), ('_rlsCompSM', '_rlsCompSV', s.RLS_COMP_SLOW_HL)):
            mean = getattr(s, mn); var = getattr(s, vn); a = 1 - 0.5 ** (1.0 / hl); old = mean.copy()
            mean = (1 - a) * mean + a * obs; var = (1 - a) * var + a * (obs - old) ** 2
            setattr(s, mn, mean); setattr(s, vn, np.maximum(var, 1e-5))
        s._rlsCompCount += 1

    # ---- FRLS reduced-rank factor learner ----
    def _frls_one(s, rets, k):
        # Same RLS idea but in a k-factor subspace (fewer parameters -> more stable).
        T = rets.shape[1]
        if T < 65: return np.zeros(s.nInst)
        z = np.apply_along_axis(s._zc, 0, rets); st = s._frls.get(int(k))
        if st is None:                              # warm-start this factor count
            init = z[:, -min(s.RLS_INIT, T):]; _, vec = np.linalg.eigh(np.cov(init)); V = vec[:, -min(int(k), s.nInst - 1):]; kk = V.shape[1]
            bs = np.zeros((kk, s.nInst)); bf = np.zeros((kk, s.nInst)); ps = np.eye(kk) / s.FRLS_RIDGE; pf = np.eye(kk) / s.FRLS_RIDGE
            for t in range(max(0, T - 1 - s.RLS_INIT), T - 1):
                x = V.T @ z[:, t]; y = z[:, t + 1]
                bs, ps = s._rls_step(bs, ps, x, y, s.FRLS_LAM_SLOW); bf, pf = s._rls_step(bf, pf, x, y, s.FRLS_LAM_FAST)
            st = {'V': V, 'bs': bs, 'bf': bf, 'ps': ps, 'pf': pf, 'last': T}; s._frls[int(k)] = st
        elif T > st['last']:                        # incremental update
            for yi in range(st['last'], T):
                xi = yi - 1
                if xi >= 0:
                    x = st['V'].T @ z[:, xi]; y = z[:, yi]
                    st['bs'], st['ps'] = s._rls_step(st['bs'], st['ps'], x, y, s.FRLS_LAM_SLOW)
                    st['bf'], st['pf'] = s._rls_step(st['bf'], st['pf'], x, y, s.FRLS_LAM_FAST)
            st['last'] = T
        x = st['V'].T @ z[:, -1]; return (1 - s.FRLS_FAST_BLEND) * s._zc(x @ st['bs']) + s.FRLS_FAST_BLEND * s._zc(x @ st['bf'])

    def _frls_signal(s, rets):
        # Blend the FRLS forecasts across the two factor counts (FRLS_KS).
        if s.FRLS_WEIGHT <= 0: return np.zeros(s.nInst)
        sigs = [s._zc(s._frls_one(rets, k)) for k in s.FRLS_KS]; s._currentFRLS = tuple(x.copy() for x in sigs)
        if len(sigs) == 1: return sigs[0]
        w = s.FRLS_KBLEND / s.FRLS_KBLEND.sum(); return np.tensordot(w, np.asarray(sigs), axes=(0, 0))

    def _frls_quality_update(s, realized):
        # Track realised quality of the FRLS sub-models (for their online weighting).
        if s._prevFRLST is None: return
        nn = len(s._prevFRLST)
        if s._frlsQFM is None or len(s._frlsQFM) != nn:
            s._frlsQFM = np.zeros(nn); s._frlsQFV = np.ones(nn); s._frlsQSM = np.zeros(nn); s._frlsQSV = np.ones(nn); s._frlsQCount = 0
        obs = np.clip(np.array([s._payoff(s._prevFRLST[i], realized, None if s._olderFRLST is None else s._olderFRLST[i]) for i in range(nn)]), -0.05, 0.05)
        for mn, vn, hl in (('_frlsQFM', '_frlsQFV', 15.0), ('_frlsQSM', '_frlsQSV', 80.0)):
            mean = getattr(s, mn); var = getattr(s, vn); a = 1 - 0.5 ** (1.0 / hl); old = mean.copy()
            mean = (1 - a) * mean + a * obs; var = (1 - a) * var + a * (obs - old) ** 2
            setattr(s, mn, mean); setattr(s, vn, np.maximum(var, 1e-5))
        s._frlsQCount += 1

    # ---- online mixing of the 4 base forecasts ----
    def _o4_update(s, realized):
        # Track realised quality of each of the 4 base experts (feeds their weights).
        if s._prev4T is None: return
        obs = np.array([s._payoff(s._prev4T[i], realized, None if s._older4T is None else s._older4T[i]) for i in range(4)]); obs = np.clip(obs, -0.05, 0.05)
        for mn, vn, hl in (('_o4FM', '_o4FV', 15.0), ('_o4SM', '_o4SV', 80.0)):
            mean = getattr(s, mn); var = getattr(s, vn); a = 1 - 0.5 ** (1.0 / hl); old = mean.copy()
            mean = (1 - a) * mean + a * obs; var = (1 - a) * var + a * (obs - old) ** 2
            setattr(s, mn, mean); setattr(s, vn, np.maximum(var, 1e-5))
        s._o4Count += 1

    def _o4_weights(s):
        # Blend weights for [lead-lag, factor, ridge, factor-long]: start from the
        # BLEND4 prior, tilt toward whichever experts have been working recently
        # (shrinkage-damped and floor/cap-bounded so no expert is zeroed or blows up).
        prior = s.BLEND4 / (s.BLEND4.sum() + EPS)
        if s.ONLINE4_ETA <= 0 or s._o4Count < s.ONLINE4_MIN: return prior
        qf = s._o4FM / np.sqrt(s._o4FV + 1e-5); qs = s._o4SM / np.sqrt(s._o4SV + 1e-5); q = 0.55 * qf + 0.45 * qs
        shrink = s._o4Count / (s._o4Count + 40.0); logits = np.log(prior + EPS) + s.ONLINE4_ETA * shrink * np.clip(q, -1.25, 1.25)
        logits -= logits.max(); ad = np.exp(logits); ad /= ad.sum(); ad = np.clip(ad, s.ONLINE4_FLOOR, s.ONLINE4_CAP); ad /= ad.sum()
        out = (1 - s.ONLINE4_BLEND) * prior + s.ONLINE4_BLEND * ad; return out / out.sum()

    # ---- the per-day core: compute all signals -> blend -> size -> book ----
    def _core(s, prc):
        n, T = prc.shape
        if T < 65: return np.zeros(n, dtype=int)         # not enough history yet
        rets = np.diff(np.log(prc), axis=1); cp = prc[:, -1]
        s._rls_comp_update(rets[:, -1]); s._frls_quality_update(rets[:, -1])  # learn from yesterday's outcome

        # 1) compute every forecast family (each z-scored to a common scale)
        llp = s._leadlag(rets); ll = s._zc(llp) if llp is not None else np.zeros(n)
        fac = s._zc(s._factor(rets, s.FACTOR_W, s.FACTOR_K, 10.0))
        ridge = s._zc(s._resid_ridge(rets))
        fac_long = s._zc(s._factor(rets, s.FACTOR_LONG_W, s.FACTOR_LONG_K, s.FACTOR_LONG_RIDGE))
        rls = s._zc(s._rls(rets)); frls = s._zc(s._frls_signal(rets))

        # 2) blend: online-weighted base four, then add RLS and FRLS at fixed weights
        s._o4_update(rets[:, -1]); bw = s._o4_weights()
        experts = np.asarray((ll, fac, ridge, fac_long)); base = bw @ experts
        rw = s.RLS_WEIGHT; fw = float(np.clip(s.FRLS_WEIGHT, 0, max(0, 1 - rw)))
        blend = (1 - rw - fw) * base + rw * rls + fw * frls
        s._prev4Sig = tuple(x.copy() for x in experts)

        # 3) rank names by conviction; give size via a SMOOTH rank cutoff (no hard cliff)
        z = s._zc(blend); order = np.argsort(-np.abs(z)); ranks = np.empty(n, int); ranks[order] = np.arange(n)
        rm = 1.0 / (1.0 + np.exp((ranks - s.N_ACTIVE) / s.RANK_SOFT)); raw = np.sign(z) * rm
        if s.INVVOL_POWER > 0:                            # optional volatility-normalisation (off)
            vol = s._ewma_std(rets, s.VOL_HL); iv = 1.0 / np.power(vol, s.INVVOL_POWER)
            iv = np.clip(iv / np.median(iv), 1.0 / s.INVVOL_CAP, s.INVVOL_CAP); raw *= iv
        raw[0] *= s.ALGO_MULT                             # optional ALGO tilt (currently 1.0)

        # 4) turbulence brake: shrink size when short-term market vol spikes vs its baseline
        scale = 1.0; mkt = rets.mean(0)
        if mkt.shape[0] >= 60:
            sv = mkt[-10:].std() + EPS; lv = mkt[-60:].std() + EPS
            scale *= float(np.clip(1.15 - 0.35 * max(sv / lv - 1, 0), 0.6, 1.15))

        # 5) convert to dollars, strip net exposure (market-neutral), apply the brake
        dollars = raw * s.dlrLimit
        if s.NEUTRALIZE > 0:
            net = float(np.sum(dollars)); gross = float(np.sum(np.abs(dollars))) + EPS
            dollars = dollars - s.NEUTRALIZE * net * (np.abs(dollars) / gross)
        dollars *= scale; shares = dollars / cp

        # 6) smooth vs yesterday, apply no-trade band, clip to per-instrument limits
        newPos = (1 - s.SMOOTH) * shares + s.SMOOTH * s._prevPos; lim = (s.dlrLimit / cp).astype(int)
        if s.NO_TRADE > 0:
            hold = np.abs(newPos - s._prevPos) < s.NO_TRADE * np.maximum(lim, 1); newPos[hold] = s._prevPos[hold]
        newPos = np.clip(newPos, -lim, lim); s._prevPos = newPos.copy()

        # 7) record each sub-model's would-be book so tomorrow can score it (adaptation)
        s._older4T = s._prev4T; s._prev4T = tuple(s._target_dollars(x) for x in experts)
        s._olderFRLST = s._prevFRLST; s._prevFRLST = tuple(s._target_dollars(x) for x in s._currentFRLS) if s._currentFRLS is not None else None
        s._olderRLSSlowT = s._prevRLSSlowT; s._prevRLSSlowT = s._target_dollars(s._currentRLSSlow) if s._currentRLSSlow is not None else None
        s._olderRLSFastT = s._prevRLSFastT; s._prevRLSFastT = s._target_dollars(s._currentRLSFast) if s._currentRLSFast is not None else None
        return newPos.astype(int)

    def getPosition(s, prc):
        # Public entry. On the FIRST call, replay recent history once so the online
        # learners (RLS/FRLS/expert weights) are warmed up even if the grader starts
        # the process fresh - then hand back to the real book (which starts flat).
        if not s._bootstrapped:
            nt = prc.shape[1]
            if s.BOOTSTRAP_DAYS > 0 and nt > 65:
                start = max(65, nt - s.BOOTSTRAP_DAYS)
                if nt > s.FACTOR_LONG_W + 1: start = max(start, s.FACTOR_LONG_W + 1)
                for t in range(start, nt): s._core(prc[:, :t])
                s._prevPos = np.zeros(s.nInst)
            s._bootstrapped = True
        return s._core(prc)


# =============================================================================
# ENGINE 2 - RobustCore : lean, low-parameter ensemble (the insurance leg)
# =============================================================================
class RobustCore:
    def __init__(s, nInst=51):
        s.nInst = nInst; s.dlrLimit = np.full(nInst, 10_000.0); s.dlrLimit[0] = 100_000.0
        s.LL_WINDOWS = ((90, 40), (120, 60), (180, 90), (240, 90))  # multi-window lead-lag (window, half-life)
        s.CORR_THRESH = 0.02                                        # weak-entry noise filter
        s.REV_LB = (3, 5, 8)                                        # reversal lookbacks (ensemble)
        s.FACTOR_W = 120; s.FACTOR_K = 3                            # factor lead-lag window / # factors
        s.VOL_HL = 40                                              # volatility half-life
        s.BASE_W = np.array((0.68, 0.24, 0.08))                    # prior weights: [lead-lag, reversal, factor]
        s.ADAPT_HL = 40                                           # half-life for performance-based reweighting
        s.N_ACTIVE = 22; s.RANK_SOFT = 3.0                         # smooth rank cutoff (how many names, how soft)
        s.INVVOL_CAP = 2.5; s.NEUTRALIZE = 1.0; s.SMOOTH = 0.1     # vol-norm cap / market-neutral / position EMA
        s._prevPos = np.zeros(nInst); s._pfd = None; s._perf = {'ll': [], 'rev': [], 'fac': []}

    @staticmethod
    def _zc(v):
        # Cross-sectional z-score.
        sd = v.std(); return (v - v.mean()) / (sd + EPS) if sd > EPS else v * 0.0

    @staticmethod
    def _ewma_std(rets, hl):
        # EWMA per-instrument volatility.
        T = rets.shape[1]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(T)[::-1]; w /= w.sum()
        mu = (rets * w).sum(1, keepdims=True); var = ((rets - mu) ** 2 * w).sum(1); return np.sqrt(var) + EPS

    def _ll_window(s, rets, window, hl):
        # One rolling EWMA lead-lag forecast (same idea as GPTCore, single window).
        T = rets.shape[1]
        if T < window + 2: window = T - 1
        r = rets[:, -window:]; Tw = r.shape[1]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(Tw - 1)[::-1]; w /= w.sum()
        X, Y = r[:, :-1], r[:, 1:]
        xmu = (X * w).sum(1, keepdims=True); xsd = np.sqrt(((X - xmu) ** 2 * w).sum(1)) + EPS
        ymu = (Y * w).sum(1, keepdims=True); ysd = np.sqrt(((Y - ymu) ** 2 * w).sum(1)) + EPS
        Xs = (X - xmu) / xsd[:, None]; Ys = (Y - ymu) / ysd[:, None]; LL = ((Xs * w) @ Ys.T).T; np.fill_diagonal(LL, 0.0)
        if s.CORR_THRESH > 0: LL[np.abs(LL) < s.CORR_THRESH] = 0.0
        return LL @ ((r[:, -1] - xmu[:, 0]) / xsd)

    def _beta(s, rets, window):
        # Each asset's market beta over a window (used to market-residualise the reversal).
        r = rets[:, -window:] if rets.shape[1] > window else rets; mkt = r.mean(0)
        return ((r * (mkt - mkt.mean())).mean(1)) / (np.var(mkt) + EPS), mkt

    def _rev(s, rets, lb):
        # Vol-adjusted, market-residual cross-sectional reversal: short recent
        # residual winners / long losers, volatility-normalised so volatile names
        # don't dominate. (Independent of the lead-lag signal -> diversifies risk.)
        if rets.shape[1] < lb + 5: return np.zeros(s.nInst)
        beta, _ = s._beta(rets, max(lb * 3, 40)); r = rets[:, -lb:]; mkt = r.mean(0)
        resid = r - beta[:, None] * mkt[None, :]; cum = resid.sum(1); vol = s._ewma_std(rets, s.VOL_HL)
        z = cum / (vol * np.sqrt(lb)); return -(z - z.mean())

    def _factor(s, rets, window, k):
        # Factor lead-lag (lagged principal factors predict the cross-section).
        if rets.shape[1] < window + 1: return np.zeros(s.nInst)
        r = rets[:, -window:]; r0 = r - r.mean(1, keepdims=True); _, vec = np.linalg.eigh(np.cov(r0)); V = vec[:, -k:]; F = V.T @ r0
        Ft, Yt = F[:, :-1].T, r0[:, 1:].T; Fmu, Fsd = Ft.mean(0), Ft.std(0) + EPS
        B = np.linalg.lstsq((Ft - Fmu) / Fsd, Yt, rcond=None)[0]; return ((F[:, -1] - Fmu) / Fsd) @ B

    def _weights(s):
        # Adapt the [lead-lag, reversal, factor] mix toward whichever family has the
        # best recent realised, vol-normalised forecast quality (bounded via exp/clip).
        w = s.BASE_W.copy(); bonus = np.ones(3)
        for i, key in enumerate(['ll', 'rev', 'fac']):
            p = s._perf[key]
            if len(p) >= 10:
                arr = np.array(p); ww = 0.5 ** (np.arange(len(arr))[::-1] / s.ADAPT_HL); ww /= ww.sum()
                m = (arr * ww).sum(); sd = np.sqrt(((arr - m) ** 2 * ww).sum()) + EPS; bonus[i] = np.exp(np.clip(m / sd, -0.8, 0.8))
        w = w * bonus; return w / w.sum()

    def getPosition(s, prc):
        # Per-day core for the robust leg: 3 families -> adaptive blend -> vol-normalised,
        # smooth-rank, market-neutral sizing -> smooth -> clip to limits.
        n, T = prc.shape
        if T < 65: return np.zeros(n, dtype=int)
        rets = np.diff(np.log(prc), axis=1); cp = prc[:, -1]
        ll = np.mean([s._zc(s._ll_window(rets, w, hl)) for w, hl in s.LL_WINDOWS], 0)     # multi-window lead-lag
        rev = np.mean([s._zc(s._rev(rets, lb)) for lb in s.REV_LB], 0)                    # reversal ensemble
        fac = s._zc(s._factor(rets, s.FACTOR_W, s.FACTOR_K))                              # factor lead-lag
        if s._pfd is not None:                                                            # learn each family's quality
            rz = rets[:, -1] / s._ewma_std(rets, s.VOL_HL)
            for key, fd in zip(['ll', 'rev', 'fac'], s._pfd):
                a = np.abs(fd) > 1e-9
                if a.any():
                    s._perf[key].append(float(np.mean(np.sign(fd[a]) * rz[a])))
                    if len(s._perf[key]) > 60: s._perf[key].pop(0)
        w = s._weights(); blend = w[0] * ll + w[1] * rev + w[2] * fac; s._pfd = (ll.copy(), rev.copy(), fac.copy())
        z = s._zc(blend); order = np.argsort(-np.abs(z)); ranks = np.empty(n, int); ranks[order] = np.arange(n)
        rm = 1.0 / (1.0 + np.exp((ranks - s.N_ACTIVE) / s.RANK_SOFT)); raw = np.sign(z) * rm
        iv = 1.0 / s._ewma_std(rets, s.VOL_HL); iv = np.clip(iv / np.median(iv), 1.0 / s.INVVOL_CAP, s.INVVOL_CAP); raw *= iv  # vol-normalise
        scale = 1.0; mkt = rets.mean(0)
        if mkt.shape[0] >= 60:                                                            # turbulence brake
            sv = mkt[-10:].std() + EPS; lv = mkt[-60:].std() + EPS; scale *= float(np.clip(1.15 - 0.35 * max(sv / lv - 1, 0), 0.6, 1.15))
        dollars = raw * s.dlrLimit; net = float(np.sum(dollars)); gross = float(np.sum(np.abs(dollars))) + EPS
        dollars = dollars - s.NEUTRALIZE * net * (np.abs(dollars) / gross); dollars *= scale; shares = dollars / cp  # market-neutral
        newPos = (1 - s.SMOOTH) * shares + s.SMOOTH * s._prevPos; lim = (s.dlrLimit / cp).astype(int)
        newPos = np.clip(newPos, -lim, lim); s._prevPos = newPos.copy(); return newPos.astype(int)


# =============================================================================
# BLENDED ENTRY POINT (the function the grader calls each day)
# =============================================================================
_gpt = GPTCore()          # adaptive high-score engine (instantiated once at import)
_robust = RobustCore()    # robust insurance leg (instantiated once at import)
BLEND_W = 0.80            # 1.0 = pure adaptive (max eval score); lower = more robust-leg insurance

def getMyPosition(prcSoFar):
    # Run both engines and blend their target share positions, then re-clip to the
    # per-instrument dollar limits. BLEND_W trades raw score for cross-regime safety.
    cp = prcSoFar[:, -1]
    lim = (_gpt.dlrLimit / cp).astype(int)
    pg = _gpt.getPosition(prcSoFar).astype(float)
    pr = _robust.getPosition(prcSoFar).astype(float)
    return np.clip(BLEND_W * pg + (1.0 - BLEND_W) * pr, -lim, lim).astype(int)
