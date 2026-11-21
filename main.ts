// gta6-trophy-rarity-forecast-01: forecast every GTA VI PS5 trophy's Exophase earn rate at the case's target.
//
// Model, in one paragraph. A rate is earners / players_tracked. The case history gives both series at every
// house read, so the program fits their dynamics instead of pinning the median to the last read (the house
// baseline's error): players_tracked grows like a power law in time since release (log-log least squares,
// recent window blended with the full span), and each trophy's earners (rate x players at each read) grow
// like a power law too, with the per-trophy exponent shrunk toward its tier group's and clamped so earners
// never shrink and never outrun the player pool for long. The median is earners(target) / players(target).
// The spread is an honest, wide logit-normal: sigma ~ 1.0 per sqrt-day of horizon (the baseline's 0.5 left
// it with ~0 coverage when a refresh moved every rate), plus quadrature terms for small earner counts
// (rounding/Poisson noise on the logit scale) and for fit quality, mildly asymmetric at the extremes.
// Everything is a pure function of the case file: no clock, no randomness, byte-identical reruns.
//
// Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), in name order; writes <id>.out to
// $MAKEFUN_OUTPUT_DIR (else /output): {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Fallback quantile levels if a case ever asks for something else. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Logit spread per square-root day of horizon: deliberately ~2x the house baseline's 0.5. */
const SIGMA_PER_SQRT_DAY = 1.0;
/** Generic log-log growth exponent for players_tracked when the history is too flat to fit one. */
const ALPHA_PRIOR = 0.55;

interface TrophyMeta {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels?: number[];
  horizon_minutes: number;
  target: string;
  game?: { release_date?: string };
  trophies: Array<TrophyMeta>;
  history: Array<Read>;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-4 in z). Deterministic. */
function zFor(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  let q: number, r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= 1 - pLow) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

/** Least-squares slope of y on x; NaN when the x-span is degenerate. */
function lsSlope(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return NaN;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
  if (sxx <= 1e-12) return NaN;
  return sxy / sxx;
}

/** Residual standard deviation of the LS fit of y on x (0 when the fit is exact or undefined). */
function lsResidSd(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 3) return 0;
  const slope = lsSlope(xs, ys);
  if (!Number.isFinite(slope)) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) { const r = ys[i] - (intercept + slope * xs[i]); ss += r * r; }
  return Math.sqrt(ss / (n - 2));
}

interface TrophyForecast {
  median: number; // percent
  sigmaDn: number; // logit units, for quantiles below the median
  sigmaUp: number; // logit units, for quantiles above the median
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const levels = Array.isArray(c.levels) && c.levels.length === 5 ? c.levels : LEVELS;
  const zs = levels.map(zFor);
  const horizonDays = Math.max(c.horizon_minutes, 1) / 1440;
  const history = Array.isArray(c.history) ? c.history : [];
  const nT = c.trophies.length;

  // Time axis: hours since release (game.release_date, else the first read), shifted so x = ln(1 + t) is safe.
  const releaseMs = c.game?.release_date ? Date.parse(c.game.release_date) : NaN;
  const firstReadMs = history.length > 0 ? Date.parse(history[0].at) : NaN;
  const originMs = Number.isFinite(releaseMs) ? releaseMs : firstReadMs;
  const hoursAt = (ms: number): number => (ms - originMs) / 3_600_000;
  const targetMs = Date.parse(c.target);
  const lastReadMs = history.length > 0 ? Date.parse(history[history.length - 1].at) : NaN;
  const tTarget = Number.isFinite(targetMs) && Number.isFinite(originMs)
    ? hoursAt(targetMs)
    : (Number.isFinite(lastReadMs) ? hoursAt(lastReadMs) : 0) + c.horizon_minutes / 60;
  const xTarget = Math.log(Math.max(1 + tTarget, 0.05));

  // Collapse the step-function history to distinct states (Exophase refreshes are the only real moves).
  interface State { t: number; x: number; players: number; rates: Array<number | null> }
  const states: State[] = [];
  for (const r of history) {
    const t = hoursAt(Date.parse(r.at));
    const prev = states[states.length - 1];
    const same = prev !== undefined && prev.players === r.players_tracked
      && prev.rates.length === r.rates.length && prev.rates.every((v, i) => v === r.rates[i]);
    if (!same) states.push({ t, x: Math.log(Math.max(1 + t, 0.05)), players: r.players_tracked, rates: r.rates });
  }

  const flat = states.length < 3;
  const lastState = states[states.length - 1];
  const pLast = lastState !== undefined && lastState.players > 0 ? lastState.players : 0;

  // --- Players: log-log growth exponent, recent window blended with the full span. ---
  let alpha = ALPHA_PRIOR;
  if (states.length >= 2 && pLast > 0) {
    const xs = states.map((s) => s.x);
    const ys = states.map((s) => Math.log(Math.max(s.players, 1)));
    const full = lsSlope(xs, ys);
    const recent = states.filter((s) => s.t >= states[states.length - 1].t - 24);
    const recSlope = recent.length >= 2 ? lsSlope(recent.map((s) => s.x), recent.map((s) => Math.log(Math.max(s.players, 1)))) : NaN;
    const aFull = Number.isFinite(full) ? full : ALPHA_PRIOR;
    alpha = clamp(0.5 * aFull + 0.5 * (Number.isFinite(recSlope) ? recSlope : aFull), 0, 1.6);
  }
  const xLast = lastState !== undefined ? lastState.x : xTarget;
  const growth = Math.exp(alpha * (xTarget - xLast));
  const pTarget = pLast > 0 ? pLast * clamp(growth, 1, 60) : 0;

  // --- Per-trophy earner fits (raw), then tier-group means for shrinkage. ---
  interface Fit { betaRaw: number; residSd: number; nPos: number; lastE: number; lastRate: number | null; lastX: number }
  const fits: Array<Fit | null> = [];
  for (let i = 0; i < nT; i++) {
    let lastRate: number | null = null;
    let lastE = 0;
    let lastX = xLast;
    const xs: number[] = [];
    const ys: number[] = [];
    let nPos = 0;
    for (const s of states) {
      const rate = i < s.rates.length ? s.rates[i] : null;
      if (rate === null || rate === undefined) continue;
      const e = (rate * s.players) / 100;
      lastRate = rate;
      lastE = e;
      lastX = s.x;
      if (e > 0.5) { xs.push(s.x); ys.push(Math.log(e)); nPos++; }
    }
    if (lastRate === null) { fits.push(null); continue; }
    let betaRaw = NaN;
    let residSd = 0;
    if (nPos >= 3 && xs[xs.length - 1] - xs[0] > 0.15) {
      betaRaw = lsSlope(xs, ys);
      residSd = lsResidSd(xs, ys);
    }
    fits.push({ betaRaw, residSd, nPos, lastE, lastRate, lastX });
  }
  const tierBeta = new Map<string, number>();
  {
    const acc = new Map<string, { wSum: number; bSum: number }>();
    for (let i = 0; i < nT; i++) {
      const f = fits[i];
      if (f === null || !Number.isFinite(f.betaRaw)) continue;
      const w = clamp((f.nPos - 2) / 4, 0, 1) * clamp(0.3 + f.lastE / 30, 0, 1);
      if (w <= 0) continue;
      const tier = c.trophies[i].tier;
      const a = acc.get(tier) ?? { wSum: 0, bSum: 0 };
      a.wSum += w; a.bSum += w * f.betaRaw;
      acc.set(tier, a);
    }
    for (const [tier, a] of acc) tierBeta.set(tier, a.bSum / a.wSum);
  }

  const baseSigma = SIGMA_PER_SQRT_DAY * Math.sqrt(horizonDays);
  const forecasts: TrophyForecast[] = [];
  for (let i = 0; i < nT; i++) {
    const f = fits[i];
    const tier = c.trophies[i].tier;
    const groupBeta = tierBeta.get(tier) ?? alpha;
    let median: number;
    let lastE = 0;
    let fitPenalty = 0.35;
    if (f === null || pTarget <= 0) {
      // No usable history for this trophy: a small rate and a very wide spread.
      median = 0.5;
      fitPenalty = 0.6;
    } else {
      lastE = f.lastE;
      const w = Number.isFinite(f.betaRaw)
        ? clamp((f.nPos - 2) / 4, 0, 1) * clamp(0.3 + f.lastE / 30, 0, 1)
        : 0;
      const beta = clamp(w * (Number.isFinite(f.betaRaw) ? f.betaRaw : groupBeta) + (1 - w) * groupBeta, 0, alpha + 0.9);
      let eTarget = f.lastE * Math.exp(beta * (xTarget - f.lastX));
      eTarget = Math.max(eTarget, f.lastE); // earners do not un-earn
      const lastShare = pLast > 0 ? f.lastE / pLast : 0;
      const shareCap = Math.min(0.999, Math.max(lastShare, lastShare * 4 + 0.15));
      eTarget = Math.min(eTarget, pTarget * shareCap, pTarget);
      median = clampPct((100 * eTarget) / pTarget);
      if (f.lastE <= 0.5) {
        // Essentially nobody has it yet: a couple of earners by the target, no more precision than that.
        median = clampPct(Math.max(median, 150 / pTarget));
      }
      // Keep the median's move from the last rate inside what a week can plausibly do on the logit scale,
      // and lean thin-evidence medians back toward the last rate.
      if (f.lastRate !== null && f.lastRate > 0) {
        const lLast = logit(f.lastRate);
        let lMed = clamp(logit(median), lLast - 2.6, lLast + 2.6);
        if (w < 0.5) lMed = lMed + (0.5 - w) * (lLast - lMed);
        if (flat) lMed -= 0.35 * Math.sqrt(horizonDays); // a stale page's next refresh broadens the pool: rates sag
        median = invLogit(lMed);
      }
      fitPenalty = Number.isFinite(f.betaRaw) ? Math.min(0.5, f.residSd * 0.8) : 0.35;
    }
    const countSigma = Math.min(0.55, 1 / Math.sqrt(Math.max(lastE, 1)));
    let sigma = Math.sqrt(baseSigma * baseSigma + countSigma * countSigma + fitPenalty * fitPenalty);
    sigma = clamp(sigma, 0.35, 2.4);
    const sigmaUp = sigma * (median < 2 ? 1.15 : 1) * (median > 60 ? 0.75 : 1);
    const sigmaDn = sigma * (median > 50 ? 1.15 : 1) * (median < 0.5 ? 0.9 : 1);
    forecasts.push({ median, sigmaDn, sigmaUp });
  }

  return c.trophies.map((t, i) => {
    const fc = forecasts[i];
    const l = logit(fc.median);
    const q: number[] = [];
    let prev = 0;
    for (let j = 0; j < zs.length; j++) {
      const z = zs[j];
      const v = round4(clampPct(invLogit(l + z * (z < 0 ? fc.sigmaDn : fc.sigmaUp))));
      const nonDec = Math.max(v, prev);
      q.push(nonDec);
      prev = nonDec;
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
