# GTA VI trophy rarity forecast

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast every PS5 GTA VI
trophy's Exophase earn rate at the case's target, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy, scored by mean pinball loss on the
logit scale. `main.ts` (Node 22, standard library only) reads every `<id>.in`
in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## The model

A rate is **earners / players_tracked**, and the case history reports both at
every read — so instead of pinning the median to the last read (the house
baseline's approach, which a single Exophase refresh left with ~0 coverage),
the program fits the two series' dynamics and divides their forecasts:

- **Players.** Collapse the history to distinct page states (the page only
  changes at Exophase refreshes), then fit `ln players` against
  `ln(1 + hours since release)` by least squares — a power-law growth curve.
  The full-span slope is blended with the last-24-hours slope and clamped, and
  extrapolated to the target.
- **Earners.** Per trophy, earners = rate × players at each read. Fit the same
  log-log growth exponent per trophy, shrunk toward its tier group's exponent
  (platinum / gold / silver / bronze) by an evidence weight that grows with the
  number of positive reads and the earner count; clamp so earners never shrink
  and never outrun the player pool. Thin histories fall back to the group.
  The median is `earners(target) / players(target)`, with the median's logit
  move from the last rate capped at what a week can plausibly do. If the page
  never changed in the history (a stale page), the median also sags slightly:
  the next refresh broadens the tracked pool, which pushes most rates down.
  A trophy with no usable history of its own is forecast from its tier
  group's prior — the median last rate of that tier's observed trophies —
  with a wider spread than any fitted trophy gets, instead of one flat rate
  for every tier.
- **Spread.** A logit-normal around the median with
  sigma ≈ 1.0·√(horizon days) — about twice the house baseline's 0.5, whose
  90-minute practice band covered nothing — plus quadrature terms for
  small-count noise (`1/√earners`, capped) and fit quality, mildly asymmetric
  at the extremes (more upside for tiny rates, more downside for large ones),
  total sigma capped at 2.4 logit units.

Everything is a pure function of the case file — no clock, no randomness — so
reruns are byte-identical, and the whole forecast takes milliseconds.

## Layout

- `main.ts` — the whole entry.
- `LICENSE` — MIT (from `makefun init`).
