"""BHWS evidence layer.

Pure, network-free helpers that sit beside backend.py. Nothing in here makes a
trading decision. The modules describe evidence (what was observed, when it
became public, how fresh it is, where it sits in its own history) and the
house playbook state, so the dashboard and the report can show the reasoning
instead of a single verdict.

Modules
  cot_calendar  CFTC observation date vs public release date, freshness
  percentile    the house percentile (trailing 156 reports, mid-rank)
  provenance    per-input source / observed / released / retrieved stamps
  playbook      Issue 11 "extreme, level, turn, add" state machine
  exposure      correlation-aware effective bets (participation ratio)
  riskreward    reward:risk with frictions and break-even hit rate
  grounding     checks AI text against the numbers it was given
  readout       per-market read that keeps bias, setup and trade apart
"""
__version__ = "0.1.0"
