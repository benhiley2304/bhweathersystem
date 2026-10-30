"""Thin alias: the detector lives in bhws_core.zones so the API and the research share one implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhws_core.zones import *  # noqa: F401,F403
from bhws_core.zones import detect, Zone, atr, live, mark_nesting, reaction, DEFAULTS  # noqa: F401
