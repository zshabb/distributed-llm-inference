"""Shared test configuration.

Hypothesis profiles implement the test tiers (ENGINEERING_STANDARDS §6.1):
* ``fast`` (default): keeps the full fast suite well under 60 s.
* ``focused``: run after changing a component, e.g.
  ``HYPOTHESIS_PROFILE=focused uv run --frozen pytest tests/unit/kv``.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "fast", max_examples=60, stateful_step_count=40, deadline=None, derandomize=False
)
settings.register_profile(
    "focused",
    max_examples=2000,
    stateful_step_count=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))
