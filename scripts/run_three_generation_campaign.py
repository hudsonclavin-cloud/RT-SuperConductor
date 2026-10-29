#!/usr/bin/env python3
"""Compatibility entry point; this now runs review triage, not generations."""

from __future__ import annotations

import sys

from build_review_campaign import main


if __name__ == "__main__":
    print(
        "Notice: this legacy script name does not run learning generations. "
        "It creates a paper-review campaign and reports zero learning cycles.",
        file=sys.stderr,
    )
    raise SystemExit(main())
