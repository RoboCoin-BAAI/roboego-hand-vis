#!/usr/bin/env python3
"""CLI for the raw headband integrity check."""

from __future__ import annotations

from roboego_hand_vis.quality.raw_headband_integrity import build_arg_parser, main

__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    main()
