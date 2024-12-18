"""
This module provides the main interface for initializing and running ICS (Iterative Coefficient Splitting) processes.

The module includes:
- Initialization processes for both standard and efficient ICS implementations.
- Functions to run standard and efficient ICS.
"""

from .ics import run_ics
from .efficient_ics import run_efficient_ics
from .ics_prepare import init_ics, init_efficient_ics

__all__ = ["init_ics", "run_ics", "init_efficient_ics", "run_efficient_ics"]
