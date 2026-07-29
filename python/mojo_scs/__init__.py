"""Conic splitting kernels and an SCS-compatible solver implemented in Mojo."""

from ._lib import build
from .kernels import matvec, project_cone
from .solver import (
    FAILED,
    INDETERMINATE,
    INFEASIBLE,
    INFEASIBLE_INACCURATE,
    SIGINT,
    SOLVED,
    SOLVED_INACCURATE,
    UNBOUNDED,
    UNBOUNDED_INACCURATE,
    UNFINISHED,
    SCS,
    solve,
)

__version__ = "0.1.0"
__all__ = [
    "SCS",
    "solve",
    "build",
    "matvec",
    "project_cone",
    "SOLVED",
    "SOLVED_INACCURATE",
    "UNFINISHED",
    "UNBOUNDED",
    "INFEASIBLE",
    "INDETERMINATE",
    "FAILED",
    "SIGINT",
    "UNBOUNDED_INACCURATE",
    "INFEASIBLE_INACCURATE",
]
