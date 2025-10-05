"""Label generation."""

from .triple_barrier import (  # noqa: F401
    LabelDiagnostics,
    TripleBarrierConfig,
    apply_triple_barrier,
    enforce_label_sanity,
    summarize_label_distribution,
)

__all__ = [
    "TripleBarrierConfig",
    "apply_triple_barrier",
    "summarize_label_distribution",
    "LabelDiagnostics",
    "enforce_label_sanity",
]
