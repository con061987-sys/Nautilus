"""Triton ↔ TVM MetaSchedule bridge.

This bridge extracts metadata from Triton kernels at JIT time,
constructs equivalent TVM TIR templates for MetaSchedule tuning,
and maps the resulting optimal configurations back to Triton
compiler options.

Architecture: Config bridge (not full IR conversion).
Intercepts at the Python level via JITFunction wrapping,
feeds TVM MetaSchedule, maps tuning records to Triton Config objects.
"""

from .bridge_orchestrator import TritonTVMBridge

__all__ = ["TritonTVMBridge"]
