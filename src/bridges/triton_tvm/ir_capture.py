"""IR capture from Triton's real compilation pipeline.

This module owns the bridge between Triton's TTGIR output and the
TVM MetaSchedule adapter. It reads the captured IR from the
backend.compiler's capture buffer, classifies it, and forwards
to the appropriate template constructor.

Architecture:
    Triton Backend Plugin (C++/Python) → capture buffer
        → IRClassifier → BoundsExtractor
            → TIRTemplateBuilder (uses real extracted bounds)
                → TVM MetaSchedule Adapter
                    → ConfigMapper → Triton recompile
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto

from src.common.logging import get_logger

from .backend import CAPTURE_KEY_FMT

logger = get_logger(__name__)


class KernelKind(Enum):
    """Classification of Triton kernel by op structure."""
    MATMUL = auto()           # tt.dot present
    ATTENTION = auto()        # matmul + softmax + matmul pattern
    REDUCTION = auto()        # tt.reduce dominates
    ELEMENTWISE = auto()      # pointwise ops only
    SCAN = auto()             # tt.scan
    PERSISTENT = auto()       # while loops with persistent threads
    BROADCAST = auto()        # tt.broadcast
    TRANSPOSE = auto()        # tt.trans
    UNKNOWN = auto()


@dataclass
class IRBounds:
    """Mathematical bounds extracted from real TTGIR.

    These are NOT template bounds — they come from the actual
    tensor shape attributes in the captured MLIR.
    """
    # For matmul
    m: int | None = None
    n: int | None = None
    k: int | None = None

    # For reduction
    reduce_size: int | None = None
    keep_size: int | None = None

    # For elementwise
    total_elements: int | None = None

    # Universal
    block_size: tuple[int, ...] = field(default_factory=tuple)
    data_dtype: str = "float32"
    tensor_ranks: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.data_dtype:
            self.data_dtype = "float32"


@dataclass
class CapturedKernelIR:
    """The fully-processed IR from a Triton kernel, ready for tuning."""
    source_hash: str
    target: str
    stage_name: str
    ir_text: str

    # Parsed from the IR
    kind: KernelKind = KernelKind.UNKNOWN
    bounds: IRBounds = field(default_factory=IRBounds)

    # All ops seen in the IR (in order)
    ops_seen: list[str] = field(default_factory=list)

    # All tensor types seen
    tensor_types: list[tuple[tuple[int, ...], str]] = field(default_factory=list)

    # num_warps, num_stages from the IR (if set)
    num_warps: int | None = None
    num_stages: int | None = None

    # Capture metadata
    capture_time_ms: float = 0.0

    @property
    def cache_key(self) -> str:
        """Stable key for caching tuning results for this kernel+target."""
        parts = [
            self.source_hash,
            self.target,
            self.kind.name,
            f"m{self.bounds.m}", f"n{self.bounds.n}", f"k{self.bounds.k}",
            self.bounds.data_dtype,
        ]
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class IRCapture:
    """Reads captured IR from the backend plugin and processes it.

    Single entry point: capture_for_source(source_hash, target).
    Reads from the backend's capture buffer, classifies, extracts
    bounds, and returns a fully-populated CapturedKernelIR.
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency
        from .bounds_extractor import BoundsExtractor
        from .ir_classifier import IRClassifier
        self.classifier = IRClassifier()
        self.extractor = BoundsExtractor()
        self._last_processed: dict[str, CapturedKernelIR] = {}

    def capture_for_source(
        self,
        source_hash: str,
        target: str,
    ) -> CapturedKernelIR | None:
        """Read the most recent captured IR for a given source/target.

        Returns None if no IR has been captured yet (compile hasn't run).
        """
        from .backend.compiler import TVMBackend

        start = time.perf_counter()
        buffer = TVMBackend.get_capture_buffer()

        # Build the exact lookup prefix from the shared format. Writers
        # (hooks.py, compiler.py) store entries as
        #   nautilus:ttgir:{source_hash[:16]}:{kernel_name}
        # We match on the hash portion, since the kernel name is not
        # passed in by the caller. Using the same format string
        # guarantees writers and readers stay in lockstep.
        key_prefix = CAPTURE_KEY_FMT.format(
            source_hash=source_hash[:16],
            kernel_name="",
        )

        ir_text: str | None = None
        stage_name = "ttgir"

        for key, value in buffer.items():
            if key.startswith(key_prefix):
                ir_text = value
                break

        if ir_text is None:
            return None

        # Process the IR
        result = self._process_ir(ir_text, source_hash, target, stage_name)
        result.capture_time_ms = (time.perf_counter() - start) * 1000

        cache_key = f"{source_hash}:{target}"
        self._last_processed[cache_key] = result
        return result

    def capture_from_text(
        self,
        ir_text: str,
        source_hash: str,
        target: str,
        stage_name: str = "ttgir",
    ) -> CapturedKernelIR:
        """Process IR text directly (for testing or manual injection)."""
        start = time.perf_counter()
        result = self._process_ir(ir_text, source_hash, target, stage_name)
        result.capture_time_ms = (time.perf_counter() - start) * 1000
        return result

    def _process_ir(
        self,
        ir_text: str,
        source_hash: str,
        target: str,
        stage_name: str,
    ) -> CapturedKernelIR:
        """Classify and extract bounds from captured IR text."""
        result = CapturedKernelIR(
            source_hash=source_hash,
            target=target,
            stage_name=stage_name,
            ir_text=ir_text,
        )

        # Step 1: Classify (returns just the KernelKind here).
        result.kind = self.classifier.classify_kind(ir_text)
        result.ops_seen = self.classifier.collect_ops(ir_text)
        result.tensor_types = self.classifier.collect_tensor_types(ir_text)
        result.num_warps, result.num_stages = self._extract_compile_attrs(ir_text)

        # Step 2: Extract bounds
        result.bounds = self.extractor.extract(ir_text, result.kind)

        logger.info(
            "IR processed: kind=%s, ops=%d, tensors=%d, m=%s, n=%s, k=%s",
            result.kind.name, len(result.ops_seen), len(result.tensor_types),
            result.bounds.m, result.bounds.n, result.bounds.k,
        )

        return result

    def _extract_compile_attrs(self, ir_text: str) -> tuple[int | None, int | None]:
        """Extract num_warps and num_stages from TTGIR module attributes."""
        num_warps: int | None = None
        num_stages: int | None = None

        # Triton sets these as module attributes: ttg.num-warps = 4
        m = re.search(r'ttg\.num-warps\s*=\s*(\d+)', ir_text)
        if m:
            num_warps = int(m.group(1))
        m = re.search(r'ttg\.num-stages\s*=\s*(\d+)', ir_text)
        if m:
            num_stages = int(m.group(1))

        return num_warps, num_stages
