"""Maps TVM MetaSchedule tuning records to Triton Config objects.

After MetaSchedule runs evolutionary search on a TIR template, its
database contains tuning records with traces (schedule primitives and
their sampled decisions). This module extracts the tile sizes, thread
binding patterns, and pipeline depths from those traces and translates
them into Triton compiler options.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MappedTuningConfig:
    """Triton compiler tuning config derived from TVM MetaSchedule output.

    This is the bridge's output format — a set of parameters that
    Triton's compiler will use for the optimized kernel.
    """
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1
    enable_fp_fusion: bool = True
    max_num_imprecise_acc: int = 0

    def to_triton_config(
        self,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Convert to a triton.Config object suitable for @triton.autotune.

        Args:
            extra_kwargs: Additional kernel meta-parameters to include
                (e.g., BLOCK_SIZE, GROUP_SIZE).

        Returns:
            triton.Config with the mapped parameters.
        """
        import triton  # local import: config_mapper must load without triton installed

        kwargs: dict[str, Any] = {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        return triton.Config(
            kwargs=kwargs,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            num_ctas=self.num_ctas,
        )

    @classmethod
    def defaults(cls) -> MappedTuningConfig:
        """Return sensible defaults when no TVM tuning is available."""
        return cls()


class ConfigMapper:
    """Maps TVM MetaSchedule tuning records to Triton Config objects.

    The mapping works by reading the 'trace' field of a tuning record,
    which contains a sequence of schedule primitive applications. From
    these we extract:
      - MultiLevelTiling decisions → block_m, block_n, block_k
      - AutoBind decisions → num_warps (derived from thread binding)
      - Pipeline decisions → num_stages
    """

    def map_record(self, tvm_record: Any, metadata: Any | None = None) -> MappedTuningConfig:
        """Map a TVM tuning record to a Triton tuning config.

        Args:
            tvm_record: A TVM MetaSchedule TuningRecord object or
                its JSON-serialized trace dict.
            metadata: Optional kernel metadata for context.

        Returns:
            MappedTuningConfig with Triton-compatible parameters.
        """
        trace = self._extract_trace(tvm_record)
        if trace is None:
            return MappedTuningConfig.defaults()

        decisions = trace.get("decisions", {})
        instructions = trace.get("instructions", [])

        # Extract tiling decisions from MultiLevelTiling
        tile_sizes = self._extract_tile_sizes(instructions, decisions)
        thread_info = self._extract_thread_binding(instructions, decisions)

        return MappedTuningConfig(
            block_m=tile_sizes.get("m", 128),
            block_n=tile_sizes.get("n", 128),
            block_k=tile_sizes.get("k", 32),
            num_warps=thread_info.get("num_warps", 4),
            num_stages=thread_info.get("num_stages", 3),
            num_ctas=thread_info.get("num_ctas", 1),
        )

    def map_json_record(
        self,
        json_record: str,
        metadata: Any | None = None,
    ) -> MappedTuningConfig:
        """Map a JSON-serialized TVM tuning record.

        Args:
            json_record: JSON string of a tuning record.
            metadata: Optional kernel metadata.

        Returns:
            MappedTuningConfig.
        """
        parsed = json.loads(json_record)
        return self.map_record(parsed, metadata)

    # ------------------------------------------------------------------
    # Internal mapping logic
    # ------------------------------------------------------------------

    def _extract_trace(self, record: Any) -> dict[str, Any] | None:
        """Extract the serialized trace from a tuning record.

        Handles four cases:
          1. TVM record object with ``.trace`` attribute
          2. Raw trace dict with ``instructions`` and ``decisions`` keys
          3. Dict with a ``trace`` key wrapping the above
          4. Bare decisions dict (e.g. ``{"tile_m": [1,2,4]}``)
        """
        if hasattr(record, "trace"):
            trace = record.trace
            if hasattr(trace, "__dict__"):
                return trace.__dict__
            if isinstance(trace, dict):
                return trace
        if isinstance(record, dict):
            # Case 2: record IS the trace
            if "instructions" in record or "decisions" in record:
                return record
            # Case 3: record has a "trace" key wrapping the real dict
            trace_val = record.get("trace")
            if isinstance(trace_val, dict):
                return trace_val
            # Case 4: record is just a decisions dict
            decisions_val = record.get("decisions")
            if isinstance(decisions_val, dict):
                return decisions_val
        return None

    def _extract_tile_sizes(
        self,
        instructions: list[Any],
        decisions: dict[str, Any],
    ) -> dict[str, int]:
        """Extract tile sizes from MultiLevelTiling decisions.

        TVM's MultiLevelTiling produces decisions like:
          {"tile_m": [1, 4, 8], "tile_n": [1, 2, 16], "tile_k": [4, 8]}

        The product of each dimension's splits gives the total tile size.
        """
        tile_sizes: dict[str, int] = {}

        for key, value in decisions.items():
            key_lower = key.lower()
            if "tile" in key_lower and isinstance(value, (list, tuple)):
                # TVM stores tile decisions as lists of factors;
                # the product is the actual tile size
                dim = None
                for d in ("m", "n", "k"):
                    if d in key_lower:
                        dim = d
                        break
                if dim:
                    product = 1
                    for factor in value:
                        if isinstance(factor, (int, float)):
                            product *= int(factor)
                    tile_sizes[dim] = product

        return tile_sizes

    def _extract_thread_binding(
        self,
        instructions: list[Any],
        decisions: dict[str, Any],
    ) -> dict[str, int]:
        """Derive num_warps, num_stages from thread binding decisions.

        TVM binds loops to threadIdx.x / blockIdx.x. The extent of
        threadIdx.x binding tells us the thread count. Dividing by
        32 (warp size) gives num_warps.

        Pipeline stages come from the MultiLevelTiling 'stages' decision.
        """
        result: dict[str, int] = {}

        for key, value in decisions.items():
            k = key.lower()
            if "stage" in k and isinstance(value, int):
                result["num_stages"] = value
            if "thread" in k and isinstance(value, (int, list, tuple)):
                if isinstance(value, int):
                    result["num_warps"] = max(value // 32, 1)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    total_threads = 1
                    for v in value:
                        if isinstance(v, (int, float)):
                            total_threads *= int(v)
                    result["num_warps"] = max(total_threads // 32, 1)

        return result
