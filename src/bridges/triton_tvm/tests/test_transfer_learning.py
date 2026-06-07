"""Tests for the cross-vendor transfer learning engine."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.bridges.triton_tvm.config_cache import ConfigCache
from src.bridges.triton_tvm.config_mapper import MappedTuningConfig
from src.bridges.triton_tvm.transfer_learning import (
    TRANSFER_MATRICES,
    PerformanceDB,
    TransferEngine,
    TransferredConfig,
    transfer_config,
)

# ---------------------------------------------------------------------------
# Property-based strategies
# ---------------------------------------------------------------------------

_VALID_VENDORS = st.sampled_from(["nvidia", "amd", "intel", "apple"])

_sane_tile = st.integers(min_value=16, max_value=512)
_sane_warps = st.integers(min_value=1, max_value=64)
_sane_stages = st.integers(min_value=1, max_value=8)
_sane_ctas = st.integers(min_value=1, max_value=4)

config_dict_strategy = st.fixed_dictionaries(
    {  # type: ignore[arg-type]
        "block_m": _sane_tile,
        "block_n": _sane_tile,
        "block_k": _sane_tile,
        "num_warps": _sane_warps,
        "num_stages": _sane_stages,
        "num_ctas": _sane_ctas,
    },
)


# ---------------------------------------------------------------------------
# Test TransferredConfig
# ---------------------------------------------------------------------------


class TestTransferredConfig:
    """TransferredConfig construction and behaviour."""

    def test_default_construction(self) -> None:
        config = MappedTuningConfig(block_m=64, block_n=128)
        result = TransferredConfig(config=config, confidence=0.85)
        assert result.config.block_m == 64
        assert result.config.block_n == 128
        assert result.confidence == 0.85
        assert result.seed_config == {}
        assert result.source_vendor == ""
        assert result.target_vendor == ""

    def test_should_tune_below_threshold(self) -> None:
        config = MappedTuningConfig()
        result = TransferredConfig(config=config, confidence=0.4)
        assert result.should_tune() is True

    def test_should_tune_above_threshold(self) -> None:
        config = MappedTuningConfig()
        result = TransferredConfig(config=config, confidence=0.8)
        assert result.should_tune() is False

    def test_should_tune_custom_threshold(self) -> None:
        config = MappedTuningConfig()
        result = TransferredConfig(config=config, confidence=0.7)
        assert result.should_tune(threshold=0.5) is False
        assert result.should_tune(threshold=0.9) is True

    def test_as_tune_kwargs(self) -> None:
        config = MappedTuningConfig(block_m=64, block_n=128, num_warps=8)
        result = TransferredConfig(
            config=config,
            confidence=0.7,
            seed_config={"BLOCK_SIZE_M": 64, "num_warps": 8},
        )
        kwargs = result.as_tune_kwargs()
        assert kwargs["BLOCK_SIZE_M"] == 64
        assert kwargs["num_warps"] == 8


# ---------------------------------------------------------------------------
# Test TransferEngine — matrix transfers
# ---------------------------------------------------------------------------


class TestTransferEngineMatrices:
    """Transfer matrix application for known vendor pairs."""

    def _make_engine(self) -> TransferEngine:
        return TransferEngine()

    def test_nvidia_to_amd(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=128, block_n=128, block_k=64,
            num_warps=8, num_stages=4,
        )
        result = engine.transfer("nvidia", "amd", src_config)

        # AMD wavefronts are 64-wide vs Nvidia 32-wide, so num_warps
        # should be lower (warp_scale = 32/64 = 0.5).
        assert result.config.num_warps < src_config.num_warps

        # Confidence for nvidia→amd should be in a reasonable range.
        assert 0.5 <= result.confidence <= 1.0

        # Tile sizes should be multiples of 16.
        assert result.config.block_m % 16 == 0
        assert result.config.block_n % 16 == 0
        assert result.config.block_k % 16 == 0

        # Source/target should be recorded.
        assert result.source_vendor == "nvidia"
        assert result.target_vendor == "amd"

    def test_nvidia_to_intel(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=128, block_n=128, block_k=64,
            num_warps=8, num_stages=4,
        )
        result = engine.transfer("nvidia", "intel", src_config)

        # Intel has no matrix cores → tile sizes should be smaller
        # (SIMD path tends toward smaller tiles).
        assert result.config.block_m <= src_config.block_m

        # Intel has lower bandwidth → fewer pipeline stages.
        assert result.config.num_stages <= src_config.num_stages

        # Confidence lower than nvidia→amd.
        assert result.confidence < 0.6

    def test_nvidia_to_apple(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=256, block_n=128, block_k=64,
            num_warps=16, num_stages=5,
        )
        result = engine.transfer("nvidia", "apple", src_config)

        # Apple has much less shared memory → smaller K tiles.
        assert result.config.block_k <= src_config.block_k

        # Apple confidence is lower.
        assert result.confidence < 0.6
        assert result.target_vendor == "apple"

    def test_amd_to_nvidia(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=64, block_n=64, block_k=32,
            num_warps=8, num_stages=3,
        )
        result = engine.transfer("amd", "nvidia", src_config)

        # Nvidia has smaller warp size (32 vs 64) → more warps.
        # warp_scale = 64/32 = 2.0, so 8 * 2.0 * sqrt(0.5) / 0.9 ≈ 12.6 → snapped to 16
        assert result.config.num_warps >= src_config.num_warps
        assert result.config.num_warps > 8

        # Nvidia has tensor cores → M/N tiles should be at least as large.
        assert result.config.block_m >= src_config.block_m

        # Tile sizes are multiples of 16.
        assert result.config.block_m % 16 == 0
        assert result.config.block_n % 16 == 0

    def test_amd_to_intel(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=128, block_n=128, block_k=64,
            num_warps=8, num_stages=4,
        )
        result = engine.transfer("amd", "intel", src_config)

        # Intel has no matrix cores → smaller tiles.
        assert result.config.block_m <= src_config.block_m
        # Lower bandwidth → fewer stages.
        assert result.config.num_stages <= src_config.num_stages

    def test_intel_to_nvidia(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(
            block_m=64, block_n=64, block_k=32,
            num_warps=4, num_stages=2,
        )
        result = engine.transfer("intel", "nvidia", src_config)

        # Nvidia has much higher bandwidth → more stages.
        assert result.config.num_stages > src_config.num_stages

    def test_same_vendor_identity_not_implemented(self) -> None:
        # Same-vendor transfers are not in TRANSFER_MATRICES
        # (they should use the original config directly).
        engine = self._make_engine()
        src_config = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "nvidia", src_config)
        # Falls back to identity with very low confidence.
        assert result.confidence < 0.3
        assert result.config.block_m == src_config.block_m

    # ------------------------------------------------------------------
    # Dict config normalisation
    # ------------------------------------------------------------------

    def test_dict_source_config(self) -> None:
        engine = self._make_engine()
        src = {"block_m": 128, "block_n": 64, "block_k": 32,
               "num_warps": 8, "num_stages": 4}
        result = engine.transfer("nvidia", "amd", src)
        # Config should be adapted: nvidia→amd has warp_scale=0.5,
        # so num_warps should be lower and tiles scaled.
        assert result.config.num_warps < 8  # adapted down for AMD
        assert result.config.num_stages > 0
        assert result.config.block_m > 0
        assert result.source_vendor == "nvidia"
        assert result.target_vendor == "amd"

    def test_partial_dict_source_config(self) -> None:
        engine = self._make_engine()
        src = {"block_m": 256, "num_warps": 16}  # missing fields → defaults
        result = engine.transfer("nvidia", "amd", src)
        # Adapted: nvidia→amd shrinks warps (warp_scale=0.5).
        assert result.config.num_warps <= 16  # adapted (same or lower) for AMD
        assert result.config.block_m > 0
        assert result.config.block_n > 0  # filled from defaults
        assert result.config.num_stages > 0

    # ------------------------------------------------------------------
    # Unknown pair fallback
    # ------------------------------------------------------------------

    def test_unknown_vendor_pair(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(block_m=128)
        result = engine.transfer("unknown_vendor", "amd", src_config)
        assert result.confidence == 0.25  # identity fallback
        assert result.config.block_m == 128  # unchanged

    def test_case_insensitive_vendor_names(self) -> None:
        engine = self._make_engine()
        src_config = MappedTuningConfig(block_m=128)
        result = engine.transfer("NVIDIA", "AMD", src_config)
        # Transfer should work with case-insensitive vendor names.
        assert result.source_vendor == "nvidia"
        assert result.target_vendor == "amd"
        assert result.confidence >= 0.5
        assert result.config.block_m > 0


# ---------------------------------------------------------------------------
# Test adapt_tile
# ---------------------------------------------------------------------------


class TestAdaptTile:
    """Tile-size scaling across vendors."""

    def _make_engine(self) -> TransferEngine:
        return TransferEngine()

    def test_nvidia_to_amd_tiles_shrink(self) -> None:
        engine = self._make_engine()
        m, n, k = engine.adapt_tile(256, 128, 64, "nvidia", "amd")
        # With warp_scale=0.5 and matrix_core_scale=0.9,
        # combined_scale_mn = sqrt(0.5*0.9) ≈ 0.67, so tiles should shrink.
        assert m < 256
        assert m >= 16
        assert n < 128
        assert n >= 16
        assert k >= 8

    def test_amd_to_nvidia_tiles_grow(self) -> None:
        engine = self._make_engine()
        m, n, _k = engine.adapt_tile(64, 64, 32, "amd", "nvidia")
        # AMD→Nvidia warp_scale=2.0 → tiles grow.
        assert m > 64
        assert m <= 512
        assert n > 64
        assert n <= 512

    def test_tiles_rounded_to_16(self) -> None:
        engine = self._make_engine()
        m, n, k = engine.adapt_tile(100, 100, 48, "nvidia", "amd")
        assert m % 16 == 0
        assert n % 16 == 0
        assert k % 16 == 0

    def test_identity_for_unknown_pair(self) -> None:
        engine = self._make_engine()
        m, n, k = engine.adapt_tile(128, 64, 32, "nvidia", "unknown")
        assert m == 128
        assert n == 64
        assert k == 32

    def test_minimum_tile_size(self) -> None:
        engine = self._make_engine()
        m, n, k = engine.adapt_tile(16, 16, 8, "nvidia", "apple")
        # Apple has much smaller shared memory, but minimum should be 16.
        assert m >= 16
        assert n >= 16
        assert k >= 8
        assert k <= 256


# ---------------------------------------------------------------------------
# Test validation helpers
# ---------------------------------------------------------------------------


class TestValidation:
    """validate_pair and list_available_targets."""

    def test_validate_known_pair(self) -> None:
        assert TransferEngine.validate_pair("nvidia", "amd") is True
        assert TransferEngine.validate_pair("nvidia", "intel") is True

    def test_validate_unknown_pair(self) -> None:
        assert TransferEngine.validate_pair("nvidia", "unknown") is False
        assert TransferEngine.validate_pair("foo", "bar") is False

    def test_validate_case_insensitive(self) -> None:
        assert TransferEngine.validate_pair("NVIDIA", "AMD") is True

    def test_list_targets_for_nvidia(self) -> None:
        targets = TransferEngine.list_available_targets("nvidia")
        assert "amd" in targets
        assert "intel" in targets
        assert "apple" in targets

    def test_list_targets_for_amd(self) -> None:
        targets = TransferEngine.list_available_targets("amd")
        assert "nvidia" in targets
        assert "intel" in targets

    def test_list_targets_unknown(self) -> None:
        targets = TransferEngine.list_available_targets("unknown")
        assert targets == []


# ---------------------------------------------------------------------------
# Test confidence scoring
# ---------------------------------------------------------------------------


class TestConfidence:
    """Confidence score computations."""

    def test_nvidia_to_amd_confidence(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "amd", src)
        # Base similarity is 0.65 + small coverage_bonus.
        assert 0.60 <= result.confidence <= 0.85

    def test_nvidia_to_intel_confidence(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "intel", src)
        # Lower confidence — dissimilar architectures.
        assert result.confidence < 0.60

    def test_amd_to_nvidia_confidence(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("amd", "nvidia", src)
        assert 0.60 <= result.confidence <= 0.85

    def test_confidence_with_historical_data(self) -> None:
        perf_db = MagicMock(spec=PerformanceDB)
        perf_db.get_historical_accuracy.return_value = 0.9

        engine = TransferEngine(perf_db=perf_db)
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "amd", src, kernel_hash="abc")

        # With historical accuracy of 0.9, confidence should be higher
        # than without history.
        assert result.confidence > 0.70

    def test_confidence_with_low_historical_data(self) -> None:
        perf_db = MagicMock(spec=PerformanceDB)
        perf_db.get_historical_accuracy.return_value = 0.3

        engine = TransferEngine(perf_db=perf_db)
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "amd", src, kernel_hash="abc")

        # Historical accuracy drags confidence down.
        engine_no_history = TransferEngine()
        result_no_history = engine_no_history.transfer(
            "nvidia", "amd", src,
        )
        assert result.confidence < result_no_history.confidence

    def test_historical_data_ignored_without_kernel_hash(self) -> None:
        perf_db = MagicMock(spec=PerformanceDB)
        perf_db.get_historical_accuracy.return_value = 0.9

        engine = TransferEngine(perf_db=perf_db)
        src = MappedTuningConfig(block_m=128)
        result = engine.transfer("nvidia", "amd", src)  # no kernel_hash

        # Without kernel_hash, perf_db is not consulted.
        perf_db.get_historical_accuracy.assert_not_called()
        assert result.confidence < 0.9


# ---------------------------------------------------------------------------
# Test ConfigCache integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    """TransferEngine with ConfigCache."""

    def test_transfer_caches_result(self, tmp_path) -> None:
        cache = ConfigCache(cache_dir=tmp_path / "transfer_cache")
        engine = TransferEngine(config_cache=cache)
        src = MappedTuningConfig(block_m=128, block_n=128)
        result = engine.transfer(
            "nvidia", "amd", src, kernel_hash="test_kernel",
        )

        # The cache should now contain the transferred config.
        cached = cache.get("test_kernel", "amd", "transferred")
        assert cached is not None
        assert cached["source_vendor"] == "nvidia"
        assert cached["config"]["block_m"] == result.config.block_m
        assert cached["config"]["block_n"] == result.config.block_n

    def test_transfer_without_cache_key_skips_cache(self, tmp_path) -> None:
        cache = ConfigCache(cache_dir=tmp_path / "transfer_cache2")
        engine = TransferEngine(config_cache=cache)
        src = MappedTuningConfig(block_m=128)
        engine.transfer("nvidia", "amd", src)  # no kernel_hash

        # Cache should be empty.
        assert len(list(cache.cache_dir.glob("*.json"))) == 0

    def test_cache_config_persisted(self, tmp_path) -> None:
        cache = ConfigCache(cache_dir=tmp_path / "transfer_cache3")
        engine = TransferEngine(config_cache=cache)
        src = MappedTuningConfig(block_m=64, num_warps=16)
        engine.transfer("nvidia", "intel", src, kernel_hash="kernel_1")

        # Read the cached file and verify contents.
        cached = cache.get("kernel_1", "intel", "transferred")
        assert cached is not None
        assert "confidence" in cached
        assert cached["config"]["num_warps"] <= 16  # adapted (same or lower)


# ---------------------------------------------------------------------------
# Test transfer_config convenience function
# ---------------------------------------------------------------------------


class TestTransferConfigFunction:
    """One-shot transfer_config convenience function."""

    def test_basic_transfer(self) -> None:
        src = MappedTuningConfig(block_m=128, block_n=128)
        result = transfer_config("nvidia", "amd", src)
        assert isinstance(result, TransferredConfig)
        assert result.source_vendor == "nvidia"
        assert result.target_vendor == "amd"
        assert 0.5 <= result.confidence <= 1.0

    def test_with_engine_override(self) -> None:
        engine = TransferEngine(default_threshold=0.5)
        src = {"block_m": 256, "num_warps": 16}
        result = transfer_config("nvidia", "intel", src, engine=engine)
        assert isinstance(result, TransferredConfig)

    def test_with_cache(self, tmp_path) -> None:
        cache = ConfigCache(cache_dir=tmp_path / "fn_cache")
        src = MappedTuningConfig(block_m=128)
        transfer_config(
            "nvidia", "amd", src,
            kernel_hash="fn_test",
            config_cache=cache,
        )
        # Cache should be populated.
        cached = cache.get("fn_test", "amd", "transferred")
        assert cached is not None


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestTransferProperties:
    """Invariant properties that should hold for all transfers."""

    @given(
        source_vendor=_VALID_VENDORS,
        target_vendor=_VALID_VENDORS,
        config=config_dict_strategy,
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_tile_sizes_are_positive_multiples_of_16(
        self,
        source_vendor: str,
        target_vendor: str,
        config: dict[str, Any],
    ) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(**config)
        result = engine.transfer(source_vendor, target_vendor, src)

        # All tile sizes should be positive.
        assert result.config.block_m >= 16
        assert result.config.block_n >= 16
        assert result.config.block_k >= 8

        # Alignment only guaranteed for cross-vendor transfers
        # (same-vendor uses identity fallback which preserves input).
        if source_vendor != target_vendor:
            assert result.config.block_m % 16 == 0
            assert result.config.block_n % 16 == 0
            assert result.config.block_k % 16 == 0

    @given(
        source_vendor=_VALID_VENDORS,
        target_vendor=_VALID_VENDORS,
        config=config_dict_strategy,
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_warp_stages_are_sane(
        self,
        source_vendor: str,
        target_vendor: str,
        config: dict[str, Any],
    ) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(**config)
        result = engine.transfer(source_vendor, target_vendor, src)

        assert 1 <= result.config.num_warps <= 64
        assert 1 <= result.config.num_stages <= 8
        assert 1 <= result.config.num_ctas <= 4

    @given(
        source_vendor=_VALID_VENDORS,
        target_vendor=_VALID_VENDORS,
        config=config_dict_strategy,
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_confidence_in_bounds(
        self,
        source_vendor: str,
        target_vendor: str,
        config: dict[str, Any],
    ) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(**config)
        result = engine.transfer(source_vendor, target_vendor, src)

        assert 0.0 <= result.confidence <= 1.0

    @given(
        source_vendor=_VALID_VENDORS,
        target_vendor=_VALID_VENDORS,
        config=config_dict_strategy,
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_transfer_is_deterministic(
        self,
        source_vendor: str,
        target_vendor: str,
        config: dict[str, Any],
    ) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(**config)

        result1 = engine.transfer(source_vendor, target_vendor, src)
        result2 = engine.transfer(source_vendor, target_vendor, src)

        assert result1.config == result2.config
        assert result1.confidence == result2.confidence

    @given(config=config_dict_strategy)
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_adapt_tile_bounds(self, config: dict[str, Any]) -> None:
        engine = TransferEngine()
        m, n, k = engine.adapt_tile(
            config["block_m"],
            config["block_n"],
            config["block_k"],
            "nvidia",
            "amd",
        )
        # Adapted tiles should be within reasonable GPU tile bounds.
        assert 16 <= m <= 512
        assert 16 <= n <= 512
        assert 8 <= k <= 256
        assert m % 16 == 0
        assert n % 16 == 0
        assert k % 16 == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_minimal_config(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig()  # all defaults
        result = engine.transfer("nvidia", "amd", src)
        # Config is adapted via transfer matrix, so values != defaults
        assert result.config.block_m >= 16
        assert result.config.block_n >= 16
        assert result.config.block_k >= 8
        assert 0.5 <= result.confidence <= 1.0

    def test_maximal_config(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(
            block_m=512, block_n=512, block_k=256,
            num_warps=64, num_stages=8, num_ctas=4,
        )
        result = engine.transfer("nvidia", "amd", src)
        # Should clamp to sane values.
        assert result.config.block_m <= 512
        assert result.config.block_n <= 512
        assert result.config.block_k <= 256
        assert result.config.num_warps <= 40  # AMD max wavefronts
        assert result.config.num_stages <= 8

    def test_very_small_tiles(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(
            block_m=16, block_n=16, block_k=8,
            num_warps=2, num_stages=1,
        )
        result = engine.transfer("nvidia", "amd", src)
        # Minimum tile sizes.
        assert result.config.block_m >= 16
        assert result.config.block_n >= 16
        assert result.config.block_k >= 8
        assert result.config.num_warps >= 1
        assert result.config.num_stages >= 1

    def test_perf_db_exception_handled_gracefully(self) -> None:
        perf_db = MagicMock(spec=PerformanceDB)
        perf_db.get_historical_accuracy.side_effect = RuntimeError("DB down")

        engine = TransferEngine(perf_db=perf_db)
        src = MappedTuningConfig(block_m=128)
        # Should not raise — just logs warning.
        result = engine.transfer("nvidia", "amd", src, kernel_hash="abc")
        assert 0.5 <= result.confidence <= 1.0

    def test_num_ctas_defaults_to_one_for_non_nvidia(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(block_m=128, num_ctas=4)
        result = engine.transfer("nvidia", "amd", src)
        # num_ctas is a Hopper feature → defaults to 1 for AMD.
        assert result.config.num_ctas == 1

    def test_seed_config_contents(self) -> None:
        engine = TransferEngine()
        src = MappedTuningConfig(block_m=64, block_n=128, block_k=32,
                                 num_warps=8, num_stages=4)
        result = engine.transfer("nvidia", "amd", src)
        seed = result.seed_config
        assert seed["BLOCK_SIZE_M"] == result.config.block_m
        assert seed["BLOCK_SIZE_N"] == result.config.block_n
        assert seed["BLOCK_SIZE_K"] == result.config.block_k
        assert seed["num_warps"] == result.config.num_warps
        assert seed["num_stages"] == result.config.num_stages


# ---------------------------------------------------------------------------
# TRANSFER_MATRICES structural tests
# ---------------------------------------------------------------------------


class TestTransferMatrices:
    """Structural invariants of TRANSFER_MATRICES."""

    def test_all_pairs_have_required_keys(self) -> None:
        required = {"warp_scale", "shared_memory_ratio", "bw_ratio",
                    "register_ratio", "matrix_core_scale"}
        for pair, matrix in TRANSFER_MATRICES.items():
            missing = required - set(matrix.keys())
            assert not missing, f"{pair} missing keys: {missing}"

    def test_no_self_pairs(self) -> None:
        for src, tgt in TRANSFER_MATRICES:
            assert src != tgt, f"self-transfer {src}→{tgt} should not exist"

    def test_symmetry_not_required_but_plausible(self) -> None:
        """Reverse pairs should also exist."""
        pairs = set(TRANSFER_MATRICES.keys())
        for src, tgt in TRANSFER_MATRICES:
            assert (tgt, src) in pairs, (
                f"missing reverse pair {tgt}→{src}"
            )

    def test_all_vendors_covered(self) -> None:
        known_vendors = {"nvidia", "amd", "intel", "apple"}
        sources = {s for s, _ in TRANSFER_MATRICES}
        targets = {t for _, t in TRANSFER_MATRICES}
        assert sources == known_vendors
        assert targets == known_vendors

    def test_warp_scale_is_positive(self) -> None:
        for pair, matrix in TRANSFER_MATRICES.items():
            assert matrix["warp_scale"] > 0, f"{pair} has non-positive warp_scale"

    def test_shared_memory_ratio_is_positive(self) -> None:
        for pair, matrix in TRANSFER_MATRICES.items():
            assert matrix["shared_memory_ratio"] > 0, f"{pair} has non-positive smem_ratio"
