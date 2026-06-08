"""Integration tests for expert-guided auto-tuning.

Tests the full expert-guided tuning system:

* :mod:`src.bridges.triton_tvm.expert_rules` — vendor-specific rulesets,
  target matching, filter functions, and search space construction
* :mod:`src.bridges.triton_tvm.search_strategy` — strategy selection
  per (kernel_type, vendor) pair, registration, serialisation
* :mod:`src.bridges.triton_tvm.config_cache` — persistent config caching
  with content-addressable keys, atomic writes, invalidation
* :mod:`src.bridges.triton_tvm.kernel_templates` — import verification
  for all vendor-specific kernel templates
* :mod:`src.bridges.triton_tvm.bridge_orchestrator` — cache integration
  and ConfigCache roundtrip through the bridge layer

All tests use mocked TVM — no real GPU hardware or TVM installation required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.common.primitives import Vendor

# ===================================================================
# SECTION 1 — Expert Rules Tests
# ===================================================================


class TestExpertRulesLoad:
    """Verify that expert rules load correctly for all 4 vendors."""

    def test_all_vendor_rules_available(self) -> None:
        """All 4 vendors (and aliases) must be registered."""
        from src.bridges.triton_tvm.expert_rules import available_vendors

        vendors = available_vendors()
        # Check that all core vendor identifiers are present
        for expected in ("h100", "mi300x", "gaudi", "apple"):
            assert expected in vendors, f"Missing vendor: {expected}"
        # Check aliases
        for alias in ("h200", "mi250", "gaudi2", "gaudi3", "m3", "m4"):
            assert alias in vendors, f"Missing alias: {alias}"

    def test_get_vendor_rules_h100(self) -> None:
        """H100 rules must have Tensor Core enabled with correct defaults."""
        from src.bridges.triton_tvm.expert_rules import get_vendor_rules

        rules = get_vendor_rules("h100")
        assert rules is not None
        assert rules.vendor == "h100"
        assert rules.display_name == "Nvidia H100 (Hopper)"
        # Matmul rules should reflect Tensor Core
        assert rules.matmul.matrix_core == "mma"
        assert rules.matmul.enable_tma is True
        assert rules.matmul.confidence >= 0.9
        # Occupancy should have 32-thread warps
        assert rules.occupancy.warp_size == 32
        assert rules.occupancy.max_warps_per_sm == 64

    def test_get_vendor_rules_mi300x(self) -> None:
        """MI300X rules must have CDNA3 Matrix Core with 64-thread wavefronts."""
        from src.bridges.triton_tvm.expert_rules import get_vendor_rules

        rules = get_vendor_rules("mi300x")
        assert rules is not None
        assert rules.vendor == "mi300x"
        # Matrix Core with 64-thread wavefronts
        assert rules.matmul.matrix_core == "mfma_16x16x16"
        assert rules.occupancy.warp_size == 64
        assert rules.matmul.enable_tma is None

    def test_get_vendor_rules_gaudi(self) -> None:
        """Gaudi rules should be SIMD-only with no matrix core."""
        from src.bridges.triton_tvm.expert_rules import get_vendor_rules

        rules = get_vendor_rules("gaudi")
        assert rules is not None
        assert rules.vendor == "gaudi"
        assert rules.matmul.matrix_core is None
        assert rules.matmul.enable_tma is None
        # Gaudi has smaller tile sizes
        assert max(rules.matmul.tile_m) <= 128

    def test_get_vendor_rules_apple(self) -> None:
        """Apple rules should have unified memory characteristics."""
        from src.bridges.triton_tvm.expert_rules import get_vendor_rules

        rules = get_vendor_rules("apple")
        assert rules is not None
        assert rules.vendor == "apple"
        # Apple has smaller tiles due to limited threadgroup memory
        assert max(rules.matmul.tile_m) <= 64
        assert max(rules.matmul.tile_k) <= 32
        # Unified memory — no discrete L2
        assert rules.memory.l2_cache_size == 0

    def test_get_vendor_rules_unknown_returns_none(self) -> None:
        """Unknown vendor must return None."""
        from src.bridges.triton_tvm.expert_rules import get_vendor_rules

        assert get_vendor_rules("nonexistent_vendor") is None
        assert get_vendor_rules("") is None

    def test_match_target_all_vendors(self) -> None:
        """TVM target strings must map to correct vendor rules."""
        from src.bridges.triton_tvm.expert_rules import match_target

        test_cases: list[tuple[str, str]] = [
            ("nvidia/nvidia-h100", "h100"),
            ("nvidia/nvidia-a100", "h100"),
            ("nvidia/sm_90", "h100"),
            ("rocm/gfx942", "mi300x"),
            ("rocm/gfx90a", "mi300x"),
            ("intel/gaudi-2", "gaudi"),
            ("intel/gaudi-3", "gaudi"),
            ("apple/m3-gpu", "apple"),
            ("apple/metal", "apple"),
        ]
        for target_str, expected_vendor in test_cases:
            rules = match_target(target_str)
            assert rules is not None, f"No match for target: {target_str}"
            assert (
                rules.vendor == expected_vendor
            ), f"Expected {expected_vendor}, got {rules.vendor} for {target_str}"

    def test_match_target_unknown_returns_none(self) -> None:
        """Unrecognised target strings must return None."""
        from src.bridges.triton_tvm.expert_rules import match_target

        assert match_target("") is None
        assert match_target("totally/bogus-target-string") is None
        assert match_target("vulkan") is None

    def test_filter_matmul_configs_respects_dimensions(self) -> None:
        """Filtered matmul tiles must all be ≤ problem dimensions."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_matmul_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assert rules is not None

        # Small problem: only small tiles survive
        filtered = filter_matmul_configs(rules, m=64, n=128, k=32)
        assert all(t <= 64 for t in filtered.tile_m)
        assert all(t <= 128 for t in filtered.tile_n)
        assert all(t <= 32 for t in filtered.tile_k)
        # Larger tiles that don't fit must have been removed
        assert 256 not in filtered.tile_m
        assert 64 in filtered.tile_m

    def test_filter_matmul_configs_large_problem(self) -> None:
        """Large problem should retain all tile candidates."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_matmul_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assert rules is not None

        filtered = filter_matmul_configs(rules, m=4096, n=4096, k=4096)
        assert filtered.tile_m == rules.matmul.tile_m
        assert filtered.tile_n == rules.matmul.tile_n
        assert filtered.tile_k == rules.matmul.tile_k
        # Non-dimension params are preserved unchanged
        assert filtered.num_warps == rules.matmul.num_warps
        assert filtered.num_stages == rules.matmul.num_stages
        assert filtered.enable_tma == rules.matmul.enable_tma

    def test_filter_attention_configs_respects_sequence_lengths(self) -> None:
        """Filtered attention blocks must be ≤ sequence dimensions."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_attention_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assert rules is not None

        filtered = filter_attention_configs(rules, seq_len_q=64, seq_len_k=64, head_dim=32)
        assert all(b <= 64 for b in filtered.block_m)
        assert all(b <= 64 for b in filtered.block_n)
        assert all(b <= 32 for b in filtered.block_k)
        # 128 is too large for a 64-length sequence
        assert 128 not in filtered.block_m

    def test_build_search_space_kwargs(self) -> None:
        """Search space kwargs should include all tunable parameters."""
        from src.bridges.triton_tvm.expert_rules import (
            build_search_space_kwargs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assert rules is not None

        kwargs = build_search_space_kwargs(rules)
        # All parameters with >1 candidate should be present
        assert "tile_m" in kwargs
        assert "tile_n" in kwargs
        assert "tile_k" in kwargs
        assert "num_warps" in kwargs
        assert "num_stages" in kwargs
        # All values should be lists
        for key, val in kwargs.items():
            assert isinstance(val, list), f"{key} should be list, got {type(val)}"
            assert len(val) >= 2, f"{key} should have at least 2 candidates"


class TestExpertRulesValidation:
    """Edge cases and validation for expert rules."""

    def test_matmul_rules_confidence_zero_raises(self) -> None:
        """Confidence of 0.0 must raise ValueError."""
        from src.bridges.triton_tvm.expert_rules import MatmulRules

        with pytest.raises(ValueError, match="Confidence"):
            MatmulRules(
                tile_m=(64,),
                tile_n=(64,),
                tile_k=(32,),
                num_warps=(4,),
                num_stages=(3,),
                confidence=0.0,
            )

    def test_matmul_rules_empty_tile_raises(self) -> None:
        """Empty tile tuple must raise ValueError."""
        from src.bridges.triton_tvm.expert_rules import MatmulRules

        with pytest.raises(ValueError, match="tile_m"):
            MatmulRules(
                tile_m=(),
                tile_n=(64,),
                tile_k=(32,),
                num_warps=(4,),
                num_stages=(3,),
            )

    def test_occupancy_rules_bad_warp_size_raises(self) -> None:
        """Warp size not 32 or 64 must raise."""
        from src.bridges.triton_tvm.expert_rules import OccupancyRules

        with pytest.raises(ValueError, match="warp_size"):
            OccupancyRules(
                max_warps_per_sm=32,
                max_threads_per_sm=1024,
                max_registers_per_sm=65536,
                warp_size=128,
            )

    def test_memory_rules_negative_bandwidth_raises(self) -> None:
        """Negative HBM bandwidth must raise."""
        from src.bridges.triton_tvm.expert_rules import MemoryRules

        with pytest.raises(ValueError, match="hbm_bandwidth"):
            MemoryRules(
                shared_memory_per_block=1024,
                max_shared_memory=65536,
                l1_cache_size=1024,
                l2_cache_size=0,
                hbm_bandwidth=-1.0,
            )

    def test_occupancy_max_occupancy_warps_register_bound(self) -> None:
        """Occupancy calculation must respect register limits."""
        from src.bridges.triton_tvm.expert_rules import OccupancyRules

        rules = OccupancyRules(
            max_warps_per_sm=64,
            max_threads_per_sm=2048,
            max_registers_per_sm=65536,
            warp_size=32,
        )
        # Each thread uses 64 registers → 65536 / (64 * 32) = 32 warps
        assert rules.max_occupancy_warps(registers_per_thread=64) == 32
        # Each thread uses 256 registers → 65536 / (256 * 32) = 8 warps
        assert rules.max_occupancy_warps(registers_per_thread=256) == 8
        # Should not exceed hardware max
        assert rules.max_occupancy_warps(registers_per_thread=1) == 64

    def test_occupancy_max_occupancy_warps_smem_bound(self) -> None:
        """Occupancy calculation must respect shared memory limits."""
        from src.bridges.triton_tvm.expert_rules import OccupancyRules

        rules = OccupancyRules(
            max_warps_per_sm=64,
            max_threads_per_sm=2048,
            max_registers_per_sm=65536,
            warp_size=32,
        )
        # 64 KB smem, 16 KB per block → 4 blocks max → at least 64/4 = 16 warps each
        warps = rules.max_occupancy_warps(
            registers_per_thread=32,
            shared_mem_per_block=16 * 1024,
            max_shared_memory=64 * 1024,
        )
        assert warps > 0


class TestExpertRulesPropertyBased:
    """Property-based tests for expert rules using Hypothesis."""

    @given(
        m=st.integers(min_value=32, max_value=4096),
        n=st.integers(min_value=32, max_value=4096),
        k=st.integers(min_value=16, max_value=4096),
    )
    def test_filter_matmul_configs_property(self, m: int, n: int, k: int) -> None:
        """Filtered tile sizes must always be ≤ problem dimensions and non-empty."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_matmul_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assume(rules is not None)
        # Ensure at least one tile size fits each dimension
        assume(m >= min(rules.matmul.tile_m))
        assume(n >= min(rules.matmul.tile_n))
        assume(k >= min(rules.matmul.tile_k))

        filtered = filter_matmul_configs(rules, m=m, n=n, k=k)
        assert len(filtered.tile_m) >= 1
        assert len(filtered.tile_n) >= 1
        assert len(filtered.tile_k) >= 1
        assert all(t <= m for t in filtered.tile_m)
        assert all(t <= n for t in filtered.tile_n)
        assert all(t <= k for t in filtered.tile_k)

    @given(
        seq_len_q=st.integers(min_value=32, max_value=4096),
        seq_len_k=st.integers(min_value=32, max_value=4096),
        head_dim=st.integers(min_value=32, max_value=256),
    )
    def test_filter_attention_configs_property(
        self, seq_len_q: int, seq_len_k: int, head_dim: int
    ) -> None:
        """Filtered attention block sizes must be ≤ dimensions and non-empty."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_attention_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assume(rules is not None)
        assume(seq_len_q >= min(rules.attention.block_m))
        assume(seq_len_k >= min(rules.attention.block_n))
        assume(head_dim >= min(rules.attention.block_k))

        filtered = filter_attention_configs(
            rules, seq_len_q=seq_len_q, seq_len_k=seq_len_k, head_dim=head_dim
        )
        assert len(filtered.block_m) >= 1
        assert len(filtered.block_n) >= 1
        assert len(filtered.block_k) >= 1
        assert all(b <= seq_len_q for b in filtered.block_m)
        assert all(b <= seq_len_k for b in filtered.block_n)
        assert all(b <= head_dim for b in filtered.block_k)


# ===================================================================
# SECTION 2 — Search Strategy Tests
# ===================================================================


class TestSearchStrategy:
    """Verify that search strategies produce correct configs per kernel x vendor."""

    def test_get_strategy_all_combos_unique(self) -> None:
        """Every (kernel type, vendor) combination must return a strategy."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        for kt in KernelType:
            if kt is KernelType.UNKNOWN:
                continue
            for v in Vendor:
                if v is Vendor.UNKNOWN:
                    continue
                strategy = get_strategy(kt, v)
                assert strategy is not None, f"No strategy for {kt.name} x {v.value}"
                assert strategy.population_size >= 1

    def test_get_strategy_matmul_nvidia_vs_amd_different(self) -> None:
        """Matmul strategies for Nvidia vs AMD must differ in population size."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        nvidia = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)
        amd = get_strategy(KernelType.MATMUL, Vendor.AMD)
        _intel = get_strategy(KernelType.MATMUL, Vendor.INTEL)
        apple = get_strategy(KernelType.MATMUL, Vendor.APPLE)

        # Nvidia has the largest population (tensor-core heavy exploration)
        assert nvidia.population_size > amd.population_size
        # Apple has the smallest (limited tile-gang configs)
        assert nvidia.population_size > apple.population_size

    def test_get_strategy_different_kernels_different(self) -> None:
        """Different kernel types for same vendor must differ."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        matmul = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)
        elementwise = get_strategy(KernelType.ELEMENTWISE, Vendor.NVIDIA)
        reduction = get_strategy(KernelType.REDUCTION, Vendor.NVIDIA)

        # Matmul has the largest search space
        assert matmul.max_trials > elementwise.max_trials
        assert matmul.max_trials > reduction.max_trials
        # Elementwise is very small
        assert elementwise.max_trials < reduction.max_trials

    def test_get_strategy_elementwise_small_across_vendors(self) -> None:
        """Elementwise strategies should all be fast (small search spaces)."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        for v in Vendor:
            if v is Vendor.UNKNOWN:
                continue
            s = get_strategy(KernelType.ELEMENTWISE, v)
            assert s.max_trials <= 150, f"Elementwise x {v.value} has too many trials"
            assert s.population_size <= 48

    def test_get_strategy_unknown_fallback(self) -> None:
        """Unknown kernel type + vendor returns the default strategy."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        strategy = get_strategy(KernelType.UNKNOWN, Vendor.UNKNOWN)
        assert strategy.population_size == 64
        assert strategy.mutation_rate == 0.15
        assert strategy.crossover_rate == 0.75

    def test_get_strategy_with_string_vendor(self) -> None:
        """String vendor input must be normalised correctly."""
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy

        nvidia = get_strategy(KernelType.MATMUL, "nvidia")
        amd = get_strategy(KernelType.MATMUL, "amd")
        assert nvidia.population_size == 256
        assert amd.population_size == 128

    def test_get_strategy_with_any_object(self) -> None:
        """Any object with .name attribute matching a KernelType must work."""
        from src.bridges.triton_tvm.search_strategy import get_strategy

        class FakeKind:
            name = "MATMUL"

        strategy = get_strategy(FakeKind(), Vendor.NVIDIA)
        assert strategy.population_size == 256

    def test_register_and_override_strategy(self) -> None:
        """Registering a new strategy works; override guard prevents duplication."""
        from src.bridges.triton_tvm.search_strategy import (
            KernelType,
            SearchStrategy,
            get_strategy,
            register_strategy,
        )

        custom = SearchStrategy(
            population_size=999,
            description="Custom test strategy",
        )

        # Register
        register_strategy(KernelType.SCAN, Vendor.AMD, custom, override=True)
        retrieved = get_strategy(KernelType.SCAN, Vendor.AMD)
        assert retrieved.population_size == 999
        assert retrieved.description == "Custom test strategy"

        # Without override=True, duplicate should raise
        with pytest.raises(KeyError, match="already registered"):
            register_strategy(KernelType.SCAN, Vendor.AMD, custom, override=False)

    def test_list_strategies_returns_all(self) -> None:
        """list_strategies must return all registered entries."""
        from src.bridges.triton_tvm.search_strategy import list_strategies

        all_strategies = list_strategies()
        # Must contain all the primary kernel x vendor combos
        assert ("MATMUL", "nvidia") in all_strategies
        assert ("MATMUL", "amd") in all_strategies
        assert ("ATTENTION", "intel") in all_strategies
        assert ("ELEMENTWISE", "apple") in all_strategies

    def test_strategy_to_tune_kwargs(self) -> None:
        """strategy_to_tune_kwargs must produce correct MetaSchedule kwargs."""
        from src.bridges.triton_tvm.search_strategy import (
            SearchStrategy,
            strategy_to_tune_kwargs,
        )

        strategy = SearchStrategy(population_size=64, max_trials=500)
        kwargs = strategy_to_tune_kwargs(strategy)
        assert kwargs["num_trials_per_iter"] == 64
        assert kwargs["max_trials_global"] == 500

        # With overrides
        kwargs = strategy_to_tune_kwargs(strategy, override_population_size=128, override_max_trials=1000)
        assert kwargs["num_trials_per_iter"] == 128
        assert kwargs["max_trials_global"] == 1000

    def test_strategy_validation(self) -> None:
        """Invalid strategy parameters must raise ValueError."""
        from src.bridges.triton_tvm.search_strategy import SearchStrategy

        with pytest.raises(ValueError, match="population_size"):
            SearchStrategy(population_size=0)
        with pytest.raises(ValueError, match="mutation_rate"):
            SearchStrategy(mutation_rate=-0.1)
        with pytest.raises(ValueError, match="crossover_rate"):
            SearchStrategy(crossover_rate=1.5)
        with pytest.raises(ValueError, match="elite_ratio"):
            SearchStrategy(elite_ratio=-0.1)
        with pytest.raises(ValueError, match="max_trials"):
            SearchStrategy(max_trials=0)

    def test_strategy_num_trials_properties(self) -> None:
        """Convenience properties must align with fields."""
        from src.bridges.triton_tvm.search_strategy import SearchStrategy

        s = SearchStrategy(population_size=128, max_trials=800)
        assert s.num_trials_per_iter == 128
        assert s.max_trials_global == 800


class TestSearchStrategyPropertyBased:
    """Property-based tests for search strategies."""

    @given(
        population_size=st.integers(min_value=1, max_value=1024),
        mutation_rate=st.floats(min_value=0.0, max_value=1.0),
        crossover_rate=st.floats(min_value=0.0, max_value=1.0),
        elite_ratio=st.floats(min_value=0.0, max_value=1.0),
        max_trials=st.integers(min_value=1, max_value=10000),
    )
    def test_strategy_roundtrip_to_dict(
        self,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        elite_ratio: float,
        max_trials: int,
    ) -> None:
        """Strategy serialisation roundtrip must preserve all fields."""
        from src.bridges.triton_tvm.search_strategy import SearchStrategy

        s = SearchStrategy(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_ratio=elite_ratio,
            max_trials=max_trials,
        )
        d = s.to_dict()
        assert d["population_size"] == population_size
        assert d["mutation_rate"] == mutation_rate
        assert d["crossover_rate"] == crossover_rate
        assert d["elite_ratio"] == elite_ratio
        assert d["max_trials"] == max_trials


# ===================================================================
# SECTION 3 — Config Cache Tests
# ===================================================================


class TestConfigCache:
    """Verify that config cache correctly caches and retrieves."""

    def test_cache_get_set_roundtrip(self, tmp_path: Path) -> None:
        """Set then get must return the same config data."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        config = {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 8, "num_stages": 4}
        cache.set("hash123", "nvidia", "sm_90", config)
        retrieved = cache.get("hash123", "nvidia", "sm_90")
        assert retrieved is not None
        assert retrieved["block_m"] == 128
        assert retrieved["block_n"] == 128
        assert retrieved["num_warps"] == 8

    def test_cache_miss(self, tmp_path: Path) -> None:
        """Unknown key must return None."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        assert cache.get("nonexistent", "nvidia", "sm_90") is None

    def test_cache_key_deterministic(self) -> None:
        """Same inputs must produce the same cache key."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        key1 = ConfigCache._make_key("hash123", "nvidia", "sm_90")
        key2 = ConfigCache._make_key("hash123", "nvidia", "sm_90")
        assert key1 == key2
        assert len(key1) == 32  # 32 hex characters
        assert key1.isalnum()  # hex string

    def test_cache_key_different_vendor(self) -> None:
        """Different vendor must produce a different cache key."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        key_nvidia = ConfigCache._make_key("hash123", "nvidia", "sm_90")
        key_amd = ConfigCache._make_key("hash123", "amd", "gfx942")
        assert key_nvidia != key_amd

    def test_cache_key_different_arch(self) -> None:
        """Different architecture must produce a different cache key."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        key_sm90 = ConfigCache._make_key("hash123", "nvidia", "sm_90")
        key_sm80 = ConfigCache._make_key("hash123", "nvidia", "sm_80")
        assert key_sm90 != key_sm80

    def test_cache_invalidate(self, tmp_path: Path) -> None:
        """Invalidated entry must return None."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        cache.set("hash123", "nvidia", "sm_90", {"block_m": 128})
        assert cache.get("hash123", "nvidia", "sm_90") is not None
        cache.invalidate("hash123", "nvidia", "sm_90")
        assert cache.get("hash123", "nvidia", "sm_90") is None

    def test_cache_clear_all(self, tmp_path: Path) -> None:
        """Clear all must remove all cache entries."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        cache.set("hash1", "nvidia", "sm_90", {"block_m": 64})
        cache.set("hash2", "amd", "gfx942", {"block_m": 128})
        removed = cache.clear_all()
        assert removed == 2
        assert cache.get("hash1", "nvidia", "sm_90") is None
        assert cache.get("hash2", "amd", "gfx942") is None

    def test_cache_corrupted_file(self, tmp_path: Path) -> None:
        """Corrupted cache file must be silently removed and return None."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        # Write a file that will match the key but contains invalid JSON
        key = ConfigCache._make_key("bad", "nvidia", "sm_90")
        bogus_path = cache.cache_dir / f"{key}.json"
        bogus_path.parent.mkdir(parents=True, exist_ok=True)
        bogus_path.write_text("{invalid json!!!", encoding="utf-8")

        result = cache.get("bad", "nvidia", "sm_90")
        assert result is None
        # Corrupted file should have been deleted
        assert not bogus_path.exists()

    def test_cache_atomic_write(self, tmp_path: Path) -> None:
        """Cache writes must be atomic (tempfile + rename)."""
        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache = ConfigCache(cache_dir=tmp_path / "cache")
        config = {"block_m": 256, "block_n": 256, "block_k": 64}
        cache.set("atomic_test", "nvidia", "sm_90", config)

        # Verify the file exists and is proper JSON
        _key = ConfigCache._make_key("atomic_test", "nvidia", "sm_90")
        entries = list(cache.cache_dir.glob("*.json"))
        assert len(entries) == 1

        # No .tmp files should remain after atomic write
        assert not list(cache.cache_dir.glob("*.tmp"))


class TestConfigCachePropertyBased:
    """Property-based tests for config cache."""

    @given(
        block_m=st.integers(min_value=16, max_value=1024),
        block_n=st.integers(min_value=16, max_value=1024),
        block_k=st.integers(min_value=16, max_value=256),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20,
    )
    def test_cache_set_get_roundtrip_property(
        self, tmp_path: Path, block_m: int, block_n: int, block_k: int
    ) -> None:
        """Any valid config set into cache must be retrievable unchanged."""
        # Use a unique subdir per iteration so each generated example
        # starts with a clean cache (tmp_path is shared across examples).
        import uuid

        from src.bridges.triton_tvm.config_cache import ConfigCache

        cache_dir = tmp_path / str(uuid.uuid4())
        cache = ConfigCache(cache_dir=cache_dir)
        config = {"block_m": block_m, "block_n": block_n, "block_k": block_k, "num_warps": 8}
        cache.set("prop_test", "nvidia", "sm_90", config)
        retrieved = cache.get("prop_test", "nvidia", "sm_90")
        assert retrieved is not None
        assert retrieved["block_m"] == block_m
        assert retrieved["block_n"] == block_n
        assert retrieved["block_k"] == block_k


# ===================================================================
# SECTION 4 — Kernel Template Import Tests
# ===================================================================


class TestKernelTemplatesImport:
    """Verify that all kernel templates import correctly."""

    def test_matmul_templates_import(self) -> None:
        """All vendor matmul templates must import."""
        from src.bridges.triton_tvm.kernel_templates import (
            matmul_apple,
            matmul_gaudi,
            matmul_h100,
            matmul_mi300x,
        )

        assert cast(Any, matmul_h100).__name__ == "matmul_h100"
        assert cast(Any, matmul_mi300x).__name__ == "matmul_mi300x"
        assert cast(Any, matmul_gaudi).__name__ == "matmul_gaudi"
        assert cast(Any, matmul_apple).__name__ == "matmul_apple"

    def test_attention_templates_import(self) -> None:
        """All vendor attention templates must import."""
        from src.bridges.triton_tvm.kernel_templates import (
            attention_apple,
            attention_gaudi,
            attention_h100,
            attention_mi300x,
        )

        assert cast(Any, attention_h100).__name__ == "attention_h100"
        assert cast(Any, attention_mi300x).__name__ == "attention_mi300x"
        assert cast(Any, attention_gaudi).__name__ == "attention_gaudi"
        assert cast(Any, attention_apple).__name__ == "attention_apple"

    def test_normalization_templates_import(self) -> None:
        """All normalization templates must import."""
        from src.bridges.triton_tvm.kernel_templates import layer_norm, rms_norm, softmax

        assert cast(Any, layer_norm).__name__ == "layer_norm"
        assert cast(Any, rms_norm).__name__ == "rms_norm"
        assert cast(Any, softmax).__name__ == "softmax"

    def test_templates_are_triton_jit(self) -> None:
        """All templates must be @triton.jit decorated functions."""
        from src.bridges.triton_tvm.kernel_templates import (
            matmul_apple,
            matmul_gaudi,
            matmul_h100,
            matmul_mi300x,
        )

        for fn in (matmul_h100, matmul_mi300x, matmul_gaudi, matmul_apple):
            # JITFunction is the type of @triton.jit decorated functions
            from triton.runtime.jit import JITFunction

            assert isinstance(fn, JITFunction), f"{cast(Any, fn).__name__} is not @triton.jit"


# ===================================================================
# SECTION 5 — Integration Tests
# ===================================================================


class TestExpertRulesWithMetaScheduleAdapter:
    """Verify expert_rules.integrate() works with MetaSchedule adapter."""

    def test_build_search_space_kwargs_integration(self) -> None:
        """build_search_space_kwargs output must be usable by MetaSchedule adapter."""
        from src.bridges.triton_tvm.expert_rules import (
            build_search_space_kwargs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("mi300x")
        assert rules is not None

        kwargs = build_search_space_kwargs(rules)
        # The adapter accepts tile_m, tile_n, tile_k, num_warps, num_stages
        assert isinstance(kwargs, dict)
        # All values must be lists of ints
        for key, values in kwargs.items():
            for v in values:
                assert isinstance(v, int), f"{key} contains non-int: {v}"

    def test_expert_rules_filter_then_search_space(self) -> None:
        """Filtering then building search space must produce valid kwargs."""
        from src.bridges.triton_tvm.expert_rules import (
            build_search_space_kwargs,
            filter_matmul_configs,
            get_vendor_rules,
        )

        rules = get_vendor_rules("h100")
        assert rules is not None

        # Filter for a small problem
        filtered = filter_matmul_configs(rules, m=128, n=256, k=64)
        kwargs = build_search_space_kwargs(
            type("FakeVendorRules", (), {"matmul": filtered})()
        )
        # After filtering, tile_m with only one value shouldn't appear
        if len(filtered.tile_m) <= 1:
            assert "tile_m" not in kwargs

    def test_match_target_feeds_get_vendor_rules(self) -> None:
        """match_target output must be usable with get_vendor_rules."""
        from src.bridges.triton_tvm.expert_rules import match_target

        target_strs = [
            "nvidia/nvidia-h100",
            "rocm/gfx942",
            "intel/gaudi-2",
            "apple/m3-gpu",
        ]
        for target_str in target_strs:
            rules = match_target(target_str)
            assert rules is not None, f"No rules for {target_str}"
            # Must have all expected sub-rules
            assert rules.matmul is not None
            assert rules.attention is not None
            assert rules.elementwise is not None
            assert rules.memory is not None
            assert rules.occupancy is not None


class TestSearchStrategyWithBridgeOrchestrator:
    """Verify search strategy integrates with bridge orchestrator."""

    def test_search_strategy_with_auto_tuning_bridge(
        self, auto_tuning_bridge: Any
    ) -> None:
        """Auto-tuning bridge must accept strategies when tuning."""
        from src.bridges.triton_tvm.search_strategy import (
            KernelType,
            get_strategy,
        )
        from src.common.primitives import Vendor

        bridge = auto_tuning_bridge
        strategy = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)

        assert strategy is not None
        assert strategy.population_size >= 1

        # The bridge's tvm_adapter.tune accepts max_trials / num_trials_per_iter
        # which can be derived from the strategy
        if bridge.tvm_adapter is not None:
            assert hasattr(bridge.tvm_adapter, "tune")

    def test_strategy_to_tune_kwargs_with_bridge(
        self, auto_tuning_bridge: Any
    ) -> None:
        """Strategy kwargs must be compatible with MetaScheduleAdapter.tune."""
        from src.bridges.triton_tvm.search_strategy import (
            KernelType,
            get_strategy,
            strategy_to_tune_kwargs,
        )
        from src.common.primitives import Vendor

        strategy = get_strategy(KernelType.ATTENTION, Vendor.INTEL)
        kwargs = strategy_to_tune_kwargs(strategy)

        # These are the params MetaScheduleAdapter.tune accepts
        assert "num_trials_per_iter" in kwargs
        assert "max_trials_global" in kwargs
        assert kwargs["num_trials_per_iter"] >= 1
        assert kwargs["max_trials_global"] >= 1


class TestConfigCacheWithBridgeOrchestrator:
    """Verify config cache integration with bridge orchestrator."""

    def test_config_cache_in_bridge_orchestrator(
        self, auto_tuning_bridge: Any
    ) -> None:
        """Bridge orchestrator must have a ConfigCache instance."""
        bridge = auto_tuning_bridge
        assert hasattr(bridge, "config_cache")

        # The config_cache should be usable
        config = {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3}
        bridge.config_cache.set("test_key", "nvidia", "sm_90", config)
        retrieved = bridge.config_cache.get("test_key", "nvidia", "sm_90")
        assert retrieved is not None
        assert retrieved["block_m"] == 64

    def test_vendor_arch_from_target(self) -> None:
        """_vendor_arch_from_target must match common target strings."""
        from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge

        test_cases: list[tuple[str, str, str]] = [
            ("nvidia/nvidia-h100", "nvidia", "h100"),
            ("nvidia/nvidia-a100", "nvidia", "a100"),
            ("rocm/gfx942", "amd", "gfx942"),
            ("intel/gaudi-2", "intel", "gaudi2"),
            ("apple/m3-gpu", "apple", "apple_m3"),
            ("cuda", "nvidia", "generic"),
        ]
        for target, expected_vendor, expected_arch in test_cases:
            vendor, arch = TritonTVMBridge._vendor_arch_from_target(target)
            assert vendor == expected_vendor, f"{target}: expected vendor {expected_vendor}, got {vendor}"
            assert arch == expected_arch, f"{target}: expected arch {expected_arch}, got {arch}"

    def test_config_cache_integration_with_bridge_tune(
        self, auto_tuning_bridge: Any
    ) -> None:
        """Bridge's config_cache must support full set/get roundtrip.

        Note: ``_tuning_chain()`` does NOT write to ``config_cache`` — that
        happens in ``tune()`` and ``tune_with_real_ir()`` after the chain
        returns.  This test verifies the bridge's ConfigCache instance
        works as expected (it's the same ``ConfigCache`` class tested above,
        wired into the bridge).
        """
        from src.bridges.triton_tvm.bridge_orchestrator import TritonTVMBridge
        from src.bridges.triton_tvm.config_mapper import MappedTuningConfig

        bridge = auto_tuning_bridge
        vendor, arch = TritonTVMBridge._vendor_arch_from_target("nvidia/nvidia-h100")
        assert vendor == "nvidia"
        assert arch == "h100"

        # ConfigCache set/get via bridge instance
        import dataclasses

        config = MappedTuningConfig(
            block_m=128, block_n=256, block_k=64, num_warps=8, num_stages=4
        )
        config_dict = dataclasses.asdict(config)
        bridge.config_cache.set("bridge_test_key", vendor, arch, config_dict)
        cached = bridge.config_cache.get("bridge_test_key", vendor, arch)
        assert cached is not None
        assert cached["block_m"] == 128
        assert cached["block_n"] == 256
        assert cached["num_warps"] == 8


class TestFullPipelineComposition:
    """High-level integration tests composing multiple components."""

    def test_expert_rules_and_search_strategy_composition(self) -> None:
        """Expert rules + search strategy must produce coherent tuning parameters."""
        from src.bridges.triton_tvm.expert_rules import (
            build_search_space_kwargs,
            filter_matmul_configs,
            get_vendor_rules,
        )
        from src.bridges.triton_tvm.search_strategy import KernelType, get_strategy
        from src.common.primitives import Vendor

        # Get rules for H100 and filter for a typical problem
        rules = get_vendor_rules("h100")
        assert rules is not None

        problem_m, problem_n, problem_k = 4096, 4096, 4096
        filtered = filter_matmul_configs(rules, m=problem_m, n=problem_n, k=problem_k)

        # Get the search strategy
        strategy = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)

        # Build search space kwargs from the filtered rules
        kwargs = build_search_space_kwargs(
            type("FakeVendorRules", (), {"matmul": filtered})()
        )

        # Strategy + filtered search space must be coherent
        assert strategy.population_size >= len(kwargs.get("tile_m", [64]))
        # All filtered tiles must be valid
        for t in filtered.tile_m:
            assert t <= problem_m
        for t in filtered.tile_n:
            assert t <= problem_n
        for t in filtered.tile_k:
            assert t <= problem_k

    @given(
        m=st.integers(min_value=64, max_value=2048),
        n=st.integers(min_value=64, max_value=2048),
        k=st.integers(min_value=32, max_value=1024),
    )
    def test_full_expert_guided_pipeline_property(
        self, m: int, n: int, k: int
    ) -> None:
        """Full pipeline: rules → filter → strategy must produce valid config."""
        from src.bridges.triton_tvm.expert_rules import (
            filter_matmul_configs,
            get_vendor_rules,
        )
        from src.bridges.triton_tvm.search_strategy import (
            KernelType,
            get_strategy,
            strategy_to_tune_kwargs,
        )
        from src.common.primitives import Vendor

        rules = get_vendor_rules("h100")
        assume(rules is not None)
        assume(m >= min(rules.matmul.tile_m))
        assume(n >= min(rules.matmul.tile_n))
        assume(k >= min(rules.matmul.tile_k))

        # Step 1: Filter rules
        filtered = filter_matmul_configs(rules, m=m, n=n, k=k)
        assert len(filtered.tile_m) >= 1
        assert len(filtered.tile_n) >= 1
        assert len(filtered.tile_k) >= 1

        # Step 2: Get strategy
        strategy = get_strategy(KernelType.MATMUL, Vendor.NVIDIA)
        assert strategy.cache_enabled is True

        # Step 3: Convert to tune kwargs
        kwargs = strategy_to_tune_kwargs(strategy)
        assert kwargs["num_trials_per_iter"] >= 1
        assert kwargs["max_trials_global"] >= 1
        # NVIDIA matmul has the largest population
        assert kwargs["num_trials_per_iter"] >= 128
