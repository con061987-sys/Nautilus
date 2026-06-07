"""Tests for the TVMOptions backend configuration."""

from __future__ import annotations

import pytest

from src.bridges.triton_tvm.backend.options import TVMOptions


class TestTVMOptions:
    """Tests for the TVMOptions dataclass."""

    def test_defaults(self) -> None:
        """Default options should be sensible."""
        opts = TVMOptions()
        assert opts.max_trials == 64
        assert opts.search_strategy == "evolutionary"
        assert opts.cost_model == "xgb"
        assert opts.use_hybrid_autotune is False

    def test_invalid_max_trials(self) -> None:
        """max_trials < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_trials"):
            TVMOptions(max_trials=0)

    def test_invalid_search_strategy(self) -> None:
        """Unknown search strategy should raise."""
        with pytest.raises(ValueError, match="search_strategy"):
            TVMOptions(search_strategy="unknown_strategy")

    def test_invalid_cost_model(self) -> None:
        """Unknown cost model should raise."""
        with pytest.raises(ValueError, match="cost_model"):
            TVMOptions(cost_model="deep_mind_v3")

    def test_to_dict_roundtrip(self) -> None:
        """to_dict → from_dict should preserve the configuration."""
        original = TVMOptions(
            target="nvidia/nvidia-h100",
            max_trials=128,
            num_trials_per_iter=32,
            search_strategy="replay-trace",
            cost_model="mlp",
            use_hybrid_autotune=True,
            work_dir="/custom/path",
        )
        d = original.to_dict()
        restored = TVMOptions.from_dict(d)
        assert restored.target == original.target
        assert restored.max_trials == original.max_trials
        assert restored.num_trials_per_iter == original.num_trials_per_iter
        assert restored.search_strategy == original.search_strategy
        assert restored.cost_model == original.cost_model
        assert restored.use_hybrid_autotune == original.use_hybrid_autotune
        assert restored.work_dir == original.work_dir

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys in the dict should be ignored (or moved to extras)."""
        opts = TVMOptions.from_dict(
            {
                "max_trials": 128,
                "unknown_key": "value",
            }
        )
        assert opts.max_trials == 128
        assert "unknown_key" in opts.extra

    def test_to_dict_includes_extras(self) -> None:
        """Custom extras should appear in the dict output."""
        opts = TVMOptions(extra={"custom_flag": True, "tune_until": 0.99})
        d = opts.to_dict()
        assert d["custom_flag"] is True
        assert d["tune_until"] == 0.99
