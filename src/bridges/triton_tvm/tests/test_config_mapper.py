"""Tests for the TVM-to-Triton config mapper."""

from __future__ import annotations

import json

from src.bridges.triton_tvm.config_mapper import ConfigMapper, MappedTuningConfig


class TestMappedTuningConfig:
    """Tuning config construction and Triton conversion."""

    def test_defaults(self) -> None:
        """Default config should use sensible values."""
        config = MappedTuningConfig.defaults()
        assert config.block_m == 128
        assert config.block_n == 128
        assert config.block_k == 32
        assert config.num_warps == 4
        assert config.num_stages == 3

    def test_to_triton_config(self) -> None:
        """to_triton_config should produce a valid triton.Config."""
        config = MappedTuningConfig(
            block_m=64,
            block_n=128,
            block_k=32,
            num_warps=8,
            num_stages=4,
        )
        triton_cfg = config.to_triton_config({"BLOCK_SIZE": 64})
        assert triton_cfg.kwargs["BLOCK_SIZE_M"] == 64
        assert triton_cfg.kwargs["BLOCK_SIZE_N"] == 128
        assert triton_cfg.kwargs["BLOCK_SIZE_K"] == 32
        assert triton_cfg.kwargs["BLOCK_SIZE"] == 64
        assert triton_cfg.num_warps == 8
        assert triton_cfg.num_stages == 4
        assert triton_cfg.num_ctas == 1


class TestConfigMapper:
    """Mapping TVM tuning records to Triton configs."""

    def test_map_with_tile_decisions(self) -> None:
        """TVM trace with tile decisions should map correctly."""
        mapper = ConfigMapper()
        trace = {
            "instructions": [{"type": "MultiLevelTiling"}],
            "decisions": {
                "tile_m": [1, 2, 4],  # product = 8
                "tile_n": [1, 2, 4],  # product = 8
                "tile_k": [2, 4],  # product = 8
                "stages": 4,
            },
        }
        config = mapper.map_record(trace)
        assert config.block_m == 8
        assert config.block_n == 8
        assert config.block_k == 8
        assert config.num_stages == 4

    def test_map_with_thread_decisions(self) -> None:
        """TVM trace with thread binding should map num_warps."""
        mapper = ConfigMapper()
        trace = {
            "instructions": [],
            "decisions": {
                "thread_binding": [128, 1],  # 128 threads → 128/32 = 4 warps
            },
        }
        config = mapper.map_record(trace)
        assert config.num_warps == 4  # 128*1 = 128 threads, /32 = 4 warps

    def test_map_empty_trace(self) -> None:
        """Empty trace should return default config."""
        mapper = ConfigMapper()
        config = mapper.map_record({})
        assert config == MappedTuningConfig.defaults()

    def test_map_none_record(self) -> None:
        """None trace should return default config."""
        mapper = ConfigMapper()
        config = mapper.map_record(None)
        assert config == MappedTuningConfig.defaults()

    def test_map_json_record(self) -> None:
        """JSON-serialized record should parse correctly."""
        mapper = ConfigMapper()
        json_record = json.dumps(
            {
                "decisions": {"tile_m": [1, 8, 2], "tile_n": [1, 4, 4]},
            }
        )
        config = mapper.map_json_record(json_record)
        assert config.block_m == 16  # 1*8*2
        assert config.block_n == 16  # 1*4*4

    def test_map_real_tvm_like_trace(self) -> None:
        """Simulate a real TVM tuning record structure."""
        mapper = ConfigMapper()

        # Simulate what TVM's database returns
        class FakeRecord:
            class FakeTrace:
                def __init__(self) -> None:
                    self.__dict__ = {
                        "instructions": [{"type": "MultiLevelTiling"}, {"type": "AutoBind"}],
                        "decisions": {
                            "tile_m": [2, 4, 4],  # 32
                            "tile_n": [2, 4, 4],  # 32
                            "tile_k": [4, 4],  # 16
                            "thread_binding": [256, 1],  # 256 threads = 8 warps
                            "stages": 5,
                        },
                    }

            def __init__(self) -> None:
                self.trace = self.FakeTrace()

        config = mapper.map_record(FakeRecord())
        assert config.block_m == 32
        assert config.block_n == 32
        assert config.block_k == 16
        assert config.num_warps == 8
        assert config.num_stages == 5
