"""Tests for src.runtime.async_checkpointer — race fix + security hardening.

Covers:
  H-18  Race condition in _checkpoint_loop: claim under lock, clear
        under lock, save outside lock; concurrent request_checkpoint
        calls must not be silently dropped.
  H-19  torch.load(weights_only=True) — no full-pickle RCE.
  H-20  Legacy-Python-pickle fallback removed; opt-in via
        CheckpointConfig.unsafe_pickle_fallback=True for tests.

The tests are stdlib-only (no torch) and exercise the public API
plus the private serializers to keep coverage focused on the
hardening surface.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))

_errors_mod = importlib.import_module("src.common.errors")
DependencyMissingError = _errors_mod.DependencyMissingError

_async_checkpointer_mod = importlib.import_module("src.runtime.async_checkpointer")
SCHEMA_VERSION = _async_checkpointer_mod.SCHEMA_VERSION
AsyncCheckpointer = _async_checkpointer_mod.AsyncCheckpointer
CheckpointConfig = _async_checkpointer_mod.CheckpointConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    p = tmp_path / "checkpoints"
    p.mkdir()
    return p


def _state(d: dict) -> dict:
    """Plain-dict state for serialization tests."""
    return d


# ---------------------------------------------------------------------------
# H-18: race condition
# ---------------------------------------------------------------------------


class TestCheckpointLoopRace:
    """Concurrent request_checkpoint() must not lose requests.

    The original loop read _pending_model and _pending_optimizer
    outside the lock; a concurrent request_checkpoint could
    overwrite those references after the read and before the
    clear, causing the new request to be silently dropped.

    The checkpointer is documented to coalesce rapid requests
    onto the LATEST model (only one checkpoint per interval);
    the race fix guarantees that the LATEST model is what
    gets saved, not that every request triggers a save.
    """

    def test_latest_model_saved_under_contention(
        self,
        storage_path: Path,
    ) -> None:
        """N rapid requests coalesce to 1 checkpoint, but the
        saved checkpoint must reflect the LATEST request — the
        race fix guarantees we don't save a stale object.
        """
        cfg = CheckpointConfig(
            interval_seconds=0.05,
            storage_path=str(storage_path),
        )
        cp = AsyncCheckpointer(cfg)
        cp.start()
        try:
            for i in range(20):
                cp.request_checkpoint(_state({"step": i, "w": [float(i)] * 8}))
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with cp._lock:
                    pending = cp._pending_model
                if pending is None and len(cp._checkpoints) >= 1:
                    break
                time.sleep(0.05)
            with cp._lock:
                assert cp._pending_model is None, "Loop did not clear pending slot under lock"
                assert len(cp._checkpoints) == 1, (
                    f"Expected 1 coalesced checkpoint, got {len(cp._checkpoints)}"
                )
                saved = cp._checkpoints[0]
            model_bytes = (saved.path / "model.state").read_bytes()
            loaded = cp._deserialize_state(model_bytes)
            assert loaded["step"] == 19, (
                f"Latest step was 19, but checkpoint saved step "
                f"{loaded['step']} — race caused a stale save"
            )
        finally:
            cp.stop()

    def test_claim_clears_pending_atomically(
        self,
        storage_path: Path,
    ) -> None:
        """The loop clears the pending slot under the same lock
        the caller uses to set it. Verified by running the
        loop thread to completion of one save.
        """
        cfg = CheckpointConfig(
            interval_seconds=0.05,
            storage_path=str(storage_path),
        )
        cp = AsyncCheckpointer(cfg)
        cp.start()
        try:
            cp.request_checkpoint(_state({"step": 0}))
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with cp._lock:
                    if cp._pending_model is None and len(cp._checkpoints) >= 1:
                        break
                time.sleep(0.05)
            with cp._lock:
                assert cp._pending_model is None
                assert cp._pending_optimizer is None
                assert len(cp._checkpoints) == 1
        finally:
            cp.stop()

    def test_request_checkpoint_locked_writes(
        self,
        storage_path: Path,
    ) -> None:
        """request_checkpoint writes model/optimizer under self._lock."""
        cfg = CheckpointConfig(
            interval_seconds=60.0,
            storage_path=str(storage_path),
        )
        cp = AsyncCheckpointer(cfg)
        cp.request_checkpoint(_state({"a": 1}), _state({"lr": 0.1}))
        with cp._lock:
            assert cp._pending_model == {"a": 1}
            assert cp._pending_optimizer == {"lr": 0.1}
            assert cp._pending_event.is_set()


# ---------------------------------------------------------------------------
# H-19 / H-20: torch weights_only=True + pickle fallback removed
# ---------------------------------------------------------------------------


class TestDeserializeSecurity:
    """The deserializer must never run pickle deserialisation on
    attacker bytes by default."""

    def test_no_torch_no_msgpack_default_raises(
        self,
        storage_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With torch+msgpack hidden, default config raises
        DependencyMissingError (not the legacy-pickle fallback)."""
        cfg = CheckpointConfig(
            storage_path=str(storage_path),
            unsafe_pickle_fallback=False,
        )
        cp = AsyncCheckpointer(cfg)
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "msgpack", None)
        payload = b"\x80\x04K\x01."
        with pytest.raises(DependencyMissingError):
            cp._deserialize_state(payload)

    def test_unsafe_pickle_fallback_opt_in(
        self,
        storage_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests can opt into pickle via CheckpointConfig."""
        cfg = CheckpointConfig(
            storage_path=str(storage_path),
            unsafe_pickle_fallback=True,
        )
        cp = AsyncCheckpointer(cfg)
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "msgpack", None)
        payload = pickle.dumps({"hello": "world"})
        assert cp._deserialize_state(payload) == {"hello": "world"}

    def test_unsafe_pickle_fallback_default_false(
        self,
        storage_path: Path,
    ) -> None:
        cfg = CheckpointConfig()
        assert cfg.unsafe_pickle_fallback is False

    def test_serialize_deserialize_roundtrip(
        self,
        storage_path: Path,
    ) -> None:
        """End-to-end save + load via msgpack path (no torch)."""
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        state = _state({"layers": [1, 2, 3], "name": "tiny"})
        info = cp.checkpoint_now(state)
        assert info.checkpoint_id > 0
        # Deserialize the bytes that were written
        data = (info.path / "model.state").read_bytes()
        loaded = cp._deserialize_state(data)
        assert loaded == state

    def test_torch_load_called_with_weights_only_true(
        self,
        storage_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If torch is available, weights_only=True is passed."""
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        calls: list[dict] = []

        class _FakeIO:
            def __init__(self, b: bytes) -> None:
                self.b = b

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_torch_load(buf, **kwargs):
            calls.append({"weights_only": kwargs.get("weights_only")})
            return {"from_torch": True}

        class _FakeTorch:
            load = staticmethod(fake_torch_load)

        fake_torch = _FakeTorch()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        result = cp._deserialize_state(b"anything")
        assert result == {"from_torch": True}
        assert calls, "torch.load was not called"
        assert calls[0]["weights_only"] is True, (
            f"weights_only was {calls[0]['weights_only']!r}, must be True"
        )


# ---------------------------------------------------------------------------
# Sanity: round-trip the schema, public API still works
# ---------------------------------------------------------------------------


class TestPublicAPIUnchanged:
    def test_schema_version(self) -> None:
        assert SCHEMA_VERSION == 2

    def test_checkpoint_info_creation(
        self,
        storage_path: Path,
    ) -> None:
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        info = cp.checkpoint_now(_state({"x": 1}))
        assert info.path.exists()
        assert (info.path / "meta.json").exists()
        assert (info.path / "model.state").exists()

    def test_recover_latest_returns_state(
        self,
        storage_path: Path,
    ) -> None:
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        state = _state({"w": [1.0, 2.0, 3.0]})
        cp.checkpoint_now(state)
        result = cp.recover_latest()
        assert result is not None
        model_state, _ = result
        assert model_state == state

    def test_on_node_failure_triggers_recovery(
        self,
        storage_path: Path,
    ) -> None:
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        cp.checkpoint_now(_state({"v": 7}))
        assert cp.on_node_failure("node-0") is True

    def test_on_node_failure_no_checkpoint(
        self,
        storage_path: Path,
    ) -> None:
        cfg = CheckpointConfig(storage_path=str(storage_path))
        cp = AsyncCheckpointer(cfg)
        assert cp.on_node_failure("node-0") is False
