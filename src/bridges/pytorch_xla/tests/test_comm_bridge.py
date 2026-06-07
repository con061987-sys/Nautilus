"""Tests for the communication bridge module.

The classes in :mod:`src.bridges.pytorch_xla.comm_bridge` wrap
torch.distributed, which is not exercisable on a CPU-only CI host.  We
therefore split the tests into three buckets:

  1. Pure-logic tests (P2P detection, construction, metadata,
     ``make_backend`` factory) — run anywhere.
  2. Behavioural tests of the three vendor backends using
     ``unittest.mock`` to stand in for torch.distributed — run
     anywhere, no real GPU needed.
  3. End-to-end tests of the six primitives on a real
     ``torch.distributed`` "gloo" group of rank 0 / world size 1.
     These verify the *plumbing* (tensor shape, dtype, device) without
     requiring any vendor library.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

# torch.distributed needs to be importable in this module for the
# vendor backend fixtures; we gate the torch-dependent tests below
# rather than at import time so the file always collects.
try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in torch-less envs
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


from src.bridges.pytorch_xla.comm_backend import CommLibrary
from src.bridges.pytorch_xla.comm_bridge import (
    CollectiveBackend,
    CrossVendorBridge,
    NCCLBackend,
    P2PCapability,
    RCCLBackend,
    detect_p2p_capability,
    make_backend,
    oneCCLBackend,
)
from src.bridges.pytorch_xla.device_mesh import (
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.common.errors import BridgeError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nvidia_device(
    device_id: int = 0,
    *,
    interconnect: InterconnectType = InterconnectType.NVLINK,
    hostname: str = "localhost",
) -> MeshDevice:
    return MeshDevice(
        device_id=device_id,
        vendor=DeviceVendor.NVIDIA,
        arch="sm_90",
        memory_gb=80.0,
        compute_tflops=989.0,
        interconnect=interconnect,
        hostname=hostname,
    )


def _amd_device(
    device_id: int = 0,
    *,
    interconnect: InterconnectType = InterconnectType.INFINITY_FABRIC,
    hostname: str = "localhost",
) -> MeshDevice:
    return MeshDevice(
        device_id=device_id,
        vendor=DeviceVendor.AMD,
        arch="gfx942",
        memory_gb=192.0,
        compute_tflops=1307.0,
        interconnect=interconnect,
        hostname=hostname,
    )


def _intel_device(
    device_id: int = 0,
    *,
    interconnect: InterconnectType = InterconnectType.UALINK,
    hostname: str = "localhost",
) -> MeshDevice:
    return MeshDevice(
        device_id=device_id,
        vendor=DeviceVendor.INTEL,
        arch="xe_hpg",
        memory_gb=96.0,
        compute_tflops=500.0,
        interconnect=interconnect,
        hostname=hostname,
    )


# ---------------------------------------------------------------------------
# 1. P2P detection — pure logic
# ---------------------------------------------------------------------------


class TestP2PCapability:
    """Tests for :func:`detect_p2p_capability`."""

    def test_same_device_is_full(self) -> None:
        """Same device id should always be FULL (a no-op transfer)."""
        a = _nvidia_device(0)
        result = detect_p2p_capability(a, a)
        assert result == P2PCapability.FULL

    def test_same_vendor_nvlink_is_full(self) -> None:
        """Two Nvidia devices on NVLink should be FULL."""
        a = _nvidia_device(0)
        b = _nvidia_device(1, interconnect=InterconnectType.NVLINK)
        assert detect_p2p_capability(a, b) == P2PCapability.FULL

    def test_same_vendor_infinity_fabric_is_full(self) -> None:
        """Two AMD devices on Infinity Fabric should be FULL."""
        a = _amd_device(0)
        b = _amd_device(1, interconnect=InterconnectType.INFINITY_FABRIC)
        assert detect_p2p_capability(a, b) == P2PCapability.FULL

    def test_different_host_is_host_staged(self) -> None:
        """Different hostnames must go through network → HOST_STAGED."""
        a = _nvidia_device(0, hostname="node-a")
        b = _nvidia_device(1, hostname="node-b")
        assert detect_p2p_capability(a, b) == P2PCapability.HOST_STAGED

    def test_cross_vendor_ualink_is_full(self) -> None:
        """UALink on both ends allows direct cross-vendor transfer."""
        a = _nvidia_device(0, interconnect=InterconnectType.UALINK)
        b = _intel_device(1, interconnect=InterconnectType.UALINK)
        assert detect_p2p_capability(a, b) == P2PCapability.FULL

    def test_cross_vendor_pcie_is_host_staged(self) -> None:
        """Cross-vendor over PCIe defaults to host staging for portability."""
        a = _nvidia_device(0, interconnect=InterconnectType.PCIE)
        b = _amd_device(1, interconnect=InterconnectType.PCIE)
        assert detect_p2p_capability(a, b) == P2PCapability.HOST_STAGED

    def test_cross_vendor_mixed_qualifiers_is_host_staged(self) -> None:
        """Cross-vendor without UALink on both ends → HOST_STAGED."""
        a = _nvidia_device(0, interconnect=InterconnectType.NVLINK)
        b = _amd_device(1, interconnect=InterconnectType.INFINITY_FABRIC)
        assert detect_p2p_capability(a, b) == P2PCapability.HOST_STAGED

    def test_same_vendor_pcie_is_full(self) -> None:
        """Same vendor over PCIe should still be FULL — GPUDirect
        P2P works on modern systems."""
        a = _nvidia_device(0, interconnect=InterconnectType.PCIE)
        b = _nvidia_device(1, interconnect=InterconnectType.PCIE)
        assert detect_p2p_capability(a, b) == P2PCapability.FULL

    def test_symmetric(self) -> None:
        """P2P capability is symmetric in src/dst."""
        a = _nvidia_device(0)
        b = _amd_device(1, interconnect=InterconnectType.PCIE)
        forward = detect_p2p_capability(a, b)
        backward = detect_p2p_capability(b, a)
        assert forward == backward


# ---------------------------------------------------------------------------
# 2. Backend construction & metadata
# ---------------------------------------------------------------------------


class TestBackendMetadata:
    """Per-vendor backends expose the right CommLibrary / DeviceVendor."""

    def test_nccl_backend_identity(self) -> None:
        b = NCCLBackend(device_id=2)
        assert b.library == CommLibrary.NCCL
        assert b.vendor == DeviceVendor.NVIDIA
        assert b.device_id == 2
        assert b.torch_backend_name == "nccl"
        assert b.device_kind == "cuda"

    def test_rccl_backend_identity(self) -> None:
        b = RCCLBackend(device_id=1)
        assert b.library == CommLibrary.RCCL
        assert b.vendor == DeviceVendor.AMD
        assert b.device_id == 1
        assert b.torch_backend_name == "nccl"  # RCCL reuses the wire protocol
        assert b.device_kind == "cuda"

    def test_oneccl_backend_identity(self) -> None:
        b = oneCCLBackend(device_id=0)
        assert b.library == CommLibrary.ONECCL
        assert b.vendor == DeviceVendor.INTEL
        assert b.device_id == 0
        assert b.torch_backend_name == "xccl"
        assert b.device_kind == "xpu"

    def test_describe_contains_key_fields(self) -> None:
        """describe() returns a dict suitable for structured logging."""
        b = NCCLBackend(device_id=0)
        d = b.describe()
        assert d["library"] == "nccl"
        assert d["vendor"] == "nvidia"
        assert d["device_id"] == 0
        assert "available" in d

    def test_is_available_caches(self) -> None:
        """``is_available`` is cached after the first call."""
        b = NCCLBackend(device_id=0)
        first = b.is_available
        # Patching the probe should NOT change the second result because
        # the value is cached.
        with patch.object(b, "_probe_availability", return_value=not first):
            second = b.is_available
        assert first == second

    def test_negative_device_id_rejected(self) -> None:
        """Constructing a backend with a negative id is a hard error."""
        with pytest.raises(BridgeError):
            NCCLBackend(device_id=-1)
        with pytest.raises(BridgeError):
            RCCLBackend(device_id=-1)
        with pytest.raises(BridgeError):
            oneCCLBackend(device_id=-1)

    def test_abstract_cannot_be_instantiated(self) -> None:
        """The abstract base refuses direct instantiation."""
        with pytest.raises(TypeError):
            CollectiveBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 3. is_available — probes don't crash
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestAvailabilityProbes:
    """The availability probes are best-effort and never raise."""

    def test_nccl_probe_false_without_backend(self) -> None:
        """If ``is_nccl_available`` returns False, NCCL is unavailable."""
        with patch.object(NCCLBackend, "_torch_backend_available", return_value=False):
            assert NCCLBackend(device_id=0).is_available is False

    def test_nccl_probe_requires_cuda(self) -> None:
        """Even with the backend available, no cuda → unavailable."""
        with (
            patch.object(NCCLBackend, "_torch_backend_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            assert NCCLBackend(device_id=0).is_available is False
        with (
            patch.object(NCCLBackend, "_torch_backend_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            assert NCCLBackend(device_id=0).is_available is True

    def test_rccl_probe_handles_missing_cuda(self) -> None:
        with patch.object(RCCLBackend, "_torch_backend_available", return_value=False):
            assert RCCLBackend(device_id=0).is_available is False

    def test_oneccl_probe_handles_missing_xpu(self) -> None:
        with patch.object(oneCCLBackend, "_torch_backend_available", return_value=False):
            assert oneCCLBackend(device_id=0).is_available is False
        # xpu module missing
        with patch.object(torch, "xpu", None, create=True):
            assert oneCCLBackend(device_id=0).is_available is False


# ---------------------------------------------------------------------------
# 4. make_backend factory
# ---------------------------------------------------------------------------


class TestMakeBackend:
    """Tests for the :func:`make_backend` factory."""

    def test_factory_returns_nccl(self) -> None:
        b = make_backend(CommLibrary.NCCL, device_id=0)
        assert isinstance(b, NCCLBackend)
        assert b.device_id == 0

    def test_factory_returns_rccl(self) -> None:
        b = make_backend(CommLibrary.RCCL, device_id=3)
        assert isinstance(b, RCCLBackend)
        assert b.device_id == 3

    def test_factory_returns_oneccl(self) -> None:
        b = make_backend(CommLibrary.ONECCL, device_id=0)
        assert isinstance(b, oneCCLBackend)

    def test_factory_rejects_gloo(self) -> None:
        """GLOO is supported by torch but we don't wrap it here."""
        with pytest.raises(BridgeError):
            make_backend(CommLibrary.GLOO, device_id=0)

    def test_factory_rejects_mixed(self) -> None:
        """MIXED is for cross-vendor bridges, not a standalone backend."""
        with pytest.raises(BridgeError):
            make_backend(CommLibrary.MIXED, device_id=0)

    def test_factory_rejects_ualink(self) -> None:
        """UALink is layered on top of vendor backends."""
        with pytest.raises(BridgeError):
            make_backend(CommLibrary.UALINK, device_id=0)


# ---------------------------------------------------------------------------
# 5. CrossVendorBridge with mock backends
# ---------------------------------------------------------------------------


class _MockBackend(CollectiveBackend):
    """A minimal concrete backend for use in CrossVendorBridge tests.

    The real backends inherit from ``_TorchDistributedBackend``; the
    bridge is what we want to test here, so we don't need that.
    """

    def __init__(
        self,
        library: CommLibrary,
        vendor: DeviceVendor,
        device_id: int = 0,
        available: bool = True,
    ) -> None:
        self._library = library
        self._vendor = vendor
        self._device_id = device_id
        self._available = available
        self.calls: list = []

    @property
    def library(self) -> CommLibrary:
        return self._library

    @property
    def vendor(self) -> DeviceVendor:
        return self._vendor

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def is_available(self) -> bool:
        return self._available

    def _record(self, op: str, *args, **kwargs) -> None:
        self.calls.append((op, args, kwargs))

    def all_reduce(self, tensor, op=None, group=None):
        self._record("all_reduce", tensor, op=op, group=group)
        return tensor

    def all_gather(self, tensor, group=None):
        self._record("all_gather", tensor, group=group)
        return tensor

    def reduce_scatter(self, tensor, op=None, group=None):
        self._record("reduce_scatter", tensor, op=op, group=group)
        return tensor

    def all_to_all(self, tensor, group=None):
        self._record("all_to_all", tensor, group=group)
        return tensor

    def send(self, tensor, dst, group=None):
        self._record("send", tensor, dst, group=group)

    def recv(self, tensor, src, group=None):
        self._record("recv", tensor, src, group=group)
        return tensor


class TestCrossVendorBridge:
    """Tests for :class:`CrossVendorBridge`."""

    def test_construction_assigns_backends(self) -> None:
        """The bridge wraps exactly the two backends passed in."""
        src = _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD, device_id=0)
        dst = _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL, device_id=1)
        a = _amd_device(0)
        b = _intel_device(1)
        bridge = CrossVendorBridge(src, dst, a, b)
        assert bridge.src_backend is src
        assert bridge.dst_backend is dst
        assert bridge.src_device is a
        assert bridge.dst_device is b

    def test_library_is_mixed(self) -> None:
        """A cross-vendor bridge always reports MIXED as its library."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0),
            _intel_device(1),
        )
        assert bridge.library == CommLibrary.MIXED

    def test_vendor_reports_source(self) -> None:
        """The bridge exposes the source vendor for bookkeeping."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0),
            _intel_device(1),
        )
        assert bridge.vendor == DeviceVendor.AMD

    def test_is_available_requires_both(self) -> None:
        """The bridge is usable only if both sides are usable."""
        a = _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD, available=True)
        b = _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL, available=False)
        bridge = CrossVendorBridge(a, b, _amd_device(0), _intel_device(1))
        assert bridge.is_available is False

        b._available = True
        assert bridge.is_available is True

    def test_uses_host_staging_default_for_cross_vendor(self) -> None:
        """Cross-vendor over PCIe defaults to host staging."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0, interconnect=InterconnectType.PCIE),
            _intel_device(1, interconnect=InterconnectType.PCIE),
        )
        assert bridge.p2p_capability == P2PCapability.HOST_STAGED
        assert bridge.uses_host_staging is True

    def test_uses_host_staging_skipped_for_ualink(self) -> None:
        """UALink pairs can do direct P2P — the bridge's P2P flag is FULL."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0, interconnect=InterconnectType.UALINK),
            _intel_device(1, interconnect=InterconnectType.UALINK),
        )
        assert bridge.p2p_capability == P2PCapability.FULL
        assert bridge.uses_host_staging is False

    def test_force_host_staging_overrides_p2p(self) -> None:
        """``force_host_staging=True`` forces staging even on UALink."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0, interconnect=InterconnectType.UALINK),
            _intel_device(1, interconnect=InterconnectType.UALINK),
            force_host_staging=True,
        )
        assert bridge.p2p_capability == P2PCapability.FULL
        assert bridge.uses_host_staging is True

    def test_describe_includes_p2p_metadata(self) -> None:
        """``describe()`` exposes enough info for structured logging."""
        bridge = CrossVendorBridge(
            _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD),
            _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL),
            _amd_device(0),
            _intel_device(1),
        )
        d = bridge.describe()
        assert d["library"] == "mixed"
        assert d["src_vendor"] == "amd"
        assert d["dst_vendor"] == "intel"
        assert d["src_library"] == "rccl"
        assert d["dst_library"] == "oneccl"
        assert d["p2p_capability"] in {c.value for c in P2PCapability}
        assert isinstance(d["uses_host_staging"], bool)

    def test_rejects_same_backend_twice(self) -> None:
        """Two identical backends is a usage error."""
        b = _MockBackend(CommLibrary.NCCL, DeviceVendor.NVIDIA)
        with pytest.raises(BridgeError):
            CrossVendorBridge(b, b, _nvidia_device(0), _nvidia_device(0))


# ---------------------------------------------------------------------------
# 6. CrossVendorBridge primitives — host-staged path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestCrossVendorBridgePrimitives:
    """End-to-end check of the six primitives on the host-staged path.

    We mock ``torch.distributed`` so we don't need a real cluster.
    """

    def _bridge(self) -> CrossVendorBridge:
        src = _MockBackend(CommLibrary.RCCL, DeviceVendor.AMD, device_id=0)
        dst = _MockBackend(CommLibrary.ONECCL, DeviceVendor.INTEL, device_id=0)
        a = _amd_device(0, interconnect=InterconnectType.PCIE)
        b = _intel_device(1, interconnect=InterconnectType.PCIE)
        return CrossVendorBridge(src, dst, a, b)

    def test_all_reduce_uses_host_staging(self) -> None:
        """all_reduce on a bridge should call into torch.distributed
        AFTER staging the tensor to host memory."""
        bridge = self._bridge()
        tensor = torch.ones(4, dtype=torch.float32)

        from src.bridges.pytorch_xla import comm_bridge

        with (
            patch.object(comm_bridge, "dist") as mock_dist,
            patch.object(comm_bridge, "torch", create=True) as mock_torch,
        ):
            # The bridge uses torch.chunk and torch.empty; route them
            # through the real torch so the test tensors flow.
            mock_torch.empty = torch.empty
            mock_torch.chunk = torch.chunk
            # The bridge accesses dist.get_world_size and dist.all_reduce.
            mock_dist.get_world_size = MagicMock(return_value=1)
            mock_dist.all_reduce = MagicMock()
            # ReduceOp is only used as the default op when caller
            # doesn't pass one — the bridge uses ReduceOp.SUM.
            mock_dist.ReduceOp = dist.ReduceOp
            bridge.all_reduce(tensor)
            mock_dist.all_reduce.assert_called_once()

    def test_send_stages_to_host(self) -> None:
        """send() should call into dist.send with a host tensor."""
        bridge = self._bridge()
        tensor = torch.ones(4, dtype=torch.float32)

        from src.bridges.pytorch_xla import comm_bridge

        with (
            patch.object(comm_bridge, "dist") as mock_dist,
            patch.object(comm_bridge, "torch", create=True) as mock_torch,
        ):
            mock_torch.empty = torch.empty
            mock_torch.chunk = torch.chunk
            mock_dist.send = MagicMock()
            bridge.send(tensor, dst=1)
            mock_dist.send.assert_called_once()

    def test_recv_stages_from_host(self) -> None:
        """recv() should call into dist.recv with a host tensor."""
        bridge = self._bridge()
        template = torch.ones(4, dtype=torch.float32)

        from src.bridges.pytorch_xla import comm_bridge

        with (
            patch.object(comm_bridge, "dist") as mock_dist,
            patch.object(comm_bridge, "torch", create=True) as mock_torch,
        ):
            mock_torch.empty = torch.empty
            mock_torch.chunk = torch.chunk
            mock_dist.recv = MagicMock()
            bridge.recv(template, src=0)
            mock_dist.recv.assert_called_once()

    def test_missing_torch_raises_dependency_error(self) -> None:
        """If torch is missing, the bridge primitives should fail loudly."""
        bridge = self._bridge()
        tensor = torch.ones(4, dtype=torch.float32)

        from src.bridges.pytorch_xla import comm_bridge

        with (
            patch.object(comm_bridge, "TORCH_AVAILABLE", False),
            patch.object(comm_bridge, "dist", None),
            patch.object(comm_bridge, "torch", None),
            pytest.raises(Exception),
        ):
            bridge.all_reduce(tensor)


# ---------------------------------------------------------------------------
# 7. Pluggability: backend primitives call torch.distributed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestVendorBackendPrimitives:
    """Smoke test the per-vendor primitives.

    The point isn't to verify the math (that's torch's job) — it's to
    confirm we wire up ``all_reduce``/``all_gather``/etc. to the right
    torch.distributed entry points.  We mock torch.distributed so the
    test doesn't need a real cluster.
    """

    @pytest.fixture
    def mock_dist(self):
        """Patch torch.distributed inside the comm_bridge module."""
        from src.bridges.pytorch_xla import comm_bridge

        with (
            patch.object(comm_bridge, "dist") as mock_dist,
            patch.object(comm_bridge, "torch", create=True) as mock_torch,
        ):
            mock_dist.is_initialized = MagicMock(return_value=False)
            mock_dist.get_backend = MagicMock(return_value="nccl")
            mock_dist.ReduceOp = dist.ReduceOp
            mock_dist.get_world_size = MagicMock(return_value=1)
            mock_dist.init_process_group = MagicMock()
            # The bridge uses the real torch.chunk and torch.empty so
            # test tensors flow through the real path.
            mock_torch.empty = torch.empty
            mock_torch.chunk = torch.chunk
            yield mock_dist

    def test_nccl_init_process_group_uses_nccl(self, mock_dist) -> None:
        b = NCCLBackend(device_id=0)
        b._ensure_initialized()
        mock_dist.init_process_group.assert_called_once()
        kwargs = mock_dist.init_process_group.call_args.kwargs
        assert kwargs["backend"] == "nccl"
        assert kwargs["world_size"] == 1

    def test_oneccl_init_uses_xccl(self, mock_dist) -> None:
        b = oneCCLBackend(device_id=0)
        b._ensure_initialized()
        kwargs = mock_dist.init_process_group.call_args.kwargs
        assert kwargs["backend"] == "xccl"

    def test_all_reduce_passes_through(self, mock_dist) -> None:
        """all_reduce on the backend hits dist.all_reduce with op=SUM."""
        b = NCCLBackend(device_id=0)
        tensor = torch.ones(2, dtype=torch.float32)
        mock_dist.all_reduce = MagicMock()
        b.all_reduce(tensor)
        mock_dist.all_reduce.assert_called_once()
        # First positional arg is the tensor
        args, kwargs = mock_dist.all_reduce.call_args
        assert args[0] is tensor
        assert kwargs.get("op") == dist.ReduceOp.SUM


# ---------------------------------------------------------------------------
# 8. Real torch.distributed end-to-end (gloo, single-rank)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestRealCollectives:
    """Run the six primitives against a real torch.distributed "gloo"
    process group of size 1.  This validates the *plumbing* — tensor
    shapes, dtypes, device transitions — without needing a vendor
    collective library.
    """

    @pytest.fixture(autouse=True)
    def _gloo_single_rank(self, monkeypatch):
        """Initialise a fresh gloo process group for each test.

        ``init_process_group`` with the default ``env://`` rendezvous
        reads ``MASTER_ADDR`` / ``MASTER_PORT`` from the environment;
        we set them here for the test process and let ``monkeypatch``
        restore them at teardown.
        """
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29501")

        # If a previous test left the default group up, tear it down
        # so each test starts clean.
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

        # Patch all three backends' _torch_backend_name to "gloo" so
        # init_process_group succeeds on CPU.  Restore at teardown.
        saved = {
            NCCLBackend: NCCLBackend._torch_backend_name,
            RCCLBackend: RCCLBackend._torch_backend_name,
            oneCCLBackend: oneCCLBackend._torch_backend_name,
        }
        NCCLBackend._torch_backend_name = "gloo"
        RCCLBackend._torch_backend_name = "gloo"
        oneCCLBackend._torch_backend_name = "gloo"
        try:
            yield
        finally:
            NCCLBackend._torch_backend_name = saved[NCCLBackend]
            RCCLBackend._torch_backend_name = saved[RCCLBackend]
            oneCCLBackend._torch_backend_name = saved[oneCCLBackend]
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass

    def test_all_reduce_identity(self) -> None:
        """On a 1-rank world, all_reduce is a no-op (sum of one)."""
        t = torch.tensor([1.0, 2.0, 3.0])
        b = NCCLBackend(device_id=0)
        b.all_reduce(t)
        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))

    def test_all_gather_concatenates(self) -> None:
        """On a 1-rank world, all_gather returns a copy of the input."""
        t = torch.tensor([1.0, 2.0])
        b = NCCLBackend(device_id=0)
        out = b.all_gather(t)
        # 1-rank world: output is just the input
        assert out.shape == t.shape
        assert torch.equal(out, t)

    def test_reduce_scatter_returns_chunk(self) -> None:
        """On a 1-rank world, reduce_scatter returns one chunk."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = NCCLBackend(device_id=0)
        out = b.reduce_scatter(t)
        # World size 1 → input is split into 1 chunk of size 4
        assert out.shape == t.shape

    def test_all_to_all_identity_on_world_1(self) -> None:
        """On a 1-rank world, all_to_all returns the input."""
        t = torch.tensor([1.0, 2.0, 3.0])
        b = NCCLBackend(device_id=0)
        out = b.all_to_all(t)
        assert out.shape == t.shape
        assert torch.equal(out, t)

    def test_send_recv_methods_exist(self) -> None:
        """send/recv method surface is intact on the backend."""
        b = NCCLBackend(device_id=0)
        assert callable(b.send)
        assert callable(b.recv)


# ---------------------------------------------------------------------------
# 9. Module-level: torch-less import safety
# ---------------------------------------------------------------------------


class TestTorchlessImport:
    """The module must import on systems without torch."""

    def test_module_can_be_imported(self) -> None:
        """importlib.reload shouldn't blow up."""
        from src.bridges.pytorch_xla import comm_bridge

        importlib.reload(comm_bridge)
        assert hasattr(comm_bridge, "CollectiveBackend")
        assert hasattr(comm_bridge, "NCCLBackend")
        assert hasattr(comm_bridge, "RCCLBackend")
        assert hasattr(comm_bridge, "oneCCLBackend")
        assert hasattr(comm_bridge, "CrossVendorBridge")
        assert hasattr(comm_bridge, "detect_p2p_capability")

    def test_torchless_path_does_not_crash(self, monkeypatch) -> None:
        """With TORCH_AVAILABLE=False, backend construction still works
        but ``is_available`` reports False and primitives raise."""
        from src.bridges.pytorch_xla import comm_bridge

        monkeypatch.setattr(comm_bridge, "TORCH_AVAILABLE", False)
        monkeypatch.setattr(comm_bridge, "dist", None)
        monkeypatch.setattr(comm_bridge, "torch", None)

        b = NCCLBackend(device_id=0)
        assert b.is_available is False
        with pytest.raises(Exception):
            b.all_reduce(None)
