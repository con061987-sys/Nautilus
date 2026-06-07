"""NCCL ↔ RCCL ↔ oneCCL communication bridge.

This module is the execution side of the heterogeneous communication
stack.  :mod:`.comm_backend` decides *which* library to use for a given
collective; this module actually dispatches the call and translates
between vendor-specific memory layouts when a cluster mixes vendors.

Key abstractions:

  * :class:`CollectiveBackend` — abstract base class with the six
    standard primitives: ``all_reduce``, ``all_gather``,
    ``reduce_scatter``, ``all_to_all``, ``send``, ``recv``.
  * :class:`NCCLBackend` — Nvidia (torch.distributed "nccl").
  * :class:`RCCLBackend` — AMD ROCm (exposed by torch.distributed as
    "nccl" when running on a ROCm build of PyTorch).
  * :class:`OneCCLBackend` — Intel XPU / oneAPI (torch.distributed
    "xccl").
  * :class:`CrossVendorBridge` — bridges two per-vendor backends when
    the cluster is heterogeneous.  Falls back to host (CPU) memory
    staging when P2P is not viable.

Why three concrete backends instead of one dispatcher?  Each vendor
library has subtly different semantics around tensor dtypes, supported
operations, and what counts as "available".  Wrapping them in their
own class lets us probe availability once at startup and report a
clear ``BridgeError`` if the cluster asks for a backend that isn't
installed.  See ``NautilusError`` in :mod:`src.common.errors` for the
structured error contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.common.errors import BridgeError, DependencyMissingError
from src.common.logging import get_logger

from .comm_backend import CommLibrary
from .device_mesh import DeviceVendor, InterconnectType, MeshDevice

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional torch import — we want the module to import on systems where
# PyTorch is not installed so that test collection succeeds on bare CI.
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in torch-less envs
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# P2P capability detection
# ---------------------------------------------------------------------------


class P2PCapability(str, Enum):
    """P2P communication capability between two devices.

    * ``FULL`` — direct device-to-device transfer is supported and
      efficient (NVLink, Infinity Fabric, UALink, same-host PCIe).
    * ``HOST_STAGED`` — devices can talk, but only through CPU memory
      staging.  Necessary for cross-host or cross-vendor-no-UALink
      links.
    * ``UNSUPPORTED`` — no path between these two devices exists.
    """

    FULL = "full"
    HOST_STAGED = "host_staged"
    UNSUPPORTED = "unsupported"


def detect_p2p_capability(
    src: MeshDevice,
    dst: MeshDevice,
) -> P2PCapability:
    """Return the P2P capability between two mesh devices.

    The decision tree is:

    1. Same device → FULL (a no-op, but cheaper than UNSUPPORTED).
    2. Different hostnames → must traverse the network → HOST_STAGED.
    3. Same-vendor with NVLink / Infinity Fabric / UALink → FULL.
    4. Cross-vendor with UALink on both ends → FULL (UALink is the
       designed cross-vendor interconnect).
    5. Cross-vendor over PCIe / Ethernet → HOST_STAGED (P2P works
       through the host bridge, but going via CPU is more portable).
    6. Anything else → HOST_STAGED.
    """
    if src.device_id == dst.device_id:
        return P2PCapability.FULL

    if src.hostname != dst.hostname:
        return P2PCapability.HOST_STAGED

    high_bw_intra = {
        InterconnectType.NVLINK,
        InterconnectType.INFINITY_FABRIC,
    }
    if src.vendor == dst.vendor and src.interconnect in high_bw_intra:
        return P2PCapability.FULL

    if src.interconnect == InterconnectType.UALINK and dst.interconnect == InterconnectType.UALINK:
        return P2PCapability.FULL

    if (
        src.vendor != dst.vendor
        and src.interconnect == InterconnectType.UALINK
        and dst.interconnect == InterconnectType.UALINK
    ):
        return P2PCapability.FULL

    # Same-host PCIe P2P works for same-vendor, but for cross-vendor we
    # stage through host memory to avoid platform-specific GPUDirect
    # P2P issues.
    if src.vendor == dst.vendor:
        return P2PCapability.FULL

    return P2PCapability.HOST_STAGED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    """Raise a typed error if torch is missing.

    We want the module to import on systems without torch (so test
    collection still works), but the moment you actually try to do a
    collective, we should fail loudly with a structured error.
    """
    if not TORCH_AVAILABLE or torch is None or dist is None:
        raise DependencyMissingError(
            "torch is not installed; cannot run collectives",
            context={"module": "src.bridges.pytorch_xla.comm_bridge"},
        )


def _device_string_for_vendor(vendor: DeviceVendor) -> str:
    """Return the torch device-kind string for a vendor."""
    if vendor == DeviceVendor.NVIDIA:
        return "cuda"
    if vendor == DeviceVendor.AMD:
        return "cuda"  # ROCm reuses the cuda device kind
    if vendor == DeviceVendor.INTEL:
        return "xpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CollectiveBackend(ABC):
    """Abstract interface for the six standard collective primitives.

    Every implementation MUST support:

      * :meth:`all_reduce`  — in-place reduce across a group.
      * :meth:`all_gather`  — gather and concatenate across a group.
      * :meth:`reduce_scatter` — combine reduce + scatter.
      * :meth:`all_to_all`  — exchange chunks with every peer.
      * :meth:`send` / :meth:`recv` — point-to-point transfers.

    All methods accept an optional ``group`` argument for callers that
    want a non-default process group.  ``None`` means "use whatever
    process group is currently active".
    """

    def __init__(self, device_id: int = 0) -> None:
        """Initialize the backend with a device id.

        Args:
            device_id: The local device id this backend is bound to.
        """
        self._device_id = device_id

    @property
    @abstractmethod
    def library(self) -> CommLibrary:
        """The :class:`CommLibrary` this backend wraps."""

    @property
    @abstractmethod
    def vendor(self) -> DeviceVendor:
        """The :class:`DeviceVendor` this backend targets."""

    @property
    @abstractmethod
    def device_id(self) -> int:
        """The local device id this backend is bound to."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True iff this backend can be used on the current system.

        Probes for both the torch.distributed backend and the
        underlying hardware (cuda/xpu runtime).  The result is cached
        after the first call.
        """

    @abstractmethod
    def all_reduce(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        """All-reduce ``tensor`` in-place.  Returns the same tensor."""

    @abstractmethod
    def all_gather(
        self,
        tensor: Any,
        group: Any = None,
    ) -> Any:
        """All-gather.  Returns a new tensor with world_size copies."""

    @abstractmethod
    def reduce_scatter(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        """Reduce-scatter.  Returns a new tensor with rank's chunk."""

    @abstractmethod
    def all_to_all(
        self,
        tensor: Any,
        group: Any = None,
    ) -> Any:
        """All-to-all.  Returns a new tensor with permuted chunks."""

    @abstractmethod
    def send(
        self,
        tensor: Any,
        dst: int,
        group: Any = None,
    ) -> None:
        """Send ``tensor`` to rank ``dst`` (blocking)."""

    @abstractmethod
    def recv(
        self,
        tensor: Any,
        src: int,
        group: Any = None,
    ) -> Any:
        """Receive into ``tensor`` from rank ``src`` (blocking)."""

    # Optional, but useful for callers that need to know what they're
    # getting before they commit to a collective.
    def describe(self) -> dict[str, Any]:
        """Return a dict describing this backend for logging."""
        return {
            "library": self.library.value,
            "vendor": self.vendor.value,
            "device_id": self.device_id,
            "available": self.is_available,
        }


# ---------------------------------------------------------------------------
# Shared base for the three torch.distributed wrappers
# ---------------------------------------------------------------------------


@dataclass
class _InitState:
    """Cached process-group init state.

    We use a sentinel to remember whether ``init_process_group`` has
    been called on this process; calling it twice raises in torch.
    """

    initialized: bool = False
    backend_used: str = ""


class _TorchDistributedBackend(CollectiveBackend):
    """Common implementation for the three vendor backends.

    Each vendor backend differs only in:

      * its :class:`CommLibrary` / :class:`DeviceVendor` identity,
      * the torch.distributed backend name it passes to
        ``init_process_group``,
      * how it probes for availability.

    All the actual collective wiring is here so the subclasses stay
    short.
    """

    _library: CommLibrary
    _vendor: DeviceVendor
    _torch_backend_name: str
    _device_kind: str

    def __init__(self, device_id: int) -> None:
        if device_id < 0:
            raise BridgeError(
                "device_id must be >= 0",
                context={"backend": self._library.value, "device_id": device_id},
            )
        self._device_id = device_id
        self._availability_cache: bool | None = None
        self._init_state = _InitState()

    # --- Properties ----------------------------------------------------

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
        if self._availability_cache is None:
            self._availability_cache = self._probe_availability()
        return self._availability_cache

    @property
    def torch_backend_name(self) -> str:
        """The string passed to ``torch.distributed.init_process_group``."""
        return self._torch_backend_name

    @property
    def device_kind(self) -> str:
        """The torch device kind (cuda / xpu / cpu) this backend targets."""
        return self._device_kind

    # --- Subclass hooks -----------------------------------------------

    def _probe_availability(self) -> bool:
        """Probe whether this backend is usable on the current system.

        Subclasses override to check the vendor-specific runtime.
        Default: only require torch.distributed to expose the backend
        name; the subclass still must verify the runtime.
        """
        return self._torch_backend_available()

    def _torch_backend_available(self) -> bool:
        """True iff torch.distributed exposes the named backend."""
        if not TORCH_AVAILABLE or dist is None:
            return False
        checker = getattr(dist, f"is_{self._torch_backend_name}_available", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:  # pragma: no cover - defensive
            return False

    # --- Lazy process-group init --------------------------------------

    def _ensure_initialized(self) -> None:
        """Make sure ``torch.distributed`` is up with the right backend.

        On a 1-rank world this is a no-op for actual collectives but
        still required by torch; the resulting "group of one" lets
        callers exercise the API paths without a multi-rank cluster.
        """
        if not TORCH_AVAILABLE or dist is None:
            _require_torch()
        if self._init_state.initialized:
            return
        if dist.is_initialized():
            self._init_state.initialized = True
            self._init_state.backend_used = str(dist.get_backend())
            return
        try:
            dist.init_process_group(
                backend=self._torch_backend_name,
                rank=0,
                world_size=1,
            )
        except Exception as exc:  # pragma: no cover - depends on cluster
            raise BridgeError(
                f"failed to init torch.distributed with {self._torch_backend_name}",
                context={
                    "backend": self._torch_backend_name,
                    "vendor": self._vendor.value,
                },
                cause=exc,
            ) from exc
        self._init_state.initialized = True
        self._init_state.backend_used = self._torch_backend_name

    def _world_size(self, group: Any) -> int:
        if not TORCH_AVAILABLE or dist is None:
            _require_torch()
        try:
            return int(dist.get_world_size(group))
        except Exception:
            return 1

    # --- The six primitives -------------------------------------------

    def all_reduce(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        _require_torch()
        self._ensure_initialized()
        if op is None:
            op = dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    def all_gather(self, tensor: Any, group: Any = None) -> Any:
        _require_torch()
        self._ensure_initialized()
        world_size = self._world_size(group)
        out = torch.empty(
            (tensor.shape[0] * world_size, *tuple(tensor.shape[1:])),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        chunks = list(torch.chunk(out, world_size))
        dist.all_gather(chunks, tensor, group=group)
        return out

    def reduce_scatter(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        _require_torch()
        self._ensure_initialized()
        if op is None:
            op = dist.ReduceOp.SUM
        world_size = self._world_size(group)
        input_list = list(torch.chunk(tensor, world_size))
        output = torch.empty_like(input_list[0])
        dist.reduce_scatter(output, input_list, op=op, group=group)
        return output

    def all_to_all(self, tensor: Any, group: Any = None) -> Any:
        _require_torch()
        self._ensure_initialized()
        world_size = self._world_size(group)
        input_list = list(torch.chunk(tensor, world_size))
        output_list = [torch.empty_like(c) for c in input_list]
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=0)

    def send(self, tensor: Any, dst: int, group: Any = None) -> None:
        _require_torch()
        self._ensure_initialized()
        dist.send(tensor, dst, group=group)

    def recv(self, tensor: Any, src: int, group: Any = None) -> Any:
        _require_torch()
        self._ensure_initialized()
        dist.recv(tensor, src, group=group)
        return tensor


# ---------------------------------------------------------------------------
# NCCL — Nvidia Collective Communication Library
# ---------------------------------------------------------------------------


class NCCLBackend(_TorchDistributedBackend):
    """NCCL backend for Nvidia GPUs.

    NCCL is the canonical GPU collective library for Nvidia hardware
    and the only one that fully exploits NVLink.  The torch.distributed
    backend name is ``"nccl"``; cuda tensors are required for actual
    collectives.
    """

    _library = CommLibrary.NCCL
    _vendor = DeviceVendor.NVIDIA
    _torch_backend_name = "nccl"
    _device_kind = "cuda"

    def _probe_availability(self) -> bool:
        """NCCL needs both the backend name *and* a working cuda runtime."""
        if not self._torch_backend_available():
            return False
        cuda_ok = getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
        return cuda_ok


# ---------------------------------------------------------------------------
# RCCL — ROCm Communication Library (AMD)
# ---------------------------------------------------------------------------


class RCCLBackend(_TorchDistributedBackend):
    """RCCL backend for AMD ROCm GPUs.

    torch.distributed exposes RCCL via the ``"nccl"`` backend name
    when the active PyTorch build is ROCm-flavored (the same wire
    protocol, just AMD's implementation).  We detect ROCm by looking
    for AMD/Radeon strings in the cuda device name — this is the
    standard probe and survives across ROCm versions.
    """

    _library = CommLibrary.RCCL
    _vendor = DeviceVendor.AMD
    _torch_backend_name = "nccl"  # RCCL reuses the "nccl" backend name
    _device_kind = "cuda"  # ROCm presents itself as cuda to torch

    def _probe_availability(self) -> bool:
        """RCCL needs the nccl backend name *and* an AMD device on cuda."""
        if not self._torch_backend_available():
            return False
        if not (torch.cuda.is_available()):
            return False
        try:
            name = torch.cuda.get_device_name(0).lower()
        except Exception:
            return False
        return "amd" in name or "radeon" in name or "gfx" in name


# ---------------------------------------------------------------------------
# oneCCL — Intel oneAPI Collective Communications Library
# ---------------------------------------------------------------------------


class OneCCLBackend(_TorchDistributedBackend):
    """oneCCL backend for Intel XPU devices (Intel Max / Gaudi / PVC).

    torch.distributed uses the backend name ``"xccl"`` for oneCCL
    (the ``x`` comes from oneAPI).  This backend requires the
    ``torch.xpu`` module to be importable, which only happens on Intel
    PyTorch builds.
    """

    _library = CommLibrary.ONECCL
    _vendor = DeviceVendor.INTEL
    _torch_backend_name = "xccl"
    _device_kind = "xpu"

    def _probe_availability(self) -> bool:
        """XCCL needs the xccl backend name *and* the xpu runtime."""
        if not self._torch_backend_available():
            return False
        xpu_mod = getattr(torch, "xpu", None)
        if xpu_mod is None:
            return False
        try:
            return bool(xpu_mod.is_available())
        except Exception:  # pragma: no cover - defensive
            return False


# ---------------------------------------------------------------------------
# Cross-vendor bridge with host-based staging
# ---------------------------------------------------------------------------


class CrossVendorBridge(CollectiveBackend):
    """Bridges collectives across two different vendor backends.

    torch.distributed cannot put an AMD device and an Intel device in
    the same process group — the underlying library is different.  This
    class implements the standard fix: stage through host memory.

    For ``all_reduce`` over an AMD + Intel cluster, the flow is:

        AMD tensor ─► host CPU tensor (RCCL all_reduce) ─►
            Intel tensor (GLOO broadcast, or oneCCL send/recv)

    That is O(2 * tensor_size) extra host memory traffic per
    collective.  It's slow, but it works without specialized
    cross-vendor interconnects.

    The bridge also surfaces a :attr:`p2p_capability` so callers can
    decide whether to use direct P2P (e.g. on a UALink-attached pair)
    or to force host staging.
    """

    def __init__(
        self,
        src_backend: CollectiveBackend,
        dst_backend: CollectiveBackend,
        src_device: MeshDevice,
        dst_device: MeshDevice,
        *,
        force_host_staging: bool = False,
    ) -> None:
        if src_backend is dst_backend:
            raise BridgeError(
                "CrossVendorBridge requires two *different* vendor backends",
                context={
                    "src_vendor": src_backend.vendor.value,
                    "dst_vendor": dst_backend.vendor.value,
                },
            )
        if src_backend.vendor == dst_backend.vendor:
            # Homogeneous — use a single backend instead.
            logger.warning(
                "CrossVendorBridge constructed with two same-vendor backends; "
                "use the vendor backend directly for better performance"
            )
        self.src_backend = src_backend
        self.dst_backend = dst_backend
        self.src_device = src_device
        self.dst_device = dst_device
        self._force_host_staging = force_host_staging
        self._p2p = detect_p2p_capability(src_device, dst_device)

    # --- Identity ----------------------------------------------------

    @property
    def library(self) -> CommLibrary:
        return CommLibrary.MIXED

    @property
    def vendor(self) -> DeviceVendor:
        # The bridge is "of" the source vendor for bookkeeping.
        return self.src_backend.vendor

    @property
    def device_id(self) -> int:
        return self.src_backend.device_id

    @property
    def is_available(self) -> bool:
        return bool(self.src_backend.is_available and self.dst_backend.is_available)

    @property
    def p2p_capability(self) -> P2PCapability:
        """How this bridge can move data between the two devices."""
        return self._p2p

    @property
    def uses_host_staging(self) -> bool:
        """True iff the bridge will route data through host memory."""
        return self._force_host_staging or self._p2p != P2PCapability.FULL

    # --- Staging helpers ---------------------------------------------

    def _stage_to_host(self, tensor: Any) -> Any:
        """Move a device tensor to host memory (CPU)."""
        _require_torch()
        return tensor.cpu()

    def _stage_from_host(self, host_tensor: Any, target_template: Any) -> Any:
        """Move a host tensor onto a target device, matching ``target_template``."""
        _require_torch()
        return host_tensor.to(target_template.device)

    # --- The six primitives (staged) ---------------------------------

    def all_reduce(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        if op is None and TORCH_AVAILABLE and dist is not None:
            op = dist.ReduceOp.SUM
        host = self._stage_to_host(tensor)
        dist.all_reduce(host, op=op, group=group)
        return self._stage_from_host(host, tensor)

    def all_gather(self, tensor: Any, group: Any = None) -> Any:
        host = self._stage_to_host(tensor)
        world_size = self._world_size(group)
        out_host = torch.empty(
            (host.shape[0] * world_size, *tuple(host.shape[1:])),
            dtype=host.dtype,
            device="cpu",
        )
        chunks = list(torch.chunk(out_host, world_size))
        dist.all_gather(chunks, host, group=group)
        return self._stage_from_host(out_host, tensor)

    def reduce_scatter(
        self,
        tensor: Any,
        op: Any = None,
        group: Any = None,
    ) -> Any:
        if op is None and TORCH_AVAILABLE and dist is not None:
            op = dist.ReduceOp.SUM
        host = self._stage_to_host(tensor)
        world_size = self._world_size(group)
        input_list = list(torch.chunk(host, world_size))
        out_host = torch.empty_like(input_list[0])
        dist.reduce_scatter(out_host, input_list, op=op, group=group)
        return self._stage_from_host(out_host, tensor)

    def all_to_all(self, tensor: Any, group: Any = None) -> Any:
        host = self._stage_to_host(tensor)
        world_size = self._world_size(group)
        input_list = list(torch.chunk(host, world_size))
        output_list = [torch.empty_like(c) for c in input_list]
        dist.all_to_all(output_list, input_list, group=group)
        return self._stage_from_host(torch.cat(output_list, dim=0), tensor)

    def send(self, tensor: Any, dst: int, group: Any = None) -> None:
        host = self._stage_to_host(tensor)
        dist.send(host, dst, group=group)

    def recv(self, tensor: Any, src: int, group: Any = None) -> Any:
        host_template = self._stage_to_host(tensor)
        host = torch.empty_like(host_template)
        dist.recv(host, src, group=group)
        return self._stage_from_host(host, tensor)

    # --- Helpers -----------------------------------------------------

    def _world_size(self, group: Any) -> int:
        _require_torch()
        try:
            return int(dist.get_world_size(group))
        except Exception:
            return 1

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "src_vendor": self.src_backend.vendor.value,
                "dst_vendor": self.dst_backend.vendor.value,
                "src_library": self.src_backend.library.value,
                "dst_library": self.dst_backend.library.value,
                "p2p_capability": self.p2p_capability.value,
                "uses_host_staging": self.uses_host_staging,
                "force_host_staging": self._force_host_staging,
            }
        )
        return base


# ---------------------------------------------------------------------------
# Factory: pick a backend for a (device, library) pair
# ---------------------------------------------------------------------------


_BACKEND_REGISTRY: dict[CommLibrary, type[CollectiveBackend]] = {
    CommLibrary.NCCL: NCCLBackend,
    CommLibrary.RCCL: RCCLBackend,
    CommLibrary.ONECCL: OneCCLBackend,
}


def make_backend(
    library: CommLibrary,
    device_id: int = 0,
) -> CollectiveBackend:
    """Construct the appropriate backend for ``library``.

    Raises :class:`BridgeError` if the library is not one of the three
    vendor libraries this module wraps (e.g. GLOO, MIXED, UALINK).
    Callers wanting GLOO should construct it directly; UALink is
    layered on top of the vendor backends and is selected automatically
    by :class:`CrossVendorBridge`.
    """
    cls = _BACKEND_REGISTRY.get(library)
    if cls is None:
        raise BridgeError(
            f"no concrete backend for library={library.value!r}",
            context={
                "library": library.value,
                "supported": [lib.value for lib in _BACKEND_REGISTRY],
            },
        )
    return cls(device_id=device_id)


__all__ = [
    "TORCH_AVAILABLE",
    "CollectiveBackend",
    "CrossVendorBridge",
    "NCCLBackend",
    "OneCCLBackend",
    "P2PCapability",
    "RCCLBackend",
    "detect_p2p_capability",
    "make_backend",
]
