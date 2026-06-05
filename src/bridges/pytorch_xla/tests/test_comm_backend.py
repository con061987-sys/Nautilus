"""Tests for the communication backend module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
)
from src.bridges.pytorch_xla.comm_backend import (
    CollectiveOp,
    CommBackend,
    CommGroup,
    CommLibrary,
)


class TestCommBackend:
    """Tests for the CommBackend class."""

    def test_homogeneous_mesh_no_bridges(self) -> None:
        """A homogeneous mesh should have no cross-vendor bridges."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        assert len(comm._groups) == 1  # One Nvidia group
        assert len(comm._cross_vendor_bridges) == 0  # No bridges
        assert comm.mesh.is_heterogeneous is False

    def test_heterogeneous_mesh_has_bridges(self) -> None:
        """A heterogeneous mesh should create cross-vendor bridges."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        assert comm.mesh.is_heterogeneous is True
        assert len(comm._groups) == 2  # Nvidia + AMD groups
        assert len(comm._cross_vendor_bridges) == 1  # One bridge

    def test_select_library_for_vendor(self) -> None:
        """_select_library_for_vendor should pick the right library."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[1])
        comm = CommBackend(mesh)
        assert comm._select_library_for_vendor(DeviceVendor.NVIDIA) == CommLibrary.NCCL
        assert comm._select_library_for_vendor(DeviceVendor.AMD) == CommLibrary.RCCL
        assert comm._select_library_for_vendor(DeviceVendor.INTEL) == CommLibrary.ONECCL
        assert comm._select_library_for_vendor(DeviceVendor.CPU) == CommLibrary.GLOO

    def test_select_library_for_op_homogeneous(self) -> None:
        """select_library_for_op should use the vendor's library for same-vendor ops."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        library = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [0, 1])
        assert library == CommLibrary.NCCL

    def test_select_library_for_op_mixed(self) -> None:
        """select_library_for_op should use MIXED for cross-vendor ops."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        library = comm.select_library_for_op(CollectiveOp.ALL_REDUCE, [0, 1])
        assert library == CommLibrary.MIXED

    def test_select_library_for_op_empty(self) -> None:
        """select_library_for_op with no devices should return GLOO."""
        devices = []
        mesh = DeviceMesh(devices=devices, mesh_shape=[0])
        comm = CommBackend(mesh)
        library = comm.select_library_for_op(CollectiveOp.BARRIER, [])
        assert library == CommLibrary.GLOO

    def test_get_stats(self) -> None:
        """get_stats should return a useful summary."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        stats = comm.get_stats()
        assert stats["num_groups"] == 2
        assert stats["num_cross_vendor_bridges"] == 1
        assert stats["is_heterogeneous"] is True

    def test_bridge_is_marked_cross_vendor(self) -> None:
        """Cross-vendor bridges should have is_cross_vendor_bridge=True."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        comm = CommBackend(mesh)
        for bridge in comm._cross_vendor_bridges.values():
            assert bridge.is_cross_vendor_bridge is True
            assert bridge.library == CommLibrary.MIXED
