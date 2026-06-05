"""Tests for the device mesh module."""

from __future__ import annotations

import pytest

from src.bridges.pytorch_xla.device_mesh import (
    DeviceMesh,
    DeviceVendor,
    InterconnectType,
    MeshDevice,
    MeshTopology,
)


class TestMeshDevice:
    """Tests for the MeshDevice class."""

    def test_mesh_device_creation(self) -> None:
        """A MeshDevice should store all hardware info."""
        device = MeshDevice(
            device_id=0,
            vendor=DeviceVendor.NVIDIA,
            arch="sm_90",
            memory_gb=80.0,
            compute_tflops=989.0,
            interconnect=InterconnectType.NVLINK,
        )
        assert device.device_id == 0
        assert device.vendor == DeviceVendor.NVIDIA
        assert device.arch == "sm_90"
        assert device.memory_gb == 80.0
        assert device.display_name == "nvidia:0(sm_90)"

    def test_mesh_device_metadata(self) -> None:
        """A MeshDevice can store custom metadata."""
        device = MeshDevice(
            device_id=1,
            vendor=DeviceVendor.AMD,
            arch="gfx942",
            memory_gb=192.0,
            compute_tflops=1307.0,
            interconnect=InterconnectType.INFINITY_FABRIC,
            metadata={"location": "rack-1"},
        )
        assert device.metadata["location"] == "rack-1"


class TestMeshTopology:
    """Tests for the MeshTopology class."""

    def test_topology_uniform(self) -> None:
        """A uniform topology should have is_uniform=True."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100, 100],
                [100, 0, 100],
                [100, 100, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_topology_non_uniform(self) -> None:
        """A non-uniform topology should have is_uniform=False."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 900, 64],
                [900, 0, 900],
                [64, 900, 0],
            ],
        )
        assert topo.is_uniform is False


class TestDeviceMesh:
    """Tests for the DeviceMesh class."""

    def test_device_mesh_creation(self) -> None:
        """A DeviceMesh should aggregate multiple MeshDevices."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        assert mesh.num_devices == 2
        assert mesh.is_heterogeneous is True
        assert DeviceVendor.NVIDIA in mesh.vendors
        assert DeviceVendor.AMD in mesh.vendors

    def test_homogeneous_mesh(self) -> None:
        """A mesh with one vendor should not be heterogeneous."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2])
        assert mesh.is_heterogeneous is False

    def test_get_devices_by_vendor(self) -> None:
        """get_devices_by_vendor should filter correctly."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
            MeshDevice(2, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[3])
        amd_devices = mesh.get_devices_by_vendor(DeviceVendor.AMD)
        assert len(amd_devices) == 2

    def test_vendor_mesh_shape(self) -> None:
        """vendor_mesh_shape should return per-vendor counts."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
            MeshDevice(2, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[3])
        shapes = mesh.vendor_mesh_shape()
        assert shapes[DeviceVendor.NVIDIA] == [1]
        assert shapes[DeviceVendor.AMD] == [2]

    def test_total_devices(self) -> None:
        """total_devices should compute the product of mesh_shape."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[2, 4])
        assert mesh.total_devices == 8

    def test_to_dict(self) -> None:
        """to_dict should serialise the mesh metadata."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[1])
        d = mesh.to_dict()
        assert d["num_devices"] == 1
        assert d["mesh_shape"] == [1]
        assert "nvidia" in d["vendors"]
        assert d["is_heterogeneous"] is False
