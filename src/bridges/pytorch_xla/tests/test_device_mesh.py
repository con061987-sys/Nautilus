"""Tests for the device mesh module."""

from __future__ import annotations

import random

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

    def test_uniform_3x3_with_zero_diagonal_regression(self) -> None:
        """Regression: a 3x3 uniform mesh whose diagonal is 0 must be uniform.

        The original bug used ``row.index(first)`` which would match the
        diagonal cell (value 0) rather than skipping it, so a perfectly
        uniform mesh with a zero diagonal was wrongly reported as
        non-uniform.
        """
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100, 100],
                [100, 0, 100],
                [100, 100, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_uniform_2x2(self) -> None:
        """A 2x2 uniform mesh should be detected as uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 64],
                [64, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_uniform_4x4(self) -> None:
        """A 4x4 uniform mesh should be detected as uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 900, 900, 900],
                [900, 0, 900, 900],
                [900, 900, 0, 900],
                [900, 900, 900, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_single_device_is_uniform(self) -> None:
        """A 1x1 matrix has no off-diagonal cells; treated as uniform."""
        topo = MeshTopology(bandwidth_matrix=[[0]])
        assert topo.is_uniform is True

    def test_empty_topology_is_uniform(self) -> None:
        """An empty topology (no bandwidth, no latency) is trivially uniform."""
        topo = MeshTopology()
        assert topo.is_uniform is True

    def test_only_latency_is_uniform(self) -> None:
        """A topology with only a latency matrix defaults to uniform."""
        topo = MeshTopology(
            latency_matrix=[
                [0, 5, 5],
                [5, 0, 5],
                [5, 5, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_jagged_rows_are_not_uniform(self) -> None:
        """A jagged bandwidth matrix must be reported as non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100, 100],
                [100, 0],
                [100, 100, 0],
            ],
        )
        assert topo.is_uniform is False

    def test_non_square_matrix_is_not_uniform(self) -> None:
        """A rectangular (non-square) matrix must be non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100, 100, 100],
                [100, 0, 100, 100],
                [100, 100, 0, 100],
            ],
        )
        assert topo.is_uniform is False

    def test_non_uniform_two_tiers(self) -> None:
        """A mesh mixing two bandwidth tiers (NVLink + PCIe) is non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 900, 900, 64],
                [900, 0, 900, 64],
                [900, 900, 0, 64],
                [64, 64, 64, 0],
            ],
        )
        assert topo.is_uniform is False

    def test_tolerance_boundary_uniform(self) -> None:
        """Off-diagonal values within 1 GB/s of each other count as uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100.0, 100.5],
                [100.4, 0, 100.9],
                [100.7, 100.2, 0],
            ],
        )
        assert topo.is_uniform is True

    def test_tolerance_boundary_non_uniform(self) -> None:
        """Off-diagonal values differing by more than 1 GB/s are non-uniform."""
        topo = MeshTopology(
            bandwidth_matrix=[
                [0, 100, 102],
                [100, 0, 100],
                [100, 100, 0],
            ],
        )
        assert topo.is_uniform is False

    def test_random_uniform_meshes_match_manual_check(self) -> None:
        """Property-based: 100 random uniform meshes are all detected as uniform.

        For each randomly generated square matrix whose off-diagonal cells
        are within the tolerance of a single base bandwidth value, both
        ``MeshTopology.is_uniform`` and the manual reference check must
        agree.
        """
        rng = random.Random(0x4E41_5554_4C55_5300)  # "NAUTILUS"

        base_bandwidths = [12.5, 64.0, 200.0, 800.0, 900.0]
        sizes = [2, 3, 4, 5, 6, 8]
        tolerance = MeshTopology._BANDWIDTH_TOLERANCE_GBPS
        safe_jitter_amplitude = tolerance * 0.4

        def manual_uniform(matrix: list[list[float]]) -> bool:
            off_diag = [
                matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j
            ]
            if not off_diag:
                return True
            return (max(off_diag) - min(off_diag)) < tolerance

        for trial in range(100):
            n = rng.choice(sizes)
            base = rng.choice(base_bandwidths)

            matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = 0.0
                    else:
                        matrix[i][j] = base + rng.uniform(
                            -safe_jitter_amplitude, safe_jitter_amplitude
                        )

            topo = MeshTopology(bandwidth_matrix=matrix)
            expected = manual_uniform(matrix)
            assert topo.is_uniform is expected, (
                f"trial {trial}: n={n} base={base} expected={expected} got={topo.is_uniform}"
            )
            assert topo.is_uniform is True, (
                f"trial {trial}: random uniform mesh detected as non-uniform"
            )

    def test_random_non_uniform_meshes_match_manual_check(self) -> None:
        """Property-based: meshes with two distinct tiers are non-uniform.

        Generate matrices where some off-diagonal cells use one bandwidth
        tier and others use a second, well-separated tier, and assert
        ``is_uniform`` is False. ``n >= 3`` to guarantee both tiers
        actually appear on off-diagonal cells.
        """
        rng = random.Random(0x4E41_5554_4C55_5301)

        tier_pairs = [(12.5, 64.0), (64.0, 200.0), (200.0, 800.0), (800.0, 900.0)]
        sizes = [3, 4, 5, 6, 8]

        def manual_uniform(matrix: list[list[float]]) -> bool:
            off_diag = [
                matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j
            ]
            if not off_diag:
                return True
            tolerance = MeshTopology._BANDWIDTH_TOLERANCE_GBPS
            return (max(off_diag) - min(off_diag)) < tolerance

        for trial in range(50):
            n = rng.choice(sizes)
            low, high = rng.choice(tier_pairs)

            matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = 0.0
                    elif (i + j) % 2 == 0:
                        matrix[i][j] = low
                    else:
                        matrix[i][j] = high

            topo = MeshTopology(bandwidth_matrix=matrix)
            assert topo.is_uniform is False, (
                f"trial {trial}: n={n} tiers={(low, high)} should be non-uniform"
            )
            assert manual_uniform(matrix) is False, (
                f"trial {trial}: matrix construction failed to produce a non-uniform mesh"
            )


class TestDeviceMesh:
    """Tests for the DeviceMesh class."""

    def test_device_mesh_creation(self) -> None:
        """A DeviceMesh should aggregate multiple MeshDevices."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(
                1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC
            ),
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
            MeshDevice(
                1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC
            ),
            MeshDevice(
                2, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC
            ),
        ]
        mesh = DeviceMesh(devices=devices, mesh_shape=[3])
        amd_devices = mesh.get_devices_by_vendor(DeviceVendor.AMD)
        assert len(amd_devices) == 2

    def test_vendor_mesh_shape(self) -> None:
        """vendor_mesh_shape should return per-vendor counts."""
        devices = [
            MeshDevice(0, DeviceVendor.NVIDIA, "sm_90", 80.0, 989.0, InterconnectType.NVLINK),
            MeshDevice(
                1, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC
            ),
            MeshDevice(
                2, DeviceVendor.AMD, "gfx942", 192.0, 1307.0, InterconnectType.INFINITY_FABRIC
            ),
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
