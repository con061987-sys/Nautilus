"""Tests for the C-API Python bindings.

The C library is *not* built in CI by default (requires CMake + a C
toolchain). These tests verify two distinct contracts:

  1. The Python module must ALWAYS be importable, even when the
     shared library is not present. This is the "graceful degradation"
     contract — calling ``is_available()`` returns False; calling a
     function that needs the library raises ``DependencyMissingError``.

  2. When the library IS present (e.g. local dev build), the
     ``compile()`` / ``triton_version()`` paths are exercised.

If the .so is not built, every test that needs it is skipped with a
clear message — never silently passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Module-level: import must always work ─────────────────────────────


class TestCApiModuleImport:
    """The Python wrapper must be importable without the .so present."""

    def test_imports_without_crash(self) -> None:
        """Importing src.c_api must not raise — the loader is lazy."""
        from src import c_api

        assert c_api is not None

    def test_exposes_public_api(self) -> None:
        from src import c_api

        # The module must always re-export these names regardless of
        # whether the library is built.
        assert hasattr(c_api, "is_available")
        assert hasattr(c_api, "compile")
        assert hasattr(c_api, "triton_version")
        assert hasattr(c_api, "TritonKernelHandle")
        assert hasattr(c_api, "CApiUnavailable")

    def test_exposes_vendor_constants(self) -> None:
        from src import c_api

        # Vendor IDs are pure-Python; must always be present
        for name in (
            "VENDOR_NVIDIA",
            "VENDOR_AMD",
            "VENDOR_INTEL",
            "VENDOR_APPLE",
            "VENDOR_HOST",
        ):
            assert hasattr(c_api, name), f"missing constant: {name}"

    def test_exposes_arch_constants(self) -> None:
        from src import c_api

        for name in (
            "ARCH_SM_90",
            "ARCH_SM_80",
            "ARCH_GFX942",
            "ARCH_GAUDI2",
        ):
            assert hasattr(c_api, name), f"missing arch: {name}"


# ── Library availability ─────────────────────────────────────────────


class TestCApiAvailability:
    """is_available() must never raise — it returns the load truth value."""

    def test_is_available_returns_bool(self) -> None:
        from src.c_api import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_is_available_does_not_raise(self) -> None:
        """Even on systems without the .so, is_available() must not raise."""
        from src.c_api import is_available

        try:
            is_available()
        except Exception as exc:
            pytest.fail(f"is_available() raised: {type(exc).__name__}: {exc}")


# ── Compile path: only meaningful when library is present ────────────


class TestCApiCompile:
    """Test the compile() wrapper. Skipped when the library isn't built."""

    def test_compile_raises_clear_error_when_unavailable(self) -> None:
        """If the .so is missing, calling compile() must raise
        DependencyMissingError (or a subclass) with a build hint.
        """
        from src.c_api import compile
        from src.common.errors import DependencyMissingError

        if compile.__module__ and _c_lib_loaded():
            pytest.skip("C library is built; this test is for the missing-lib case")

        with pytest.raises(DependencyMissingError) as exc_info:
            compile(
                source="def k(): pass",
                kernel_name="k",
                vendor=0,  # VENDOR_NVIDIA
                arch=90,  # ARCH_SM_90
                num_warps=4,
                num_stages=2,
            )
        # The error message must include a build hint
        msg = str(exc_info.value)
        assert "cmake" in msg.lower() or "build" in msg.lower() or "lib" in msg.lower(), (
            f"error message lacks a build hint: {msg!r}"
        )

    def test_triton_version_raises_when_unavailable(self) -> None:
        """triton_version() must raise DependencyMissingError when no .so."""
        from src.c_api import triton_version
        from src.common.errors import DependencyMissingError

        if _c_lib_loaded():
            pytest.skip("C library is built")

        with pytest.raises(DependencyMissingError):
            triton_version()

    def test_compile_signature_is_keyword_safe(self) -> None:
        """compile() must accept all kwargs by name (mirrors the C-API)."""
        from src.c_api import compile
        from src.common.errors import DependencyMissingError

        # Should not raise TypeError; only the underlying lib load should
        # raise. The signature itself must accept every keyword.
        try:
            compile(
                source="x",
                kernel_name="k",
                vendor=0,
                arch=90,
                num_warps=4,
                num_stages=2,
                block_m=64,
                block_n=64,
                block_k=32,
            )
        except DependencyMissingError:
            pass  # Expected when no .so
        except TypeError as exc:
            pytest.fail(f"compile() rejected kwargs: {exc}")


# ── TritonKernelHandle lifecycle ──────────────────────────────────────


class TestTritonKernelHandle:
    """TritonKernelHandle is a context manager / RAII handle."""

    def test_handle_class_exists(self) -> None:
        from src.c_api import TritonKernelHandle

        assert TritonKernelHandle is not None

    def test_handle_destructor_swallows_lib_missing(self) -> None:
        """The destructor (via __del__) must not crash when the .so is
        missing — the production code wraps release() in try/except for
        this exact reason.
        """
        if _c_lib_loaded():
            pytest.skip("C library is built; lib-missing path not exercised")

        from src.c_api import TritonKernelHandle

        # Construct with a fake c_handle. Letting it go out of scope
        # triggers __del__ → release() → _load_c_lib() which raises.
        # The production __del__ swallows that exception.
        handle = TritonKernelHandle(c_handle=0xDEADBEEF)
        del handle
        # If we get here, the destructor did not propagate the error.

    def test_handle_context_manager_releases(self) -> None:
        """When the .so IS built, the context manager protocol must
        release the handle on exit.
        """
        if not _c_lib_loaded():
            pytest.skip("C library not built; __exit__ path not exercised here")

        from src.c_api import TritonKernelHandle

        handle = TritonKernelHandle(c_handle=0xDEADBEEF)
        with handle as h:
            assert h is handle
        assert handle._released is True


# ── Loader probes for the actual .so file ────────────────────────────


class TestCApiLibrarySearch:
    """The library loader searches a fixed list of paths. Verify the
    search list is sane even when the .so doesn't exist.
    """

    def test_search_paths_listed(self) -> None:
        from src import c_api

        # The module exposes the search list via _C_LIB_PATHS.
        # Check it includes at least the conventional locations.
        paths = c_api._C_LIB_PATHS
        assert isinstance(paths, list)
        # At least one path must mention the library name
        assert any("nautilus_c_api" in p for p in paths)

    def test_loader_handles_missing_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset NAUTILUS_C_LIB must not crash the loader."""
        monkeypatch.delenv("NAUTILUS_C_LIB", raising=False)
        # Force a fresh load by clearing the cached lib
        import src.c_api as c_api_mod
        from src.c_api import _load_c_lib
        from src.common.errors import DependencyMissingError

        c_api_mod._C_LIB = None
        c_api_mod._C_LIB_LOAD_ERROR = None

        with pytest.raises(DependencyMissingError):
            _load_c_lib()

    def test_loader_handles_nonexistent_env_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """NAUTILUS_C_LIB pointing at a non-existent file must raise
        DependencyMissingError, not OSError.
        """
        fake_path = tmp_path / "does_not_exist.so"
        monkeypatch.setenv("NAUTILUS_C_LIB", str(fake_path))

        import src.c_api as c_api_mod
        from src.c_api import _load_c_lib
        from src.common.errors import DependencyMissingError

        c_api_mod._C_LIB = None
        c_api_mod._C_LIB_LOAD_ERROR = None

        with pytest.raises(DependencyMissingError):
            _load_c_lib()


# ── Helpers ───────────────────────────────────────────────────────────


def _c_lib_loaded() -> bool:
    """Check whether the C library is currently loaded (i.e. buildable)."""
    import src.c_api as c_api_mod

    return c_api_mod._C_LIB is not None
