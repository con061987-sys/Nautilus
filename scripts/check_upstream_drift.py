#!/usr/bin/env python3
"""scripts/check_upstream_drift.py — Detect upstream dependency drift.

Compares the pinned versions in pyproject.toml against the latest
available versions on PyPI. If any pinned version is no longer the
latest, flags it as DRIFT. The CI opens a GitHub issue when this
happens.

Exits 0 if no drift, 1 if drift detected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import tomllib


NAUTILUS_ROOT = Path(__file__).resolve().parent.parent


# Packages that are particularly sensitive to version drift
SENSITIVE_PACKAGES = [
    "torch", "triton", "apache-tvm", "torch_xla",
    "aotriton", "networkx", "pyzmq",
]


def _load_pinned_versions() -> dict[str, str]:
    """Return {package_name: pinned_version} from pyproject.toml."""
    data = tomllib.loads((NAUTILUS_ROOT / "pyproject.toml").read_text())
    pinned: dict[str, str] = {}
    for section in ("dependencies",):
        for spec in data["project"].get(section, []):
            name, _, ver = re.split(r"(<=|>=|==|~=|<|>|!=)", spec, maxsplit=1)
            pinned[name.strip().lower()] = ver.strip()
    for extras in data["project"].get("optional-dependencies", {}).values():
        for spec in extras:
            name, _, ver = re.split(r"(<=|>=|==|~=|<|>|!=)", spec, maxsplit=1)
            pinned.setdefault(name.strip().lower(), ver.strip())
    return pinned


def _get_latest_version(package: str) -> str | None:
    """Fetch the latest version of `package` from PyPI. Returns None on error."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"warn: failed to fetch {package}: {exc}", file=sys.stderr)
        return None
    return data.get("info", {}).get("version")


def _strip_pin_op(pin: str) -> str:
    """Strip leading operators from a pin (e.g. '==2.4.1' -> '2.4.1')."""
    return re.sub(r"^([<>=!~]+)", "", pin).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", action="store_true", help="Output Markdown for GitHub issue")
    args = parser.parse_args()

    pinned = _load_pinned_versions()
    drift: list[dict[str, Any]] = []

    print(f"Checking {len(SENSITIVE_PACKAGES)} sensitive packages for drift...")
    for pkg in SENSITIVE_PACKAGES:
        if pkg not in pinned:
            print(f"  {pkg}: not pinned in pyproject.toml — DRIFT")
            drift.append({
                "package": pkg,
                "pinned": "(not pinned)",
                "latest": _get_latest_version(pkg) or "(unknown)",
                "status": "MISSING",
            })
            continue
        pinned_ver = _strip_pin_op(pinned[pkg])
        latest = _get_latest_version(pkg)
        if latest is None:
            print(f"  {pkg}: could not fetch latest version")
            continue
        if pinned_ver != latest:
            print(f"  {pkg}: pinned={pinned_ver}, latest={latest} — DRIFT")
            drift.append({
                "package": pkg,
                "pinned": pinned_ver,
                "latest": latest,
                "status": "OUT_OF_DATE",
            })
        else:
            print(f"  {pkg}: pinned={pinned_ver} — OK")

    if not drift:
        print()
        print("No drift detected.")
        if args.md:
            print("## No drift")
        return 0

    print()
    print(f"DRIFT DETECTED in {len(drift)} package(s)")
    if args.md:
        print("\n## DRIFT")
        print("\n| Package | Pinned | Latest |")
        print("|---|---|---|")
        for d in drift:
            print(f"| {d['package']} | {d['pinned']} | {d['latest']} |")
        print("\nAction required: bump the pin in pyproject.toml and verify CI passes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
