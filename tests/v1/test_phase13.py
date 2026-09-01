"""
AI Router v1
PHASE 13 — FULL PRODUCTION HARDENING REGRESSION

Runs every v1 regression test from the repository root so that
project-level imports such as `router`, `providers`, and
`router_api` resolve correctly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

TESTS_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = TESTS_DIR.parent.parent


# ============================================================
# TEST SUITE
# ============================================================

TESTS = [
    (
        "Phase 10 usage",
        "test_phase10_usage.py",
    ),
    (
        "Phase 11 runtime accounting",
        "test_phase11.py",
    ),
    (
        "Phase 12 intelligent routing",
        "test_phase12.py",
    ),
    (
        "Runtime bootstrap",
        "test_runtime_bootstrap.py",
    ),
    (
        "Runtime lifecycle",
        "test_runtime_lifecycle.py",
    ),
    (
        "API error mapping",
        "test_api_error_mapping.py",
    ),
    (
        "API error semantics",
        "test_api_error_semantics.py",
    ),
    (
        "Request IDs",
        "test_request_ids.py",
    ),
    (
        "Concurrency",
        "test_concurrency.py",
    ),
]


# ============================================================
# TEST RUNNER
# ============================================================

def run_test(
    name: str,
    filename: str,
) -> bool:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    test_path = (
        TESTS_DIR
        / filename
    )

    if not test_path.exists():

        print(
            f"[FAIL] Test file not found: "
            f"{test_path}"
        )

        return False

    # --------------------------------------------------------
    # Make project-root imports available to subprocesses.
    # --------------------------------------------------------

    environment = os.environ.copy()

    current_pythonpath = (
        environment.get(
            "PYTHONPATH",
            "",
        )
    )

    project_root_string = str(
        PROJECT_ROOT
    )

    pythonpath_entries = [
        project_root_string
    ]

    if current_pythonpath:

        pythonpath_entries.append(
            current_pythonpath
        )

    environment["PYTHONPATH"] = (
        os.pathsep.join(
            pythonpath_entries
        )
    )

    # --------------------------------------------------------
    # Execute from repository root.
    # --------------------------------------------------------

    result = subprocess.run(
        [
            sys.executable,
            str(test_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )

    passed = (
        result.returncode == 0
    )

    print()

    if passed:

        print(
            f"[PASS] {name}"
        )

    else:

        print(
            f"[FAIL] {name}"
        )

    return passed


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 70)
    print(
        "PHASE 13 — FULL PRODUCTION HARDENING REGRESSION"
    )
    print("=" * 70)

    print(
        f"project root: {PROJECT_ROOT}"
    )

    print(
        f"test directory: {TESTS_DIR}"
    )

    failures: list[str] = []

    for name, filename in TESTS:

        passed = run_test(
            name,
            filename,
        )

        if not passed:

            failures.append(
                f"{name} ({filename})"
            )

    print()
    print("=" * 70)
    print("PHASE 13 — SUMMARY")
    print("=" * 70)

    total = len(TESTS)

    passed_count = (
        total
        - len(failures)
    )

    print(
        f"passed: {passed_count}/{total}"
    )

    print(
        f"failed: {len(failures)}"
    )

    if failures:

        print()
        print(
            "FAILED TESTS:"
        )

        for failure in failures:

            print(
                f"  - {failure}"
            )

        print()
        print(
            "PHASE 13 — RESULT: FAIL"
        )

        return 1

    print()
    print(
        "All production-hardening regressions passed."
    )

    print()
    print(
        "PHASE 13 — RESULT: PASS"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
