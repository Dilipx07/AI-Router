"""
AI Router v1
PHASE 13 — FULL PRODUCTION HARDENING REGRESSION
"""

from __future__ import annotations

import subprocess
import sys


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


def run_test(
    name: str,
    filename: str,
) -> bool:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            filename,
        ],
        check=False,
    )

    passed = result.returncode == 0

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


def main() -> int:

    print("=" * 70)
    print("PHASE 13 — FULL PRODUCTION HARDENING REGRESSION")
    print("=" * 70)

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
    passed_count = total - len(failures)

    print(
        f"passed: {passed_count}/{total}"
    )

    print(
        f"failed: {len(failures)}"
    )

    if failures:

        print()
        print("FAILED TESTS:")

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
