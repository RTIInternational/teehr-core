"""Run the shared conformance cases with teehr-core's engine.

python -m teehr_core.conformance path/to/teehr-shared-eval-spec/conformance/cases
"""

from __future__ import annotations

import sys

from teehr_core.conformance import run_cases


def main(argv: list[str]) -> int:
    """Run the cases in ``argv[0]``; print each outcome; return 1 if any failed."""
    if len(argv) != 1:
        print(__doc__)
        return 2
    outcomes = run_cases(argv[0])
    for o in outcomes:
        print(f"{'pass' if o.passed else 'FAIL'}  {o.case}")
        for p in o.problems[:20]:
            print(f"      {p}")
    failed = sum(not o.passed for o in outcomes)
    print(f"\n{len(outcomes) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
