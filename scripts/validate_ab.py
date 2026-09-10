#!/usr/bin/env python3
"""Compare two `jackdaw validate` transcripts to attribute failures.

`jackdaw validate` needs a live BalatroBot server, so it cannot run in CI or
on a dev machine without Balatro. When it reports failures after a code
change, the only question that matters is: **were these failures already
there?** Diffing the raw transcripts is unhelpful — they interleave timing
noise and per-scenario detail. This compares the *sets of failing scenarios*.

Workflow (engine change on `jackdaw/engine/hand_eval.py` as the example)::

    git stash push jackdaw/engine/hand_eval.py
    jackdaw validate > before.txt 2>&1
    git stash pop
    jackdaw validate > after.txt 2>&1
    uv run scripts/validate_ab.py before.txt after.txt

An empty "newly failing" set exonerates the change: whatever is red was red
already, and belongs to the upstream fork rather than to you.

Interpretation notes:
- Some scenarios may be genuinely flaky against a live game (animation
  timing, RNG in the live client). A scenario appearing in *both* newly
  failing and newly passing across repeat runs is flaky, not caused.
- If in doubt, run each side twice and intersect: a real regression fails
  consistently.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Summary lines look like:  "    [x] scenario_name"  /  "    [v] scenario_name"
# The mark characters vary by terminal encoding, so match structurally.
_SUMMARY_ROW = re.compile(r"^\s{4}\[(.)\]\s+(\S+)\s*$")
_CATEGORY_ROW = re.compile(r"^\s{2}(\w+):\s+(PASS|FAIL)\s+\((\d+)/(\d+)\)\s*$")
_TOTAL_ROW = re.compile(r"^Total:\s+(\d+)/(\d+)\s+passed\s*$")


def parse(path: Path) -> tuple[dict[str, bool], tuple[int, int] | None]:
    """Return ({scenario: passed}, (passed, total)) from a validate transcript.

    The per-scenario summary rows are marked with a glyph whose exact
    character depends on the terminal encoding (Windows consoles mangle it),
    so rather than hardcoding one, collect the raw marks and work out which
    means "pass" from the reported totals.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    in_summary = False
    marks: dict[str, str] = {}
    totals: tuple[int, int] | None = None

    for line in text.splitlines():
        if line.strip() == "SUMMARY":
            in_summary = True
            continue
        total_match = _TOTAL_ROW.match(line)
        if total_match:
            totals = (int(total_match.group(1)), int(total_match.group(2)))
            continue
        if not in_summary:
            continue
        row = _SUMMARY_ROW.match(line)
        if row:
            marks[row.group(2)] = row.group(1)

    if not marks:
        return {}, totals

    counts: dict[str, int] = {}
    for mark in marks.values():
        counts[mark] = counts.get(mark, 0) + 1
    if len(counts) == 1:
        # Every scenario shares one mark: all passed, or all failed. Only the
        # totals can tell which.
        all_passed = bool(totals and totals[0] == totals[1])
        return {name: all_passed for name in marks}, totals
    if totals:
        # The mark whose count matches the reported pass count is "pass".
        pass_mark = min(counts, key=lambda m: abs(counts[m] - totals[0]))
    else:
        pass_mark = max(counts, key=lambda m: counts[m])
    return {name: mark == pass_mark for name, mark in marks.items()}, totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="transcript from the UNCHANGED code")
    parser.add_argument("after", type=Path, help="transcript from the CHANGED code")
    args = parser.parse_args()

    before, before_totals = parse(args.before)
    after, after_totals = parse(args.after)

    if not before or not after:
        raise SystemExit(
            "Could not parse a SUMMARY section from one of the transcripts. "
            "Capture the full output of `jackdaw validate`, including the summary."
        )

    def fmt(totals: tuple[int, int] | None) -> str:
        return f"{totals[0]}/{totals[1]} passed" if totals else "unknown"

    print(f"before ({args.before.name}): {fmt(before_totals)}, {len(before)} scenarios")
    print(f"after  ({args.after.name}): {fmt(after_totals)}, {len(after)} scenarios")

    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    if only_before or only_after:
        print("\n!! the two runs did not cover the same scenarios")
        for name in only_before:
            print(f"   only in before: {name}")
        for name in only_after:
            print(f"   only in after:  {name}")

    shared = sorted(set(before) & set(after))
    newly_failing = [n for n in shared if before[n] and not after[n]]
    newly_passing = [n for n in shared if not before[n] and after[n]]
    already_failing = [n for n in shared if not before[n] and not after[n]]

    print(f"\nfailing before AND after (pre-existing): {len(already_failing)}")
    for name in already_failing:
        print(f"   = {name}")

    print(f"\nnewly passing after the change: {len(newly_passing)}")
    for name in newly_passing:
        print(f"   + {name}")

    print(f"\n*** NEWLY FAILING (caused by the change): {len(newly_failing)} ***")
    for name in newly_failing:
        print(f"   - {name}")

    if newly_failing:
        print(
            "\nVerdict: the change introduced failures. Re-run both sides once "
            "more to rule out live-client flakiness before concluding."
        )
        sys.exit(1)
    print(
        "\nVerdict: no scenario regressed. Any red above was already red on the "
        "unchanged code and belongs to the upstream fork, not to this change."
    )


if __name__ == "__main__":
    main()
