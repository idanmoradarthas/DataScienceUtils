"""Generate pull request report for Tier 2 cross-skill similarity check.

This script parses the enforced similarity-check report and the margin similarity-check
report (run with a lower threshold such as 0.65), formats a Markdown summary containing
the enforced check status, threshold, and top 10 near-threshold pairs, and appends the
summary to --summary-file.

The script always exits 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_similarity_report(report_path: Path) -> tuple[str, list[dict]]:
    """Parse a skillevaluator-similarity.json file.

    Returns:
        (overall_status, findings_list)

    """
    if not report_path.is_file():
        return "", []
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        status = data.get("overall_status", "")
        findings = []
        for r in data.get("results", []):
            findings.extend(r.get("findings", []))
        return status, findings
    except Exception:
        return "", []


def process_similarity_report(
    enforced_file: Path,
    margin_file: Path,
    threshold: float,
    summary_file: Path | None,
    margin_threshold: float = 0.65,
) -> None:
    """Format and write similarity check summary."""
    lines: list[str] = [
        "## Tier 2 similarity (blocking)",
        "",
    ]

    enforced_status, _ = parse_similarity_report(enforced_file)
    if not enforced_status:
        enforced_label = "unknown / missing"
    else:
        enforced_label = enforced_status.upper()

    lines.append(f"- **Enforced Check Status**: `{enforced_label}` (threshold: `{threshold}`)")
    lines.append("")

    lines.append("### Near-threshold similarity (margin probe)")
    lines.append("")

    margin_status, margin_findings = parse_similarity_report(margin_file)
    if not margin_status and not margin_findings:
        lines.append("margin unavailable")
    else:
        # Extract pairs with similarity below threshold
        pairs: list[tuple[float, str, str]] = []
        for f in margin_findings:
            meta = f.get("metadata", {})
            score = meta.get("score")
            if score is None:
                continue
            try:
                score_val = float(score)
            except (ValueError, TypeError):
                continue

            if score_val < threshold:
                entry_a = meta.get("entry_a") or meta.get("path_a") or ""
                entry_b = meta.get("entry_b") or meta.get("path_b") or ""
                if entry_a and entry_b:
                    pairs.append((score_val, str(entry_a), str(entry_b)))

        if not pairs:
            lines.append(f"No pair scored between {margin_threshold} and {threshold}.")
        else:
            # Sort descending by score, take top 10
            pairs.sort(key=lambda x: x[0], reverse=True)
            top_pairs = pairs[:10]
            for score_val, entry_a, entry_b in top_pairs:
                lines.append(f"- `{entry_a} <-> {entry_b}` : `{score_val:.3f}`")

    lines.append("")

    content = "\n".join(lines) + "\n"
    if summary_file:
        try:
            with summary_file.open("a", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            print(f"Warning: Could not append to summary file '{summary_file}': {exc}", file=sys.stderr)
            print(content)
    else:
        print(content)


def main() -> None:
    """CLI entry point for pr_similarity_report."""
    parser = argparse.ArgumentParser(description="Generate Tier 2 similarity summary report.")
    parser.add_argument("--enforced", required=True, help="Path to enforced skillevaluator-similarity.json")
    parser.add_argument("--margin", required=True, help="Path to margin skillevaluator-similarity.json")
    parser.add_argument("--threshold", type=float, default=0.75, help="Enforced threshold (default: 0.75)")
    parser.add_argument("--margin-threshold", type=float, default=0.65, help="Margin probe threshold (default: 0.65)")
    parser.add_argument("--summary-file", help="Path to append Markdown summary")

    args = parser.parse_args()

    process_similarity_report(
        enforced_file=Path(args.enforced),
        margin_file=Path(args.margin),
        threshold=args.threshold,
        summary_file=Path(args.summary_file) if args.summary_file else None,
        margin_threshold=args.margin_threshold,
    )


if __name__ == "__main__":
    main()
