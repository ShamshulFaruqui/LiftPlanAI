"""
Rule-Based Lift Group Classifier — Stage 3 of the LiftPlan AI pipeline.

Classifies a lift group's type (passenger, service, firefighter, goods)
purely from its group ID (e.g. "EL01", "SL02", "FFL01") — NOT from any
explicit "(PASSENGER)"-style label in the drawing text. This matters
because real drawings won't always spell the type out next to the group
ID; often the ID's own prefix convention is the only signal available.

This is deliberately a separate, independent signal from what the text
extractor already reads (the explicit type label, when present). Used
together, they let the system corroborate a classification from two
sources, or fall back to this classifier when no explicit label exists —
matching the proposal's description of a "hybrid lift group classifier
combining rule-based keyword logic with a supervised ML model."

The rule-based layer here IS the keyword-logic half; a supervised model
could be layered on top later for cases the rules can't resolve (e.g.
non-standard or ambiguous group ID conventions from a given architectural
office), consistent with the phased approach in the proposal's Methods
section.

Usage:
    python rule_based_classifier.py --group-id "FFL01"
    python rule_based_classifier.py --evaluate --ground-truth ../../data/annotations/synthetic_ground_truth.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Prefix -> lift type. Order matters: longer, more specific prefixes must
# be checked before shorter ones they contain (e.g. "FFL" before any
# hypothetical bare "F", "SL"/"SE"/"SV" before a bare "S"), otherwise the
# shorter prefix would win a naive first-match search. This mirrors the
# same specificity principle used in the file triage keyword lexicon.
PREFIX_TYPE_MAP = [
    ("FFL", "firefighter"),
    ("FF", "firefighter"),
    ("SL", "service"),
    ("SE", "service"),
    ("SV", "service"),
    ("GL", "goods"),
    ("EL", "passenger"),
    ("PL", "passenger"),
    ("BED", "bed"),
    ("STR", "bed"),
]


def classify_group_id(group_id: str) -> str:
    """
    Extract the leading alphabetic prefix from a group ID and classify its
    type. Handles compound IDs like "EL01-EL02" or "EL01-EL02..." by
    classifying from the first token only, since a single group's ID
    prefix is consistent across all its cars in this project's convention.

    Returns None if no known prefix matches, rather than guessing — an
    unrecognised prefix should be flagged for engineer review, not
    silently misclassified.
    """
    if not group_id:
        return None

    first_token = re.split(r"[-.]", group_id.strip())[0]
    prefix_letters = re.match(r"[A-Za-z]+", first_token)
    if not prefix_letters:
        return None
    prefix = prefix_letters.group(0).upper()

    for known_prefix, lift_type in PREFIX_TYPE_MAP:
        if prefix.startswith(known_prefix):
            return lift_type

    return None


def evaluate_against_ground_truth(ground_truth_path: Path, verbose: bool = True) -> dict:
    """
    Classify every group's type from its ID alone (ignoring the ground
    truth's explicit lift_type label during classification, using it only
    to check correctness afterward) and report accuracy.
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    metrics = {"total_groups": 0, "correct": 0, "unrecognised": [], "misclassified": []}

    for project in ground_truth:
        for group in project["groups"]:
            metrics["total_groups"] += 1
            predicted = classify_group_id(group["group_id"])
            actual = group["lift_type"]

            if predicted is None:
                metrics["unrecognised"].append((project["project_id"], group["group_id"], actual))
            elif predicted == actual:
                metrics["correct"] += 1
            else:
                metrics["misclassified"].append(
                    (project["project_id"], group["group_id"], actual, predicted)
                )

    n = metrics["total_groups"]
    print("=" * 60)
    print("RULE-BASED CLASSIFIER ACCURACY (group ID -> type, no label used)")
    print("=" * 60)
    print(f"Correct:       {metrics['correct']}/{n} ({metrics['correct']/n*100:.1f}%)")
    print(f"Unrecognised:  {len(metrics['unrecognised'])}/{n}")
    print(f"Misclassified: {len(metrics['misclassified'])}/{n}")

    if verbose and metrics["unrecognised"]:
        print("\nUnrecognised group IDs:")
        for proj_id, gid, actual in metrics["unrecognised"][:10]:
            print(f"  {proj_id}: {gid!r} (true type: {actual})")

    if verbose and metrics["misclassified"]:
        print("\nMisclassifications:")
        for proj_id, gid, actual, predicted in metrics["misclassified"][:10]:
            print(f"  {proj_id}: {gid!r} -> predicted {predicted}, actual {actual}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 3: Rule-Based Lift Group Classifier")
    parser.add_argument("--group-id", type=str, help="Classify a single group ID, e.g. 'FFL01'")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run classification against the full synthetic ground truth")
    parser.add_argument("--ground-truth", type=str, default="data/annotations/synthetic_ground_truth.json")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_against_ground_truth(Path(args.ground_truth))
    elif args.group_id:
        result = classify_group_id(args.group_id)
        print(f"{args.group_id!r} -> {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
