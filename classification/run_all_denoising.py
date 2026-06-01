#!/usr/bin/env python3
"""
Orchestration script to run the full classification pipeline for all denoising methods.

Adapted from VEP_classification_comp/run_all_denoising.py for the AMBER dataset.

Usage:
    python run_all_denoising.py
    python run_all_denoising.py --skip-classify --skip-merge
    python run_all_denoising.py --method RAW --classification bysub_notime
    python run_all_denoising.py --recordings standard
    python run_all_denoising.py --recordings artifact --artifact-conditions X4 X6
    python run_all_denoising.py --recordings standard artifact --artifact-conditions X4
    python run_all_denoising.py --recordings X1 X4 X8
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BALANCING,
    DENOISING_METHODS,
    MA_WIN,
    TIME_SWITCH,
    get_task_id,
    resolve_recordings,
)


def run_command(cmd, label):
    """Run a subprocess command and report timing."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    t_start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
    elapsed = time.time() - t_start
    if result.returncode != 0:
        print(f"  ERROR: {label} failed with return code {result.returncode}")
        return False
    print(f"  Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline for all denoising methods"
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Skip feature extraction"
    )
    parser.add_argument(
        "--skip-classify", action="store_true", help="Skip classification"
    )
    parser.add_argument(
        "--skip-merge", action="store_true", help="Skip merging results"
    )
    parser.add_argument(
        "--skip-compare", action="store_true", help="Skip denoising comparison"
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=None,
        help="Specific method(s) to process (default: all)",
    )
    parser.add_argument(
        "--classification",
        default=None,
        help="Classification type (default: allsubs_notime)",
    )
    parser.add_argument(
        "--feature-type",
        choices=["stat", "temporal", "both"],
        default="both",
        help="Feature extraction type",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help=(
            "Which RSVP recordings to include. Options: "
            "'standard' (X1,X2), 'artifact' (X4,X6,X8), 'all' (default), "
            "or individual task codes (e.g. X1 X4 X8). "
            "Can combine groups: --recordings standard artifact"
        ),
    )
    parser.add_argument(
        "--artifact-conditions",
        nargs="+",
        default=None,
        help=(
            "When 'artifact' is in --recordings, which conditions to include. "
            "Any subset of: X4, X6, X8. Default: all three."
        ),
    )
    args = parser.parse_args()

    # Resolve recordings to task list and task_id
    tasks = resolve_recordings(args.recordings, args.artifact_conditions)
    task_id = get_task_id(tasks)
    print(f"Resolved recordings to tasks: {tasks}")
    print(f"Task set identifier: {task_id}")

    methods = args.method or list(DENOISING_METHODS.keys())
    classification_type = args.classification or ("allsubs")

    # Build task arguments for subprocess calls
    tasks_args = ["--tasks"] + tasks
    task_set_args = ["--task-set", task_id]

    # Step 1: Feature extraction
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("  STEP 1: Feature extraction")
        print("=" * 60)
        for method in methods:
            run_command(
                [
                    sys.executable,
                    "extract_features.py",
                    method,
                    "--feature-type",
                    args.feature_type,
                ]
                + tasks_args,
                f"Feature extraction - {method} (tasks: {task_id})",
            )

    # Step 2: Classification
    if not args.skip_classify:
        print("\n" + "=" * 60)
        print("  STEP 2: Running classification for all denoising methods")
        print("=" * 60)
        for method in methods:
            run_command(
                [
                    sys.executable,
                    "classify_main.py",
                    method,
                    "--classification",
                    classification_type,
                    "--time-switch",
                    TIME_SWITCH,
                ]
                + task_set_args,
                f"Classification - {method} (tasks: {task_id})",
            )

    # Step 3: Merge results
    if not args.skip_merge:
        print("\n" + "=" * 60)
        print("  STEP 3: Merging results for all denoising methods")
        print("=" * 60)
        for method in methods:
            run_command(
                [
                    sys.executable,
                    "merge_results.py",
                    method,
                    "--classification",
                    classification_type,
                ]
                + task_set_args,
                f"Merge results - {method} (tasks: {task_id})",
            )

    # Step 4: Compare denoising methods
    if not args.skip_compare:
        print("\n" + "=" * 60)
        print("  STEP 4: Comparing denoising methods")
        print("=" * 60)
        run_command(
            [
                sys.executable,
                "compare_denoising.py",
                "--classification",
                classification_type,
            ]
            + task_set_args,
            "Compare denoising methods (tasks: {})".format(task_id),
        )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
