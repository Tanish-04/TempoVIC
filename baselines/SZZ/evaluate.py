"""
Evaluate SZZ baseline results against ground truth.

Computes precision, recall, and F1 across the entire dataset (not per-project).
Uses 12-character prefix matching for commit SHA comparison.

Usage:
    python evaluate.py --method b --time run1
    python evaluate.py --method v --time run1
    python evaluate.py --method ag --time run1
    python evaluate.py --method ma --time run1
"""

import os
import sys
import json
import argparse

from setting import *


def convert_project_name(project):
    return project.replace("/", "_") if "/" in project else project


def evaluate(lang: str, szz_method: str, time: str):
    """
    Evaluate SZZ results against ground truth for the entire dataset.

    Metrics:
        Precision = matched_commits / total_szz_commits
        Recall    = matched_commits / total_gt_commits
        F1        = harmonic mean of precision and recall
    """
    BANNER_W = 100

    data_file = os.path.join(DATA_FOLDER, f'verified_cve_with_versions_{lang}.json')
    if not os.path.exists(data_file):
        print(f"[ERROR] Dataset file not found: {data_file}")
        return None

    with open(data_file) as fin:
        labeled_items = json.load(fin)

    print(f"\nSZZ Method: {szz_method}-SZZ")
    print(f"Language: {lang}")
    print(f"Dataset: {len(labeled_items)} CVEs")
    print(f"{'=' * BANNER_W}")

    # Accumulators
    total_cves        = 0
    total_matches     = 0
    total_gt_commits  = 0
    total_szz_commits = 0
    szz_failures      = 0
    no_gt_count       = 0

    # Per-project breakdown (for informational table only)
    project_stats = {}

    for item in labeled_items:
        project  = item['project']
        cve_id   = item.get('cve_id', 'N/A')
        pro_name = convert_project_name(project)

        # Load SZZ results for this project
        result_path = os.path.join(
            WORK_DIR,
            f"results/{szz_method}-szz/{lang}/{time}/{szz_method}-{pro_name}.json"
        )

        if not os.path.exists(result_path):
            szz_failures += 1
            continue

        try:
            with open(result_path) as fin:
                szz_results = json.load(fin)
        except (json.JSONDecodeError, OSError):
            szz_failures += 1
            continue

        # Collect ground truth inducing commits for this CVE
        gt_commits = set()
        szz_commits = set()
        has_szz_result = True

        for fd in item['fixing_details']:
            fixing_commit = fd['fixing_commit']

            if fixing_commit not in szz_results:
                has_szz_result = False
                continue

            szz_output = szz_results[fixing_commit]

            # V-SZZ returns list of dicts with previous_commits
            if szz_method == 'v' and isinstance(szz_output, list) and szz_output and isinstance(szz_output[0], dict):
                for record in szz_output:
                    prev = record.get("previous_commits", [])
                    if prev:
                        last = prev[-1]
                        commit_sha = last.get("commit", last) if isinstance(last, dict) else last
                        szz_commits.add(commit_sha)
            elif isinstance(szz_output, list):
                szz_commits.update(szz_output)
            else:
                szz_commits.add(szz_output)

            for ic in fd['inducing_commits']:
                gt_commits.add(ic['commit_id'])

        if not has_szz_result and not szz_commits:
            szz_failures += 1
            continue

        if len(gt_commits) == 0:
            no_gt_count += 1
            continue

        total_cves += 1

        # 12-char prefix matching
        gt_short  = {c[:12] for c in gt_commits}
        szz_short = {c[:12] for c in szz_commits}
        matches   = len(gt_short & szz_short)

        total_matches     += matches
        total_gt_commits  += len(gt_commits)
        total_szz_commits += len(szz_commits)

        # Track per-project stats
        if project not in project_stats:
            project_stats[project] = {
                'cves': 0, 'matches': 0,
                'gt': 0, 'szz': 0,
            }
        project_stats[project]['cves']    += 1
        project_stats[project]['matches'] += matches
        project_stats[project]['gt']      += len(gt_commits)
        project_stats[project]['szz']     += len(szz_commits)

    # Compute metrics
    precision = total_matches / total_szz_commits if total_szz_commits > 0 else 0.0
    recall    = total_matches / total_gt_commits  if total_gt_commits  > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Print results
    print(f"\n{'=' * BANNER_W}")
    print(f"  RESULTS: {szz_method.upper()}-SZZ")
    print(f"{'=' * BANNER_W}")
    print(f"  {'CVEs evaluated:':<30} {total_cves:>8}")
    print(f"  {'SZZ failures:':<30} {szz_failures:>8}")
    print(f"  {'No ground truth:':<30} {no_gt_count:>8}")
    print(f"  {'Total GT commits:':<30} {total_gt_commits:>8}")
    print(f"  {'Total SZZ commits:':<30} {total_szz_commits:>8}")
    print(f"  {'Matched commits:':<30} {total_matches:>8}")
    print(f"  {'Precision:':<30} {precision:>8.4f}")
    print(f"  {'Recall:':<30} {recall:>8.4f}")
    print(f"  {'F1:':<30} {f1:>8.4f}")

    # Per-project breakdown table
    if project_stats:
        print(f"\n{'=' * BANNER_W}")
        print(f"  PER-PROJECT BREAKDOWN")
        print(f"{'=' * BANNER_W}")
        print(f"  {'Project':<30} {'CVEs':<8} {'Match':<8} {'GT':<8} {'SZZ':<8} {'Prec':<10} {'Recall':<10} {'F1':<10}")
        print(f"  {'-' * (BANNER_W - 4)}")
        for proj in sorted(project_stats.keys()):
            s = project_stats[proj]
            p = s['matches'] / s['szz'] if s['szz'] > 0 else 0.0
            r = s['matches'] / s['gt']  if s['gt']  > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            print(f"  {proj:<30} {s['cves']:<8} {s['matches']:<8} {s['gt']:<8} {s['szz']:<8} {p:<10.4f} {r:<10.4f} {f:<10.4f}")
        print(f"  {'-' * (BANNER_W - 4)}")
        print(f"  {'TOTAL':<30} {total_cves:<8} {total_matches:<8} {total_gt_commits:<8} {total_szz_commits:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

    return {
        'method': szz_method,
        'cves_evaluated': total_cves,
        'szz_failures': szz_failures,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SZZ baseline results')

    parser.add_argument('--method', type=str, default='b',
                        choices=['b', 'ag', 'ma', 'v'],
                        help='SZZ variant to evaluate (default: b)')
    parser.add_argument('--language', type=str, default='C',
                        help='Programming language (default: C)')
    parser.add_argument('--time', type=str, default='run1',
                        help='Run identifier matching main.py --time (default: run1)')

    args = parser.parse_args()

    results = evaluate(lang=args.language, szz_method=args.method, time=args.time)