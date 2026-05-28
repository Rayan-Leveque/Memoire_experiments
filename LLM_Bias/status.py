#!/usr/bin/env python3
import csv, os
from collections import Counter
from pathlib import Path

RESULTS = Path("data/results")
TOTAL = 700

counts = Counter()
for f in sorted(RESULTS.glob("behavioral_results_*.csv")):
    with open(f) as fh:
        for row in csv.DictReader(fh):
            counts[row["model"]] += 1

print(f"{'Model':<35} {'Rows':>6}  {'%':>6}  {'Status'}")
print("-" * 60)
for model, count in sorted(counts.items(), key=lambda x: -x[1]):
    pct = count / TOTAL * 100
    status = "✓ done" if count >= TOTAL else f"{TOTAL - count} left"
    print(f"{model:<35} {count:>6}  {pct:>5.1f}%  {status}")
