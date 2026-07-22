"""
Delete Sentinel-2 image triplets that have no corresponding valid AERONET row
in Middle_East_AOD_550nm.csv. This covers two cases:

  1. Sites removed from the dataset (too few matched images).
  2. Acquisitions whose only matching AERONET measurement was dropped during
     AOD_550nm computation (AOD_500nm or AOD_675nm was invalid).

For each triplet in Sentinel2_L1C/{site}/{yyyymmdd_HHMMSS}_*.tif, the script
checks whether any row in the final CSV matches that site within ±30 minutes.
If none matches, all three resolution files are deleted.

Run with --dry-run to preview without deleting.
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_FILE    = os.path.join(BASE_DIR, "..", "Middle_East_AOD_550nm.csv")
IMAGE_DIR   = os.path.join(BASE_DIR, "..", "Sentinel2_L1C")
TIME_WINDOW = timedelta(minutes=30)

DRY_RUN = "--dry-run" in sys.argv


def parse_aeronet_dt(date_str, time_str):
    d, m, y = date_str.split(":")
    return datetime.strptime(f"{y}-{m}-{d} {time_str}", "%Y-%m-%d %H:%M:%S")


def load_valid_times():
    valid = defaultdict(list)
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        site_i = header.index("AERONET_Site")
        date_i = next(i for i, h in enumerate(header) if "Date"    in h)
        time_i = next(i for i, h in enumerate(header) if "Time(hh" in h)
        for row in reader:
            try:
                dt = parse_aeronet_dt(row[date_i], row[time_i])
                valid[row[site_i]].append(dt)
            except (ValueError, IndexError):
                continue
    for site in valid:
        valid[site].sort()
    print(f"Loaded {sum(len(v) for v in valid.values()):,} valid AERONET rows across {len(valid):,} sites.")
    return valid


def has_match(site, img_dt, valid):
    return any(
        abs((img_dt - t).total_seconds()) <= TIME_WINDOW.total_seconds()
        for t in valid.get(site, [])
    )


def main():
    valid = load_valid_times()

    deleted_triplets = 0
    deleted_files    = 0
    kept_triplets    = 0

    for site in sorted(os.listdir(IMAGE_DIR)):
        site_dir = os.path.join(IMAGE_DIR, site)
        if not os.path.isdir(site_dir):
            continue

        stems = set(
            f.rsplit("_", 1)[0]
            for f in os.listdir(site_dir)
            if f.endswith(".tif")
        )

        for stem in sorted(stems):
            try:
                img_dt = datetime.strptime(stem, "%Y%m%d_%H%M%S")
            except ValueError:
                continue

            if has_match(site, img_dt, valid):
                kept_triplets += 1
                continue

            deleted_triplets += 1
            for res in [10, 20, 60]:
                path = os.path.join(site_dir, f"{stem}_{res}m.tif")
                if os.path.exists(path):
                    if DRY_RUN:
                        print(f"[DRY-RUN] DELETE {path}")
                    else:
                        os.remove(path)
                    deleted_files += 1

    mode = "[DRY-RUN] Would delete" if DRY_RUN else "Deleted"
    print(f"\nDone.")
    print(f"  Kept     : {kept_triplets:,} triplets")
    print(f"  {mode} : {deleted_triplets:,} triplets  ({deleted_files:,} files)")
    if DRY_RUN:
        print("\nRe-run without --dry-run to apply deletions.")


if __name__ == "__main__":
    main()
