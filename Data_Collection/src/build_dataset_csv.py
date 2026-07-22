"""
Build Dataset/dataset.csv — one row per Sentinel-2 image.

For each image, all AERONET measurements within ±30 minutes are collected.
Following the NASA Dark Target validation protocol (and Sayer et al. 2020,
Virtanen et al. 2018), multiple measurements are averaged into a single
representative AOD value. Images with fewer than 2 measurements are dropped.

Non-AOD columns (site, lat, lon, elevation, solar zenith, day-of-year) are
taken from the measurement closest in time to the satellite overpass.

Output columns: all 10 original columns + image_path (relative to project root).
One row per image.
"""

import csv
import os
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_IN      = os.path.join(BASE_DIR, "..", "temp_csv_files", "Middle_East_AOD_550nm.csv")
IMAGE_DIR   = os.path.join(BASE_DIR, "..", "Dataset", "images")
CSV_OUT     = os.path.join(BASE_DIR, "..", "Dataset", "dataset.csv")
TIME_WINDOW = timedelta(minutes=30)
MIN_MEASUREMENTS = 2

PATH_PREFIX = "Data_Collection/Dataset/images"


def parse_aeronet_dt(date_str, time_str):
    d, m, y = date_str.split(":")
    return datetime.strptime(f"{y}-{m}-{d} {time_str}", "%Y-%m-%d %H:%M:%S")


def load_images_by_site():
    """Returns {site: sorted list of (datetime, relative_path)}."""
    images = defaultdict(list)
    for site in os.listdir(IMAGE_DIR):
        site_dir = os.path.join(IMAGE_DIR, site)
        if not os.path.isdir(site_dir):
            continue
        for fname in sorted(os.listdir(site_dir)):
            if not fname.endswith(".tif"):
                continue
            try:
                dt = datetime.strptime(fname[:-4], "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            images[site].append((dt, f"{PATH_PREFIX}/{site}/{fname}"))
    for site in images:
        images[site].sort(key=lambda x: x[0])
    return images


def find_closest(aero_dt, site_images):
    """Binary search — returns (path, diff_seconds) of closest image within ±30 min."""
    if not site_images:
        return None, None
    dts = [x[0] for x in site_images]
    pos = bisect_left(dts, aero_dt)
    best_path, best_diff = None, float("inf")
    for i in [pos - 1, pos]:
        if 0 <= i < len(site_images):
            diff = abs((site_images[i][0] - aero_dt).total_seconds())
            if diff <= TIME_WINDOW.total_seconds() and diff < best_diff:
                best_diff = diff
                best_path = site_images[i][1]
    return best_path, best_diff


def main():
    print("Loading image index...")
    images_by_site = load_images_by_site()
    print(f"  {sum(len(v) for v in images_by_site.values()):,} images across {len(images_by_site):,} sites.")

    print("Matching AERONET rows to images...")
    image_rows = defaultdict(list)
    rows_in = rows_no_match = 0

    with open(CSV_IN, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        site_i = header.index("AERONET_Site")
        date_i = next(i for i, h in enumerate(header) if "Date"    in h)
        time_i = next(i for i, h in enumerate(header) if "Time(hh" in h)
        aod_i  = header.index("AOD_550nm")

        for row in reader:
            rows_in += 1
            try:
                aero_dt = parse_aeronet_dt(row[date_i], row[time_i])
            except (ValueError, IndexError):
                rows_no_match += 1
                continue
            site = row[site_i].strip()
            path, diff = find_closest(aero_dt, images_by_site.get(site, []))
            if path is None:
                rows_no_match += 1
                continue
            image_rows[path].append((diff, row))

    print("Aggregating...")
    kept = dropped_single = 0

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header + ["image_path"])

        for path, entries in sorted(image_rows.items()):
            if len(entries) < MIN_MEASUREMENTS:
                dropped_single += 1
                continue

            # Mean AOD over all matched measurements
            mean_aod = round(sum(float(r[aod_i]) for _, r in entries) / len(entries), 6)

            # Metadata from the measurement closest in time to the overpass
            _, closest_row = min(entries, key=lambda x: x[0])

            out_row = closest_row[:]
            out_row[aod_i] = mean_aod
            writer.writerow(out_row + [path])
            kept += 1

    print(f"\nDone.")
    print(f"  Input rows                    : {rows_in:,}")
    print(f"  Unmatched (no overpass)       : {rows_no_match:,}")
    print(f"  Final rows (>=2 measurements) : {kept:,}")
    print(f"  Dropped (<2 measurements)     : {dropped_single:,}")
    print(f"  Output: {CSV_OUT}")


if __name__ == "__main__":
    main()
