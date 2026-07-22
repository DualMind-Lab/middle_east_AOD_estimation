"""
Merge each Sentinel-2 acquisition triplet (_10m, _20m, _60m) into a single
13-band GeoTIFF at 10m resolution using nearest-neighbour upsampling.

Input  : Data_Collection/Sentinel2_L1C/{site}/{yyyymmdd_HHMMSS}_*.tif
Output : Data_Collection/Dataset/images/{site}/{yyyymmdd_HHMMSS}.tif

Band order in output (wavelength order):
  0  B01  (coastal aerosol,  60m → upsampled)
  1  B02  (blue,             10m)
  2  B03  (green,            10m)
  3  B04  (red,              10m)
  4  B05  (red-edge 1,       20m → upsampled)
  5  B06  (red-edge 2,       20m → upsampled)
  6  B07  (red-edge 3,       20m → upsampled)
  7  B08  (NIR,              10m)
  8  B8A  (narrow NIR,       20m → upsampled)
  9  B09  (water vapour,     60m → upsampled)
  10 B10  (SWIR cirrus,      60m → upsampled)
  11 B11  (SWIR 1,           20m → upsampled)
  12 B12  (SWIR 2,           20m → upsampled)
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "..", "Sentinel2_L1C")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Dataset", "images")
MAX_WORKERS = 8


def merge_triplet(site_dir: str, stem: str, out_path: str) -> bool:
    """Load the three resolution TIFFs, upsample to 10m, stack 13 bands, save."""
    p10 = os.path.join(site_dir, f"{stem}_10m.tif")
    p20 = os.path.join(site_dir, f"{stem}_20m.tif")
    p60 = os.path.join(site_dir, f"{stem}_60m.tif")

    if not all(os.path.exists(p) for p in [p10, p20, p60]):
        return False

    img10 = tifffile.imread(p10)  # (H,   W,   4)  B02 B03 B04 B08
    img20 = tifffile.imread(p20)  # (H/2, W/2, 6)  B05 B06 B07 B8A B11 B12
    img60 = tifffile.imread(p60)  # (H/6, W/6, 3)  B01 B09 B10

    h, w = img10.shape[:2]

    # Upsample 20m and 60m to 10m using nearest-neighbour
    img20_up = cv2.resize(img20, (w, h), interpolation=cv2.INTER_NEAREST)
    img60_up = cv2.resize(img60, (w, h), interpolation=cv2.INTER_NEAREST)

    # Stack in wavelength order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
    merged = np.stack([
        img60_up[:, :, 0],  # B01
        img10[:, :, 0],     # B02
        img10[:, :, 1],     # B03
        img10[:, :, 2],     # B04
        img20_up[:, :, 0],  # B05
        img20_up[:, :, 1],  # B06
        img20_up[:, :, 2],  # B07
        img10[:, :, 3],     # B08
        img20_up[:, :, 3],  # B8A
        img60_up[:, :, 1],  # B09
        img60_up[:, :, 2],  # B10
        img20_up[:, :, 4],  # B11
        img20_up[:, :, 5],  # B12
    ], axis=-1).astype(np.uint16)   # shape: (H, W, 13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, merged)
    return True


def collect_tasks():
    tasks = []
    for site in sorted(os.listdir(INPUT_DIR)):
        site_dir = os.path.join(INPUT_DIR, site)
        if not os.path.isdir(site_dir):
            continue
        stems = set(
            f.rsplit("_", 1)[0]
            for f in os.listdir(site_dir)
            if f.endswith(".tif")
        )
        out_site_dir = os.path.join(OUTPUT_DIR, site)
        for stem in sorted(stems):
            out_path = os.path.join(out_site_dir, f"{stem}.tif")
            if os.path.exists(out_path):
                continue  # already merged
            tasks.append((site_dir, stem, out_path))
    return tasks


def main():
    tasks = collect_tasks()
    print(f"Triplets to merge: {len(tasks):,}")
    if not tasks:
        print("Nothing to do.")
        return

    done = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(merge_triplet, s, st, o): (s, st) for s, st, o in tasks}
        with tqdm(total=len(tasks), unit="triplet") as bar:
            for future in as_completed(futures):
                ok = future.result()
                if ok:
                    done += 1
                else:
                    failed += 1
                bar.update(1)

    print(f"\nDone.  Merged: {done:,}  |  Failed: {failed:,}")


if __name__ == "__main__":
    main()
