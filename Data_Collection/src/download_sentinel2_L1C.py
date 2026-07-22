import csv
import os
import math
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import tifffile
from dotenv import load_dotenv
from sentinelhub import (
    SHConfig, SentinelHubCatalog, SentinelHubRequest,
    DataCollection, MimeType, BBox, CRS, bbox_to_dimensions
)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

config = SHConfig()
config.sh_client_id     = os.getenv("CLIENT_ID")
config.sh_client_secret = os.getenv("CLIENT_SECRET")
config.sh_base_url      = "https://sh.dataspace.copernicus.eu"
config.sh_token_url     = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_FILE    = os.path.join(BASE_DIR, "..", "Middle_East_AOD_2015.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "Sentinel2_L1C")

HALF_KM     = 2500          # 2.5 km each side → 5×5 km patch
TIME_WINDOW = timedelta(minutes=30)
MAX_WORKERS = 8             # parallel threads

# Sites already fully downloaded — skip entirely to save API requests
EXCLUDE_SITES = {
    "SEDE_BOKER",
    "CUT-TEPAK",
    "Cairo_EMA_2",
    "Eilat",
    "El_Farafra",
    "IMS-METU-ERDEMLI",
    "KAUST_Campus",
    "KITcube_Masada",
    "Kuwait_University",
    "Masdar_Institute",
    "Mezaira",
    "Nicosia",
    "TUBITAK_UZAY_Ankara",
    "Tuz_Golu_3",
    # IASBS and Nes_Ziona intentionally excluded — need re-download (too few matches)
}

# All 13 S2 L1C bands grouped by native resolution
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B01", "B09", "B10"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "..", "download_log.txt")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

def make_evalscript(bands: list) -> str:
    """Generate an evalscript that returns the given bands scaled to UINT16 (0-10000)."""
    bands_js  = ", ".join(f'"{b}"' for b in bands)
    return_js = ", ".join(f"sample.{b} * 10000" for b in bands)
    return f"""
//VERSION=3
function setup() {{
    return {{
        input:  [{{ bands: [{bands_js}] }}],
        output: {{ bands: {len(bands)}, sampleType: "UINT16" }}
    }};
}}
function evaluatePixel(sample) {{
    return [{return_js}];
}}
"""

S2_L1C = DataCollection.SENTINEL2_L1C.define_from(
    "SENTINEL2_L1C_CDSE",
    service_url=config.sh_base_url,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_bbox(lat: float, lon: float) -> BBox:
    """Return a 5×5 km BBox centred on lat/lon."""
    d_lat = HALF_KM / 111_320
    d_lon = HALF_KM / (111_320 * math.cos(math.radians(lat)))
    return BBox(
        bbox=(lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat),
        crs=CRS.WGS84,
    )


def parse_aeronet_dt(date_str: str, time_str: str) -> datetime:
    """Parse dd:mm:yyyy and hh:mm:ss into a datetime."""
    d, m, y = date_str.split(":")
    return datetime.strptime(f"{y}-{m}-{d} {time_str}", "%Y-%m-%d %H:%M:%S")


def image_path(site: str, s2_dt: datetime, resolution: int) -> str:
    folder = os.path.join(OUTPUT_DIR, site)
    os.makedirs(folder, exist_ok=True)
    fname = s2_dt.strftime("%Y%m%d_%H%M%S") + f"_{resolution}m.tif"
    return os.path.join(folder, fname)


def already_downloaded(site: str, s2_dt: datetime) -> bool:
    # Consider fully downloaded only if all 3 resolution files exist
    return all(
        os.path.exists(image_path(site, s2_dt, res))
        for res in [10, 20, 60]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def load_csv() -> dict:
    """
    Returns a dict:
      { (site, date_str): {"lat": float, "lon": float, "times": [datetime, ...]} }
    """
    data = defaultdict(lambda: {"lat": None, "lon": None, "times": []})

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        lat_idx  = next(i for i, h in enumerate(header) if "Latitude"  in h)
        lon_idx  = next(i for i, h in enumerate(header) if "Longitude" in h)
        date_idx = next(i for i, h in enumerate(header) if "Date"      in h)
        time_idx = next(i for i, h in enumerate(header) if "Time(hh"   in h)
        site_idx = 0

        for row in reader:
            try:
                site = row[site_idx].strip()
                lat  = float(row[lat_idx])
                lon  = float(row[lon_idx])
                dt   = parse_aeronet_dt(row[date_idx], row[time_idx])
            except (ValueError, IndexError):
                continue

            key = (site, row[date_idx])
            data[key]["lat"] = lat
            data[key]["lon"] = lon
            data[key]["times"].append(dt)

    log.info(f"Loaded {len(data):,} unique (site, date) combinations.")
    return data


def download_resolution(site: str, s2_dt: datetime, bbox: BBox,
                         bands: list, resolution: int,
                         t_start: datetime, t_end: datetime) -> bool:
    """Download one resolution group and save as GeoTIFF. Returns True on success."""
    path = image_path(site, s2_dt, resolution)
    if os.path.exists(path):
        return True  # already have this resolution

    try:
        request = SentinelHubRequest(
            evalscript=make_evalscript(bands),
            input_data=[SentinelHubRequest.input_data(
                data_collection=S2_L1C,
                time_interval=(t_start.isoformat() + "Z", t_end.isoformat() + "Z"),
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=resolution),
            config=config,
        )
        data = request.get_data()[0]
        tifffile.imwrite(path, data.astype(np.uint16))
        return True
    except Exception as e:
        log.error(f"  Download failed [{resolution}m] for {site} @ {s2_dt}: {e}")
        return False


def download(site: str, s2_dt: datetime, bbox: BBox) -> bool:
    """Download all 13 bands across 3 resolution GeoTIFFs in parallel. Returns True if all succeed."""
    t_start = s2_dt - timedelta(seconds=1)
    t_end   = s2_dt + timedelta(seconds=1)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [
            ex.submit(download_resolution, site, s2_dt, bbox, BANDS_10M, 10, t_start, t_end),
            ex.submit(download_resolution, site, s2_dt, bbox, BANDS_20M, 20, t_start, t_end),
            ex.submit(download_resolution, site, s2_dt, bbox, BANDS_60M, 60, t_start, t_end),
        ]
        results = [f.result() for f in futures]
    return all(results)


def process_group(idx, site, date_str, info):
    """Process one (site, date) group: catalog search + download if matched."""
    results_summary = {"downloaded": 0, "skipped": 0, "failed": 0}

    lat, lon   = info["lat"], info["lon"]
    aero_times = info["times"]
    bbox       = make_bbox(lat, lon)

    d, m, y   = date_str.split(":")
    day_start = datetime(int(y), int(m), int(d), 0, 0, 0)
    day_end   = day_start + timedelta(days=1)

    catalog = SentinelHubCatalog(config=config)
    try:
        items = list(catalog.search(
            S2_L1C,
            bbox=bbox,
            time=(day_start.isoformat() + "Z", day_end.isoformat() + "Z"),
            fields={"include": ["id", "properties.datetime"], "exclude": []},
        ))
    except Exception as e:
        log.warning(f"Catalog search failed for {site} {date_str}: {e}")
        return results_summary

    for item in items:
        s2_dt_str = item["properties"]["datetime"]
        s2_dt     = datetime.fromisoformat(s2_dt_str.replace("Z", "+00:00")).replace(tzinfo=None)

        matched = any(abs((s2_dt - at).total_seconds()) <= TIME_WINDOW.total_seconds()
                      for at in aero_times)
        if not matched:
            continue

        if already_downloaded(site, s2_dt):
            results_summary["skipped"] += 1
            log.info(f"[{idx}] SKIP {site} {s2_dt} (already exists)")
            continue

        log.info(f"[{idx}] DOWNLOAD {site} {s2_dt}")
        ok = download(site, s2_dt, bbox)
        if ok:
            results_summary["downloaded"] += 1
        else:
            results_summary["failed"] += 1

    return results_summary


def main():
    groups = load_csv()
    total  = {"downloaded": 0, "skipped": 0, "failed": 0}
    lock   = Lock()

    tasks = [
        (idx, site, date_str, info)
        for idx, ((site, date_str), info) in enumerate(groups.items())
        if site not in EXCLUDE_SITES
    ]
    log.info(f"Excluded sites: {EXCLUDE_SITES}. Remaining tasks: {len(tasks):,}")

    log.info(f"Starting with {MAX_WORKERS} parallel workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_group, idx, site, date_str, info): idx
            for idx, site, date_str, info in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                for k in total:
                    total[k] += result[k]
            if i % 500 == 0:
                log.info(
                    f"Progress: {i:,}/{len(tasks):,} groups done | "
                    f"Downloaded: {total['downloaded']} | "
                    f"Skipped: {total['skipped']} | "
                    f"Failed: {total['failed']}"
                )

    log.info(
        f"\nDone. Downloaded: {total['downloaded']} | "
        f"Skipped: {total['skipped']} | Failed: {total['failed']}"
    )


if __name__ == "__main__":
    main()
