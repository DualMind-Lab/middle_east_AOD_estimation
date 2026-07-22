import csv
import os
import math

INPUT_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Middle_East_AOD_2015.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Middle_East_AOD_550nm.csv")

KEEP_COLS = [
    "AERONET_Site",
    "Date(dd:mm:yyyy)",
    "Time(hh:mm:ss)",
    "Day_of_Year",
    "Day_of_Year(Fraction)",
    "Site_Latitude(Degrees)",
    "Site_Longitude(Degrees)",
    "Site_Elevation(m)",
    "Solar_Zenith_Angle(Degrees)",
]

TARGET_WL = 550.0  # nm


def compute_aod550(aod_500, aod_675):
    """
    Interpolate AOD at 550nm using the Angstrom exponent formula:
        AOD(λ) = AOD(λ_ref) * (λ / λ_ref) ^ (-α)

    Only uses rows where both AOD_500nm and AOD_675nm are valid (> 0).
    α is derived directly from the two wavelengths.
    Returns None if either value is missing.
    """
    try:
        a500 = float(aod_500)
        a675 = float(aod_675)
    except (ValueError, TypeError):
        return None

    if a500 <= 0 or a675 <= 0:
        return None

    alpha = -math.log(a500 / a675) / math.log(500.0 / 675.0)
    aod550 = a500 * (TARGET_WL / 500.0) ** (-alpha)
    return round(aod550, 6)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)

        # Find column indices
        idx = {h: i for i, h in enumerate(header)}
        aod500_i = idx["AOD_500nm"]
        aod675_i = idx["AOD_675nm"]
        keep_i     = [idx[c] for c in KEEP_COLS]

        out_header = KEEP_COLS[:5] + ["AOD_550nm"] + KEEP_COLS[5:]

        total = 0
        kept  = 0
        dropped = 0

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(out_header)

            for row in reader:
                total += 1
                aod550 = compute_aod550(row[aod500_i], row[aod675_i])

                if aod550 is None or aod550 <= 0:
                    dropped += 1
                    continue

                out_row = (
                    [row[keep_i[i]] for i in range(5)]   # cols 0-4
                    + [aod550]                             # AOD_550nm
                    + [row[keep_i[i]] for i in range(5, len(keep_i))]  # remaining cols
                )
                writer.writerow(out_row)
                kept += 1

    print(f"Done.")
    print(f"  Total rows    : {total:,}")
    print(f"  Kept          : {kept:,}")
    print(f"  Dropped (no valid AOD): {dropped:,}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
