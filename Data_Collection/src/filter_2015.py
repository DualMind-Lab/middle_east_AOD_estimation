import csv
import os

INPUT_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Middle_East_AOD.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Middle_East_AOD_2015.csv")

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)

        date_idx = next(i for i, h in enumerate(header) if "Date" in h)

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

            kept = 0
            total = 0
            for row in reader:
                total += 1
                try:
                    # format: dd:mm:yyyy
                    year = int(row[date_idx].split(":")[2])
                except (ValueError, IndexError):
                    continue
                if year >= 2015:
                    writer.writerow(row)
                    kept += 1

    print(f"Done! Kept {kept:,} rows out of {total:,} total.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
