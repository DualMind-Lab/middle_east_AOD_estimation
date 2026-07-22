import csv
import os
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "All_Sites_Times_All_Points_AOD20.dat")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Middle_East_AOD.csv")
HEADER_ROW_INDEX = 6  # first 6 lines are metadata, line 7 is the column header

# Verified from aeronet_locations_v3.txt (fetched 2026-04-19)
MIDDLE_EAST_SITES = {
    # Egypt
    "Cairo_University", "Cairo_EMA", "Cairo_EMA_2",
    "El_Farafra", "Qena_SVU",
    # Israel
    "SEDE_BOKER", "Nes_Ziona", "Dead_Sea", "Migal", "Amiaz",
    "Shizafon", "EIN_TAMAR", "Weizmann_Institute", "Haifa",
    "Eilat", "Technion_Haifa_IL", "Tel-Aviv_University", "KITcube_Masada",
    # Jordan
    "KITcube_WadiRum",
    # Iraq
    "Baghdad",
    # Iran
    "IASBS",
    # Kuwait
    "Kuwait_University", "Kuwait_University_2", "Kuwait_Airport",
    "Kuwait_Inst_Sci_Res", "Shagaya_Park",
    # UAE
    "Abu_Dhabi", "Dhabi", "Al_Ain", "Mussafa", "Umm_Al_Quwain", "MAARCO",
    "SMART", "SMART_POL", "Al_Dhafra", "Mezaira", "Al_Khaznah",
    "Hamim", "Jabal_Hafeet", "Al_Qlaa", "Saih_Salam", "Dhadnah",
    "Abu_Al_Bukhoosh", "Dalma", "Sir_Bu_Nuair", "Zakum_Field",
    "Masdar_Institute", "Masdar_Institute_2",
    "DEWA_ResearchCentre", "Khalifa_University",
    # Saudi Arabia
    "Solar_Village", "KAUST_Campus", "Riyadh_Airport_SDSC",
    "Rafha_Airport_SDSC", "Hada_El-Sham", "Najran_Airport_SDSC",
    # Bahrain
    "Bahrain", "Bahrain_2",
    # Oman
    "Muscat", "SQU_Oman", "University_of_Nizwa",
    # Turkey
    "IMS-METU-ERDEMLI", "Haymana_Ankara", "Ankara",
    "Ankara_TUBITAK", "TUBITAK_UZAY_Ankara",
    "Tuz_Golu", "Tuz_Golu_2", "Tuz_Golu_3",
    # Cyprus (incl. Akrotiri and Dhekelia)
    "Nicosia", "Nicosia_CY", "Chalhalbara", "Troodos_CAO",
    "AgiaMarina_Xyliatou", "Agia_Marina_Xyliatou", "Agia_Marina",
    "Pafos", "Pafos_CY", "Paphos", "Paphos_CY",
    "Larnaca", "Larnaca_CY", "CUT-TEPAK",
}

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for _ in range(HEADER_ROW_INDEX):
            next(fin)

        reader = csv.reader(fin)
        header = next(reader)

        print(f"Filtering for {len(MIDDLE_EAST_SITES)} known Middle East sites...")

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

            kept = 0
            total = 0
            found_sites = set()

            for row in reader:
                total += 1
                if total % 500_000 == 0:
                    print(f"  Processed {total:,} rows, kept {kept:,}...")
                if not row:
                    continue
                site = row[0].strip()
                if site in MIDDLE_EAST_SITES:
                    writer.writerow(row)
                    kept += 1
                    found_sites.add(site)

    print(f"\nDone! Kept {kept:,} rows out of {total:,} total.")
    print(f"Sites found in data: {sorted(found_sites)}")
    missing = MIDDLE_EAST_SITES - found_sites
    if missing:
        print(f"Sites in list but NOT in data: {sorted(missing)}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
