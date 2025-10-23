"""
Add approximate latitude & longitude for Berlin districts + sub-districts
to your existing CSVs (jobs, events, courses).
"""

import pandas as pd
from pathlib import Path

# --- Folder containing your data files ---
DATA_DIR = Path(__file__).resolve().parent
FILES = [
    "berlin_tech_jobs.csv",
    "berlin_tech_events.csv",
    "german_courses_berlin.csv",
]

# --- Approximate coordinates for Berlin boroughs & neighborhoods ---
BERLIN_DISTRICTS = {
    # Central areas
    "Mitte": (52.5200, 13.4050),
    "Moabit": (52.5260, 13.3400),
    "Tiergarten": (52.5140, 13.3500),
    "Wedding": (52.5473, 13.3559),
    "Hansaviertel": (52.5186, 13.3477),

    # Friedrichshain–Kreuzberg
    "Friedrichshain": (52.5150, 13.4540),
    "Kreuzberg": (52.4964, 13.4184),

    # Pankow borough
    "Prenzlauer Berg": (52.5395, 13.4243),
    "Pankow": (52.5680, 13.4010),
    "Weißensee": (52.5550, 13.4650),

    # Charlottenburg–Wilmersdorf
    "Charlottenburg": (52.5050, 13.3030),
    "Wilmersdorf": (52.4800, 13.3150),
    "Grunewald": (52.4830, 13.2560),

    # Neukölln
    "Neukölln": (52.4800, 13.4380),
    "Britz": (52.4483, 13.4387),
    "Rixdorf": (52.4711, 13.4409),

    # Tempelhof–Schöneberg
    "Schöneberg": (52.4838, 13.3500),
    "Tempelhof": (52.4700, 13.3850),
    "Mariendorf": (52.4325, 13.3870),
    "Lichtenrade": (52.3986, 13.4079),

    # Treptow–Köpenick
    "Treptow": (52.4850, 13.4690),
    "Adlershof": (52.4330, 13.5330),
    "Köpenick": (52.4436, 13.5784),

    # Lichtenberg
    "Lichtenberg": (52.5150, 13.4990),
    "Hohenschönhausen": (52.5561, 13.5014),

    # Marzahn–Hellersdorf
    "Marzahn": (52.5440, 13.5600),
    "Hellersdorf": (52.5380, 13.6040),

    # Reinickendorf
    "Reinickendorf": (52.5910, 13.3240),
    "Tegel": (52.5760, 13.2910),

    # Spandau
    "Spandau": (52.5380, 13.2010),
    "Staaken": (52.5300, 13.1500),

    # Steglitz–Zehlendorf
    "Steglitz": (52.4560, 13.3210),
    "Zehlendorf": (52.4330, 13.2570),
    "Dahlem": (52.4550, 13.2900),
}

def find_district(text: str):
    """Try to match a Berlin district/subdistrict in the text."""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for dist in BERLIN_DISTRICTS:
        if dist.lower() in text_lower:
            return dist
    if "berlin" in text_lower:
        return "Mitte"  # default to central Berlin
    return None


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add district + lat/lon columns."""
    df = df.copy()
    if "location" not in df.columns:
        print("⚠️ No 'location' column found, skipping.")
        return df

    df["district"] = df["location"].apply(find_district)
    df["latitude"] = df["district"].map(lambda d: BERLIN_DISTRICTS.get(d, (None, None))[0])
    df["longitude"] = df["district"].map(lambda d: BERLIN_DISTRICTS.get(d, (None, None))[1])

    added = df["latitude"].notna().sum()
    print(f"✅ Added coordinates for {added} rows ({len(df)} total)")
    return df


if __name__ == "__main__":
    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"❌ File not found: {path}")
            continue

        print(f"\n📂 Processing {path.name}...")
        df = pd.read_csv(path)
        df_new = add_coordinates(df)

        # Optional: Backup before overwrite
        backup = path.with_suffix(".backup.csv")
        path.rename(backup)
        df_new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved updated file: {path.name}  (Backup: {backup.name})")

    print("\n🎉 All done! Each file now includes district, latitude, and longitude columns.")
