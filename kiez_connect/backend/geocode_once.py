import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- robust base path (works in script *and* notebook) ----------
try:
    BASE_DIR = os.path.dirname(__file__)  # when run as a script/module
except NameError:
    # when run in a REPL/notebook: try CWD/backend, else CWD
    cwd = os.getcwd()
    BASE_DIR = os.path.join(cwd, "backend") if os.path.basename(cwd).lower() != "backend" else cwd

DATA_DIR = os.path.join(BASE_DIR, "data")
EV = os.path.join(DATA_DIR, "berlin_tech_events.csv")
JB = os.path.join(DATA_DIR, "berlin_tech_jobs.csv")

print("Using BASE_DIR:", BASE_DIR)
print("Events CSV:", EV)
print("Jobs   CSV:", JB)

def smart_read(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-16"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def geocode_file(path: str, address_candidates):
    if not os.path.exists(path):
        print("File not found:", path)
        return
    df = smart_read(path)

    # pick first address-like column that exists
    addr_col = None
    for c in address_candidates:
        if c in df.columns:
            addr_col = c
            break
    if addr_col is None:
        print("No address column among", address_candidates, "in", path)
        return

    # set up geocoder (Nominatim needs a user_agent)
    geolocator = Nominatim(user_agent="kiez_connect_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    lats, lons = [], []
    for i, addr in enumerate(df[addr_col].fillna("").astype(str).tolist(), start=1):
        if not addr.strip():
            lats.append(None); lons.append(None)
            continue
        q = f"{addr}, Berlin, Germany"
        try:
            loc = geocode(q)
            if loc:
                lats.append(loc.latitude); lons.append(loc.longitude)
            else:
                lats.append(None); lons.append(None)
        except Exception:
            lats.append(None); lons.append(None)
        if i % 10 == 0:
            print(f"  ...geocoded {i} rows")

    df["lat"] = lats
    df["lon"] = lons
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved with lat/lon → {path}")

def main():
    # events: your columns looked like Title | Link | Date & Time | Location
    geocode_file(EV, ["address", "Location"])
    # jobs: your columns: ... | company | location | date_posted | ...
    geocode_file(JB, ["address", "location", "Location"])

if __name__ == "__main__":
    main()
