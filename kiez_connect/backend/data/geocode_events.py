import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

DATA_DIR = os.path.dirname(__file__)
path = os.path.join(DATA_DIR, "berlin_tech_events.csv")

print(f"📂 Loading: {path}")
df = pd.read_csv(path)
print("Initial columns:", df.columns.tolist())

geolocator = Nominatim(user_agent="kiez_connect_event_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

latitudes, longitudes = [], []

for i, addr in enumerate(df["Location"].fillna("").astype(str)):
    if not addr.strip():
        latitudes.append(None)
        longitudes.append(None)
        continue
    query = f"{addr}, Berlin, Germany"
    try:
        loc = geocode(query)
        if loc:
            latitudes.append(loc.latitude)
            longitudes.append(loc.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)
    except Exception:
        latitudes.append(None)
        longitudes.append(None)
    if i % 10 == 0:
        print(f"  ...processed {i} rows")

df["latitude"] = latitudes
df["longitude"] = longitudes
out_path = path.replace(".csv", "_geo.csv")
df.to_csv(out_path, index=False)
print(f"✅ Saved geocoded file to {out_path}")
