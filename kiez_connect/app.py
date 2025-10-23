import os
from pathlib import Path
import unicodedata
import pandas as pd
import streamlit as st
import pydeck as pdk

# =====================================================
# Robust data directory resolver (works everywhere)
# =====================================================
def resolve_data_dir() -> Path:
    """
    Find backend/data folder safely.
    Works in Streamlit, Python, and Jupyter (no __file__ issues).
    """
    # 1) Environment variable override (expand user, accept file or dir)
    env = os.environ.get("KC_DATA_DIR")
    if env:
        p = Path(os.path.expanduser(env))
        if p.exists():
            # If user pointed to a file, return its parent
            return p if p.is_dir() else p.parent

    # 2) Base dir depends on whether __file__ exists
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path(os.getcwd())

    # 3) Check possible locations
    candidates = [
        base / "backend" / "data",
        base / "data",
        Path.cwd() / "backend" / "data",
        Path(r"C:\Users\krupa\Desktop\Bootcamp\project_keiz_connect\kiez_connect\backend\data"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return base / "backend" / "data"

DATA_DIR = resolve_data_dir()

# =====================================================
# Berlin District Coordinates (preloaded fallback)
# =====================================================
DISTRICT_CENTROIDS = {
    "mitte": (52.5200, 13.4050),
    "kreuzberg": (52.4986, 13.4030),
    "neukölln": (52.4751, 13.4386),
    "friedrichshain": (52.5156, 13.4549),
    "charlottenburg": (52.5070, 13.3040),
    "wilmersdorf": (52.4895, 13.3157),
    "schöneberg": (52.4832, 13.3477),
    "tempelhof": (52.4675, 13.4036),
    "pankow": (52.5693, 13.4010),
    "prenzlauer berg": (52.5380, 13.4247),
    "spandau": (52.5511, 13.1999),
    "steglitz": (52.4560, 13.3220),
    "treptow": (52.4816, 13.4764),
    "köpenick": (52.4429, 13.5756),
    "marzahn": (52.5450, 13.5690),
    "hellersdorf": (52.5345, 13.6132),
    "reinickendorf": (52.5870, 13.3260),
    "moabit": (52.5303, 13.3390),
    "wedding": (52.5496, 13.3551),
    "berlin": (52.5200, 13.4050),
}
# Precompute a normalized mapping to make detection robust to punctuation/umlauts/hyphens
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Normalize unicode (e.g., ü -> u), lowercase, remove punctuation and extra spaces
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # Replace common separators with spaces and strip punctuation
    for ch in "-_\\/.,":
        s = s.replace(ch, " ")
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(part for part in s.split() if part)

# Map normalized key -> original key
_NORMALIZED_DISTRICT_MAP = { _normalize_text(k): k for k in DISTRICT_CENTROIDS.keys() }
DISTRICT_KEYS = sorted(_NORMALIZED_DISTRICT_MAP.keys(), key=len, reverse=True)

# =====================================================
# Helpers
# =====================================================
def detect_district(text: str):
    """Detect a district name in free text. Returns the canonical district key from DISTRICT_CENTROIDS.
    Uses normalized matching to handle umlauts, hyphens, and punctuation.
    """
    if not isinstance(text, str):
        return None
    norm = _normalize_text(text)
    for nd in DISTRICT_KEYS:
        if nd and nd in norm:
            # return the canonical original key
            return _NORMALIZED_DISTRICT_MAP.get(nd)
    if "berlin" in norm:
        return "berlin"
    return None


def bake_coords(df_in: pd.DataFrame) -> pd.DataFrame:
    """Ensure lat/lon exist, using district approximations for missing ones."""
    df = df_in.copy()
    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA
    if "district" not in df.columns:
        df["district"] = pd.NA

    for i in df.index:
        if pd.isna(df.at[i, "latitude"]) or pd.isna(df.at[i, "longitude"]):
            d = (
                detect_district(str(df.at[i, "district"]))
                or detect_district(str(df.at[i, "location"]))
                or "berlin"
            )
            lat, lon = DISTRICT_CENTROIDS.get(d, DISTRICT_CENTROIDS["berlin"])
            df.at[i, "latitude"] = lat
            df.at[i, "longitude"] = lon
            if pd.isna(df.at[i, "district"]) or not str(df.at[i, "district"]).strip():
                df.at[i, "district"] = d.title()
    return df


# -------------------------------------------------
# Geocoding helpers (use cache, optional live geocoding)
# -------------------------------------------------
def _load_geocode_cache(data_dir: Path):
    p = data_dir / "geocode_cache.parquet"
    mapping = {}
    if not p.exists():
        return mapping
    try:
        df = pd.read_parquet(p)
        # try to find address/lat/lon columns
        cols = {c.lower(): c for c in df.columns}
        addr = cols.get("query") or cols.get("address") or cols.get("location")
        lat = cols.get("lat") or cols.get("latitude")
        lon = cols.get("lon") or cols.get("longitude")
        if addr and lat and lon:
            for _, r in df[[addr, lat, lon]].iterrows():
                k = _normalize_text(str(r[addr]))
                try:
                    mapping[k] = (float(r[lat]), float(r[lon]))
                except Exception:
                    continue
    except Exception:
        # can't read cache — ignore
        pass
    return mapping


def _save_geocode_cache(data_dir: Path, mapping: dict):
    # Save mapping to parquet as columns query, latitude, longitude
    try:
        import pyarrow as pa  # ensure dependency
        import pyarrow.parquet as pq
        rows = []
        for q, (lat, lon) in mapping.items():
            rows.append({"query": q, "latitude": lat, "longitude": lon})
        if rows:
            df = pd.DataFrame(rows)
            out = data_dir / "geocode_cache.parquet"
            df.to_parquet(out, index=False)
    except Exception:
        # writing cache is best-effort
        pass


def _geocode_with_cache(addr: str, data_dir: Path, cache: dict):
    k = _normalize_text(addr)
    if not k:
        return None
    if k in cache:
        return cache[k]
    # try live geocoding (best-effort)
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
    except Exception:
        return None
    try:
        geolocator = Nominatim(user_agent="kiez_connect_runtime_geocoder")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        q = f"{addr}, Berlin, Germany"
        loc = geocode(q)
        if loc:
            val = (float(loc.latitude), float(loc.longitude))
            cache[k] = val
            _save_geocode_cache(data_dir, cache)
            return val
    except Exception:
        return None
    return None


def ensure_geocoded(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    # Try to fill latitude/longitude for rows that have an address-like column
    # but are missing coordinates. This uses an on-disk cache and optional
    # live geocoding; it's best-effort and rate-limited.
    cache = _load_geocode_cache(data_dir)
    changed = False
    for i in df.index:
        if pd.notna(df.at[i, "latitude"]) and pd.notna(df.at[i, "longitude"]):
            continue
        # pick the best textual candidate
        candidate = None
        for col in ("location", "address", "district", "title"):
            if col in df.columns and pd.notna(df.at[i, col]):
                candidate = str(df.at[i, col])
                if candidate.strip():
                    break
        if not candidate:
            continue
        coords = _geocode_with_cache(candidate, data_dir, cache)
        if coords:
            lat, lon = coords
            df.at[i, "latitude"] = lat
            df.at[i, "longitude"] = lon
            changed = True
    return df


# =====================================================
# Load data safely
# =====================================================
@st.cache_data
def load_data(data_dir: Path):
    def _smart_read(path: Path) -> pd.DataFrame:
        # Try utf-8, then fallback to latin-1; preserving original behavior while
        # being tolerant to encoding differences.
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        # last resort
        return pd.read_csv(path)

    try:
        # Prefer geocoded variants if they exist (e.g. berlin_tech_events_geo.csv)
        ev_base = data_dir / "berlin_tech_events.csv"
        ev_geo = data_dir / (ev_base.stem + "_geo.csv")
        events = _smart_read(ev_geo if ev_geo.exists() else ev_base)

        jb_base = data_dir / "berlin_tech_jobs.csv"
        jb_geo = data_dir / (jb_base.stem + "_geo.csv")
        jobs = _smart_read(jb_geo if jb_geo.exists() else jb_base)

        courses = _smart_read(data_dir / "german_courses_berlin.csv")
    except Exception as e:
        st.error(f"❌ Could not load CSV files from {data_dir}\n\nError: {e}")
        return pd.DataFrame()

    for df, t in [(jobs, "job"), (events, "event"), (courses, "course")]:
        df.columns = [c.strip().lower() for c in df.columns]
        df["type"] = t

    merged = pd.concat([jobs, events, courses], ignore_index=True)
    # Normalize column names (already done per-file below), then try to enrich
    # missing coordinates from a geocode cache if available, then fall back to
    # district centroids.
    merged = bake_coords(merged)

    # If any rows are still missing latitude/longitude, try to enrich from a
    # parquet cache that maps address/query -> coords (optional).
    try:
        cache_path = data_dir / "geocode_cache.parquet"
        if cache_path.exists():
            try:
                cache_df = pd.read_parquet(cache_path)
                # detect likely column names in cache
                cache_cols = {c.lower(): c for c in cache_df.columns}
                addr_col = None
                for name in ("query", "address", "location", "addr", "query_string"):
                    if name in cache_cols:
                        addr_col = cache_cols[name]
                        break
                lat_col = None
                lon_col = None
                for name in ("lat", "latitude"):
                    if name in cache_cols:
                        lat_col = cache_cols[name]
                        break
                for name in ("lon", "longitude"):
                    if name in cache_cols:
                        lon_col = cache_cols[name]
                        break

                if addr_col and lat_col and lon_col:
                    # build mapping from normalized address -> (lat, lon)
                    mapping = {}
                    for _, r in cache_df[[addr_col, lat_col, lon_col]].iterrows():
                        k = _normalize_text(str(r[addr_col]))
                        try:
                            mapping[k] = (float(r[lat_col]), float(r[lon_col]))
                        except Exception:
                            continue

                    # fill missing coords from mapping using location/address/district
                    def _lookup_coord(row):
                        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                            return row["latitude"], row["longitude"]
                        candidates = []
                        for col in ("location", "address", "district", "title"):
                            if col in row and pd.notna(row[col]):
                                candidates.append(str(row[col]))
                        for c in candidates:
                            k = _normalize_text(c)
                            if k in mapping:
                                return mapping[k]
                        return (row.get("latitude"), row.get("longitude"))

                    # apply lookup for rows missing lat/lon
                    mask = merged["latitude"].isna() | merged["longitude"].isna()
                    if mask.any():
                        for i in merged[mask].index:
                            lat, lon = _lookup_coord(merged.loc[i])
                            if pd.notna(lat) and pd.notna(lon):
                                merged.at[i, "latitude"] = lat
                                merged.at[i, "longitude"] = lon
            except Exception:
                # reading parquet may fail if pyarrow isn't installed — ignore
                pass
    except Exception:
        pass
    return merged


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="Kiez Connect", page_icon="💬", layout="wide")
st.title("💬 Kiez Connect – Your Berlin Chat Assistant")
st.caption("Ask about jobs, events, or German courses — instant results and map pins across Berlin.")

df = load_data(DATA_DIR)

if df.empty:
    st.warning("⚠️ No data loaded. Check your CSV files in backend/data.")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results = df.copy()

col_list, col_map = st.columns([0.45, 0.55])

# =====================================================
# Chat Input
# =====================================================
query = st.chat_input("Example: 'show jobs in Mitte' or 'find events in Kreuzberg'")

if query:
    st.chat_message("user").write(query)
    q = query.lower()

    topic = None
    if "job" in q:
        topic = "job"
    elif "event" in q:
        topic = "event"
    elif "course" in q or "german" in q:
        topic = "course"

    loc_key = detect_district(q)
    keywords = ["developer", "engineer", "data", "design", "marketing", "teacher", "python", "manager"]
    keyword = next((k for k in keywords if k in q), None)

    subset = df.copy()
    if topic:
        subset = subset[subset["type"] == topic]
    if loc_key:
        subset = subset[
            subset["district"].fillna("").str.lower().str.contains(loc_key)
            | subset["location"].fillna("").str.lower().str.contains(loc_key)
        ]
    if keyword:
        search_cols = [c for c in ["title", "company", "provider", "course_name"] if c in subset.columns]
        subset = subset[
            subset[search_cols].apply(lambda x: x.astype(str).str.lower().str.contains(keyword).any(), axis=1)
        ]

    subset = bake_coords(subset)
    st.session_state.results = subset

    if subset.empty:
        st.chat_message("assistant").write("❌ No matches found. Try another keyword or district.")
    else:
        emoji = {"job": "💼", "event": "🎉", "course": "🎓"}.get(topic, "📍")
        # Build a safe, user-friendly label for the result type
        if topic:
            label = f"{topic}s" if not topic.endswith('s') else topic
        else:
            label = "results"
        loc_label = (loc_key or "Berlin").title()
        st.chat_message("assistant").write(f"{emoji} Found **{len(subset)} {label}** in **{loc_label}**.")


# =====================================================
# Left Column – Results List
# =====================================================
with col_list:
    results = st.session_state.results
    st.subheader(f"Results ({len(results)})")

    if results.empty:
        st.info("No results yet.")
    else:
        for _, row in results.head(25).iterrows():
            title = row.get("title", "No title")
            location = row.get("district") or row.get("location") or "Berlin"
            link = (
                row.get("job_url_direct")
                or row.get("job_url")
                or row.get("link")
                or row.get("url")
            )
            if isinstance(link, str) and link.startswith("http"):
                st.markdown(f"**[{title}]({link})**  \n📍 {location}")
            else:
                st.markdown(f"**{title}**  \n📍 {location}")
            st.markdown("---")


# =====================================================
# Right Column – Map View
# =====================================================
with col_map:
    st.subheader("📍 Map View")

    # ✅ Defensive fix: make sure coordinates exist before filtering
    results = st.session_state.results.copy()

    # Create latitude/longitude columns if missing
    if "latitude" not in results.columns or "longitude" not in results.columns:
        results = bake_coords(results)

    # Fill coordinates if they’re still empty
    results["latitude"] = results["latitude"].fillna(52.5200)
    results["longitude"] = results["longitude"].fillna(13.4050)

    # Now drop rows without valid coordinates
    points = results.dropna(subset=["latitude", "longitude"])

    if points.empty:
        st.info("No map points available yet.")
    else:
        def color_for_type(t):
            t = (t or "").lower()
            return {
                "job": [0, 122, 255],
                "event": [0, 170, 90],
                "course": [255, 140, 0],
            }.get(t, [128, 128, 128])

        points["color"] = points["type"].apply(color_for_type)
        mid_lat = float(points["latitude"].mean())
        mid_lon = float(points["longitude"].mean())

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=80,
            pickable=True,
        )

        tooltip = {"html": "<b>{title}</b><br/>{district}<br/>{location}"}
        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

st.divider()
if st.button("🔄 Clear Chat"):
    st.session_state.results = df.copy()
    st.rerun()
