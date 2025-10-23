import os
from pathlib import Path
import unicodedata
import math
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


def _haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points."""
    try:
        # simple haversine
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c
    except Exception:
        return float('inf')


# Small reusable helpers used by multiple UI sections
def _first_text(row, cols, default=""):
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v and v.lower() not in {"nan", "none", "nan.0"}:
                return v
    return default


def friendly_location(row):
    for col in ("district", "location", "address"):
        if col in row and pd.notna(row[col]):
            v = str(row[col]).strip()
            if v and v.lower() not in {"nan", "none", "nan.0"}:
                return v
    return "Berlin"


def best_link(row):
    candidates = [
        "job_url_direct",
        "job_url",
        "link",
        "url",
        "company_url_direct",
        "company_url",
        "website",
        "registration",
        "appointment_url",
        "booking_url",
    ]
    for col in candidates:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if not val:
                continue
            if val.startswith("www."):
                val = "https://" + val
            if val.startswith("http://") or val.startswith("https://"):
                return val
    return None


# Safe rerun helper: some Streamlit builds expose different APIs
def safe_rerun():
    """Try to rerun the Streamlit script in a compatible way; fall back to st.stop()."""
    try:
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
                return
            except Exception:
                pass
        if hasattr(st, "rerun"):
            try:
                st.rerun()
                return
            except Exception:
                pass
    except Exception:
        # defensive: if something unexpected happens, fall through to stop
        pass
    # last-resort: stop execution and wait for user interaction
    st.stop()


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

    # --- Friendly greeting handling ---
    greetings = ("hi", "hello", "hey", "hallo", "greetings")
    if any(q.strip().startswith(g) for g in greetings):
        # Ask a friendly follow-up to engage the user
        st.chat_message("assistant").write(
            "Hey — what can I do for you? You can ask me to show jobs, events, or German courses."
        )
        # do not run the search below on a bare greeting
        # allow quick follow-up by returning early
        st.session_state.most_recent_query = q
        # Some Streamlit builds may not expose experimental_rerun; fall back to
        # stopping the script which preserves session_state for the next user
        # interaction.
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
            except Exception:
                st.stop()
        else:
            st.stop()

    topic = None
    if "job" in q:
        topic = "job"
    elif "event" in q:
        topic = "event"
    elif "course" in q or "german" in q:
        topic = "course"

    # remember last topic for UI state
    st.session_state['last_topic'] = topic or st.session_state.get('last_topic')

    loc_key = detect_district(q)
    # Follow-up prompts: if user asked for a topic (event/job/course), ask scope & display
    scope = None
    display_pref = None  # 'address' or 'map'
    if topic:
        # small dialog to ask how to scope results and what to show
        # preserve previous answers in session_state for smoother UX
        s_key = f"followup_scope_{topic}"
        d_key = f"followup_display_{topic}"
        if s_key not in st.session_state:
            st.session_state[s_key] = "all"
        if d_key not in st.session_state:
            st.session_state[d_key] = "map"

        # ensure extra per-topic state exists
        r_key = f"followup_radius_{topic}"
        u_key = f"followup_use_my_location_{topic}"
        lat_key = f"followup_my_lat_{topic}"
        lon_key = f"followup_my_lon_{topic}"
        if r_key not in st.session_state:
            st.session_state[r_key] = 3.0
        if u_key not in st.session_state:
            st.session_state[u_key] = False
        if lat_key not in st.session_state:
            st.session_state[lat_key] = 52.52
        if lon_key not in st.session_state:
            st.session_state[lon_key] = 13.405

        with st.form(key=f"followup_form_{q}", clear_on_submit=False):
            cols = st.columns([0.56, 0.44])
            with cols[0]:
                if loc_key:
                    st.write(f"Where should I search for {topic}s? (detected: {loc_key.title()})")
                    scope_choice = st.radio(
                        "Scope:",
                        [f"Only {loc_key.title()}", "All Berlin", f"Nearby {loc_key.title()} (within radius)"],
                    )
                else:
                    st.write(f"Where should I search for {topic}s?")
                    scope_choice = st.radio("Scope:", ["All Berlin", "Nearby current location (within radius)"])

                # radius control (km)
                radius_val = st.slider("Nearby radius (km)", min_value=0.5, max_value=20.0, value=float(st.session_state[r_key]), step=0.5)

                # option to use user's coordinates instead of district centroid
                use_my = st.checkbox("Use my location (enter coords below)", value=st.session_state[u_key])
                my_lat = None
                my_lon = None
                if use_my:
                    my_lat = st.number_input("My latitude", value=float(st.session_state[lat_key]), format="%.6f")
                    my_lon = st.number_input("My longitude", value=float(st.session_state[lon_key]), format="%.6f")
            with cols[1]:
                display_choice = st.radio("Show as:", ["Map pins", "Address list"], index=0)
            submitted = st.form_submit_button("Search")

        if submitted:
            # translate choices into internal flags
            if "Only" in scope_choice:
                scope = "only"
            elif "Nearby" in scope_choice:
                scope = "nearby"
            else:
                scope = "all"
            display_pref = "address" if display_choice.startswith("Address") else "map"
            # persist the choices per-topic
            st.session_state[s_key] = scope
            st.session_state[d_key] = display_pref
            st.session_state[r_key] = float(radius_val)
            st.session_state[u_key] = bool(use_my)
            if my_lat is not None and my_lon is not None:
                st.session_state[lat_key] = float(my_lat)
                st.session_state[lon_key] = float(my_lon)
        else:
            # use prior saved values if user didn't submit the form yet
            scope = st.session_state.get(s_key, "all")
            display_pref = st.session_state.get(d_key, "map")
    keywords = ["developer", "engineer", "data", "design", "marketing", "teacher", "python", "manager"]
    keyword = next((k for k in keywords if k in q), None)

    subset = df.copy()
    if topic:
        subset = subset[subset["type"] == topic]
    if loc_key:
        if scope == "only":
            subset = subset[
                subset["district"].fillna("").str.lower().str.contains(loc_key)
                | subset["location"].fillna("").str.lower().str.contains(loc_key)
            ]
        elif scope == "all":
            # no filtering, keep all Berlin results
            pass
        else:
            # nearby: compute rows within configured radius (km).
            try:
                # radius and optional user coords saved per-topic
                r_key = f"followup_radius_{topic}"
                u_key = f"followup_use_my_location_{topic}"
                lat_key = f"followup_my_lat_{topic}"
                lon_key = f"followup_my_lon_{topic}"
                radius_km = float(st.session_state.get(r_key, 3.0))

                if st.session_state.get(u_key, False):
                    lat0 = float(st.session_state.get(lat_key, DISTRICT_CENTROIDS.get(loc_key, DISTRICT_CENTROIDS["berlin"])[0]))
                    lon0 = float(st.session_state.get(lon_key, DISTRICT_CENTROIDS.get(loc_key, DISTRICT_CENTROIDS["berlin"])[1]))
                else:
                    lat0, lon0 = DISTRICT_CENTROIDS.get(loc_key, DISTRICT_CENTROIDS["berlin"])

                # ensure coords present
                subset = bake_coords(subset)
                mask = []
                for _, r in subset.iterrows():
                    try:
                        lat = float(r.get("latitude") or lat0)
                        lon = float(r.get("longitude") or lon0)
                        dkm = _haversine_km(lat0, lon0, lat, lon)
                        mask.append(dkm <= float(radius_km))
                    except Exception:
                        mask.append(False)
                subset = subset.loc[mask]
            except Exception:
                # fallback to substring matching
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

    # If the user asked for details about a course/job by name, try to find a match
    def _find_and_select(text, df):
        t = _normalize_text(text)
        if not t:
            return None
        # search title, course_name, provider, company
        for col in ("title", "course_name", "provider", "company"):
            if col in df.columns:
                for idx, val in df[col].dropna().items():
                    if t in _normalize_text(str(val)):
                        return idx
        return None

    # look for explicit detail requests like "who runs [course]" or if query contains a known title
    if any(word in q for word in ("who", "who runs", "details", "tell me about", "more info", "provider")):
        pick = _find_and_select(q, subset) or _find_and_select(q, df)
        if pick is not None:
            st.session_state.selected = int(pick)
            # show assistant message with the short details
            r = (subset if pick in subset.index else df).loc[pick]
            provider = r.get("provider") or r.get("company") or "Unknown"
            title_text = r.get("title") or r.get("course_name") or "Item"
            link = None
            for c in ("url", "link", "job_url_direct", "job_url", "registration", "appointment_url", "booking_url"):
                if c in r and pd.notna(r[c]):
                    link = str(r[c])
                    break
            msg = f"Here are the details for **{title_text}**:\n- Provider: {provider}"
            if link:
                msg += f"\n- Link: {link}"
            if pd.notna(r.get("location")):
                msg += f"\n- Location: {r.get('location')}"
            st.chat_message("assistant").write(msg)

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
        # include display preference in assistant message
        if display_pref == "address":
            show_txt = "addresses and details"
        else:
            show_txt = "map pins and details"
        st.chat_message("assistant").write(f"{emoji} Found **{len(subset)} {label}** in **{loc_label}**. Showing {show_txt}.")

        # --- Render a compact, human-friendly summary of the first few items ---
        def _render_summary(rows: pd.DataFrame, n=5):
            pieces = []
            for _, r in rows.head(n).iterrows():
                title = _first_text(r, ["title", "course_name", "provider"], default="No title")
                # find a link if present
                link = None
                for c in ("url", "link", "job_url_direct", "job_url", "registration", "appointment_url", "booking_url", "website"):
                    if c in r and pd.notna(r[c]):
                        l = str(r[c]).strip()
                        if l.startswith("www."):
                            l = "https://" + l
                        if l.startswith("http://") or l.startswith("https://"):
                            link = l
                            break
                when = None
                for c in ("when", "date", "start_time", "time"):
                    if c in r and pd.notna(r[c]):
                        when = str(r[c])
                        break
                location = friendly_location(r)
                addr_line = None
                # If user specifically asked for address list, show address field if available
                if display_pref == "address":
                    for ac in ("address", "location", "district"):
                        if ac in r and pd.notna(r[ac]):
                            addr_line = str(r[ac])
                            break
                line = f"**{title}**"
                if link:
                    line = f"[{title}]({link})"
                meta = []
                if when:
                    meta.append(f"When: {when}")
                meta.append(f"Location: {location}")
                if addr_line:
                    meta.append(f"Address: {addr_line}")
                pieces.append(line + "  \n" + " | ".join(meta))
            return "\n\n".join(pieces)

        summary_md = _render_summary(subset)
        if summary_md:
            st.chat_message("assistant").write(summary_md)


# =====================================================
# Left Column – Results List
# =====================================================
with col_list:
    results = st.session_state.results
    st.subheader(f"Results ({len(results)})")

    if results.empty:
        st.info("No results yet.")
    else:
        # Helper: choose a friendly location string (avoid 'nan' and empty values)
        def friendly_location(row):
            for col in ("district", "location", "address"):
                if col in row and pd.notna(row[col]):
                    v = str(row[col]).strip()
                    if v and v.lower() not in {"nan", "none", "nan.0"}:
                        return v
            return "Berlin"

        # Helper: find the best link for a row and normalize it for clickable markdown
        def best_link(row):
            # Common link columns across datasets
            candidates = [
                "job_url_direct",
                "job_url",
                "link",
                "url",
                "company_url_direct",
                "company_url",
                "website",
                "registration",
                "appointment_url",
                "booking_url",
            ]
            for col in candidates:
                if col in row and pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if not val:
                        continue
                    # If it's a bare domain like example.com, add scheme
                    if val.startswith("www."):
                        val = "https://" + val
                    if val.startswith("http://") or val.startswith("https://"):
                        return val
            return None

        def _first_text(row, cols, default=""):
            for c in cols:
                if c in row and pd.notna(row[c]):
                    v = str(row[c]).strip()
                    if v and v.lower() not in {"nan", "none", "nan.0"}:
                        return v
            return default

        # detect display preference from session
        display_pref_list = st.session_state.get(f"followup_display_{st.session_state.get('last_topic', '')}", None)
        # fallback if not set
        if not display_pref_list:
            # try any topic-specific stored value
            for k in st.session_state.keys():
                if k.startswith("followup_display_"):
                    display_pref_list = st.session_state[k]
                    break

        for _, row in results.head(25).iterrows():
            title = _first_text(row, ["title", "course_name", "provider"], default="No title")
            location = friendly_location(row)
            link = best_link(row)
            addr_line = None
            if display_pref_list == "address":
                for ac in ("address", "location", "district"):
                    if ac in row and pd.notna(row[ac]):
                        addr_line = str(row[ac])
                        break
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                if link:
                    st.markdown(f"**[{title}]({link})**  \n📍 {location}")
                else:
                    st.markdown(f"**{title}**  \n📍 {location}")
                if addr_line:
                    st.markdown(f"**Address:** {addr_line}")
            with cols[1]:
                key = f"details_{int(row.name)}"
                if st.button("Details", key=key):
                    st.session_state.selected = int(row.name)
                    safe_rerun()
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

    # Details panel: show full details for selected result
    selected = st.session_state.get("selected")
    with st.expander("Details", expanded=True):
        if selected is None:
            st.info("Select a row's Details button to see more information here.")
        else:
            if selected in results.index:
                r = results.loc[selected]
            elif selected in df.index:
                r = df.loc[selected]
            else:
                st.warning("Selected item is no longer available.")
                r = None
            if r is not None:
                # Render known fields cleanly
                def _fmt(k):
                    v = r.get(k)
                    if pd.isna(v):
                        return None
                    return str(v)

                st.markdown(f"### { _fmt('title') or _fmt('course_name') or 'Details' }")
                provider = _fmt('provider') or _fmt('company')
                if provider:
                    st.markdown(f"**Provider:** {provider}")
                if _fmt('location'):
                    st.markdown(f"**Location:** {_fmt('location')}")
                if _fmt('district'):
                    st.markdown(f"**District:** {_fmt('district')}")
                if _fmt('price'):
                    st.markdown(f"**Price:** {_fmt('price')}")
                if _fmt('duration'):
                    st.markdown(f"**Duration:** {_fmt('duration')}")
                if _fmt('level'):
                    st.markdown(f"**Level:** {_fmt('level')}")
                # show contact / link
                for c in ("url", "link", "job_url_direct", "job_url", "website", "registration", "appointment_url", "booking_url"):
                    if _fmt(c):
                        st.markdown(f"**Link:** [{_fmt(c)}]({_fmt(c)})")
                        break
                # show coordinates
                if _fmt('latitude') and _fmt('longitude'):
                    st.markdown(f"**Coordinates:** { _fmt('latitude') }, { _fmt('longitude') }")
                # show raw row for debug
                with st.expander("Raw data"):
                    st.json({k: (None if pd.isna(v) else v) for k, v in r.items()})

st.divider()
if st.button("🔄 Clear Chat"):
    st.session_state.results = df.copy()
    # clear any persisted followup_* keys so next chat starts fresh
    keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith("followup_")]
    for k in keys_to_remove:
        try:
            del st.session_state[k]
        except Exception:
            pass
    safe_rerun()
