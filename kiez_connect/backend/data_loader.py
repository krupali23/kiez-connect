# backend/data_loader.py
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

EVENTS_CSV  = os.path.join(DATA_DIR, "berlin_tech_events.csv")
JOBS_CSV    = os.path.join(DATA_DIR, "berlin_tech_jobs.csv")
COURSES_CSV = os.path.join(DATA_DIR, "german_courses_berlin.csv")  # optional/your new file


# ---------- core helpers ----------
def _smart_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)  # last resort


def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns and build a _blob column safely across pandas versions,
    even when there are duplicate column names (e.g., duplicated 'lat' headers).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "title","org","address","date","link","category","lat","lon","_blob"
        ])

    df = df.copy()

    # --- numeric coords: handle duplicates safely ---
    for c in ("lat", "lon", "latitude", "longitude"):
        if c in df.columns:
            try:
                s = df[c]
                # If duplicate header names exist, df[c] can be a DataFrame
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]          # take the first of the duplicates
                s = s.squeeze()               # Series if it's a single column
                df[c] = pd.to_numeric(s, errors="coerce")
            except Exception:
                # leave as-is if conversion fails
                pass

    # harmonize single lat/lon columns from latitude/longitude
    if "lat" not in df.columns and "latitude" in df.columns:
        df["lat"] = df["latitude"]
    if "lon" not in df.columns and "longitude" in df.columns:
        df["lon"] = df["longitude"]

    # --- ensure core text columns exist ---
    for col in ["title","org","address","date","link","category"]:
        if col not in df.columns:
            df[col] = ""

    # --- build search blob robustly (no Series.dtype access) ---
    try:
        dtypes_series = getattr(df, "dtypes", None)
        if dtypes_series is not None:
            obj_cols = [c for c, t in dtypes_series.items() if "object" in str(t)]
        else:
            obj_cols = list(df.columns)
    except Exception:
        obj_cols = list(df.columns)

    df["_blob"] = df[obj_cols].astype(str).apply(lambda s: " | ".join(s), axis=1).str.lower()
    return df



# ---------- loaders ----------
def load_events(path: str = EVENTS_CSV) -> pd.DataFrame:
    """
    Your events CSV typically has columns:
      Title | Link | Date & Time | Location
    """
    df = _smart_read(path)
    if df.empty: return df
    mapping = {
        "Title": "title",
        "Link": "link",
        "Date & Time": "date",
        "Location": "address",
        # allow lowercase variants too
        "title": "title",
        "link": "link",
        "date & time": "date",
        "location": "address",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return _normalize_common(df)


def load_jobs(path: str = JOBS_CSV) -> pd.DataFrame:
    """
    Jobs CSV usually has:
      title | company | location | date_posted | job_url(_direct) | job_type/company_industry
    """
    df = _smart_read(path)
    if df.empty: return df
    mapping = {}
    for src, dst in [
        ("title", "title"),
        ("company", "org"),
        ("location", "address"),
        ("date_posted", "date"),
        ("job_url_direct", "link"),
        ("job_url", "link"),
        ("job_type", "category"),
        ("company_industry", "category"),
    ]:
        if src in df.columns:
            mapping[src] = dst
    df = df.rename(columns=mapping)
    return _normalize_common(df)


def _pick_col(cols, *candidates):
    lc = [str(c).lower() for c in cols]
    for want in candidates:
        w = want.lower()
        for i, name in enumerate(lc):
            if w in name:
                return cols[i]
    return None


def load_courses(path: str = COURSES_CSV) -> pd.DataFrame:
    """
    Auto-detect common names for columns in your German courses file.
    """
    df = _smart_read(path)
    if df.empty: return df

    cols = list(df.columns)

    title   = _pick_col(cols, "course title","course","title","name","program")
    org     = _pick_col(cols, "provider","institution","institute","school","center","organisation","organization")
    address = _pick_col(cols, "address","location","campus","bezirk","district")
    date    = _pick_col(cols, "date","start","when","schedule","term","beginn","start date")
    link    = _pick_col(cols, "link","url","website","web","page")
    category= _pick_col(cols, "category","level","type")

    mapping = {}
    if title:    mapping[title]    = "title"
    if org:      mapping[org]      = "org"
    if address:  mapping[address]  = "address"
    if date:     mapping[date]     = "date"
    if link:     mapping[link]     = "link"
    if category: mapping[category] = "category"

    # fallback: if still no title, use first column
    if "title" not in mapping and cols:
        mapping[cols[0]] = "title"

    df = df.rename(columns=mapping)
    return _normalize_common(df)


def load_data() -> dict:
    return {
        "events":  load_events(),
        "jobs":    load_jobs(),
        "courses": load_courses(),  # may be empty if file not present
    }
