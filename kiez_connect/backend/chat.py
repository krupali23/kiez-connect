# backend/chat.py
import re
import hashlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BERLIN_DISTRICTS = [
    "mitte","friedrichshain","kreuzberg","pankow",
    "charlottenburg","wilmersdorf","spandau","reinickendorf",
    "neukölln","treptow","köpenick","lichtenberg",
    "marzahn","hellersdorf","tempelhof","schöneberg",
    "steglitz","zehlendorf","tiergarten","prenzlauer berg",
    "moabit","wedding","dahlem","adlershof"
]
DIST_RE = re.compile("|".join([re.escape(d) for d in BERLIN_DISTRICTS]), re.I)

# Berlin bbox (fallback coords if no lat/lon in the row)
BB_MIN_LAT, BB_MAX_LAT = 52.40, 52.62
BB_MIN_LON, BB_MAX_LON = 13.20, 13.60

def _fallback_coords(text: str):
    if not text: text = "berlin"
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    a = int(h[:8], 16) / 0xFFFFFFFF
    b = int(h[8:16], 16) / 0xFFFFFFFF
    lat = BB_MIN_LAT + a * (BB_MAX_LAT - BB_MIN_LAT)
    lon = BB_MIN_LON + b * (BB_MAX_LON - BB_MIN_LON)
    return float(lat), float(lon)

def _ensure_standard_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        return pd.DataFrame(columns=["title","org","address","date","link","category","lat","lon","_blob"])
    df = df.copy()

    # if jobs/courses/events used different names, they should be normalized by data_loader already
    for c in ["title","org","address","date","link","category"]:
        if c not in df: df[c] = ""

    # lat/lon harmonization
    if "lat" not in df and "latitude" in df: df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "lon" not in df and "longitude" in df: df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    if "lat" in df: df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df: df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # search blob
    obj_cols = [c for c in df.columns if str(df[c].dtype).startswith("object")]
    df["_blob"] = df[obj_cols].astype(str).apply(lambda s: " | ".join(s), axis=1).str.lower()
    return df

def _vectorize(df: pd.DataFrame):
    if df.empty: return None, None
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform(df["_blob"])
    return vec, X

def _rank(df, vec, mat, query, top=200):
    if df.empty or vec is None: return df.head(0)
    qv = vec.transform([query.lower()])
    sim = cosine_similarity(qv, mat)[0]
    out = df.copy()
    out["_score"] = sim
    return out.sort_values("_score", ascending=False).head(top)

def _filter_by_district(df, query):
    dists = list({m.group(0).lower() for m in DIST_RE.finditer(query)})
    if not dists or df.empty: return df
    def has_dist(t): 
        t = str(t).lower()
        return any(d in t for d in dists)
    mask = df["address"].apply(has_dist) | df["_blob"].apply(has_dist)
    return df[mask] if mask.any() else df

def _build_markers(df, kind, limit=50):
    markers = []
    for _, r in df.head(limit).iterrows():
        title   = str(r.get("title",""))
        org     = str(r.get("org",""))
        address = str(r.get("address",""))
        link    = str(r.get("link",""))
        date    = str(r.get("date",""))
        lat     = r.get("lat", None)
        lon     = r.get("lon", None)
        if pd.isna(lat) or pd.isna(lon) or lat is None or lon is None:
            lat, lon = _fallback_coords(address or title or org)
        markers.append({
            "type": kind, "title": title, "org": org, "address": address,
            "date": date, "link": link, "lat": float(lat), "lon": float(lon)
        })
    return markers

def _choose_table(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["course","courses","sprach","language","german","schule","class"]):
        return "courses"
    if any(w in q for w in ["job","jobs","hiring","role","position","apply","career"]):
        return "jobs"
    return "events"
    
  # --- put this near your other functions in chat.py ---

def handle_chat_safe(message: str, DATA: dict) -> dict:
    """
    Never raise. If smart search fails, return a simple 'safe mode' response:
    just the first N rows from the chosen table with markers.
    """
    try:
        return handle_chat(message, DATA)  # your normal smart handler
    except Exception as e:
        # Minimal, robust fallback
        try:
            intent = _choose_table(message or "")
            raw = DATA.get(intent)
            df = _ensure_standard_cols(raw)
            if df.empty:
                return {"reply": f"Sorry, no {intent} data.", "results": [], "markers": [], "intent": intent}

            # take first N rows for demo
            keep_cols = [c for c in ["title","org","address","date","link","category"] if c in df.columns]
            sample = df[keep_cols].fillna("").head(10)
            results = sample.to_dict(orient="records")
            markers = _build_markers(df.head(10), intent, limit=10)

            return {
                "reply": f"(Safe mode) Showing first {len(results)} {intent}.",
                "results": results,
                "markers": markers,
                "intent": intent,
                "error": f"{type(e).__name__}"  # optional, helps you debug later
            }
        except Exception as e2:
            # absolute last resort: echo
            return {
                "reply": "Sorry, something went wrong. Try: 'events in mitte', 'AI jobs', or 'german courses in neukölln'.",
                "results": [],
                "markers": [],
                "intent": "unknown",
                "error": f"{type(e).__name__}/{type(e2).__name__}"
            }
  

def handle_chat(message: str, DATA: dict) -> dict:
    """
    Decide which table to use (events/jobs/courses), rank rows for the query,
    and return {reply, results, markers, intent}.
    """
    q = (message or "").strip()
    if not q:
        return {"reply": "Ask me about tech events, jobs, or German courses in Berlin (e.g. 'events in Mitte', 'AI jobs', 'German courses in Neukölln')."}

    intent = _choose_table(q)

    raw = DATA.get(intent)
    df  = _ensure_standard_cols(raw if isinstance(raw, pd.DataFrame) else pd.DataFrame())
    if df.empty:
        return {"reply": f"No {intent} data loaded.", "results": [], "markers": [], "intent": intent}

    vec, mat = _vectorize(df)
    ranked = _rank(df, vec, mat, q, top=200)
    ranked = _filter_by_district(ranked, q)

    keep_cols = [c for c in ["title","org","address","date","link","category"] if c in ranked.columns]
    results = ranked[keep_cols].fillna("").head(50).to_dict(orient="records")
    markers = _build_markers(ranked, intent, limit=50)

    top = results[0]["title"] if results else ""
    reply = f"Found {len(results)} {intent}." + (f" Top: {top}" if top else "")
    return {"reply": reply, "results": results, "markers": markers, "intent": intent}
