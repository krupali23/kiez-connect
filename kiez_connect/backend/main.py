# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .data_loader import load_data
from .chat import handle_chat_safe  # safe wrapper

# If Swagger /docs is blank on your network, serve ReDoc at /docs instead
app = FastAPI(title="Kiez Connect", docs_url=None, redoc_url="/docs")

# CORS (needed so the widget on 55xx can call 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

DATA = load_data()

class ChatIn(BaseModel):
    message: str

@app.get("/health")
def health():
    import pandas as pd
    def count(df):
        return 0 if df is None or not isinstance(df, pd.DataFrame) or df.empty else len(df)
    return {"ok": True, "events": count(DATA.get("events")),
            "jobs": count(DATA.get("jobs")), "courses": count(DATA.get("courses"))}

@app.post("/chat")
def chat_api(payload: ChatIn):
    try:
        return handle_chat_safe(payload.message, DATA)
    except Exception as e:
        return {
            "reply": "Sorry, something went wrong. Try: 'events in mitte', 'AI jobs', or 'german courses in neukölln'.",
            "results": [], "markers": [], "error": f"{type(e).__name__}"
        }
