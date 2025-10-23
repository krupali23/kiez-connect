import os, re, json, hashlib
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
STORE_DIR = Path("store"); STORE_DIR.mkdir(exist_ok=True)

def chunk_text(text, size=800, overlap=100):
    text = re.sub(r"\s+", " ", str(text)).strip()
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def fingerprint(s): 
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:12]

def load_docs():
    docs = []
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in [".md", ".txt"]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            src = ""
            m = re.search(r"Source:\s*(\S+)", txt)
            if m: src = m.group(1)
            for ch in chunk_text(txt):
                docs.append({"text": ch, "title": p.stem, "url": src, "source": str(p)})
        elif p.suffix.lower() in [".csv", ".xlsx"]:
            try:
                df = pd.read_csv(p, dtype=str, encoding="utf-8").fillna("")
            except Exception:
                try:
                    df = pd.read_excel(p, dtype=str).fillna("")
                except Exception:
                    continue
            for _, r in df.iterrows():
                blob = " | ".join([str(x) for x in r.values if str(x)])
                for ch in chunk_text(blob, 500, 50):
                    docs.append({"text": ch, "title": p.stem, "url": "", "source": str(p)})
    return docs

def main():
    docs = load_docs()
    if not docs:
        print("⚠️  No files found in /data.")
        return
    print(f"Found {len(docs)} text chunks from {len(set(d['source'] for d in docs))} files")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)

    faiss.write_index(index, str(STORE_DIR / "index.faiss"))
    (STORE_DIR / "meta.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    print("✅ Index built and saved in /store")

if __name__ == "__main__":
    main()
