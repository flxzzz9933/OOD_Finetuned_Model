import argparse, json, re
from pathlib import Path
from typing import List
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_NOTES = ROOT / "data/raw/notes"
RAW_CODE  = ROOT / "data/raw/code"
INTERIM   = ROOT / "data/interim"
INTERIM.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_MIN = 300
WINDOW    = 800  # fixed-size window; safe and predictable

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="ignore")

def discover_files(include_repo_notes: bool, only_glob: str|None, max_files: int|None) -> List[Path]:
    cand = [p for p in RAW_NOTES.rglob("*") if p.suffix.lower() in {".txt", ".md"}]
    if include_repo_notes and RAW_CODE.exists():
        cand += [p for p in RAW_CODE.rglob("notes*.txt")]
    if only_glob:
        pat = only_glob.lower()
        cand = [p for p in cand if pat in p.name.lower()]
    seen, out = set(), []
    for p in cand:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); out.append(p)
    if max_files is not None:
        out = out[:max_files]
    return out

def split_fixed_windows(text: str, window=WINDOW, min_len=CHUNK_MIN) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text)
    chunks = [text[i:i+window] for i in range(0, len(text), window)]
    return [c for c in chunks if len(c) >= min_len]

def tag_topic(t: str):
    kw = {
        "mvc": ["mvc", "controller", "view", "observer"],
        "strategy": ["strategy", "strategy pattern"],
        "factory": ["factory", "factory method", "abstract factory"],
        "observer": ["observer", "notify", "subscriber", "listener"],
        "decorator": ["decorator", "wrap", "component"],
        "adapter": ["adapter", "adaptee", "target interface"],
        "testing": ["junit", "unit test", "mock", "fixture"],
        "gui": ["swing", "gui", "panel", "frame", "key listener"],
        "command": ["command pattern", "execute", "undo", "macro"],
    }
    tags, low = [], t.lower()
    for tag, words in kw.items():
        if any(w in low for w in words):
            tags.append(tag)
    return tags

def embed(texts: List[str], batch: int):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_NAME)
    dim = m.get_sentence_embedding_dimension()
    X = np.zeros((len(texts), dim), dtype="float32")
    for i in range(0, len(texts), batch):
        j = min(len(texts), i+batch)
        X[i:j] = m.encode(texts[i:j], batch_size=min(64, batch), normalize_embeddings=True).astype("float32")
        print(f"  embedded {j}/{len(texts)}")
    return X

def main():
    ap = argparse.ArgumentParser(description="SAFE build of context index (no tqdm, fixed-window chunking).")
    ap.add_argument("--include-repo-notes", action="store_true")
    ap.add_argument("--only-glob", type=str, default=None, help="Substring to filter filenames")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of files (debug)")
    ap.add_argument("--no-embed", action="store_true", help="Skip embeddings for speed")
    ap.add_argument("--embed-batch", type=int, default=512)
    args = ap.parse_args()

    print("==> Discovering files")
    files = discover_files(args.include_repo_notes, args.only_glob, args.max_files)
    print(f"Found {len(files)} file(s).")
    if not files:
        print("No inputs. Put .txt/.md in data/raw/notes/.")
        return

    rows, texts = [], []
    print("==> Chunking (fixed windows)")
    for idx, p in enumerate(files, 1):
        rel = str(p.relative_to(ROOT))
        t = read_text(p)
        print(f"[{idx}/{len(files)}] {rel}  ({len(t):,} chars)")
        chunks = split_fixed_windows(t)
        print(f"    -> {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            rows.append({
                "chunk_id": f"{rel}::chunk{i:04d}",
                "source": rel,
                "text": c,
                "topics": tag_topic(c)
            })
            texts.append(c)

    out_chunks = INTERIM / "context_chunks.jsonl"
    with open(out_chunks, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved chunks -> {out_chunks} (total {len(rows)})")

    if args.no_embed:
        print("==> Skipping embeddings (dry run). Done.")
        return

    print("==> Embedding")
    X = embed(texts, args.embed_batch)
    out_embs = INTERIM / "context_embeddings.npz"
    out_meta = INTERIM / "context_meta.json"
    np.savez_compressed(out_embs, X=X)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "dim": int(X.shape[1]), "count": int(X.shape[0])}, f)
    print("✅ Done.")
    print(f"  - {out_chunks}")
    print(f"  - {out_embs}")
    print(f"  - {out_meta}")

if __name__ == "__main__":
    main()
