import os, re, json, ujson, glob
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW_CODE = ROOT / "data/raw/code"
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

EMB_META = json.load(open(INTERIM / "context_meta.json", "r", encoding="utf-8"))
CHUNKS_PATH = INTERIM / "context_chunks.jsonl"
EMBS = np.load(INTERIM / "context_embeddings.npz")["X"]  # (N, D), normalized

# For code embedding (for retrieval), a small model works fine
CODE_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Heuristics for Java method extraction
METHOD_SIG = re.compile(r'''
    (?P<mod>(public|private|protected)\s+)?      # visibility
    (?P<static>static\s+)?                       # static
    (?P<ret>[A-Za-z_<>\[\]?]+)\s+               # return type
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*         # method name
    \((?P<params>[^)]*)\)\s*                    # params
    (\{)                                        # opening brace
''', re.VERBOSE)

# Give the model explicit tags it will see during training/inference
SYS_TAG = "<System>"
CTX_B = "<Context>"
CTX_E = "</Context>"
PFX_B = "<CodePrefix>"
PFX_E = "</CodePrefix>"
SFX_B = "<CodeSuffix>"
SFX_E = "</CodeSuffix>"
TGT_B = "<Target>"
TGT_E = "</Target>"

def iter_java_files(folder: Path):
    for p in folder.rglob("*.java"):
        yield p

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def strip_comments(code: str) -> str:
    # Keep names/structure but remove heavy comments for cleaner embedding (not for training)
    code_ = re.sub(r"/\*.*?\*/", "", code, flags=re.S)      # block comments
    code_ = re.sub(r"//.*?$", "", code_, flags=re.M)        # line comments
    return code_

def find_methods(code: str) -> List[Tuple[int,int,re.Match]]:
    """Return (start_idx_of_sig, end_idx_of_block, sig_match) for each method."""
    out = []
    for m in METHOD_SIG.finditer(code):
        start = m.start()
        # Find matching brace for the method body
        # We know there's a "{" at m.group(0) end; find its pair with a stack
        open_idx = code.find("{", m.end() - 1, m.end() + 1)
        if open_idx == -1:
            continue
        depth, i = 1, open_idx + 1
        while i < len(code) and depth > 0:
            if code[i] == "{": depth += 1
            elif code[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            out.append((start, i, m))
    return out

def select_mask_spans(code: str) -> List[Tuple[int,int]]:
    """
    Choose semantic spans to mask:
    - Prefer full method bodies (keep signature in prefix).
    - Also pick blocks named for design roles (notify/update/create/execute).
    """
    spans = []
    methods = find_methods(code)
    for (s, e, m) in methods:
        sig = code[s:e]
        name = m.group("name").lower()
        # prioritize design-relevant methods
        if any(k in name for k in ["notify", "update", "create", "execute", "handle", "dispatch", "render", "subscribe"]):
            spans.append((s, e))
        # sample others sparsely
    # fallback: take 1–2 medium methods if none matched
    if not spans and methods:
        spans = [methods[0][:2]]
        if len(methods) > 2:
            spans.append(methods[-1][:2])
    # Filter spans that are too tiny/huge
    spans = [(s, e) for (s, e) in spans if 60 <= (e - s) <= 4000]
    return spans[:2]  # at most 2 spans per file for MVP

def embed_texts(texts: List[str], model) -> np.ndarray:
    X = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return X.astype("float32")

def retrieve_context_for_code(code_text: str, topk=4) -> List[Dict]:
    model = SentenceTransformer(CODE_EMB_MODEL)
    code_emb = model.encode([strip_comments(code_text)], normalize_embeddings=True)
    sims = (code_emb @ EMBS.T)[0]  # cosine since both normalized
    top_idx = np.argsort(-sims)[:topk]

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [ujson.loads(lines[i]) for i in top_idx]

def build_example(code: str, span: Tuple[int,int], ctx_rows: List[Dict], src_path: str) -> Dict:
    s, e = span
    prefix = code[:s].rstrip()
    target = code[s:e].strip()
    suffix = code[e:].lstrip()

    ctx_text = "\n---\n".join([f'[{r["chunk_id"]}] {r["text"]}' for r in ctx_rows])

    system = f"""{SYS_TAG}
You are a CS3500 TA. Follow MVC & design-pattern conventions. Use only the provided context if relevant. Cite slide ids when appropriate in comments like // cites: [chunk_id].
{SYS_TAG.replace('<','</')}
""".strip()

    formatted = {
        "system": system,
        "context": f"{CTX_B}\n{ctx_text}\n{CTX_E}",
        "code_prefix": f"{PFX_B}\n{prefix}\n{PFX_E}",
        "code_suffix": f"{SFX_B}\n{suffix}\n{SFX_E}",
        "target":   f"{TGT_B}\n{target}\n{TGT_E}",
        "meta": {
            "source": src_path,
            "mask_span": [s, e],
            "topics": list({t for r in ctx_rows for t in r.get("topics", [])})
        }
    }
    return formatted

def main():
    out_path = PROCESSED / "fim_dataset.jsonl"
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for f in tqdm(list(iter_java_files(RAW_CODE)), desc="Masking Java"):
            code = read(f)
            spans = select_mask_spans(code)
            if not spans:
                continue
            # retrieve once per file (ok); you can also retrieve per-span if you want
            ctx_rows = retrieve_context_for_code(code, topk=4)
            for sp in spans:
                ex = build_example(code, sp, ctx_rows, str(f.relative_to(ROOT)))
                out.write(ujson.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Wrote {n_written} FIM examples to {out_path}")

if __name__ == "__main__":
    main()
