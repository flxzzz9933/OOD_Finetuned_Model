import json, ujson
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/fim_dataset.jsonl"

class FIMDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(ujson.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def join_example(ex: Dict) -> str:
    # Build a single training string per example for causal models
    return "\n".join([
        ex["system"],
        ex["context"],
        ex["code_prefix"],
        ex["code_suffix"],
        ex["target"]  # target appears after; your trainer should compute loss only on this span
    ])

def collate(batch: List[Dict]) -> Dict:
    texts = [join_example(ex) for ex in batch]
    return {"texts": texts, "metas": [ex["meta"] for ex in batch]}

if __name__ == "__main__":
    ds = FIMDataset(DATA)
    print("Examples:", len(ds))
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    for b in dl:
        print(b["texts"][0][:400], "\n---")
        break
