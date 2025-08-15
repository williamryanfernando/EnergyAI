"""
Usage:
  python data_build.py \
    --inputs /path/a.json /path/b.jsonl \
    --out /path/training_dataset.json \
    --min_chars 3 --max_chars 8000 --dedupe

It auto-detects JSON vs JSONL and maps common keys like
("prompt","completion") -> ("instruction","output").
"""

import argparse, json, sys, random
from pathlib import Path

KEY_MAPS = [
    ("instruction", "output"),
    ("prompt", "completion"),
    ("question", "answer"),
    ("input", "target"),
]

def iter_records(path: Path):
    """Yield dicts from JSON or JSONL; be permissive about shape."""
    text = path.read_text(encoding="utf-8").strip()
    items = []
    if text.startswith("{") or text.startswith("["):
        # JSON
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        items = data
    else:
        # JSONL
        items = [json.loads(line) for line in text.splitlines() if line.strip()]

    for obj in items:
        if not isinstance(obj, dict):
            continue
        yield obj

def normalize(record: dict):
    """Return {"instruction","output"} if possible, else None."""

    if "instruction" in record and "output" in record:
        return {"instruction": str(record["instruction"]).strip(),
                "output": str(record["output"]).strip()}

    for a, b in KEY_MAPS:
        if a in record and b in record:
            return {"instruction": str(record[a]).strip(),
                    "output": str(record[b]).strip()}

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=False, default=[], help="Input JSON/JSONL files")
    ap.add_argument("--glob", default="", help="Glob to add (e.g., '/mnt/data/*.jsonl')")
    ap.add_argument("--out", default="training_dataset.json", help="Output JSON file path")
    ap.add_argument("--min_chars", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=8000)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle output records")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    if args.glob:
        inputs += list(Path(".").glob(args.glob))

    # If user didn't pass inputs, try common files in /mnt/data
    if not inputs:
        candidates = []
        for ext in ("*.json", "*.jsonl"):
            candidates += list(Path("/mnt/data").glob(ext))
        inputs = candidates

    if not inputs:
        print("No input files found. Pass --inputs or put files in /mnt/data.", file=sys.stderr)
        sys.exit(2)

    print(f"Scanning {len(inputs)} file(s):")
    for p in inputs:
        print(" -", p)

    records = []
    for p in inputs:
        try:
            for rec in iter_records(p):
                norm = normalize(rec)
                if not norm:
                    continue
                ok = args.min_chars <= len(norm["instruction"]) <= args.max_chars \
                     and args.min_chars <= len(norm["output"]) <= args.max_chars
                if ok:
                    records.append(norm)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not records:
        print("No valid records found after normalization/filters.", file=sys.stderr)
        sys.exit(3)

    if args.dedupe:
        seen = set()
        deduped = []
        for r in records:
            key = (r["instruction"], r["output"])
            if key in seen: 
                continue
            seen.add(key)
            deduped.append(r)
        records = deduped

    if args.shuffle:
        random.seed(42)
        random.shuffle(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} examples to {out_path}")

if __name__ == "__main__":
    main()
