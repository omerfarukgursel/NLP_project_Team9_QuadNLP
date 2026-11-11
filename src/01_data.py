"""Step 1: build pairs.csv from the PAN 2025 multi-author dataset (v4).
Data: https://doi.org/10.5281/zenodo.20502645 (CC BY 4.0). Unzip so that
ROOT/{easy,medium,hard}/{train,validation,test}/problem-*.txt exist."""
import os, json
import pandas as pd

ROOT = "/kaggle/working/data"
OUT = "/kaggle/working/pairs.csv"
rows, bad, fixed = [], [], []

def read_sents(path, n_labels):
    raw = open(path, encoding="utf-8").read().split("\n")
    sents = [s.strip() for s in raw if s.strip()]
    if len(sents) == n_labels + 1:
        return sents, False
    # repair: a line that starts with whitespace continues the previous sentence
    merged = []
    for line in raw:
        if not line.strip():
            continue
        if line[0].isspace() and merged:
            merged[-1] += " " + line.strip()
        else:
            merged.append(line.strip())
    return merged, True

for level in ["easy", "medium", "hard"]:
    for split in ["train", "validation", "test"]:
        d = f"{ROOT}/{level}/{split}"
        files = [f for f in os.listdir(d) if f.startswith("problem-") and f.endswith(".txt")]
        for f in files:
            pid = f[len("problem-"):-len(".txt")]
            truth = json.load(open(f"{d}/truth-problem-{pid}.json", encoding="utf-8"))
            ch = truth["changes"]
            sents, was_fixed = read_sents(f"{d}/{f}", len(ch))
            # manual fix: one sentence split over two lines in this document
            if (level, split, pid) == ("hard", "train", "3207") and len(sents) == 14:
                sents = sents[:10] + [sents[10] + " " + sents[11]] + sents[12:]
                was_fixed = True
            if len(ch) != len(sents) - 1:
                bad.append((level, split, pid, len(sents), len(ch)))
                continue
            if was_fixed:
                fixed.append((level, split, pid))
            for i in range(len(ch)):
                prev = sents[i - 1] if i > 0 else ""
                nxt = sents[i + 2] if i + 2 < len(sents) else ""
                rows.append({
                    "level": level, "split": split, "doc": int(pid),
                    "pos": i, "n_authors": truth["authors"], "label": ch[i],
                    "pair_a": sents[i], "pair_b": sents[i + 1],
                    "ctx_a": f"{prev} [CTX] {sents[i]}".strip(),
                    "ctx_b": f"{sents[i + 1]} [CTX] {nxt}".strip(),
                })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("repaired documents:", fixed)
print("still mismatched:", bad)
print(df.groupby(["level", "split"]).agg(pairs=("label", "size"), change_rate=("label", "mean")).round(3))
# expected: hard train 51,061 pairs, hard test 11,077 pairs
