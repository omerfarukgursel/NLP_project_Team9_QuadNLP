"""Step 3: evaluate saved models on the frozen test set (once per model) + paired bootstrap over documents.
Needs models/ from 02_train.py --save_model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, pandas as pd, torch, json
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CSV, LEVEL, SEEDS = "/kaggle/working/pairs.csv", "hard", [1, 2, 3]
MODES = ["pair"]   # baseline only for the mid-project report; use ["pair", "context"] for the final report
df = pd.read_csv(CSV, keep_default_na=False)
te = df[(df.level == LEVEL) & (df.split == "test")].reset_index(drop=True)
print("test pairs:", len(te), "| documents:", te.doc.nunique())   # expected 11077 / 900
os.makedirs("/kaggle/working/preds", exist_ok=True)

@torch.no_grad()
def predict(path, colA, colB):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).cuda().eval()
    out = []
    for i in range(0, len(te), 64):
        b = te.iloc[i:i+64]
        enc = tok(list(b[colA]), list(b[colB]), truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(out).numpy()

cols = {"pair": ("pair_a", "pair_b"), "context": ("ctx_a", "ctx_b")}
y = te.label.values
docs, inv = np.unique(te.doc.values, return_inverse=True)
D = len(docs)
counts = {}          # (mode, seed) -> [D,4] (tp, fp, fn, tn) per doc
f1s = {m: [] for m in MODES}
for mode in MODES:
    for s in SEEDS:
        tag = f"{LEVEL}_{mode}_s{s}"
        logits = predict(f"/kaggle/working/models/{tag}", *cols[mode])
        pred = logits.argmax(-1)
        o = te[["level", "doc", "pos", "n_authors", "label"]].copy()
        o["prob1"] = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy(); o["pred"] = pred
        o.to_csv(f"/kaggle/working/preds/test_{tag}.csv", index=False)
        f = f1_score(y, pred, average="macro"); f1s[mode].append(f)
        print(tag, "test macro-F1 =", round(f, 4))
        c = np.zeros((D, 4))
        for k, m in enumerate([(pred == 1) & (y == 1), (pred == 1) & (y == 0),
                               (pred == 0) & (y == 1), (pred == 0) & (y == 0)]):
            c[:, k] = np.bincount(inv, weights=m.astype(float), minlength=D)
        counts[(mode, s)] = c

def macro_from(cnt):                      # cnt: [..., 4] = tp, fp, fn, tn
    tp, fp, fn, tn = [cnt[..., i] for i in range(4)]
    return (2*tp/(2*tp+fp+fn+1e-12) + 2*tn/(2*tn+fn+fp+1e-12)) / 2

rng = np.random.default_rng(0)
B = 1000
W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(B)])   # [B, D]
boot = {m: np.mean([macro_from(W @ counts[(m, s)]) for s in SEEDS], axis=0) for m in MODES}
res = {}
for m in MODES:
    a = np.array(f1s[m])
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    res[m] = dict(seeds=[round(x, 4) for x in a], mean=round(a.mean(), 4),
                  std=round(a.std(ddof=1), 4), ci95=[round(lo, 4), round(hi, 4)])
if len(MODES) == 2:
    gain = boot["context"] - boot["pair"]
    g = np.mean(f1s["context"]) - np.mean(f1s["pair"])
    glo, ghi = np.percentile(gain, [2.5, 97.5])
    res["gain_context_minus_pair"] = dict(mean=round(g, 4), ci95=[round(glo, 4), round(ghi, 4)],
        real=bool(g >= 0.01 and glo > 0))
print(json.dumps(res, indent=1))
json.dump(res, open("/kaggle/working/test_results.json", "w"), indent=1)
import transformers, sklearn, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| sklearn", sklearn.__version__, "| pandas", pd.__version__)
