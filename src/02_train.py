"""Step 2: fine-tune DeBERTa-v3-base. Examples:
  python 02_train.py --mode pair --levels hard --seeds 1 2 3 --save_model
  python 02_train.py --mode context --levels hard --seeds 1 2 3 --save_model
  python 02_train.py --mode pair --levels easy medium --seeds 1 2 3
Writes results.csv, preds/val_*.csv and preds/test_*.csv. Resumes: finished runs are skipped."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # keep this line first
import gc, time, shutil, json, numpy as np, pandas as pd, torch, random
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

# Settings are identical for every run: lr 2e-5, 3 epochs, batch 16, max_len 256, fp16, class-weighted CE
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="pair", choices=["pair", "context"])
ap.add_argument("--levels", nargs="+", default=["hard"])
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--save_model", action="store_true", help="keep best checkpoint in models/ (needed by 03_test.py)")
a = ap.parse_args()
MODE, LEVELS, SEEDS, EPOCHS = a.mode, a.levels, a.seeds, a.epochs
CSV, R, P = "/kaggle/working/pairs.csv", "/kaggle/working/results.csv", "/kaggle/working/preds"
assert torch.cuda.device_count() == 1, "more than one GPU visible; restart the session"
os.makedirs(P, exist_ok=True)
df_all = pd.read_csv(CSV, keep_default_na=False)
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
tok.add_special_tokens({"additional_special_tokens": ["[CTX]"]})
colA, colB = ("pair_a", "pair_b") if MODE == "pair" else ("ctx_a", "ctx_b")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def enc(batch):
    return tok(batch[colA], batch[colB], truncation=True, max_length=256)

def make(d):
    ds = Dataset.from_pandas(d[[colA, colB, "label"]])
    return ds.map(enc, batched=True, remove_columns=[colA, colB])

def macro_from(cnt):
    tp, fp, fn, tn = [cnt[..., i] for i in range(4)]
    return (2*tp/(2*tp+fp+fn+1e-12) + 2*tn/(2*tn+fn+fp+1e-12)) / 2

def summarize(level, pdir=P):
    fs, cs = [], []
    for s in SEEDS:
        p = pd.read_csv(f"{pdir}/test_{level}_{MODE}_s{s}.csv")
        y, pred = p.label.values, p.pred.values
        docs, inv = np.unique(p.doc.values, return_inverse=True)
        c = np.zeros((len(docs), 4))
        for k, m in enumerate([(pred == 1) & (y == 1), (pred == 1) & (y == 0),
                               (pred == 0) & (y == 1), (pred == 0) & (y == 0)]):
            c[:, k] = np.bincount(inv, weights=m.astype(float), minlength=len(docs))
        fs.append(f1_score(y, pred, average="macro")); cs.append(c)
    D = len(docs)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(1000)])
    boot = np.mean([macro_from(W @ c) for c in cs], axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(level=level, mode=MODE, seeds=[round(f, 4) for f in fs], mean=round(float(np.mean(fs)), 4),
                std=round(float(np.std(fs, ddof=1)), 4), ci95=[round(float(lo), 4), round(float(hi), 4)])

def run(level, seed):
    tag = f"{level}_{MODE}_s{seed}"
    t0 = time.time(); set_seed(seed)
    d = df_all[df_all.level == level]
    tr = d[d.split == "train"].reset_index(drop=True)
    va = d[d.split == "validation"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    print(f"\n=== {tag} | train {len(tr)} val {len(va)} test {len(te)} | GPUs {torch.cuda.device_count()} ===")
    ds_tr, ds_va, ds_te = make(tr), make(va), make(te)
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=2, dtype=torch.float32)
    model.resize_token_embeddings(len(tok))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[tok.convert_tokens_to_ids("[CTX]")] = emb[tok.sep_token_id].clone()
    cnt = np.bincount(tr.label.values, minlength=2)
    w = torch.tensor(len(tr) / (2 * cnt), dtype=torch.float)
    print("class weights:", w)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    args = TrainingArguments(
        output_dir=f"/kaggle/working/out_{tag}",
        learning_rate=2e-5, num_train_epochs=EPOCHS,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        weight_decay=0.0, warmup_steps=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        save_total_limit=1, save_only_model=True,
        fp16=True, seed=seed, report_to="none", logging_steps=200)
    trainer = WTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                       processing_class=tok, compute_metrics=metrics)
    trainer.train()
    row = {"level": level, "mode": MODE, "seed": seed, "smoke": False, "epochs": EPOCHS,
           "train_pairs": len(tr), "val_pairs": len(va), "steps": trainer.state.global_step,
           "gpus": torch.cuda.device_count(), "trunc_rate": 0.0}
    for name, ds, part in [("val", ds_va, va), ("test", ds_te, te)]:
        logits = trainer.predict(ds).predictions
        o = part[["level", "doc", "pos", "n_authors", "label"]].copy()
        o["prob1"] = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy(); o["pred"] = logits.argmax(-1)
        o.to_csv(f"{P}/{name}_{tag}.csv", index=False)
        row[f"{name}_macro_f1"] = round(f1_score(o.label, o.pred, average="macro"), 4)
    row["val_f1_per_epoch"] = "/".join(str(round(h["eval_macro_f1"], 4))
                                       for h in trainer.state.log_history if "eval_macro_f1" in h)
    row["minutes"] = round((time.time() - t0) / 60, 1)
    res = pd.concat([pd.read_csv(R), pd.DataFrame([row])]) if os.path.exists(R) else pd.DataFrame([row])
    res.to_csv(R, index=False)
    print(row)
    if a.save_model:
        trainer.save_model(f"/kaggle/working/models/{tag}"); tok.save_pretrained(f"/kaggle/working/models/{tag}")
    shutil.rmtree(args.output_dir, ignore_errors=True)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()

for level in LEVELS:
    for seed in SEEDS:
        done = os.path.exists(R) and ((pd.read_csv(R).level == level) & (pd.read_csv(R).mode == MODE)
                                      & (pd.read_csv(R).seed == seed) & (~pd.read_csv(R).smoke.astype(bool))).any()
        if done and os.path.exists(f"{P}/test_{level}_{MODE}_s{seed}.csv"):
            print("skipped (already done):", level, MODE, seed); continue
        run(level, seed)
    s = summarize(level)
    print("\n>>> SUMMARY", json.dumps(s))
    json.dump(s, open(f"/kaggle/working/test_results_{level}_{MODE}.json", "w"))
print(pd.read_csv(R).tail(10).to_string())
