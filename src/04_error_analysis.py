# Step 4: pair-only error analysis on VALIDATION predictions (seeds 1-3); does not touch the test set
import numpy as np, pandas as pd
va = pd.read_csv("/kaggle/working/pairs.csv", keep_default_na=False)
va = va[(va.level == "hard") & (va.split == "validation")]
va = va.assign(n_pairs=va.groupby("doc")["pos"].transform("size"))
nw = lambda s: len(str(s).split())
rows = []
for s in [1, 2, 3]:
    p = pd.read_csv(f"/kaggle/working/preds/val_hard_pair_s{s}.csv")
    m = p.merge(va[["doc", "pos", "pair_a", "pair_b", "n_pairs"]], on=["doc", "pos"])
    m["seed"] = s; rows.append(m)
a = pd.concat(rows)
err = a[a.pred != a.label].copy()
print("hata sayısı (seed başına):", err.groupby("seed").size().to_dict(), "| toplam çift/seed:", len(a)//3)

lab = a.set_index(["seed", "doc", "pos"]).label
def nb(r):   # komşu sınırlardan biri değişim mi?
    return any(lab.get((r.seed, r.doc, r.pos+d), 0) == 1 for d in (-1, 1))
err["short"] = (err.pair_a.map(nw) <= 5) | (err.pair_b.map(nw) <= 5)
err["near_change"] = err.apply(nb, axis=1)
err["edge"] = (err.pos <= 1) | (err.pos >= err.n_pairs - 2)
err["type"] = np.where(err.label == 1, "FN: kaçırılan değişim", "FP: sahte değişim")

e1 = err[err.seed == 1]
print("\n== seed 1 kategori sayıları ==")
for name, g in [("tür", e1.type), ("yazar sayısı", e1.n_authors.astype(str) + " yazar"),
                ("kısa cümle (<=5 kelime)", np.where(e1.short, "kısa", "uzun")),
                ("başka sınıra komşu", np.where(e1.near_change, "komşu değişim var", "yalnız")),
                ("belge kenarı (ilk/son 2)", np.where(e1.edge, "kenar", "orta"))]:
    print(f"\n{name}:\n", pd.Series(g).value_counts().to_string())

# Her seed'de aynı çiftte hata: tutarlı (sistematik) hatalar
key = err.groupby(["doc", "pos"]).seed.nunique()
print("\n3 seed'in 3'ünde de yanlış:", int((key == 3).sum()), "| en az 1'inde:", len(key))

pd.set_option("display.max_colwidth", 160)
print("\n== 36 örnek hata (seed 1): kategori başına ~6 ==")
ex = []
for cond, nm in [((e1.type.str.startswith("FN")) & e1.short, "FN kısa"),
                 ((e1.type.str.startswith("FN")) & ~e1.short, "FN uzun"),
                 ((e1.type.str.startswith("FP")) & e1.short, "FP kısa"),
                 ((e1.type.str.startswith("FP")) & ~e1.short, "FP uzun"),
                 (e1.near_change, "komşu değişim"), (e1.edge, "belge kenarı")]:
    s = e1[cond].sample(min(6, int(cond.sum())), random_state=0)
    for _, r in s.iterrows():
        print(f"[{nm}] doc {r.doc} pos {r.pos} yazar={r.n_authors} etiket={r.label} tahmin={r.pred}\n   A: {r.pair_a[:150]}\n   B: {r.pair_b[:150]}")
