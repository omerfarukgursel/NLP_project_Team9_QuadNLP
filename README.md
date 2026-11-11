# Does context help find where the author changes?

Team 9 (QuadNLP), NLP course project. Controlled study on the PAN 2025 Multi-Author Writing Style Analysis **hard** level:
does DeBERTa-v3-base that also sees the neighbouring sentences beat the same model that sees only the sentence pair (macro-F1)?

- Pair-only input: `s_i [SEP] s_{i+1}`
- Context input: `s_{i-1} [CTX] s_i [SEP] s_{i+1} [CTX] s_{i+2}`
- Everything else is identical (lr 2e-5, 3 epochs, batch 16, max 256 tokens, class-weighted CE, fp16, best epoch by dev macro-F1, seeds 1-3).

Data: PAN 2025 multi-author dataset v4, https://doi.org/10.5281/zenodo.20502645 (CC BY 4.0). Not included here; unzip to `/kaggle/working/data/{easy,medium,hard}/{train,validation,test}`.

## Reproduce (Kaggle, one T4 GPU, Python 3.13)
```
python src/01_data.py                                                        # builds pairs.csv (hard test: 11,077 pairs)
python src/02_train.py --mode pair --levels hard --seeds 1 2 3 --save_model  # ~27 min per seed
python src/03_test.py                                                        # test macro-F1, mean +- std, 95% bootstrap CI
python src/04_error_analysis.py                                              # error counts on validation predictions
```
Context runs: `python src/02_train.py --mode context --levels hard --seeds 1 2 3 --save_model` (~43 min per seed).
Easy and medium (pair-only, ~30 min per seed): `python src/02_train.py --mode pair --levels easy medium --seeds 1 2 3`.

## Results so far (test macro-F1, mean +- std over seeds 1-3, 95% paired bootstrap over test documents)
| System | Easy | Medium | Hard | Hard dev |
|---|---|---|---|---|
| Pair-only DeBERTa-v3-base | 0.960 +- 0.003 [0.955, 0.964] | 0.836 +- 0.005 [0.826, 0.846] | 0.846 +- 0.004 [0.838, 0.854] | 0.843 +- 0.005 |
| Context DeBERTa-v3-base | not run | not run | pending (final report) | 0.816 +- 0.005 |
| Lin et al. (2025), published | 0.958 | 0.823 | 0.830 | - |

Our runs select the checkpoint on dev; small differences to the published scores are not a claim of a better method.

`results/results_all.csv` has one row per run (all levels); per-pair predictions are in `preds/` if uploaded.

## Environment
Python 3.13.16, torch 2.11.0, transformers 5.20.2, scikit-learn 1.6.1, pandas 2.3.3.
