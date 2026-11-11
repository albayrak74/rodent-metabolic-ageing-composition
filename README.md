# Senescence, metabolism and transcriptomic age in ageing mouse and rat metabolic tissues

Code, fitted models and result tables for:

Alkan I, Albayrak M. *Bulk senescence signatures track immune composition and do not precede metabolic decline: a machine-learning analysis of ageing mouse and rat tissues.* (submitted)

## Data
All input data are public (GEO): GSE314970 (rat atlas), GSE314817 (mouse atlas), GSE226118 / GSE226117 (gastrocnemius), GSE132040 (Tabula Muris Senis bulk), GSE53960 (rat body map). Gene annotation and orthologues: NCBI Gene (`gene_info`, `gene_orthologs`). Hallmark gene sets via Enrichr; SenMayo mouse set from MSigDB (`SAUL_SEN_MAYO`, in `scripts/senmayo_mouse.txt`).

## Pipeline (`scripts/`)
| Step | Script | Purpose |
|---|---|---|
| 1 | `01_prep.py` | atlas counts → log2 CPM, rat→mouse 1:1 orthologue mapping |
| 2 | `04_prep_extra.py` | gastrocnemius series and external cohorts |
| 3 | `00_qc.py` | sample QC (r < 0.90 to tissue median profile) |
| 4 | `05_composition_leadlag.py` | programme/composition scores, composition adjustment, T50 lead–lag |
| 5 | `06_age_models.py` | elastic net / XGBoost / MLP clocks, repeated 5-fold CV |
| 6 | `07_transfer_validation.py` | final clocks, cross-species transfer, external validation, SHAP |
| 7 | `08_clock_composition.py` | composition-adjusted clocks, age acceleration |
| 8 | `11_sensitivity.py` | immune-only adjustment, FDR, *Cdkn1a* |
| 9 | `09_figures_1_3.py`, `10_figure4_table1.py` | figures and Table 1 |

`02_scores.py` and `03_t50.py` are the exploratory go/no-go analyses (change-point model) described in the Methods.

Paths in the scripts point to a local working directory; edit `D` / `W` at the top of each script.

## Fitted clocks (`models/`)
`clock_<species>_<tissue>.json`: elastic net clocks trained on all atlas samples of each tissue (rat: male Sprague Dawley; mouse: male and female C57BL/6J; 6–27 months). Apply with:

```
python scripts/apply_clock.py models/clock_rat_liver.json my_counts.tsv
```

Genes are z-scored within the dataset you score, so predictions are relative to that dataset. Rank correlation with age is the robust metric; report cell composition next to predicted age, because clock outputs and age acceleration depend on it.

## Results (`results/`)
All per-sample scores and every table behind the figures and supplementary eTables.

## Licence
Code: MIT. Models and result tables: CC BY 4.0.
