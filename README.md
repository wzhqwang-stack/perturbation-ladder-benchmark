# Baseline-disciplined stress-testing of perturbation-response engines

Code, run-level evidence and verification for the manuscript
*"Baseline-disciplined stress-testing of perturbation-response engines: a four-level generalization ladder from combinatorial CRISPR screens to a cross-system drug anchor"*.

The ladder changes one factor at a time — combinatorial hold-out within a study (T3), cross-cell-line transfer (T1), cross-study transfer (T2), a dose dimension, and cross-system/cross-modality transfer anchored on a real drug formulation (T4, Compound Kushen Injection) — and arms every level with permutation, control-passthrough and linear baselines, three seeds, and delta-centred diagnostics.

## Repository layout

```
scripts/            full pipeline (contracts, splits, level runs, aggregation, figures, T4 retrieval)
scripts/patches/    upstream compatibility patches (GEARS x scipy)
results/runs/       per-run evidence: metrics.json + per-condition CSVs for all runs (models/predictions excluded; regenerable per RUNBOOK)
results/            aggregated artifacts referenced by the manuscript (delta-Pearson appendix, retrieval null)
docs/               grid summary (BH-corrected significance table)
figures/            P1–P6 (PNG 350 dpi + PDF vector)
RUNBOOK.md          one-page reproduction guide (data download → runs → aggregation → figures)
verify_repo.py      numeric self-check: 19 assertions over the included artifacts (exit 0 = pass)
checksums.csv       SHA-256 fingerprints of every source data file, with origin URLs
requirements.txt    frozen environment (main); requirements_cpa.txt (CPA subsystem)
```

## Quickstart

```powershell
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python verify_repo.py            # 19/19 PASS, no GPU needed
```

Full reproduction (data download → five levels → aggregation → figures) follows `RUNBOOK.md`; raw data are public (figshare / GEO / HuggingFace CC0 / NCBI) and every source file is fingerprinted in `checksums.csv`.

## Data access

| Dataset | Source |
|---|---|
| norman / adamson / replogle-K562 / replogle-RPE1 (GEARS PertData) | figshare (GEARS distribution zips; see checksums.csv) |
| GO essential / gene2go | GEARS distribution |
| GSE124715, GSE125933 (bulk drug-response) | GEO |
| Tahoe-100M (CRC contract source) | HuggingFace `tahoebio/Tahoe-100M` (CC0) |
| NCBI gene_info | NCBI |

## Verification

`python verify_repo.py` re-derives every headline number in the manuscript from the included run-level artifacts: T3 GEARS on four datasets, T1 permutation/CPA, T2 GEARS/linear, control-split delta-Pearson, T4 linear baselines, the 14-query retrieval permutation null, and the BH-corrected grid counts.

## License

Code: MIT (see `LICENSE`). Data sources follow their original licenses (CC0 / public accessions).

## Citation

Manuscript under submission; citation block to be added upon acceptance.
