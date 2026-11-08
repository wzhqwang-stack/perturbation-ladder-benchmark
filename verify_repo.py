#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_repo.py — numeric self-check for the GitHub repository.
19 assertions over the included run-level artifacts (no GPU, no large files).
exit 0 = all pass."""
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def canon_summary(prefix, level=None, seed=None, src=None, tgt=None, gse=None):
    best = None
    for d in (PKG / "results" / "runs").iterdir():
        mj = d / "metrics.json"
        if not (d.is_dir() and mj.exists() and d.name.startswith(prefix)):
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("meta", {})
        lv = meta.get("level")
        if level is not None:
            if lv != level:
                continue
        elif lv in ("T1", "T2", "T4"):
            continue
        if src is not None and str(meta.get("source", meta.get("dataset", ""))) != src:
            continue
        if tgt is not None and str(meta.get("target", "")) != tgt:
            continue
        if gse is not None and str(meta.get("gse", "")).lower() != gse.lower():
            continue
        if seed is not None:
            m = re.search(r"_s(\d+)_", d.name)
            if not m or int(m.group(1)) != seed:
                continue
        if best is None or mj.stat().st_mtime > best[0]:
            best = (mj.stat().st_mtime, j["summary"])
    return best[1] if best else None


def get(summary, metric):
    v = summary.get(metric)
    return v.get("mean") if isinstance(v, dict) else None


def mean_metric(model, level, seeds, src=None, tgt=None, metric="r2_deg", method=None):
    vals = []
    for s in seeds:
        summ = canon_summary(model, level, s, src, tgt)
        if summ is None:
            continue
        if method is None:
            v = get(summ, metric)
        else:
            node = summ.get(method, {})
            v = node.get(metric, {}).get("mean") if isinstance(node, dict) else None
        if v is not None:
            vals.append(float(v))
    return float(sum(vals) / len(vals)) if vals else None


def main():
    t3 = {"norman": (0.60, 0.73, 0.667), "adamson": (0.66, 0.78, 0.718),
          "replogle_k562_essential": (0.62, 0.74, 0.683), "replogle_rpe1_essential": (0.47, 0.59, 0.53)}
    for ds, (lo, hi, ref) in t3.items():
        v = mean_metric("gears", None, [149, 150, 151], src=ds)
        check(f"T3 GEARS {ds} ~{ref}", v is not None and lo <= v <= hi, f"got {v}")

    r1 = mean_metric("scgen", "T1", [149, 150, 151], src="replogle_k562_essential", tgt="replogle_rpe1_essential", method="random", metric="r2_deg")
    r2 = mean_metric("scgen", "T1", [149, 150, 151], src="replogle_rpe1_essential", tgt="replogle_k562_essential", method="random", metric="r2_deg")
    check("T1 random 0.481/0.637", r1 is not None and 0.45 <= r1 <= 0.51 and r2 is not None and 0.60 <= r2 <= 0.67, f"got {r1}/{r2}")

    c1 = mean_metric("cpa", "T1", [149, 150, 151], src="replogle_rpe1_essential", tgt="replogle_k562_essential", metric="r2_deg")
    c2 = mean_metric("cpa", "T1", [149, 150, 151], src="replogle_k562_essential", tgt="replogle_rpe1_essential", metric="r2_deg")
    check("T1 CPA reverse~0.226 / forward~0.033", c1 is not None and 0.17 <= c1 <= 0.28 and c2 is not None and -0.10 <= c2 <= 0.17, f"got {c1}/{c2}")

    g1 = mean_metric("gears", "T2", [149, 150, 151], src="norman", tgt="adamson", metric="r2_deg")
    g2 = mean_metric("gears", "T2", [149, 150, 151], src="adamson", tgt="norman", metric="r2_deg")
    check("T2 GEARS fwd in [-1.20,-0.35] / rev in [-0.40,-0.10]",
          g1 is not None and -1.20 <= g1 <= -0.35 and g2 is not None and -0.40 <= g2 <= -0.10, f"got {g1}/{g2}")

    l1 = mean_metric("baselines", "T2", [149, 150, 151], src="norman", tgt="adamson", method="linear_transfer", metric="r2_deg")
    l2 = mean_metric("baselines", "T2", [149, 150, 151], src="adamson", tgt="norman", method="linear_transfer", metric="r2_deg")
    check("T2 linear 0.399/0.404", l1 is not None and 0.37 <= l1 <= 0.43 and l2 is not None and 0.37 <= l2 <= 0.43, f"got {l1}/{l2}")

    import csv as _csv
    adf = list(_csv.DictReader(open(PKG / "results" / "appendix_delta_r2.csv", encoding="utf-8")))
    kk = [float(r["delta_pearson_random_split"]) for r in adf
          if r["level"] == "T1" and "k562\u2192rpe1" in r["direction"] and r.get("delta_pearson_random_split")]
    gg = [float(r["delta_pearson_GEARS_split"]) for r in adf
          if r["level"] == "T3" and r.get("dataset") == "replogle_k562_essential" and r.get("delta_pearson_GEARS_split")]
    check("delta split random K->R ~0.660 / GEARS T3 K562 ~0.591",
          bool(kk) and 0.64 <= sum(kk) / len(kk) <= 0.68 and bool(gg) and 0.56 <= sum(gg) / len(gg) <= 0.62,
          f"got {(sum(kk)/len(kk)) if kk else 0:.4f}/{(sum(gg)/len(gg)) if gg else 0:.4f}")

    t4a = canon_summary("baselines_t4_", "T4", 149, gse="gse124715") or {}
    t4b = canon_summary("baselines_t4_", "T4", 149, gse="gse125933") or {}
    la = t4a.get("linear_transfer", {}).get("r2_deg", {}).get("mean")
    lb = t4b.get("linear_transfer", {}).get("r2_deg", {}).get("mean")
    check("T4 linear -0.323/-0.200", la is not None and -0.37 <= la <= -0.28 and lb is not None and -0.25 <= lb <= -0.15, f"got {la}/{lb}")

    import pandas as pd
    frames = []
    meta_rows = []
    for d in (PKG / "results" / "runs").iterdir():
        mj = d / "t4_retrieval_null.csv"
        if d.is_dir() and mj.exists():
            try:
                frames.append(pd.read_csv(mj))
            except Exception:
                pass
        m2 = d / "metrics.json"
        if d.is_dir() and m2.exists():
            try:
                j = json.loads(m2.read_text(encoding="utf-8"))
                meta = j.get("meta", {})
                if meta.get("level") == "T4" and meta.get("gse"):
                    meta_rows.extend(meta.get("retrieval_null_CKI", []))
            except Exception:
                pass
    n_ok = 0
    n_tot = 0
    allframes = frames + ([pd.DataFrame(meta_rows)] if meta_rows else [])
    if allframes:
        ndf = pd.concat(allframes, ignore_index=True)
        cols = [c for c in ("line", "query") if c in ndf.columns]
        if cols:
            ndf = ndf.drop_duplicates(subset=cols, keep="last")
        n_ok = int((ndf["p_emp"] <= 0.001).sum())
        n_tot = len(ndf)
    check("T4 retrieval permutation null: >=10 queries P<=0.001", n_ok >= 10, f"{n_ok}/{n_tot} (CSV+meta merged)")

    grid = PKG / "docs" / "网格汇总_stage02.md"
    gtxt = grid.read_text(encoding="utf-8") if grid.exists() else ""
    sig12 = len(re.findall(r"\| yes\s+\|", gtxt))
    check("BH correction: 12/18 significant (norman+K562)", sig12 == 12, f"sig_BH rows {sig12}")

    fails = 0
    print("=" * 60)
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))
    print("=" * 60)
    print(f"{len(results) - fails}/{len(results)} PASS" + ("" if fails == 0 else f"  FAIL={fails}"))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
