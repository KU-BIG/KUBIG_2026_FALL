"""Phase 3 integrated analysis (PLAN.md sec 4): pool information-asymmetry,
gap, and downstream-performance metrics across every condition run in
Phase 1/2, and test the causal chain information_asymmetry -> delta_gap ->
downstream_performance with a Baron & Kenny-style mediation regression
(rather than a bare correlation/scatter, per PLAN.md's own upgrade from the
original plan — see docs/PLAN.md sec 4).

Only src/experiments/phase2_expA_dial_swipe.py's conditions have a directly
interpretable, pre-registered "information asymmetry" axis (quantity: gene
count relative to ADT's fixed 134 dims; quality: categorical, encoded as an
ordinal proxy for how much of the ADT-relevant signal is present). Exp B
(mismatch type) and exp C (heterogeneity) are included as covariates /
secondary evidence, not folded into the same asymmetry scale, since
conflating "how many genes" with "which cell got paired with which" would
manufacture a single number without a shared unit — see the
`information_asymmetry_index` docstring below for exactly what is and
isn't comparable across rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ADT_DIM = 134


def _load_table(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"WARNING: {path} not found, skipping (run the corresponding experiment first)")
        return None


def information_asymmetry_index(row: pd.Series) -> float:
    """log(n_genes_actual / ADT_DIM) for exp A's quantity axis — the
    plan's own operationalization (GEX gene count relative to the fixed
    134-dim ADT panel). Quality-axis conditions and exp C (heterogeneity)
    get NaN here; they enter the regression through separate covariates
    instead of being coerced onto the same numeric scale.
    """
    if row.get("axis") == "quantity" and pd.notna(row.get("n_genes_actual")):
        return float(np.log(row["n_genes_actual"] / ADT_DIM))
    return float("nan")


def build_pooled_table() -> pd.DataFrame:
    frames = []

    expA = _load_table("results/tables/phase2_expA_dial_swipe.csv")
    if expA is not None:
        expA = expA.copy()
        expA["source"] = "expA"
        expA["information_asymmetry_index"] = expA.apply(information_asymmetry_index, axis=1)
        expA["heterogeneity_entropy"] = float("nan")
        frames.append(expA)

    expC = _load_table("results/tables/phase2_expC_lineage.csv")
    if expC is not None:
        expC = expC.copy()
        expC["source"] = "expC"
        expC["information_asymmetry_index"] = float("nan")
        frames.append(expC)

    if not frames:
        raise RuntimeError(
            "No experiment result tables found — run phase2_expA_dial_swipe.py "
            "and/or phase2_expC_lineage.py before phase3_integration.py"
        )
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    return pooled


def run_mediation(df: pd.DataFrame, asymmetry_col: str, gap_col: str = "delta_gap", perf_col: str = "top5_retrieval_acc"):
    """Baron & Kenny mediation steps, using only rows where `asymmetry_col`
    is defined (currently: exp A's quantity axis).
    Step 1: asymmetry -> gap
    Step 2: asymmetry -> performance
    Step 3: asymmetry + gap -> performance (mediation confirmed if gap's
    coefficient is significant and asymmetry's coefficient shrinks vs step 2)
    """
    sub = df.dropna(subset=[asymmetry_col, gap_col, perf_col]).copy()
    if len(sub) < 8:
        print(f"Not enough rows with non-null {asymmetry_col} ({len(sub)}) to run mediation regression")
        return None

    x = sm.add_constant(sub[[asymmetry_col]])
    step1 = sm.OLS(sub[gap_col], x).fit()

    step2 = sm.OLS(sub[perf_col], x).fit()

    x3 = sm.add_constant(sub[[asymmetry_col, gap_col]])
    step3 = sm.OLS(sub[perf_col], x3).fit()

    print("=== Step 1: asymmetry -> gap ===")
    print(step1.summary().tables[1])
    print("=== Step 2: asymmetry -> performance ===")
    print(step2.summary().tables[1])
    print("=== Step 3: asymmetry + gap -> performance ===")
    print(step3.summary().tables[1])

    direct_effect_shrunk = abs(step3.params[asymmetry_col]) < abs(step2.params[asymmetry_col])
    gap_significant_in_step3 = step3.pvalues[gap_col] < 0.05
    print(f"\nMediation signature present: gap significant in step 3 (p={step3.pvalues[gap_col]:.4f}) "
          f"AND direct effect shrunk ({step2.params[asymmetry_col]:.4f} -> {step3.params[asymmetry_col]:.4f}): "
          f"{gap_significant_in_step3 and direct_effect_shrunk}")

    return dict(step1=step1, step2=step2, step3=step3)


def main():
    pooled = build_pooled_table()
    out_path = "results/tables/phase3_pooled.csv"
    pooled.to_csv(out_path, index=False)
    print(f"Saved pooled table: {out_path} ({len(pooled)} rows from {pooled['source'].unique().tolist()})")

    print("\n--- Mediation analysis: exp A quantity axis ---")
    run_mediation(pooled, asymmetry_col="information_asymmetry_index")


if __name__ == "__main__":
    main()
