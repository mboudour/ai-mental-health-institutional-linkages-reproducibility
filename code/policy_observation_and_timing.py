#!/usr/bin/env python3
"""Restarted Step 5: policy citation-window sensitivity and temporal ordering.

Save as:
    new_version/computations/step5_v6_policy_window_and_timing_v2.py

Run from new_version/:
    python computations/step5_v6_policy_window_and_timing_v2.py

Inputs (read only):
    data/linked_final_outputs_v6/publication_anchor_final.pkl
    data/linked_final_outputs_v6/linked_grants_final.pkl
    data/linked_final_outputs_v6/linked_policy_documents_final.pkl
    computations/outputs/step1_v6_audit_v2/edges_publication_grant.csv
    computations/outputs/step1_v6_audit_v2/edges_publication_policy_document.csv
    computations/outputs/step4_v6_linkage_associations/publication_linkage_analysis_dataset.csv

This script does not infer causal timing. A positive year difference means only that
one Dimensions record is dated later than another. Policy documents remain citation-
linked records, not evidence of policy impact, endorsement, topical similarity, or
institutional intention.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning, SingularMatrixWarning
except ImportError as exc:
    raise SystemExit(
        "This script requires statsmodels. In your activated conda environment, run:\n"
        "conda install -y statsmodels\n"
        "Then rerun the script."
    ) from exc

SCRIPT_VERSION = "step5_v6_policy_window_and_timing_2.0_policy_count_merge_fix"
POLICY_DATA_END_YEAR = 2025
PRIMARY_POLICY_CUTOFF = 2021  # At least four observed calendar years through 2025.
ROBUSTNESS_POLICY_CUTOFF = 2020  # At least five observed calendar years through 2025.
MODEL_VARS = [
    "ethics_responsibility_primary",
    "computational_performance_primary",
    "mental_health_conditions_symptoms",
    "clinical_assessment_diagnosis",
    "treatment_intervention",
    "neurocognitive_affective_processes",
    "publication_year_centered",
    "log1p_times_cited",
]
OUTCOME = "has_policy_document_link"

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "linked_final_outputs_v6"
STEP1_DIR = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2"
STEP4_DIR = PROJECT_ROOT / "computations" / "outputs" / "step4_v6_linkage_associations"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step5_v6_policy_window_and_timing_v2"

P_FILE = DATA_DIR / "publication_anchor_final.pkl"
G_FILE = DATA_DIR / "linked_grants_final.pkl"
D_FILE = DATA_DIR / "linked_policy_documents_final.pkl"
PG_FILE = STEP1_DIR / "edges_publication_grant.csv"
PD_FILE = STEP1_DIR / "edges_publication_policy_document.csv"
ANALYSIS_FILE = STEP4_DIR / "publication_linkage_analysis_dataset.csv"
STEP4_MANIFEST = STEP4_DIR / "step4_association_manifest.json"


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Required input file(s) missing:\n" + "\n".join(missing))


def require_columns(frame: pd.DataFrame, label: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(missing)}")


def fit_policy_model(frame: pd.DataFrame, sample_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_frame = frame[[OUTCOME] + MODEL_VARS].dropna().copy()
    y = model_frame[OUTCOME].astype(float)
    x = sm.add_constant(model_frame[MODEL_VARS].astype(float), has_constant="add")
    try:
        if y.nunique() < 2:
            raise ValueError("Outcome has fewer than two observed values.")
        if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
            raise ValueError("Design matrix is rank-deficient; coefficients are not uniquely identifiable.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", SingularMatrixWarning)
            warnings.simplefilter("error", ConvergenceWarning)
            result = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="HC3")
        ci = result.conf_int()
        coefficients = pd.DataFrame({
            "analysis_sample": sample_label,
            "outcome": OUTCOME,
            "term": result.params.index,
            "coefficient_log_odds": result.params.values,
            "robust_standard_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_95_log_odds_low": ci.iloc[:, 0].values,
            "ci_95_log_odds_high": ci.iloc[:, 1].values,
        })
        coefficients["odds_ratio"] = np.exp(coefficients["coefficient_log_odds"])
        coefficients["ci_95_odds_ratio_low"] = np.exp(coefficients["ci_95_log_odds_low"])
        coefficients["ci_95_odds_ratio_high"] = np.exp(coefficients["ci_95_log_odds_high"])
        diagnostics = {
            "analysis_sample": sample_label,
            "model_status": "fit",
            "n_publications": int(len(model_frame)),
            "n_policy_linked_publications": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3),
            "n_parameters_including_intercept": int(len(result.params)),
            "aic": round(float(result.aic), 6),
            "pseudo_r2_mcfadden": round(float(1 - result.llf / result.llnull), 6) if result.llnull else None,
            "covariance_estimator": "HC3 robust",
            "interpretive_boundary": "Conditional association only; not a causal effect, policy impact, policy endorsement, or institutional-selection estimate.",
        }
        return coefficients, diagnostics
    except Exception as exc:
        empty = pd.DataFrame(columns=[
            "analysis_sample", "outcome", "term", "coefficient_log_odds", "robust_standard_error", "z_value", "p_value",
            "ci_95_log_odds_low", "ci_95_log_odds_high", "odds_ratio", "ci_95_odds_ratio_low", "ci_95_odds_ratio_high",
        ])
        diagnostics = {
            "analysis_sample": sample_label,
            "model_status": "not_fit",
            "n_publications": int(len(model_frame)),
            "n_policy_linked_publications": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3) if len(y) else None,
            "n_parameters_including_intercept": int(len(MODEL_VARS) + 1),
            "aic": None,
            "pseudo_r2_mcfadden": None,
            "covariance_estimator": "HC3 robust",
            "interpretive_boundary": f"Model did not fit: {type(exc).__name__}: {exc}",
        }
        return empty, diagnostics


def timing_summary(frame: pd.DataFrame, relation: str, lag_col: str) -> dict[str, Any]:
    available = frame[lag_col].dropna()
    nonnegative = available[available >= 0]
    negative = available[available < 0]
    return {
        "relation": relation,
        "n_edges": int(len(frame)),
        "n_edges_with_both_years": int(len(available)),
        "n_missing_year_difference": int(frame[lag_col].isna().sum()),
        "n_negative_year_differences": int(len(negative)),
        "n_zero_year_differences": int((available == 0).sum()),
        "n_positive_year_differences": int((available > 0).sum()),
        "median_year_difference_all": float(available.median()) if len(available) else None,
        "median_nonnegative_year_difference": float(nonnegative.median()) if len(nonnegative) else None,
        "p25_nonnegative_year_difference": float(nonnegative.quantile(0.25)) if len(nonnegative) else None,
        "p75_nonnegative_year_difference": float(nonnegative.quantile(0.75)) if len(nonnegative) else None,
        "maximum_nonnegative_year_difference": int(nonnegative.max()) if len(nonnegative) else None,
        "interpretation": "Record-year difference only; not a causal lag, time to impact, or proof of knowledge translation.",
    }


def main() -> None:
    require_files([P_FILE, G_FILE, D_FILE, PG_FILE, PD_FILE, ANALYSIS_FILE, STEP4_MANIFEST])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    P = pd.read_pickle(P_FILE).copy()
    G = pd.read_pickle(G_FILE).copy()
    D = pd.read_pickle(D_FILE).copy()
    pg_edges = pd.read_csv(PG_FILE, dtype=str)
    pd_edges = pd.read_csv(PD_FILE, dtype=str)
    analysis = pd.read_csv(ANALYSIS_FILE).copy()

    require_columns(P, "Final P", ["id", "year", "retain_in_final_P"])
    require_columns(G, "Linked grants", ["id", "start_year"])
    require_columns(D, "Linked policy documents", ["id", "year"])
    require_columns(pg_edges, "P–grant edges", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P–policy edges", ["publication_id", "policy_document_id"])
    require_columns(analysis, "Step 4 analysis table", ["publication_id", "year", OUTCOME] + MODEL_VARS)

    for frame, column in [(P, "id"), (G, "id"), (D, "id"), (analysis, "publication_id")]:
        frame[column] = frame[column].astype(str).str.strip()
    for frame, column in [(pg_edges, "publication_id"), (pg_edges, "grant_id"), (pd_edges, "publication_id"), (pd_edges, "policy_document_id")]:
        frame[column] = frame[column].astype(str).str.strip()

    if not P["retain_in_final_P"].astype(bool).all():
        raise SystemExit("P includes a record not retained by the final v6 literal screen.")
    if P["id"].duplicated().any() or G["id"].duplicated().any() or D["id"].duplicated().any() or analysis["publication_id"].duplicated().any():
        raise SystemExit("An input table contains duplicate primary IDs.")
    if len(P) != len(analysis) or set(P["id"]) != set(analysis["publication_id"]):
        raise SystemExit("Step 4 analysis table does not correspond one-to-one to final v6 P.")

    P_years = P[["id", "year"]].rename(columns={"id": "publication_id", "year": "publication_year"})
    G_years = G[["id", "start_year"]].rename(columns={"id": "grant_id", "start_year": "grant_start_year"})
    D_years = D[["id", "year"]].rename(columns={"id": "policy_document_id", "year": "policy_document_year"})
    for frame, column in [(P_years, "publication_year"), (G_years, "grant_start_year"), (D_years, "policy_document_year")]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    pg_timing = pg_edges.drop_duplicates().merge(P_years, on="publication_id", how="left", validate="many_to_one").merge(G_years, on="grant_id", how="left", validate="many_to_one")
    pd_timing = pd_edges.drop_duplicates().merge(P_years, on="publication_id", how="left", validate="many_to_one").merge(D_years, on="policy_document_id", how="left", validate="many_to_one")
    pg_timing["grant_start_minus_publication_year"] = pg_timing["grant_start_year"] - pg_timing["publication_year"]
    pd_timing["policy_document_minus_publication_year"] = pd_timing["policy_document_year"] - pd_timing["publication_year"]

    timing = pd.DataFrame([
        timing_summary(pg_timing, "publication_grant", "grant_start_minus_publication_year"),
        timing_summary(pd_timing, "publication_policy_document", "policy_document_minus_publication_year"),
    ])

    policy_by_pub = (
        pd_timing.groupby("publication_id", as_index=False)
        .agg(
            n_linked_policy_documents=("policy_document_id", "nunique"),
            earliest_linked_policy_document_year=("policy_document_year", "min"),
            latest_linked_policy_document_year=("policy_document_year", "max"),
            minimum_policy_record_year_difference=("policy_document_minus_publication_year", "min"),
            maximum_policy_record_year_difference=("policy_document_minus_publication_year", "max"),
        )
    )
    policy_by_pub["minimum_nonnegative_policy_record_year_difference"] = policy_by_pub["publication_id"].map(
        pd_timing.loc[pd_timing["policy_document_minus_publication_year"] >= 0]
        .groupby("publication_id")["policy_document_minus_publication_year"].min()
    )

    analysis["year"] = pd.to_numeric(analysis["year"], errors="coerce")
    if analysis["year"].isna().any():
        raise SystemExit("Step 4 analysis data contains missing publication years.")
    analysis["year"] = analysis["year"].astype(int)
    analysis["observed_policy_calendar_years_through_2025"] = POLICY_DATA_END_YEAR - analysis["year"] + 1
    # Step 4 already has a derived policy-count field. Replace it with the
    # count recomputed from the verified P–policy edge list, rather than let
    # pandas create _x/_y suffixes during the merge.
    analysis = analysis.drop(columns=["n_linked_policy_documents"], errors="ignore")
    analysis = analysis.merge(policy_by_pub, on="publication_id", how="left", validate="one_to_one")
    analysis["n_linked_policy_documents"] = analysis["n_linked_policy_documents"].fillna(0).astype(int)

    policy_year_rows = []
    for year, subset in analysis.groupby("year", sort=True):
        policy_year_rows.append({
            "publication_year": int(year),
            "n_final_P_publications": int(len(subset)),
            "n_policy_linked_publications": int(subset[OUTCOME].sum()),
            "policy_link_rate_pct": round(100 * subset[OUTCOME].mean(), 3),
            "observed_policy_calendar_years_through_2025": int(POLICY_DATA_END_YEAR - year + 1),
            "low_event_flag": bool(subset[OUTCOME].sum() < 5),
        })
    policy_year_summary = pd.DataFrame(policy_year_rows)

    model_specs = [
        ("all_final_P_citation_window_sensitivity", analysis),
        (f"publications_through_{PRIMARY_POLICY_CUTOFF}_primary", analysis.loc[analysis["year"] <= PRIMARY_POLICY_CUTOFF].copy()),
        (f"publications_through_{ROBUSTNESS_POLICY_CUTOFF}_window_robustness", analysis.loc[analysis["year"] <= ROBUSTNESS_POLICY_CUTOFF].copy()),
    ]

    coefficient_frames = []
    diagnostic_rows = []
    window_rows = []
    for label, subset in model_specs:
        coefficients, diagnostics = fit_policy_model(subset, label)
        coefficient_frames.append(coefficients)
        diagnostic_rows.append(diagnostics)
        window_rows.append({
            "analysis_sample": label,
            "publication_year_eligibility": f"<= {label.split('_')[2]}" if label.startswith("publications_through_") else "all years 2000–2025",
            "n_final_P_publications": int(len(subset)),
            "n_policy_linked_publications": int(subset[OUTCOME].sum()),
            "policy_link_rate_pct": round(100 * subset[OUTCOME].mean(), 3) if len(subset) else None,
            "minimum_observed_policy_calendar_years_through_2025": int(subset["observed_policy_calendar_years_through_2025"].min()) if len(subset) else None,
            "purpose": "Primary policy-window analysis" if label.endswith("_primary") else ("Five-year policy-window robustness check" if label.endswith("_window_robustness") else "Citation-window sensitivity only"),
        })
    model_coefficients = pd.concat(coefficient_frames, ignore_index=True)
    model_diagnostics = pd.DataFrame(diagnostic_rows)
    policy_window_summary = pd.DataFrame(window_rows)

    validation_checks = pd.DataFrame([
        {"check": "final_P_and_step4_analysis_identical_ID_set", "passed": set(P["id"]) == set(analysis["publication_id"]), "detail": ""},
        {"check": "publication_grant_timing_endpoints_have_years", "passed": int(pg_timing[["publication_year", "grant_start_year"]].isna().any(axis=1).sum()) == 0, "detail": ""},
        {"check": "publication_policy_timing_endpoints_have_years", "passed": int(pd_timing[["publication_year", "policy_document_year"]].isna().any(axis=1).sum()) == 0, "detail": ""},
        {"check": "policy_window_models_fit", "passed": bool((model_diagnostics["model_status"] == "fit").all()), "detail": "; ".join(model_diagnostics.loc[model_diagnostics["model_status"] != "fit", "interpretive_boundary"].tolist())},
        {"check": "all_year_policy_model_labelled_sensitivity", "passed": "all_final_P_citation_window_sensitivity" in set(model_diagnostics["analysis_sample"]), "detail": ""},
    ])

    pg_timing.to_csv(OUTPUT_DIR / "publication_grant_record_year_differences.csv", index=False)
    pd_timing.to_csv(OUTPUT_DIR / "publication_policy_record_year_differences.csv", index=False)
    timing.to_csv(OUTPUT_DIR / "record_year_difference_summary.csv", index=False)
    policy_by_pub.to_csv(OUTPUT_DIR / "policy_record_timing_by_publication.csv", index=False)
    policy_year_summary.to_csv(OUTPUT_DIR / "policy_linkage_by_publication_year.csv", index=False)
    policy_window_summary.to_csv(OUTPUT_DIR / "policy_citation_window_samples.csv", index=False)
    model_coefficients.to_csv(OUTPUT_DIR / "policy_window_model_coefficients.csv", index=False)
    model_diagnostics.to_csv(OUTPUT_DIR / "policy_window_model_diagnostics.csv", index=False)
    validation_checks.to_csv(OUTPUT_DIR / "temporal_validation_checks.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "source_dataset": "linked_final_outputs_v6",
        "policy_data_end_year": POLICY_DATA_END_YEAR,
        "citation_window_design": {
            "primary_sample": f"P publication years through {PRIMARY_POLICY_CUTOFF}, providing at least four observed calendar years through 2025.",
            "robustness_sample": f"P publication years through {ROBUSTNESS_POLICY_CUTOFF}, providing at least five observed calendar years through 2025.",
            "all_years_model": "Sensitivity only because recently published P records have fewer observed policy-document years.",
            "observed_window_definition": "POLICY_DATA_END_YEAR minus publication year plus one calendar year; used only as a record-coverage proxy.",
        },
        "temporal_measure": "Difference between the years stored in linked Dimensions records.",
        "interpretive_boundary": "Record-year differences are not causal lags, time to impact, policy influence, institutional selection, or proof of knowledge translation. Negative year differences are retained and reported rather than discarded from the linkage network.",
        "model_predictors": MODEL_VARS,
        "model": "Binomial logistic policy-link association models with HC3 robust standard errors.",
        "step4_model_reference": json.loads(STEP4_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
    }
    (OUTPUT_DIR / "step5_temporal_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Restarted Step 5 (v6 policy window and timing) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    for _, row in policy_window_summary.iterrows():
        print(f"{row['analysis_sample']}: {row['n_policy_linked_publications']:,} / {row['n_final_P_publications']:,} policy-linked P")
    print(f"P–grant record-year pairs: {len(pg_timing):,}")
    print(f"P–policy record-year pairs: {len(pd_timing):,}")


if __name__ == "__main__":
    main()
