#!/usr/bin/env python3
"""Restarted Step 4: publication-level linkage associations in final v6.

Save as:
    new_version/computations/step4_v6_linkage_associations.py

Run from new_version/:
    python computations/step4_v6_linkage_associations.py

Inputs (read only):
    data/linked_final_outputs_v6/publication_anchor_final.pkl
    computations/outputs/step1_v6_audit_v2/edges_publication_grant.csv
    computations/outputs/step1_v6_audit_v2/edges_publication_policy_document.csv
    computations/outputs/step2_v6_indicators/publication_text_indicators.csv
    computations/outputs/step3_v6_topic_families_v2/publication_concept_families.pkl

This is an association analysis, not a causal or predictive analysis. Grant and
policy-document linkage outcomes are recorded Dimensions relationships. A policy
edge does not demonstrate topical similarity, endorsement, policy impact,
institutional intention, or knowledge translation.
"""

from __future__ import annotations

import json
import math
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
        "This Step 4 script requires statsmodels. In your activated conda environment, run:\n"
        "conda install -y statsmodels\n"
        "Then rerun the script."
    ) from exc

SCRIPT_VERSION = "step4_v6_linkage_associations_1.0"
POLICY_LOW_EVENT_FLAG = 5
LOW_DENOMINATOR_FLAG = 30

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "linked_final_outputs_v6"
STEP1_DIR = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2"
STEP2_DIR = PROJECT_ROOT / "computations" / "outputs" / "step2_v6_indicators"
STEP3_DIR = PROJECT_ROOT / "computations" / "outputs" / "step3_v6_topic_families_v2"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step4_v6_linkage_associations"

P_FILE = DATA_DIR / "publication_anchor_final.pkl"
PG_FILE = STEP1_DIR / "edges_publication_grant.csv"
PD_FILE = STEP1_DIR / "edges_publication_policy_document.csv"
INDICATOR_FILE = STEP2_DIR / "publication_text_indicators.csv"
FAMILY_FILE = STEP3_DIR / "publication_concept_families.pkl"
STEP1_MANIFEST = STEP1_DIR / "data_manifest.json"
STEP2_MANIFEST = STEP2_DIR / "step2_indicator_manifest.json"
STEP3_MANIFEST = STEP3_DIR / "step3_topic_family_manifest.json"

TEXT_VARS = [
    "ethics_responsibility_primary",
    "computational_performance_primary",
]
FAMILY_VARS = [
    "mental_health_conditions_symptoms",
    "clinical_assessment_diagnosis",
    "treatment_intervention",
    "neurocognitive_affective_processes",
]
MODEL_VARS = TEXT_VARS + FAMILY_VARS + ["publication_year_centered", "log1p_times_cited"]
OUTCOMES = ["has_grant_link", "has_policy_document_link"]


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Required input file(s) missing:\n" + "\n".join(missing))


def require_columns(frame: pd.DataFrame, label: str, fields: list[str]) -> None:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(missing)}")


def bool_int(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.notna() & ~values.isin([0, 1])
    if invalid.any():
        raise SystemExit(f"{label} contains non-binary values.")
    return values.fillna(0).astype(int)


def rate_table(frame: pd.DataFrame, variable: str, variable_set: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(variable, dropna=False, sort=True):
        for outcome in OUTCOMES:
            events = int(group[outcome].sum())
            n = int(len(group))
            rows.append({
                "variable_set": variable_set,
                "variable": variable,
                "level": int(value) if pd.notna(value) else None,
                "outcome": outcome,
                "n_publications": n,
                "n_linked_publications": events,
                "linkage_rate_pct": round(100 * events / n, 3) if n else None,
                "low_denominator_flag": bool(n < LOW_DENOMINATOR_FLAG),
                "low_event_flag": bool(events < POLICY_LOW_EVENT_FLAG),
            })
    return pd.DataFrame(rows)


def wilson_interval(events: int, total: int) -> tuple[float | None, float | None]:
    """Wilson 95% interval for a descriptive linkage rate."""
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = events / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return 100 * (centre - half_width), 100 * (centre + half_width)


def yearly_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby("year", sort=True):
        n = len(group)
        for outcome in OUTCOMES:
            events = int(group[outcome].sum())
            ci_low, ci_high = wilson_interval(events, n)
            rows.append({
                "publication_year": int(year),
                "outcome": outcome,
                "n_publications": int(n),
                "n_linked_publications": events,
                "linkage_rate_pct": round(100 * events / n, 3),
                "wilson_ci_95_low_pct": round(ci_low, 3) if ci_low is not None else None,
                "wilson_ci_95_high_pct": round(ci_high, 3) if ci_high is not None else None,
                "low_denominator_flag": bool(n < LOW_DENOMINATOR_FLAG),
                "low_event_flag": bool(events < POLICY_LOW_EVENT_FLAG),
            })
    return pd.DataFrame(rows)


def fit_association_model(frame: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_frame = frame[[outcome] + MODEL_VARS].dropna().copy()
    y = model_frame[outcome].astype(float)
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
        output = pd.DataFrame({
            "outcome": outcome,
            "term": result.params.index,
            "coefficient_log_odds": result.params.values,
            "robust_standard_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_95_log_odds_low": ci.iloc[:, 0].values,
            "ci_95_log_odds_high": ci.iloc[:, 1].values,
        })
        output["odds_ratio"] = np.exp(output["coefficient_log_odds"])
        output["ci_95_odds_ratio_low"] = np.exp(output["ci_95_log_odds_low"])
        output["ci_95_odds_ratio_high"] = np.exp(output["ci_95_log_odds_high"])
        output = output[[
            "outcome", "term", "coefficient_log_odds", "robust_standard_error", "z_value", "p_value",
            "odds_ratio", "ci_95_odds_ratio_low", "ci_95_odds_ratio_high",
            "ci_95_log_odds_low", "ci_95_log_odds_high",
        ]]
        diagnostics = {
            "outcome": outcome,
            "model_status": "fit",
            "n_observations": int(result.nobs),
            "n_events": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3),
            "n_parameters_including_intercept": int(len(result.params)),
            "aic": round(float(result.aic), 6),
            "pseudo_r2_mcfadden": round(float(1 - result.llf / result.llnull), 6) if result.llnull else None,
            "covariance_estimator": "HC3 robust",
            "interpretation": "Conditional association only; odds ratios do not identify causal effects or institutional intent.",
        }
        return output, diagnostics
    except Exception as exc:
        empty = pd.DataFrame(columns=[
            "outcome", "term", "coefficient_log_odds", "robust_standard_error", "z_value", "p_value",
            "odds_ratio", "ci_95_odds_ratio_low", "ci_95_odds_ratio_high",
            "ci_95_log_odds_low", "ci_95_log_odds_high",
        ])
        diagnostics = {
            "outcome": outcome,
            "model_status": "not_fit",
            "n_observations": int(len(model_frame)),
            "n_events": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3) if len(y) else None,
            "n_parameters_including_intercept": int(len(MODEL_VARS) + 1),
            "aic": None,
            "pseudo_r2_mcfadden": None,
            "covariance_estimator": "HC3 robust",
            "interpretation": f"Model did not fit: {type(exc).__name__}: {exc}",
        }
        return empty, diagnostics


def main() -> None:
    input_paths = [P_FILE, PG_FILE, PD_FILE, INDICATOR_FILE, FAMILY_FILE, STEP1_MANIFEST, STEP2_MANIFEST, STEP3_MANIFEST]
    require_files(input_paths)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    publications = pd.read_pickle(P_FILE).copy()
    indicators = pd.read_csv(INDICATOR_FILE).copy()
    families = pd.read_pickle(FAMILY_FILE).copy()
    pg_edges = pd.read_csv(PG_FILE, dtype=str)
    pd_edges = pd.read_csv(PD_FILE, dtype=str)

    require_columns(publications, "Final P", ["id", "year", "times_cited", "retain_in_final_P"])
    require_columns(indicators, "Text indicators", ["publication_id"] + TEXT_VARS)
    require_columns(families, "Topic families", ["publication_id"] + FAMILY_VARS)
    require_columns(pg_edges, "P–grant edges", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P–policy-document edges", ["publication_id", "policy_document_id"])

    publications["id"] = publications["id"].astype(str).str.strip()
    indicators["publication_id"] = indicators["publication_id"].astype(str).str.strip()
    families["publication_id"] = families["publication_id"].astype(str).str.strip()
    pg_edges["publication_id"] = pg_edges["publication_id"].astype(str).str.strip()
    pd_edges["publication_id"] = pd_edges["publication_id"].astype(str).str.strip()

    if not publications["retain_in_final_P"].astype(bool).all():
        raise SystemExit("Final P contains a record not retained by the approved v6 local screen.")
    if publications["id"].duplicated().any() or indicators["publication_id"].duplicated().any() or families["publication_id"].duplicated().any():
        raise SystemExit("P, indicators, or family table contains duplicate publication IDs.")

    P_ids = set(publications["id"])
    edge_failures = int((~pg_edges["publication_id"].isin(P_ids)).sum()) + int((~pd_edges["publication_id"].isin(P_ids)).sum())
    if edge_failures:
        raise SystemExit("An edge input contains a publication endpoint outside final P. Re-run restarted Step 1.")

    analysis = publications[["id", "year", "times_cited"]].rename(columns={"id": "publication_id"}).copy()
    analysis = analysis.merge(
        indicators[["publication_id"] + TEXT_VARS], on="publication_id", how="left", validate="one_to_one"
    )
    analysis = analysis.merge(
        families[["publication_id"] + FAMILY_VARS], on="publication_id", how="left", validate="one_to_one"
    )
    if len(analysis) != len(publications):
        raise SystemExit("One-to-one input merge changed the final P denominator.")
    if analysis[TEXT_VARS + FAMILY_VARS].isna().any().any():
        missing_ids = analysis.loc[analysis[TEXT_VARS + FAMILY_VARS].isna().any(axis=1), "publication_id"].head(10).tolist()
        raise SystemExit("Indicator/family data missing for final P IDs, e.g.: " + "; ".join(missing_ids))

    for column in TEXT_VARS + FAMILY_VARS:
        analysis[column] = bool_int(analysis[column], column)

    grant_counts = pg_edges.drop_duplicates().groupby("publication_id")["grant_id"].nunique().rename("n_linked_grants")
    policy_counts = pd_edges.drop_duplicates().groupby("publication_id")["policy_document_id"].nunique().rename("n_linked_policy_documents")
    analysis = analysis.merge(grant_counts, on="publication_id", how="left", validate="one_to_one")
    analysis = analysis.merge(policy_counts, on="publication_id", how="left", validate="one_to_one")
    analysis[["n_linked_grants", "n_linked_policy_documents"]] = analysis[["n_linked_grants", "n_linked_policy_documents"]].fillna(0).astype(int)
    analysis["has_grant_link"] = analysis["n_linked_grants"].gt(0).astype(int)
    analysis["has_policy_document_link"] = analysis["n_linked_policy_documents"].gt(0).astype(int)

    analysis["year"] = pd.to_numeric(analysis["year"], errors="coerce")
    if analysis["year"].isna().any():
        raise SystemExit("Final P contains missing or non-numeric publication years.")
    analysis["year"] = analysis["year"].astype(int)
    analysis["publication_year_centered"] = analysis["year"] - analysis["year"].mean()
    analysis["times_cited"] = pd.to_numeric(analysis["times_cited"], errors="coerce")
    analysis["times_cited_missing"] = analysis["times_cited"].isna().astype(int)
    analysis["times_cited"] = analysis["times_cited"].fillna(0).clip(lower=0)
    analysis["log1p_times_cited"] = np.log1p(analysis["times_cited"])
    analysis["indicator_overlap_group"] = np.select(
        [
            (analysis[TEXT_VARS[0]] == 0) & (analysis[TEXT_VARS[1]] == 0),
            (analysis[TEXT_VARS[0]] == 1) & (analysis[TEXT_VARS[1]] == 0),
            (analysis[TEXT_VARS[0]] == 0) & (analysis[TEXT_VARS[1]] == 1),
            (analysis[TEXT_VARS[0]] == 1) & (analysis[TEXT_VARS[1]] == 1),
        ],
        ["neither", "ethics_responsibility_only", "computational_performance_only", "both"],
        default="invalid",
    )

    denominator_rows = []
    for outcome in OUTCOMES:
        events = int(analysis[outcome].sum())
        ci_low, ci_high = wilson_interval(events, len(analysis))
        denominator_rows.append({
            "outcome": outcome,
            "n_final_P_publications": len(analysis),
            "n_linked_publications": events,
            "n_unlinked_publications": int(len(analysis) - events),
            "linkage_rate_pct": round(100 * events / len(analysis), 3),
            "wilson_ci_95_low_pct": round(ci_low, 3),
            "wilson_ci_95_high_pct": round(ci_high, 3),
            "rare_outcome_flag": bool(events < 0.05 * len(analysis)),
        })
    denominators = pd.DataFrame(denominator_rows)

    text_rates = pd.concat([rate_table(analysis, variable, "primary_text_indicator") for variable in TEXT_VARS], ignore_index=True)
    family_rates = pd.concat([rate_table(analysis, variable, "supplementary_topic_family") for variable in FAMILY_VARS], ignore_index=True)
    overlap_rates = []
    for level, group in analysis.groupby("indicator_overlap_group", sort=False):
        for outcome in OUTCOMES:
            n = len(group)
            events = int(group[outcome].sum())
            overlap_rates.append({
                "indicator_overlap_group": level,
                "outcome": outcome,
                "n_publications": n,
                "n_linked_publications": events,
                "linkage_rate_pct": round(100 * events / n, 3),
                "low_denominator_flag": bool(n < LOW_DENOMINATOR_FLAG),
                "low_event_flag": bool(events < POLICY_LOW_EVENT_FLAG),
            })
    overlap_rates = pd.DataFrame(overlap_rates)
    year_rates = yearly_outcomes(analysis)

    model_outputs = []
    diagnostics = []
    for outcome in OUTCOMES:
        coefficients, diagnostic = fit_association_model(analysis, outcome)
        model_outputs.append(coefficients)
        diagnostics.append(diagnostic)
    model_coefficients = pd.concat(model_outputs, ignore_index=True)
    model_diagnostics = pd.DataFrame(diagnostics)

    validation_checks = pd.DataFrame([
        {"check": "final_P_unique_ids", "passed": not publications["id"].duplicated().any(), "detail": ""},
        {"check": "indicator_table_one_row_per_final_P", "passed": len(indicators) == len(analysis), "detail": ""},
        {"check": "topic_family_table_one_row_per_final_P", "passed": len(families) == len(analysis), "detail": ""},
        {"check": "all_text_and_topic_variables_binary", "passed": True, "detail": "validated as 0/1 before analysis"},
        {"check": "edge_publication_endpoints_in_final_P", "passed": edge_failures == 0, "detail": ""},
        {"check": "policy_outcome_low_event_flag_present", "passed": "has_policy_document_link" in set(denominators.loc[denominators["rare_outcome_flag"], "outcome"]), "detail": "policy outcome is expected to be rare and is flagged"},
        {"check": "all_requested_models_fit", "passed": bool((model_diagnostics["model_status"] == "fit").all()), "detail": "; ".join(model_diagnostics.loc[model_diagnostics["model_status"] != "fit", "interpretation"].tolist())},
    ])

    analysis.to_csv(OUTPUT_DIR / "publication_linkage_analysis_dataset.csv", index=False)
    denominators.to_csv(OUTPUT_DIR / "outcome_denominators.csv", index=False)
    text_rates.to_csv(OUTPUT_DIR / "linkage_rates_by_text_indicator.csv", index=False)
    family_rates.to_csv(OUTPUT_DIR / "linkage_rates_by_topic_family.csv", index=False)
    overlap_rates.to_csv(OUTPUT_DIR / "linkage_rates_by_indicator_overlap.csv", index=False)
    year_rates.to_csv(OUTPUT_DIR / "linkage_rates_by_publication_year.csv", index=False)
    model_coefficients.to_csv(OUTPUT_DIR / "association_model_coefficients.csv", index=False)
    model_diagnostics.to_csv(OUTPUT_DIR / "association_model_diagnostics.csv", index=False)
    validation_checks.to_csv(OUTPUT_DIR / "outcome_validation_checks.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "unit_of_analysis": "final screened P publication",
        "source_dataset": "linked_final_outputs_v6",
        "outcomes": {
            "has_grant_link": "At least one valid P–grant edge constructed from P.supporting_grant_ids and final linked grant IDs.",
            "has_policy_document_link": "At least one valid P–policy-document edge constructed from policy_documents.publication_ids intersected with final P IDs.",
        },
        "primary_predictors": TEXT_VARS,
        "supplementary_predictors": FAMILY_VARS,
        "model_adjustments": ["publication_year_centered", "log1p_times_cited"],
        "model": "Separate binomial logistic regressions for grant and policy-document linkage, with HC3 robust standard errors.",
        "low_n_rule": {"low_denominator_n": LOW_DENOMINATOR_FLAG, "low_event_n": POLICY_LOW_EVENT_FLAG},
        "interpretive_boundary": "Results are conditional associations in recorded Dimensions linkages. They do not identify causal effects, intentional selection, suppression, policy endorsement, policy impact, or semantic affinity between linked records.",
        "input_manifests": {
            "step1": json.loads(STEP1_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
            "step2": json.loads(STEP2_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
            "step3": json.loads(STEP3_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
        },
    }
    (OUTPUT_DIR / "step4_association_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Restarted Step 4 (v6 linkage associations) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    for _, row in denominators.iterrows():
        print(f"{row['outcome']}: {row['n_linked_publications']:,} / {row['n_final_P_publications']:,} ({row['linkage_rate_pct']:.3f}%)")
    print("Models fitted:", ", ".join(model_diagnostics["outcome"].tolist()))


if __name__ == "__main__":
    main()
