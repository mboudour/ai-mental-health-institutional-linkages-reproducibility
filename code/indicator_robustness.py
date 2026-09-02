#!/usr/bin/env python3
"""Restarted Step 7: pre-specified text-indicator robustness checks.

Save as:
    new_version/computations/step7_v6_indicator_robustness.py

Run from new_version/:
    python computations/step7_v6_indicator_robustness.py

This script reruns only the final linkage association models under the four text-
indicator definitions already fixed before outcome analysis in Step 2. It does not
retrieve data, alter final P, alter edge lists, or choose new terms after results.

Grant outcome: all final P publications.
Policy outcome: P through 2021 is primary (at least five observed calendar years
through the 2025 policy-data end year); P through 2020 is a five-year-window
robustness sample. All-years policy estimates are not reinterpreted here because
Step 5 labels them citation-window sensitivity only.
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

SCRIPT_VERSION = "step7_v6_indicator_robustness_1.0"
POLICY_PRIMARY_MAX_PUBLICATION_YEAR = 2021
POLICY_ROBUSTNESS_MAX_PUBLICATION_YEAR = 2020

ETHICS_PRIMARY = "ethics_responsibility_primary"
ETHICS_NARROW = "ethics_responsibility_narrow_robustness"
COMPUTATION_PRIMARY = "computational_performance_primary"
COMPUTATION_EXPANDED = "computational_performance_expanded_sensitivity"
TOPIC_VARS = [
    "mental_health_conditions_symptoms",
    "clinical_assessment_diagnosis",
    "treatment_intervention",
    "neurocognitive_affective_processes",
]
CONTROL_VARS = TOPIC_VARS + ["publication_year_centered", "log1p_times_cited"]

INDICATOR_SPECS = [
    {
        "definition": "primary_both",
        "ethics_variable": ETHICS_PRIMARY,
        "computational_variable": COMPUTATION_PRIMARY,
        "ethics_role": "primary",
        "computational_role": "primary_narrow",
    },
    {
        "definition": "ethics_narrow_only",
        "ethics_variable": ETHICS_NARROW,
        "computational_variable": COMPUTATION_PRIMARY,
        "ethics_role": "narrow_robustness",
        "computational_role": "primary_narrow",
    },
    {
        "definition": "computational_expanded_only",
        "ethics_variable": ETHICS_PRIMARY,
        "computational_variable": COMPUTATION_EXPANDED,
        "ethics_role": "primary",
        "computational_role": "expanded_sensitivity",
    },
    {
        "definition": "both_alternatives",
        "ethics_variable": ETHICS_NARROW,
        "computational_variable": COMPUTATION_EXPANDED,
        "ethics_role": "narrow_robustness",
        "computational_role": "expanded_sensitivity",
    },
]

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "linked_final_outputs_v6"
STEP1_DIR = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2"
STEP2_DIR = PROJECT_ROOT / "computations" / "outputs" / "step2_v6_indicators"
STEP3_DIR = PROJECT_ROOT / "computations" / "outputs" / "step3_v6_topic_families_v2"
STEP5_DIR = PROJECT_ROOT / "computations" / "outputs" / "step5_v6_policy_window_and_timing_v2"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step7_v6_indicator_robustness"

P_FILE = DATA_DIR / "publication_anchor_final.pkl"
PG_FILE = STEP1_DIR / "edges_publication_grant.csv"
PD_FILE = STEP1_DIR / "edges_publication_policy_document.csv"
INDICATOR_FILE = STEP2_DIR / "publication_text_indicators.csv"
FAMILY_FILE = STEP3_DIR / "publication_concept_families.pkl"
STEP2_MANIFEST = STEP2_DIR / "step2_indicator_manifest.json"
STEP5_MANIFEST = STEP5_DIR / "step5_temporal_manifest.json"


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Required input file(s) missing:\n" + "\n".join(missing))


def require_columns(frame: pd.DataFrame, label: str, fields: list[str]) -> None:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {', '.join(missing)}")


def as_binary(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise SystemExit(f"{label}: {column} must contain only 0/1 values.")
        frame[column] = values.astype(int)


def fit_model(frame: pd.DataFrame, outcome: str, model_sample: str, definition: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_frame = frame[[outcome] + predictors].dropna().copy()
    y = model_frame[outcome].astype(float)
    x = sm.add_constant(model_frame[predictors].astype(float), has_constant="add")
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
            "analysis_sample": model_sample,
            "indicator_definition": definition,
            "outcome": outcome,
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
            "analysis_sample": model_sample,
            "indicator_definition": definition,
            "outcome": outcome,
            "model_status": "fit",
            "n_publications": int(len(model_frame)),
            "n_events": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3),
            "n_parameters_including_intercept": int(len(result.params)),
            "aic": round(float(result.aic), 6),
            "pseudo_r2_mcfadden": round(float(1 - result.llf / result.llnull), 6) if result.llnull else None,
            "covariance_estimator": "HC3 robust",
            "interpretive_boundary": "Conditional association only; not causal selection, policy impact, policy endorsement, or institutional intent.",
        }
        return coefficients, diagnostics
    except Exception as exc:
        columns = [
            "analysis_sample", "indicator_definition", "outcome", "term", "coefficient_log_odds", "robust_standard_error", "z_value", "p_value",
            "ci_95_log_odds_low", "ci_95_log_odds_high", "odds_ratio", "ci_95_odds_ratio_low", "ci_95_odds_ratio_high",
        ]
        diagnostics = {
            "analysis_sample": model_sample,
            "indicator_definition": definition,
            "outcome": outcome,
            "model_status": "not_fit",
            "n_publications": int(len(model_frame)),
            "n_events": int(y.sum()),
            "event_rate_pct": round(100 * float(y.mean()), 3) if len(y) else None,
            "n_parameters_including_intercept": int(len(predictors) + 1),
            "aic": None,
            "pseudo_r2_mcfadden": None,
            "covariance_estimator": "HC3 robust",
            "interpretive_boundary": f"Model did not fit: {type(exc).__name__}: {exc}",
        }
        return pd.DataFrame(columns=columns), diagnostics


def indicator_prevalence(frame: pd.DataFrame, variable: str, definition: str, role: str) -> dict[str, Any]:
    return {
        "indicator_definition": definition,
        "indicator_role": role,
        "variable": variable,
        "n_publications": int(len(frame)),
        "n_positive": int(frame[variable].sum()),
        "percent_positive": round(100 * frame[variable].mean(), 3),
    }


def robustness_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    terms = [ETHICS_PRIMARY, ETHICS_NARROW, COMPUTATION_PRIMARY, COMPUTATION_EXPANDED]
    rows = []
    for sample in sorted(coefficients["analysis_sample"].unique()):
        for term in terms:
            selection = coefficients.loc[(coefficients["analysis_sample"] == sample) & (coefficients["term"] == term)].copy()
            if selection.empty:
                continue
            for _, row in selection.iterrows():
                rows.append({
                    "analysis_sample": sample,
                    "term": term,
                    "indicator_definition": row["indicator_definition"],
                    "odds_ratio": row["odds_ratio"],
                    "ci_95_odds_ratio_low": row["ci_95_odds_ratio_low"],
                    "ci_95_odds_ratio_high": row["ci_95_odds_ratio_high"],
                    "p_value": row["p_value"],
                    "direction": "positive" if row["coefficient_log_odds"] > 0 else "negative" if row["coefficient_log_odds"] < 0 else "zero",
                    "ci_excludes_one": bool((row["ci_95_odds_ratio_low"] > 1) or (row["ci_95_odds_ratio_high"] < 1)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    require_files([P_FILE, PG_FILE, PD_FILE, INDICATOR_FILE, FAMILY_FILE, STEP2_MANIFEST, STEP5_MANIFEST])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    P = pd.read_pickle(P_FILE).copy()
    indicators = pd.read_csv(INDICATOR_FILE).copy()
    families = pd.read_pickle(FAMILY_FILE).copy()
    pg = pd.read_csv(PG_FILE, dtype=str).drop_duplicates().copy()
    pd_edges = pd.read_csv(PD_FILE, dtype=str).drop_duplicates().copy()

    require_columns(P, "Final P", ["id", "year", "times_cited", "retain_in_final_P"])
    require_columns(indicators, "Step 2 indicators", ["publication_id", ETHICS_PRIMARY, ETHICS_NARROW, COMPUTATION_PRIMARY, COMPUTATION_EXPANDED])
    require_columns(families, "Step 3 topic families", ["publication_id"] + TOPIC_VARS)
    require_columns(pg, "P–grant edges", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P–policy-document edges", ["publication_id", "policy_document_id"])

    for frame, field in [(P, "id"), (indicators, "publication_id"), (families, "publication_id"), (pg, "publication_id"), (pd_edges, "publication_id")]:
        frame[field] = frame[field].astype(str).str.strip()
    if not P["retain_in_final_P"].astype(bool).all():
        raise SystemExit("Final P includes records outside the approved v6 literal screen.")
    if P["id"].duplicated().any() or indicators["publication_id"].duplicated().any() or families["publication_id"].duplicated().any():
        raise SystemExit("P, indicator, or topic-family table has duplicate publication IDs.")

    all_indicator_vars = [ETHICS_PRIMARY, ETHICS_NARROW, COMPUTATION_PRIMARY, COMPUTATION_EXPANDED]
    as_binary(indicators, all_indicator_vars, "Step 2 indicators")
    as_binary(families, TOPIC_VARS, "Step 3 topic families")

    analysis = P[["id", "year", "times_cited"]].rename(columns={"id": "publication_id"}).copy()
    analysis = analysis.merge(indicators[["publication_id"] + all_indicator_vars], on="publication_id", how="left", validate="one_to_one")
    analysis = analysis.merge(families[["publication_id"] + TOPIC_VARS], on="publication_id", how="left", validate="one_to_one")
    if len(analysis) != len(P) or analysis[all_indicator_vars + TOPIC_VARS].isna().any().any():
        raise SystemExit("One-to-one final-P merge failed or has missing indicator/topic values.")

    P_ids = set(analysis["publication_id"])
    if not pg["publication_id"].isin(P_ids).all() or not pd_edges["publication_id"].isin(P_ids).all():
        raise SystemExit("An edge contains a publication ID outside final P.")
    analysis["has_grant_link"] = analysis["publication_id"].isin(set(pg["publication_id"])).astype(int)
    analysis["has_policy_document_link"] = analysis["publication_id"].isin(set(pd_edges["publication_id"])).astype(int)
    analysis["year"] = pd.to_numeric(analysis["year"], errors="coerce")
    analysis["times_cited"] = pd.to_numeric(analysis["times_cited"], errors="coerce")
    if analysis["year"].isna().any():
        raise SystemExit("Final P has a missing/non-numeric publication year.")
    analysis["year"] = analysis["year"].astype(int)
    analysis["times_cited"] = analysis["times_cited"].fillna(0).clip(lower=0)
    analysis["publication_year_centered"] = analysis["year"] - analysis["year"].mean()
    analysis["log1p_times_cited"] = np.log1p(analysis["times_cited"])

    samples = [
        ("grant_all_final_P", analysis, "has_grant_link"),
        (f"policy_through_{POLICY_PRIMARY_MAX_PUBLICATION_YEAR}_primary_window", analysis.loc[analysis["year"] <= POLICY_PRIMARY_MAX_PUBLICATION_YEAR].copy(), "has_policy_document_link"),
        (f"policy_through_{POLICY_ROBUSTNESS_MAX_PUBLICATION_YEAR}_five_year_window", analysis.loc[analysis["year"] <= POLICY_ROBUSTNESS_MAX_PUBLICATION_YEAR].copy(), "has_policy_document_link"),
    ]

    prevalence_rows = []
    coefficient_frames = []
    diagnostic_rows = []
    for sample_label, sample_frame, outcome in samples:
        for specification in INDICATOR_SPECS:
            predictors = [specification["ethics_variable"], specification["computational_variable"]] + CONTROL_VARS
            coefficients, diagnostics = fit_model(sample_frame, outcome, sample_label, specification["definition"], predictors)
            coefficient_frames.append(coefficients)
            diagnostic_rows.append(diagnostics)
            prevalence_rows.extend([
                {"analysis_sample": sample_label, **indicator_prevalence(sample_frame, specification["ethics_variable"], specification["definition"], specification["ethics_role"])},
                {"analysis_sample": sample_label, **indicator_prevalence(sample_frame, specification["computational_variable"], specification["definition"], specification["computational_role"])},
            ])

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    prevalence = pd.DataFrame(prevalence_rows).drop_duplicates().sort_values(["analysis_sample", "indicator_definition", "variable"])
    summary = robustness_summary(coefficients)

    validation = pd.DataFrame([
        {"check": "final_P_unique_ids", "passed": not P["id"].duplicated().any(), "detail": ""},
        {"check": "all_indicator_definitions_pre_specified_in_step2", "passed": True, "detail": "Only the Step 2 primary/narrow/expanded columns are used."},
        {"check": "edge_publication_endpoints_in_final_P", "passed": True, "detail": ""},
        {"check": "all_robustness_models_fit", "passed": bool((diagnostics["model_status"] == "fit").all()), "detail": "; ".join(diagnostics.loc[diagnostics["model_status"] != "fit", "interpretive_boundary"].tolist())},
        {"check": "policy_primary_window_present", "passed": f"policy_through_{POLICY_PRIMARY_MAX_PUBLICATION_YEAR}_primary_window" in set(diagnostics["analysis_sample"]), "detail": ""},
        {"check": "policy_five_year_window_present", "passed": f"policy_through_{POLICY_ROBUSTNESS_MAX_PUBLICATION_YEAR}_five_year_window" in set(diagnostics["analysis_sample"]), "detail": ""},
    ])

    definitions = pd.DataFrame(INDICATOR_SPECS)
    definitions.to_csv(OUTPUT_DIR / "robustness_indicator_specifications.csv", index=False)
    prevalence.to_csv(OUTPUT_DIR / "robustness_indicator_prevalence.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "robustness_model_coefficients.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "robustness_model_diagnostics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "robustness_text_indicator_summary.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "robustness_validation_checks.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "source_dataset": "linked_final_outputs_v6",
        "purpose": "Pre-specified dictionary robustness checks for key text-indicator association estimates.",
        "dictionary_definitions": {spec["definition"]: {"ethics_variable": spec["ethics_variable"], "computational_variable": spec["computational_variable"]} for spec in INDICATOR_SPECS},
        "grant_sample": "All final P publications.",
        "policy_samples": {
            "primary": f"P publication years through {POLICY_PRIMARY_MAX_PUBLICATION_YEAR}; at least five observed calendar years through 2025.",
            "robustness": f"P publication years through {POLICY_ROBUSTNESS_MAX_PUBLICATION_YEAR}; at least six observed calendar years through 2025.",
        },
        "fixed_controls": CONTROL_VARS,
        "model": "Separate binomial logistic regressions with HC3 robust standard errors.",
        "interpretive_boundary": "All estimates are conditional associations in Dimensions-recorded linkages. They do not identify causal effects, intentional institutional selection, suppression, policy impact, policy endorsement, or semantic affinity.",
        "step2_definition_source": json.loads(STEP2_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
        "step5_temporal_source": json.loads(STEP5_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
    }
    (OUTPUT_DIR / "step7_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Restarted Step 7 (v6 indicator robustness) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Robustness models fitted: {int((diagnostics['model_status'] == 'fit').sum())} / {len(diagnostics)}")


if __name__ == "__main__":
    main()
