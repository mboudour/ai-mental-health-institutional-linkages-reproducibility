#!/usr/bin/env python3
"""Targeted robustness and supplementary reporting for the manuscript.

Save this file as:
    new_version/computations/step8_targeted_robustness_and_reporting.py

Run from the new_version directory:
    python computations/step8_targeted_robustness_and_reporting.py

Read-only inputs:
    data/linked_final_outputs_v6/publication_anchor_final.pkl
    data/linked_final_outputs_v6/linked_grants_final.pkl
    computations/outputs/step1_v6_audit_v2/edges_publication_grant.csv
    computations/outputs/step1_v6_audit_v2/edges_publication_policy_document.csv
    computations/outputs/step2_v6_indicators/publication_text_indicators.csv
    computations/outputs/step4_v6_linkage_associations/publication_linkage_analysis_dataset.csv

Outputs:
    computations/outputs/step8_targeted_robustness_and_reporting/

This script performs only the following pre-specified targeted checks:
  1. Complete-case (abstract-available) replications of the primary grant and
     policy-linkage association models.
  2. Citation-omitted and flexible-year (cubic B-spline) association models.
  3. Firth penalized-logistic sensitivity models for the two policy eligibility
     samples.
  4. Full coefficient tables, convergence/separation/influence/collinearity
     diagnostics, standardized model-based predicted-probability contrasts,
     and descriptive grant-degree distributions.
  5. Frequency tables for the already approved primary indicator dictionaries;
     this does not estimate post-hoc indicator-subfamily associations.
  6. A one-publication deletion sensitivity for the intensive configuration
     statistic, removing the publication with the largest observed contribution
     to the degree-product statistic.
  7. A descriptive summary of available language metadata.

All models are association models of Dimensions-recorded links. They do not
identify causal effects, funding decisions, institutional intention, policy
impact, endorsement, or topical affinity.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from patsy import dmatrix
    from scipy.special import expit
    from scipy.stats import norm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning, SingularMatrixWarning
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy, pandas, scipy, patsy, and statsmodels. "
        "In your activated conda environment, run:\n"
        "conda install -y numpy pandas scipy patsy statsmodels openpyxl\n"
        "Then rerun the script."
    ) from exc


# ---------------------------------------------------------------------------
# Fixed analytical definitions
# ---------------------------------------------------------------------------
SCRIPT_VERSION = "targeted_robustness_and_reporting_1.0"
RANDOM_SEED = 20260902
N_ALIGNMENT_PERMUTATIONS = 1000
PRIMARY_POLICY_CUTOFF = 2021
ROBUST_POLICY_CUTOFF = 2020
TEXT_VARS = ["ethics_responsibility_primary", "computational_performance_primary"]
FAMILY_VARS = [
    "mental_health_conditions_symptoms",
    "clinical_assessment_diagnosis",
    "treatment_intervention",
    "neurocognitive_affective_processes",
]
BASE_LINEAR_VARS = TEXT_VARS + FAMILY_VARS + ["publication_year_centered", "log1p_times_cited"]
OUTCOME_GRANT = "has_grant_link"
OUTCOME_GRANT_PREPUBLICATION = "has_grant_start_on_or_before_publication_link"
OUTCOME_POLICY = "has_policy_document_link"

ETHICS_GROUPS = {
    "Ethics (general)": {"ethic*"},
    "Fairness and bias": {"fairness", "algorithmic bias", "bias mitigation"},
    "Explainability, interpretability, transparency, and accountability": {
        "explainab*", "interpretab*", "transparen*", "accountab*"
    },
    "Privacy and consent": {"data privacy", "privacy-preserving", "informed consent"},
    "Responsibility, governance, and regulation": {
        "responsible AI", "responsible artificial intelligence", "trustworthy AI",
        "AI governance", "algorithmic governance", "AI regulation", "algorithmic regulation"
    },
    "Equity and rights": {"health equity", "human rights"},
}


# ---------------------------------------------------------------------------
# Paths and basic validation
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "linked_final_outputs_v6"
STEP1_DIR = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2"
STEP2_DIR = PROJECT_ROOT / "computations" / "outputs" / "step2_v6_indicators"
STEP4_DIR = PROJECT_ROOT / "computations" / "outputs" / "step4_v6_linkage_associations"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step8_targeted_robustness_and_reporting"

P_FILE = DATA_DIR / "publication_anchor_final.pkl"
G_FILE = DATA_DIR / "linked_grants_final.pkl"
PG_FILE = STEP1_DIR / "edges_publication_grant.csv"
PD_FILE = STEP1_DIR / "edges_publication_policy_document.csv"
INDICATOR_FILE = STEP2_DIR / "publication_text_indicators.csv"
ANALYSIS_FILE = STEP4_DIR / "publication_linkage_analysis_dataset.csv"


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Required input file(s) missing:\n" + "\n".join(missing))


def require_columns(frame: pd.DataFrame, label: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(missing)}")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def abstract_available(value: Any) -> bool:
    return bool(clean_text(value))


def bool_int(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.notna() & ~values.isin([0, 1])
    if invalid.any():
        raise SystemExit(f"{label} contains values other than 0 or 1.")
    return values.fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Logistic model construction, diagnostics, and standardized contrasts
# ---------------------------------------------------------------------------
def make_design(frame: pd.DataFrame, *, include_citations: bool, year_form: str) -> tuple[pd.DataFrame, list[str]]:
    """Return a design matrix with an intercept and named substantive columns."""
    required = TEXT_VARS + FAMILY_VARS + ["year", "publication_year_centered", "log1p_times_cited"]
    require_columns(frame, "analysis data", required)
    pieces = [frame[TEXT_VARS + FAMILY_VARS].astype(float).reset_index(drop=True)]
    term_names = TEXT_VARS + FAMILY_VARS

    if year_form == "linear":
        pieces.append(frame[["publication_year_centered"]].astype(float).reset_index(drop=True))
        term_names.append("publication_year_centered")
    elif year_form == "spline":
        spline = dmatrix(
            "bs(year, df=4, degree=3, include_intercept=False) - 1",
            {"year": frame["year"].astype(float).to_numpy()},
            return_type="dataframe",
        ).reset_index(drop=True)
        spline.columns = [f"publication_year_spline_{index + 1}" for index in range(spline.shape[1])]
        pieces.append(spline)
        term_names.extend(list(spline.columns))
    else:
        raise ValueError(f"Unknown year form: {year_form}")

    if include_citations:
        pieces.append(frame[["log1p_times_cited"]].astype(float).reset_index(drop=True))
        term_names.append("log1p_times_cited")

    design = pd.concat(pieces, axis=1)
    design = sm.add_constant(design, has_constant="add")
    return design, term_names


def vif_table(design: pd.DataFrame, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [column for column in design.columns if column != "const"]
    for column in columns:
        y = design[column].to_numpy(dtype=float)
        other = design[[name for name in columns if name != column]].to_numpy(dtype=float)
        if other.shape[1] == 0 or np.isclose(y.var(), 0):
            vif = np.nan
        else:
            other = np.column_stack([np.ones(len(other)), other])
            fitted = other @ np.linalg.lstsq(other, y, rcond=None)[0]
            residual_ss = float(np.sum((y - fitted) ** 2))
            total_ss = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1 - residual_ss / total_ss if total_ss > 0 else np.nan
            vif = np.inf if pd.notna(r_squared) and r_squared >= 1 - 1e-12 else (1 / (1 - r_squared) if pd.notna(r_squared) else np.nan)
        rows.append({"model_label": model_label, "term": column, "vif": vif})
    return pd.DataFrame(rows)


def predictor_correlations(design: pd.DataFrame, model_label: str) -> pd.DataFrame:
    columns = [column for column in design.columns if column != "const"]
    correlation = design[columns].corr(method="pearson")
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            rows.append({
                "model_label": model_label,
                "term_1": left,
                "term_2": right,
                "pearson_correlation": float(correlation.loc[left, right]),
                "absolute_correlation": float(abs(correlation.loc[left, right])),
            })
    return pd.DataFrame(rows)


def model_coefficient_table(result: Any, model_label: str, outcome: str, model_family: str, covariance: str) -> pd.DataFrame:
    confidence = result.conf_int()
    table = pd.DataFrame({
        "model_label": model_label,
        "model_family": model_family,
        "outcome": outcome,
        "term": result.params.index,
        "coefficient_log_odds": result.params.to_numpy(dtype=float),
        "standard_error": result.bse.to_numpy(dtype=float),
        "z_value": result.tvalues.to_numpy(dtype=float),
        "p_value": result.pvalues.to_numpy(dtype=float),
        "ci_95_log_odds_low": confidence.iloc[:, 0].to_numpy(dtype=float),
        "ci_95_log_odds_high": confidence.iloc[:, 1].to_numpy(dtype=float),
        "covariance_estimator": covariance,
    })
    table["odds_ratio"] = np.exp(table["coefficient_log_odds"])
    table["ci_95_odds_ratio_low"] = np.exp(table["ci_95_log_odds_low"])
    table["ci_95_odds_ratio_high"] = np.exp(table["ci_95_log_odds_high"])
    return table


def glm_diagnostics(result: Any, design: pd.DataFrame, outcome: str, model_label: str, warnings_seen: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fitted = np.asarray(result.fittedvalues, dtype=float)
    separation = {
        "perfect_separation_warning": any("PerfectSeparation" in item for item in warnings_seen),
        "convergence_warning": any("ConvergenceWarning" in item for item in warnings_seen),
        "fitted_probability_min": float(fitted.min()),
        "fitted_probability_max": float(fitted.max()),
        "near_zero_probability_n": int((fitted < 1e-8).sum()),
        "near_one_probability_n": int((fitted > 1 - 1e-8).sum()),
        "separation_note": "No perfect-separation warning was emitted by the binomial GLM fit; fitted-probability extremes are reported as a screening diagnostic, not as a proof that quasi-separation is absent.",
    }
    try:
        influence = result.get_influence(observed=True)
        cooks = np.asarray(influence.cooks_distance[0], dtype=float)
        dfbetas = np.asarray(influence.dfbetas, dtype=float)
        influence_row = {
            "model_label": model_label,
            "outcome": outcome,
            "max_cooks_distance": float(np.nanmax(cooks)),
            "n_cooks_distance_above_4_over_n": int((cooks > (4 / len(fitted))).sum()),
            "max_abs_dfbetas": float(np.nanmax(np.abs(dfbetas))),
            "influence_note": "Influence diagnostics are reported for the fitted binomial GLM. They are screening diagnostics, not case-deletion effect estimates.",
        }
    except Exception as exc:  # diagnostics must not suppress a valid model fit
        influence_row = {
            "model_label": model_label,
            "outcome": outcome,
            "max_cooks_distance": np.nan,
            "n_cooks_distance_above_4_over_n": np.nan,
            "max_abs_dfbetas": np.nan,
            "influence_note": f"Influence diagnostic unavailable: {type(exc).__name__}: {exc}",
        }
    diagnostics = pd.DataFrame([{
        "model_label": model_label,
        "model_family": "binomial_logistic",
        "outcome": outcome,
        "model_status": "fit",
        "converged": bool(getattr(result, "converged", True)),
        "n_observations": int(result.nobs),
        "n_events": int(np.asarray(result.model.endog).sum()),
        "event_rate_pct": float(100 * np.asarray(result.model.endog).mean()),
        "n_parameters_including_intercept": int(len(result.params)),
        "aic": float(result.aic),
        "log_likelihood": float(result.llf),
        "pseudo_r2_mcfadden": float(1 - result.llf / result.llnull) if result.llnull else np.nan,
        "covariance_estimator": "HC3 robust",
        **separation,
    }])
    return diagnostics, pd.DataFrame([influence_row]), vif_table(design, model_label)


def standardized_probability_contrasts(result: Any, design: pd.DataFrame, model_label: str, outcome: str) -> pd.DataFrame:
    rows = []
    for term in TEXT_VARS:
        if term not in design.columns:
            continue
        zero = design.copy()
        one = design.copy()
        zero[term] = 0.0
        one[term] = 1.0
        predicted_zero = np.asarray(result.predict(zero), dtype=float)
        predicted_one = np.asarray(result.predict(one), dtype=float)
        rows.append({
            "model_label": model_label,
            "outcome": outcome,
            "indicator": term,
            "standardized_predicted_probability_indicator_0": float(predicted_zero.mean()),
            "standardized_predicted_probability_indicator_1": float(predicted_one.mean()),
            "standardized_probability_difference": float(predicted_one.mean() - predicted_zero.mean()),
            "interpretation": "Model-based standardized contrast obtained by setting one indicator to 0 and 1 for all observations while retaining all other observed covariates. It is not a causal effect.",
        })
    return pd.DataFrame(rows)


def fit_glm(frame: pd.DataFrame, outcome: str, model_label: str, *, include_citations: bool, year_form: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    required = [outcome] + TEXT_VARS + FAMILY_VARS + ["year", "publication_year_centered", "log1p_times_cited"]
    model_frame = frame[required].dropna().reset_index(drop=True).copy()
    y = model_frame[outcome].astype(float)
    if y.nunique() < 2:
        raise ValueError(f"{model_label}: outcome has fewer than two observed values.")
    design, _ = make_design(model_frame, include_citations=include_citations, year_form=year_form)
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        raise ValueError(f"{model_label}: design matrix is rank deficient.")
    warnings_seen: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type="HC3")
        warnings_seen = [warning.category.__name__ for warning in caught]
    coefficients = model_coefficient_table(result, model_label, outcome, "binomial_logistic", "HC3 robust")
    diagnostics, influence, vif = glm_diagnostics(result, design, outcome, model_label, warnings_seen)
    correlations = predictor_correlations(design, model_label)
    contrasts = standardized_probability_contrasts(result, design, model_label, outcome)
    return coefficients, diagnostics, influence, vif, correlations, contrasts, result, design


# ---------------------------------------------------------------------------
# Firth penalized logistic regression (policy-sample sensitivity only)
# ---------------------------------------------------------------------------
def firth_penalized_loglikelihood(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    eta = np.clip(x @ beta, -35, 35)
    mu = expit(eta)
    w = np.clip(mu * (1 - mu), 1e-12, None)
    information = x.T @ (w[:, None] * x)
    sign, log_det = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(log_det):
        return -np.inf
    loglikelihood = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return loglikelihood + 0.5 * float(log_det)


def fit_firth(x: np.ndarray, y: np.ndarray, *, max_iter: int = 2000, tolerance: float = 1e-9) -> tuple[np.ndarray, np.ndarray, int, bool, float]:
    if y.min() == y.max():
        raise ValueError("Firth fit requires both outcome classes.")
    beta = np.zeros(x.shape[1], dtype=float)
    mean_y = float(y.mean())
    beta[0] = np.log(mean_y / (1 - mean_y))
    current = firth_penalized_loglikelihood(beta, x, y)
    if not np.isfinite(current):
        raise ValueError("Initial Firth penalized log-likelihood is not finite.")

    converged = False
    for iteration in range(1, max_iter + 1):
        eta = np.clip(x @ beta, -35, 35)
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        information = x.T @ (w[:, None] * x)
        if np.linalg.matrix_rank(information) < information.shape[0]:
            raise ValueError("Firth information matrix is singular.")
        information_inverse = np.linalg.inv(information)
        leverage = w * np.einsum("ij,jk,ik->i", x, information_inverse, x)
        adjusted_score = x.T @ (y - mu + leverage * (0.5 - mu))
        step = np.linalg.solve(information, adjusted_score)

        multiplier = 1.0
        proposed = beta + step
        proposed_value = firth_penalized_loglikelihood(proposed, x, y)
        while (not np.isfinite(proposed_value) or proposed_value < current) and multiplier > 2.0 ** -30:
            multiplier *= 0.5
            proposed = beta + multiplier * step
            proposed_value = firth_penalized_loglikelihood(proposed, x, y)
        if not np.isfinite(proposed_value) or proposed_value < current:
            raise RuntimeError("Firth step-halving failed to improve the penalized likelihood.")
        beta = proposed
        if np.max(np.abs(multiplier * step)) < tolerance:
            converged = True
            current = proposed_value
            break
        current = proposed_value

    eta = np.clip(x @ beta, -35, 35)
    mu = expit(eta)
    w = np.clip(mu * (1 - mu), 1e-12, None)
    information = x.T @ (w[:, None] * x)
    covariance = np.linalg.inv(information)
    return beta, covariance, iteration, converged, current


def fit_firth_policy(frame: pd.DataFrame, model_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [OUTCOME_POLICY] + TEXT_VARS + FAMILY_VARS + ["year", "publication_year_centered", "log1p_times_cited"]
    model_frame = frame[required].dropna().reset_index(drop=True).copy()
    y = model_frame[OUTCOME_POLICY].to_numpy(dtype=float)
    design, _ = make_design(model_frame, include_citations=True, year_form="linear")
    x = design.to_numpy(dtype=float)
    beta, covariance, iterations, converged, penalized_ll = fit_firth(x, y)
    se = np.sqrt(np.diag(covariance))
    z_values = beta / se
    p_values = 2 * norm.sf(np.abs(z_values))
    confidence_low = beta - 1.959963984540054 * se
    confidence_high = beta + 1.959963984540054 * se
    coefficients = pd.DataFrame({
        "model_label": model_label,
        "model_family": "firth_penalized_logistic",
        "outcome": OUTCOME_POLICY,
        "term": design.columns,
        "coefficient_log_odds": beta,
        "standard_error": se,
        "z_value": z_values,
        "p_value": p_values,
        "ci_95_log_odds_low": confidence_low,
        "ci_95_log_odds_high": confidence_high,
        "covariance_estimator": "inverse expected Fisher information; model-based Wald sensitivity",
    })
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient_log_odds"])
    coefficients["ci_95_odds_ratio_low"] = np.exp(coefficients["ci_95_log_odds_low"])
    coefficients["ci_95_odds_ratio_high"] = np.exp(coefficients["ci_95_log_odds_high"])
    diagnostics = pd.DataFrame([{
        "model_label": model_label,
        "model_family": "firth_penalized_logistic",
        "outcome": OUTCOME_POLICY,
        "model_status": "fit" if converged else "iteration_limit_reached",
        "converged": bool(converged),
        "iterations": int(iterations),
        "n_observations": int(len(y)),
        "n_events": int(y.sum()),
        "event_rate_pct": float(100 * y.mean()),
        "n_parameters_including_intercept": int(x.shape[1]),
        "penalized_log_likelihood": float(penalized_ll),
        "covariance_estimator": "inverse expected Fisher information; model-based Wald sensitivity",
        "interpretation": "Firth penalized likelihood is a rare-outcome sensitivity. Its standard errors and Wald intervals are model-based and are not HC3 robust standard errors.",
    }])
    return coefficients, diagnostics


# ---------------------------------------------------------------------------
# Supplementary descriptive outputs
# ---------------------------------------------------------------------------
def dictionary_frequencies(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_columns = {
        "ethics_responsibility_primary": [
            "ethics_responsibility_primary_matched_terms",
            "ethics_responsibility_primary_title_terms",
            "ethics_responsibility_primary_abstract_terms",
        ],
        "computational_performance_primary": [
            "computational_performance_primary_matched_terms",
            "computational_performance_primary_title_terms",
            "computational_performance_primary_abstract_terms",
        ],
    }
    term_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for indicator, columns in term_columns.items():
        if not all(column in indicators.columns for column in columns):
            raise SystemExit(f"Indicator table lacks matched-term fields for {indicator}.")
        all_terms = indicators[columns[0]].fillna("").map(lambda value: {item.strip() for item in str(value).split(";") if item.strip()})
        title_terms = indicators[columns[1]].fillna("").map(lambda value: {item.strip() for item in str(value).split(";") if item.strip()})
        abstract_terms = indicators[columns[2]].fillna("").map(lambda value: {item.strip() for item in str(value).split(";") if item.strip()})
        observed_terms = sorted(set().union(*all_terms.tolist())) if len(all_terms) else []
        for term in observed_terms:
            term_rows.append({
                "indicator": indicator,
                "term": term,
                "n_publications_any_text": int(all_terms.map(lambda terms: term in terms).sum()),
                "n_publications_title": int(title_terms.map(lambda terms: term in terms).sum()),
                "n_publications_abstract": int(abstract_terms.map(lambda terms: term in terms).sum()),
                "pct_of_publications_any_text": float(100 * all_terms.map(lambda terms: term in terms).mean()),
            })
    for group, terms in ETHICS_GROUPS.items():
        matches = all_terms if False else None  # populated below from ethics data only
        ethics_terms = indicators["ethics_responsibility_primary_matched_terms"].fillna("").map(
            lambda value: {item.strip() for item in str(value).split(";") if item.strip()}
        )
        matched = ethics_terms.map(lambda observed: bool(observed & terms))
        group_rows.append({
            "indicator": "ethics_responsibility_primary",
            "dictionary_subfamily": group,
            "component_terms": "; ".join(sorted(terms)),
            "n_publications_with_at_least_one_component_term": int(matched.sum()),
            "pct_of_publications": float(100 * matched.mean()),
            "purpose": "Descriptive composition of the primary composite indicator; not an association model.",
        })
    return pd.DataFrame(term_rows), pd.DataFrame(group_rows)


def grant_degree_distribution(analysis: pd.DataFrame) -> pd.DataFrame:
    require_columns(analysis, "analysis data", ["n_linked_grants"])
    counts = analysis["n_linked_grants"].value_counts().sort_index()
    output = pd.DataFrame({"n_linked_grants": counts.index.astype(int), "n_publications": counts.values.astype(int)})
    output["share_of_all_publications"] = output["n_publications"] / len(analysis)
    output["share_among_grant_linked_publications"] = np.where(
        output["n_linked_grants"] > 0,
        output["n_publications"] / max(1, int((analysis["n_linked_grants"] > 0).sum())),
        np.nan,
    )
    return output


def language_metadata_summary(publications: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["language", "language_title"]:
        if column not in publications.columns:
            rows.append({"field": column, "status": "not_returned", "value": None, "n_publications": None, "note": "No language metadata field was available in the publication file."})
            continue
        values = publications[column].map(clean_text)
        rows.append({"field": column, "status": "available", "value": "nonmissing", "n_publications": int(values.ne("").sum()), "note": "Metadata availability only; no language-based exclusion was applied by this script."})
        counts = values[values.ne("")].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"field": column, "status": "available", "value": value, "n_publications": int(count), "note": "As returned in Dimensions metadata."})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Configuration hub-deletion sensitivity
# ---------------------------------------------------------------------------
def alignment_statistics(publications: pd.DataFrame, pg_edges: pd.DataFrame, pd_edges: pd.DataFrame, *, n_permutations: int, seed: int) -> tuple[dict[str, float], pd.DataFrame, dict[str, int]]:
    p_ids = publications["publication_id"].astype(str).to_numpy()
    p_index = {value: index for index, value in enumerate(p_ids)}
    years = publications["year"].to_numpy(dtype=int)
    pg_indices = pg_edges["publication_id"].map(p_index).to_numpy(dtype=int)
    pd_indices = pd_edges["publication_id"].map(p_index).to_numpy(dtype=int)
    pg_degree = np.bincount(pg_indices, minlength=len(publications))
    pd_degree = np.bincount(pd_indices, minlength=len(publications))
    observed = {
        "N_both": float(((pg_degree > 0) & (pd_degree > 0)).sum()),
        "intensive_degree_product_sum": float(np.sum(pg_degree * pd_degree)),
    }
    contribution = pd.DataFrame({
        "publication_id": p_ids,
        "year": years,
        "P_G_degree": pg_degree,
        "P_D_degree": pd_degree,
        "degree_product_contribution": pg_degree * pd_degree,
    })
    rng = np.random.default_rng(seed)
    null_rows = []
    fixed_strata = 0
    permuted_strata = 0
    for _ in range(n_permutations):
        targets = np.arange(len(publications), dtype=int)
        local_fixed = 0
        local_permuted = 0
        for year in sorted(pd.unique(years)):
            indices = np.flatnonzero(years == year)
            if len(indices) < 2:
                local_fixed += 1
            else:
                targets[indices] = rng.permutation(indices)
                local_permuted += 1
        if not null_rows:
            fixed_strata = local_fixed
            permuted_strata = local_permuted
        permuted_pd_degree = np.bincount(targets[pd_indices], minlength=len(publications))
        null_rows.append({
            "N_both": float(((pg_degree > 0) & (permuted_pd_degree > 0)).sum()),
            "intensive_degree_product_sum": float(np.sum(pg_degree * permuted_pd_degree)),
        })
    null_frame = pd.DataFrame(null_rows)
    return observed, contribution, {"fixed_strata": fixed_strata, "permuted_strata": permuted_strata, "n_permutations": n_permutations, "null_N_both_mean": float(null_frame["N_both"].mean()), "null_intensive_mean": float(null_frame["intensive_degree_product_sum"].mean()), "N_both_two_sided_p": two_sided_permutation_p(observed["N_both"], null_frame["N_both"].to_numpy()), "intensive_two_sided_p": two_sided_permutation_p(observed["intensive_degree_product_sum"], null_frame["intensive_degree_product_sum"].to_numpy())}


def two_sided_permutation_p(observed: float, null: np.ndarray) -> float:
    upper = (1 + int((null >= observed).sum())) / (1 + len(null))
    lower = (1 + int((null <= observed).sum())) / (1 + len(null))
    return float(min(1.0, 2 * min(upper, lower)))


def hub_deletion_sensitivity(publications: pd.DataFrame, pg_edges: pd.DataFrame, pd_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed, contributions, full_meta = alignment_statistics(
        publications, pg_edges, pd_edges, n_permutations=N_ALIGNMENT_PERMUTATIONS, seed=RANDOM_SEED
    )
    candidates = contributions[contributions["degree_product_contribution"] > 0].sort_values(
        ["degree_product_contribution", "P_G_degree", "P_D_degree", "publication_id"],
        ascending=[False, False, False, True],
    )
    if candidates.empty:
        raise SystemExit("No publication contributes to the intensive degree-product statistic.")
    hub = candidates.iloc[0]
    deleted_publications = publications.loc[publications["publication_id"] != hub["publication_id"]].copy()
    deleted_pg = pg_edges.loc[pg_edges["publication_id"] != hub["publication_id"]].copy()
    deleted_pd = pd_edges.loc[pd_edges["publication_id"] != hub["publication_id"]].copy()
    deleted_observed, _, deleted_meta = alignment_statistics(
        deleted_publications, deleted_pg, deleted_pd, n_permutations=N_ALIGNMENT_PERMUTATIONS, seed=RANDOM_SEED
    )
    rows = []
    for statistic in ["N_both", "intensive_degree_product_sum"]:
        key = "null_N_both_mean" if statistic == "N_both" else "null_intensive_mean"
        p_key = "N_both_two_sided_p" if statistic == "N_both" else "intensive_two_sided_p"
        rows.append({
            "analysis": "full_system",
            "removed_publication_id": None,
            "statistic": statistic,
            "observed": observed[statistic],
            "null_mean": full_meta[key],
            "enrichment": observed[statistic] / full_meta[key],
            "two_sided_permutation_p": full_meta[p_key],
            "n_permutations": full_meta["n_permutations"],
            "fixed_singleton_year_strata": full_meta["fixed_strata"],
            "permuted_year_strata": full_meta["permuted_strata"],
        })
        rows.append({
            "analysis": "highest_contributor_removed",
            "removed_publication_id": hub["publication_id"],
            "statistic": statistic,
            "observed": deleted_observed[statistic],
            "null_mean": deleted_meta[key],
            "enrichment": deleted_observed[statistic] / deleted_meta[key],
            "two_sided_permutation_p": deleted_meta[p_key],
            "n_permutations": deleted_meta["n_permutations"],
            "fixed_singleton_year_strata": deleted_meta["fixed_strata"],
            "permuted_year_strata": deleted_meta["permuted_strata"],
        })
    hub_details = pd.DataFrame([{
        "removed_publication_id": hub["publication_id"],
        "publication_year": int(hub["year"]),
        "P_G_degree": int(hub["P_G_degree"]),
        "P_D_degree": int(hub["P_D_degree"]),
        "degree_product_contribution": int(hub["degree_product_contribution"]),
        "selection_rule": "Publication with the maximum observed P-G degree times P-D degree; ties resolved by P-G degree, P-D degree, then publication ID.",
    }])
    return pd.DataFrame(rows), hub_details


# ---------------------------------------------------------------------------
# Workbook helper
# ---------------------------------------------------------------------------
def write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, table in tables.items():
                table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except Exception as exc:
        raise SystemExit(f"Could not create Excel workbook {path}: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    require_files([P_FILE, G_FILE, PG_FILE, PD_FILE, INDICATOR_FILE, ANALYSIS_FILE])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    publications = pd.read_pickle(P_FILE).copy()
    grants = pd.read_pickle(G_FILE).copy()
    pg_edges = pd.read_csv(PG_FILE, dtype=str).drop_duplicates().copy()
    pd_edges = pd.read_csv(PD_FILE, dtype=str).drop_duplicates().copy()
    indicators = pd.read_csv(INDICATOR_FILE, dtype={"publication_id": str}).copy()
    analysis = pd.read_csv(ANALYSIS_FILE, dtype={"publication_id": str}).copy()

    require_columns(publications, "publication file", ["id", "year", "abstract", "retain_in_final_P"])
    require_columns(grants, "grant file", ["id", "start_year"])
    require_columns(pg_edges, "P-G edge list", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P-D edge list", ["publication_id", "policy_document_id"])
    require_columns(indicators, "indicator file", ["publication_id"] + TEXT_VARS)
    require_columns(analysis, "analysis file", ["publication_id", "year", "n_linked_grants", OUTCOME_GRANT, OUTCOME_POLICY] + TEXT_VARS + FAMILY_VARS + ["publication_year_centered", "log1p_times_cited"])

    publications["id"] = publications["id"].astype(str).str.strip()
    grants["id"] = grants["id"].astype(str).str.strip()
    for frame, column in [(pg_edges, "publication_id"), (pg_edges, "grant_id"), (pd_edges, "publication_id"), (pd_edges, "policy_document_id"), (indicators, "publication_id"), (analysis, "publication_id")]:
        frame[column] = frame[column].astype(str).str.strip()
    if not publications["retain_in_final_P"].astype(bool).all():
        raise SystemExit("The publication file contains a record outside the approved publication anchor.")
    if publications["id"].duplicated().any() or grants["id"].duplicated().any() or analysis["publication_id"].duplicated().any():
        raise SystemExit("Publication, grant, or analytic IDs are not unique.")
    if set(analysis["publication_id"]) != set(publications["id"]):
        raise SystemExit("The analytic file does not match the publication anchor by identifier.")

    publication_metadata = publications[["id", "year", "abstract"] + [column for column in ["language", "language_title"] if column in publications.columns]].copy()
    publication_metadata = publication_metadata.rename(columns={"id": "publication_id", "year": "publication_year_raw"})
    publication_metadata["abstract_available"] = publication_metadata["abstract"].map(abstract_available).astype(int)
    analysis = analysis.merge(
        publication_metadata.drop(columns=["abstract"]), on="publication_id", how="left", validate="one_to_one"
    )
    if analysis["abstract_available"].isna().any():
        raise SystemExit("Abstract-availability merge failed for one or more anchor publications.")
    analysis["abstract_available"] = analysis["abstract_available"].astype(int)
    analysis["year"] = pd.to_numeric(analysis["year"], errors="coerce")
    if analysis["year"].isna().any():
        raise SystemExit("Publication year is missing or non-numeric in the analytic file.")
    analysis["year"] = analysis["year"].astype(int)
    for column in TEXT_VARS + FAMILY_VARS + [OUTCOME_GRANT, OUTCOME_POLICY]:
        analysis[column] = bool_int(analysis[column], column)

    # Reconstruct the secondary, temporally ordered grant-link outcome.
    p_year = analysis[["publication_id", "year"]].rename(columns={"year": "publication_year"})
    g_year = grants[["id", "start_year"]].rename(columns={"id": "grant_id", "start_year": "grant_start_year"})
    g_year["grant_start_year"] = pd.to_numeric(g_year["grant_start_year"], errors="coerce")
    grant_timing = pg_edges.merge(p_year, on="publication_id", how="left", validate="many_to_one").merge(g_year, on="grant_id", how="left", validate="many_to_one")
    if grant_timing[["publication_year", "grant_start_year"]].isna().any(axis=1).any():
        raise SystemExit("Cannot construct grant timing sensitivity because a P-G tie lacks a publication or grant-start year.")
    eligible_grant_ids = set(grant_timing.loc[grant_timing["grant_start_year"] <= grant_timing["publication_year"], "publication_id"])
    analysis[OUTCOME_GRANT_PREPUBLICATION] = analysis["publication_id"].isin(eligible_grant_ids).astype(int)

    # Samples and model specifications. The complete-case criterion is abstract
    # availability, not a language exclusion.
    samples = {
        "grant_primary": analysis.copy(),
        "grant_abstract_available": analysis.loc[analysis["abstract_available"] == 1].copy(),
        "grant_start_on_or_before_publication": analysis.copy(),
        "policy_through_2021_primary": analysis.loc[analysis["year"] <= PRIMARY_POLICY_CUTOFF].copy(),
        "policy_through_2020_robustness": analysis.loc[analysis["year"] <= ROBUST_POLICY_CUTOFF].copy(),
        "policy_through_2021_abstract_available": analysis.loc[(analysis["year"] <= PRIMARY_POLICY_CUTOFF) & (analysis["abstract_available"] == 1)].copy(),
        "policy_through_2020_abstract_available": analysis.loc[(analysis["year"] <= ROBUST_POLICY_CUTOFF) & (analysis["abstract_available"] == 1)].copy(),
    }
    specifications = [
        ("grant_primary", "grant_primary", OUTCOME_GRANT, True, "linear"),
        ("grant_complete_case", "grant_abstract_available", OUTCOME_GRANT, True, "linear"),
        ("grant_no_citation_adjustment", "grant_primary", OUTCOME_GRANT, False, "linear"),
        ("grant_flexible_year", "grant_primary", OUTCOME_GRANT, True, "spline"),
        ("grant_start_on_or_before_publication", "grant_start_on_or_before_publication", OUTCOME_GRANT_PREPUBLICATION, True, "linear"),
        ("policy_2021_primary", "policy_through_2021_primary", OUTCOME_POLICY, True, "linear"),
        ("policy_2020_robustness", "policy_through_2020_robustness", OUTCOME_POLICY, True, "linear"),
        ("policy_2021_complete_case", "policy_through_2021_abstract_available", OUTCOME_POLICY, True, "linear"),
        ("policy_2020_complete_case", "policy_through_2020_abstract_available", OUTCOME_POLICY, True, "linear"),
        ("policy_2021_no_citation_adjustment", "policy_through_2021_primary", OUTCOME_POLICY, False, "linear"),
        ("policy_2020_no_citation_adjustment", "policy_through_2020_robustness", OUTCOME_POLICY, False, "linear"),
        ("policy_2021_flexible_year", "policy_through_2021_primary", OUTCOME_POLICY, True, "spline"),
        ("policy_2020_flexible_year", "policy_through_2020_robustness", OUTCOME_POLICY, True, "spline"),
    ]

    coefficient_tables: list[pd.DataFrame] = []
    diagnostic_tables: list[pd.DataFrame] = []
    influence_tables: list[pd.DataFrame] = []
    vif_tables: list[pd.DataFrame] = []
    correlation_tables: list[pd.DataFrame] = []
    contrast_tables: list[pd.DataFrame] = []
    sample_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, tuple[Any, pd.DataFrame, str]] = {}

    for model_label, sample_label, outcome, include_citations, year_form in specifications:
        frame = samples[sample_label]
        sample_rows.append({
            "model_label": model_label,
            "sample_label": sample_label,
            "outcome": outcome,
            "n_publications_before_complete_model_rows": int(len(frame)),
            "n_events_before_complete_model_rows": int(frame[outcome].sum()),
            "include_citation_adjustment": include_citations,
            "publication_year_functional_form": year_form,
            "abstract_available_only": bool("abstract_available" in sample_label),
            "interpretation": "Association model of a Dimensions-recorded linkage outcome.",
        })
        coefficients, diagnostics, influence, vif, correlations, contrasts, result, design = fit_glm(
            frame, outcome, model_label, include_citations=include_citations, year_form=year_form
        )
        coefficient_tables.append(coefficients)
        diagnostic_tables.append(diagnostics)
        influence_tables.append(influence)
        vif_tables.append(vif)
        correlation_tables.append(correlations)
        contrast_tables.append(contrasts)
        fitted_models[model_label] = (result, design, outcome)

    # Firth penalized-logistic policy sensitivity uses the original linear,
    # citation-adjusted covariate specification.
    firth_coefficients: list[pd.DataFrame] = []
    firth_diagnostics: list[pd.DataFrame] = []
    for model_label, sample_label in [
        ("policy_2021_firth", "policy_through_2021_primary"),
        ("policy_2020_firth", "policy_through_2020_robustness"),
    ]:
        coefficients, diagnostics = fit_firth_policy(samples[sample_label], model_label)
        firth_coefficients.append(coefficients)
        firth_diagnostics.append(diagnostics)

    term_frequency, ethics_subfamily_frequency = dictionary_frequencies(indicators)
    degrees = grant_degree_distribution(analysis)
    language_summary = language_metadata_summary(publications)

    configuration_publications = analysis[["publication_id", "year"]].copy()
    hub_sensitivity, hub_details = hub_deletion_sensitivity(configuration_publications, pg_edges, pd_edges)

    coefficients_all = pd.concat(coefficient_tables, ignore_index=True)
    diagnostics_all = pd.concat(diagnostic_tables, ignore_index=True)
    influence_all = pd.concat(influence_tables, ignore_index=True)
    vif_all = pd.concat(vif_tables, ignore_index=True)
    correlations_all = pd.concat(correlation_tables, ignore_index=True)
    contrasts_all = pd.concat(contrast_tables, ignore_index=True)
    firth_coefficients_all = pd.concat(firth_coefficients, ignore_index=True)
    firth_diagnostics_all = pd.concat(firth_diagnostics, ignore_index=True)
    samples_table = pd.DataFrame(sample_rows)

    # Write machine-readable and spreadsheet-ready outputs.
    tables = {
        "robustness_model_coefficients": coefficients_all,
        "firth_policy_coefficients": firth_coefficients_all,
        "model_diagnostics": diagnostics_all,
        "firth_policy_diagnostics": firth_diagnostics_all,
        "influence_diagnostics": influence_all,
        "variance_inflation_factors": vif_all,
        "predictor_correlations": correlations_all,
        "standardized_probability_contrasts": contrasts_all,
        "model_samples": samples_table,
        "grant_degree_distribution": degrees,
        "dictionary_term_frequencies": term_frequency,
        "ethics_dictionary_subfamily_frequencies": ethics_subfamily_frequency,
        "language_metadata_summary": language_summary,
        "alignment_hub_deletion_sensitivity": hub_sensitivity,
        "alignment_hub_deletion_details": hub_details,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    write_workbook(OUTPUT_DIR / "targeted_robustness_and_reporting.xlsx", tables)

    validation = pd.DataFrame([
        {"check": "publication_anchor_unique_ids", "passed": not publications["id"].duplicated().any(), "detail": f"{len(publications)} publications"},
        {"check": "analysis_matches_publication_anchor", "passed": set(analysis["publication_id"]) == set(publications["id"]), "detail": "Identifier sets checked exactly"},
        {"check": "all_glm_models_converged", "passed": bool(diagnostics_all["converged"].all()), "detail": "See model_diagnostics.csv"},
        {"check": "all_firth_models_converged", "passed": bool(firth_diagnostics_all["converged"].all()), "detail": "See firth_policy_diagnostics.csv"},
        {"check": "hub_deletion_sensitivity_completed", "passed": len(hub_sensitivity) == 4, "detail": "Full-system and highest-contributor-removed statistics for two primary alignment measures"},
    ])
    validation.to_csv(OUTPUT_DIR / "validation_checks.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_cutoffs": {"primary": PRIMARY_POLICY_CUTOFF, "robustness": ROBUST_POLICY_CUTOFF},
        "focal_indicators": TEXT_VARS,
        "supplementary_topic_families": FAMILY_VARS,
        "targeted_robustness": [
            "abstract-available complete-case models",
            "citation-omitted models",
            "cubic B-spline publication-year models",
            "Firth penalized-logistic policy models",
            "grant-start-on-or-before-publication secondary outcome",
            "highest-contributor deletion sensitivity for configuration alignment",
        ],
        "not_conducted": [
            "post-hoc indicator-subfamily association models",
            "event-history or survival analysis",
            "grant-count model conditional on having a grant link",
            "new manual coding validation study",
        ],
        "permutation_details": {
            "n_permutations": N_ALIGNMENT_PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "scheme": "one-to-one P-end permutation of complete P-D incidence profiles within publication-year strata; singleton strata fixed",
            "p_value_formula": "two-sided p = min(1, 2 * min((1 + count(null >= observed))/(1 + B), (1 + count(null <= observed))/(1 + B)))",
        },
        "interpretive_boundary": "All outputs concern Dimensions-recorded relations and conditional associations. They do not establish causal effects, institutional intentions, funding decisions, policy impact, endorsement, or topical affinity.",
    }
    (OUTPUT_DIR / "robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Targeted robustness and reporting analysis completed.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print("Primary files to inspect:")
    for name in [
        "robustness_model_coefficients.csv",
        "firth_policy_coefficients.csv",
        "model_diagnostics.csv",
        "influence_diagnostics.csv",
        "standardized_probability_contrasts.csv",
        "alignment_hub_deletion_sensitivity.csv",
        "targeted_robustness_and_reporting.xlsx",
        "validation_checks.csv",
    ]:
        print(f"  {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
