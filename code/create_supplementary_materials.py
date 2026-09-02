#!/usr/bin/env python3
"""Create LaTeX supplementary materials from verified uploaded robustness outputs."""

from pathlib import Path
import math
import re

import pandas as pd

ROOT = Path("/home/ubuntu")
UP = ROOT / "upload"
OUT = ROOT / "manuscript" / "supplementary_materials.tex"

TERM_LABELS = {
    "const": "Intercept",
    "ethics_responsibility_primary": "Ethics/responsibility indicator",
    "computational_performance_primary": "Computational-performance indicator",
    "mental_health_conditions_symptoms": "Conditions/symptoms",
    "clinical_assessment_diagnosis": "Assessment/diagnosis",
    "treatment_intervention": "Treatment/intervention",
    "neurocognitive_affective_processes": "Neurocognitive/affective",
    "publication_year_centered": "Publication year (centered)",
    "log1p_times_cited": "Log(1 + citations)",
}
MODEL_LABELS = {
    "grant_primary": "Grant linkage: primary",
    "grant_complete_case": "Grant linkage: abstract available",
    "grant_no_citation_adjustment": "Grant linkage: no citation adjustment",
    "grant_flexible_year": "Grant linkage: flexible year",
    "grant_start_on_or_before_publication": "Grant linkage: start year $\\leq$ publication year",
    "policy_2021_primary": "Policy linkage: through 2021",
    "policy_2020_robustness": "Policy linkage: through 2020",
    "policy_2021_complete_case": "Policy linkage: through 2021, abstract available",
    "policy_2020_complete_case": "Policy linkage: through 2020, abstract available",
    "policy_2021_no_citation_adjustment": "Policy linkage: through 2021, no citations",
    "policy_2020_no_citation_adjustment": "Policy linkage: through 2020, no citations",
    "policy_2021_flexible_year": "Policy linkage: through 2021, flexible year",
    "policy_2020_flexible_year": "Policy linkage: through 2020, flexible year",
    "policy_2021_firth": "Policy linkage: through 2021, Firth",
    "policy_2020_firth": "Policy linkage: through 2020, Firth",
}


def latex_escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", r"\\textbackslash{}").replace("&", r"\\&").replace("%", r"\\%").replace("_", r"\\_").replace("#", r"\\#")


def p_text(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    if value < 0.001:
        return "$<.001$"
    return f"{float(value):.3f}".lstrip("0")


def number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def or_ci(row: pd.Series) -> str:
    return f"{number(row['odds_ratio'])} ({number(row['ci_95_odds_ratio_low'])}--{number(row['ci_95_odds_ratio_high'])})"


def longtable_header(columns: str, caption: str, label: str, header: str) -> list[str]:
    return [
        r"\\begin{longtable}{" + columns + "}",
        r"\\caption{" + caption + r"}\\label{" + label + r"}\\\\",
        r"\\toprule",
        header + r" \\",
        r"\\midrule",
        r"\\endfirsthead",
        r"\\multicolumn{" + str(header.count("&") + 1) + r"}{l}{\\small\\itshape Continued from previous page} \\",
        r"\\toprule",
        header + r" \\",
        r"\\midrule",
        r"\\endhead",
        r"\\midrule",
        r"\\multicolumn{" + str(header.count("&") + 1) + r"}{r}{\\small\\itshape Continued on next page} \\",
        r"\\endfoot",
        r"\\bottomrule",
        r"\\endlastfoot",
    ]


def longtable_footer() -> list[str]:
    return [r"\\end{longtable}", ""]


def normalize_latex_commands(lines: list[str]) -> list[str]:
    """Correct doubled command backslashes while preserving LaTeX row breaks.

    Earlier generator strings used raw literals with an extra escape level. This
    normalizer reduces doubled command backslashes in the emitted source, but
    leaves a trailing ``\\\\`` row break intact.
    """
    single = chr(92)
    double = single * 2
    normalized: list[str] = []
    for line in lines:
        trailing_row_break = ""
        # Generator strings may end with two backslashes (ordinary table row)
        # or four (a command string followed by a table row break). Remove all
        # trailing pairs before normalizing commands, then restore one row break.
        trailing_pairs = 0
        while line.endswith(double):
            line = line[:-2]
            trailing_pairs += 1
        if trailing_pairs:
            trailing_row_break = double
        while double in line:
            line = line.replace(double, single)
        line = line.replace("@@LINEBREAK@@", double)
        normalized.append(line + trailing_row_break)
    return normalized


coeff = pd.read_csv(UP / "robustness_model_coefficients.csv")
firth = pd.read_csv(UP / "firth_policy_coefficients.csv")
diagnostics = pd.read_csv(UP / "model_diagnostics.csv")
firth_diagnostics = pd.read_csv(UP / "firth_policy_diagnostics.csv")
samples = pd.read_csv(UP / "model_samples.csv")
contrasts = pd.read_csv(UP / "standardized_probability_contrasts.csv")
vif = pd.read_csv(UP / "variance_inflation_factors.csv")
correlations = pd.read_csv(UP / "predictor_correlations.csv")
influence = pd.read_csv(UP / "influence_diagnostics.csv")
term_frequency = pd.read_csv(UP / "dictionary_term_frequencies.csv")
subfamilies = pd.read_csv(UP / "ethics_dictionary_subfamily_frequencies.csv")
grant_degree = pd.read_csv(UP / "grant_degree_distribution.csv")
hub = pd.read_csv(UP / "alignment_hub_deletion_sensitivity.csv")
hub_detail = pd.read_csv(UP / "alignment_hub_deletion_details.csv")
language = pd.read_csv(UP / "language_metadata_summary.csv")
screening_summary = pd.read_csv(UP / "overall_summary.csv")
screening_checks = pd.read_csv(UP / "validation_checks.csv")

lines: list[str] = [
    r"\\documentclass[11pt]{article}",
    r"\\usepackage[margin=1in]{geometry}",
    r"\\usepackage{booktabs,longtable,array}",
    r"\\usepackage[T1]{fontenc}",
    r"\\usepackage[utf8]{inputenc}",
    r"\\usepackage{amsmath}",
    r"\\usepackage{caption}",
    r"\\setlength{\\LTleft}{0pt}",
    r"\\setlength{\\LTright}{0pt}",
    r"\\begin{document}",
    r"\\begin{center}",
    r"{\\Large Supplementary Materials}@@LINEBREAK@@[0.5em]",
    r"{\\large Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research}",
    r"\\end{center}",
    "",
    r"\\section*{Supplementary Methods}",
    r"The primary association models are binomial logistic regressions with HC3 robust standard errors. Their covariates are the two primary publication-language indicators, four non-exclusive supplementary topic-family descriptors, centered publication year, and $\\log(1+\\text{times cited})$. Policy-linkage models are restricted to publications through 2021 for the primary eligibility analysis and through 2020 for the more restrictive robustness analysis. The complete-case analyses retain records with an available abstract. Citation-omitted and flexible-year specifications are sensitivity analyses that estimate different conditional associations from the primary model. Flexible year is represented by a cubic B-spline with four degrees of freedom.",
    "",
    r"Firth penalized-logistic models repeat the primary linear-year, citation-adjusted policy specifications. The Firth estimates use inverse expected Fisher information and model-based Wald intervals; they are a rare-outcome sensitivity and do not replace the HC3-robust primary estimates. For the cross-relation alignment test, each permutation is a one-to-one reassignment of complete $P$--$D$ incidence profiles within publication-year strata. Every profile is used once per permutation, strata with one publication remain fixed, and $P$--$G$ ties, policy-document endpoints, and issuer attributions are retained. The random seed is 20260902; two-sided empirical $p$ values use the plus-one correction described in the main text.",
    "",
    r"\\section*{Supplementary Results}",
]

# S1: full main models
base_models = ["grant_primary", "policy_2021_primary", "policy_2020_robustness"]
base = coeff[coeff["model_label"].isin(base_models)].copy()
base["model_order"] = base["model_label"].map({label: index for index, label in enumerate(base_models)})
term_order = ["const"] + [term for term in TERM_LABELS if term != "const"]
base["term_order"] = base["term"].map({term: index for index, term in enumerate(term_order)})
base = base.sort_values(["model_order", "term_order"])
lines += longtable_header("p{4.0cm}p{4.0cm}rrrr", "Full adjusted association models.", "tab:s1-full-models", "Model & Term & Log odds & Robust SE & OR (95\\% CI) & $p$")
for _, row in base.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS.get(row['term'], latex_escape(row['term']))} & {number(row['coefficient_log_odds'], 3)} & {number(row['standard_error'], 3)} & {or_ci(row)} & {p_text(row['p_value'])} " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: Policy models have 147 events among 2,271 publications through 2021 and 120 events among 1,550 publications through 2020. OR = odds ratio; CI = confidence interval.")
lines.append("")

# S2: all focal GLM sensitivity results
focal = coeff[coeff["term"].isin(["ethics_responsibility_primary", "computational_performance_primary"])].copy()
model_order = list(MODEL_LABELS)
focal["model_order"] = focal["model_label"].map({label: index for index, label in enumerate(model_order)})
focal["term_order"] = focal["term"].map({"ethics_responsibility_primary": 0, "computational_performance_primary": 1})
focal = focal.sort_values(["model_order", "term_order"])
lines += longtable_header("p{5.0cm}p{4.0cm}rr", "Targeted sensitivity analyses for the focal indicators.", "tab:s2-focal-sensitivities", "Specification & Indicator & OR (95\\% CI) & $p$")
for _, row in focal.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS[row['term']]} & {or_ci(row)} & {p_text(row['p_value'])} " + r"\\")
lines += longtable_footer()

# S3 Firth
firth_focal = firth[firth["term"].isin(["ethics_responsibility_primary", "computational_performance_primary"])].copy()
firth_focal["model_order"] = firth_focal["model_label"].map({"policy_2021_firth": 0, "policy_2020_firth": 1})
firth_focal["term_order"] = firth_focal["term"].map({"ethics_responsibility_primary": 0, "computational_performance_primary": 1})
firth_focal = firth_focal.sort_values(["model_order", "term_order"])
lines += longtable_header("p{4.5cm}p{4.0cm}rr", "Firth penalized-logistic sensitivity results for policy-document linkage.", "tab:s3-firth", "Specification & Indicator & OR (95\\% CI) & $p$")
for _, row in firth_focal.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS[row['term']]} & {or_ci(row)} & {p_text(row['p_value'])} " + r"\\")
lines += longtable_footer()

# S4 diagnostic table
selected_diag_labels = ["grant_primary", "policy_2021_primary", "policy_2020_robustness", "policy_2021_complete_case", "policy_2020_complete_case", "policy_2021_no_citation_adjustment", "policy_2020_no_citation_adjustment", "policy_2021_flexible_year", "policy_2020_flexible_year"]
diagnostic_table = diagnostics[diagnostics["model_label"].isin(selected_diag_labels)].copy()
diagnostic_table["model_order"] = diagnostic_table["model_label"].map({label: index for index, label in enumerate(selected_diag_labels)})
diagnostic_table = diagnostic_table.sort_values("model_order")
lines += longtable_header("p{5.1cm}rrrrrr", "Model diagnostics for principal and policy sensitivity models.", "tab:s4-diagnostics", "Model & $n$ & Events & AIC & McFadden $R^2$ & Min. fitted $p$ & Max. fitted $p$")
for _, row in diagnostic_table.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {int(row['n_observations']):,} & {int(row['n_events']):,} & {number(row['aic'], 1)} & {number(row['pseudo_r2_mcfadden'], 3)} & {number(row['fitted_probability_min'], 4)} & {number(row['fitted_probability_max'], 4)} " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: All listed binomial GLM models converged. No perfect-separation or convergence warning was emitted. Fitted-probability extrema are screening diagnostics; they do not prove the absence of quasi-separation.")
lines.append("")

# S5 probability contrasts
contrast_models = ["grant_primary", "policy_2021_primary", "policy_2020_robustness"]
contrast_table = contrasts[contrasts["model_label"].isin(contrast_models)].copy()
contrast_table["model_order"] = contrast_table["model_label"].map({label: index for index, label in enumerate(contrast_models)})
contrast_table["term_order"] = contrast_table["indicator"].map({"ethics_responsibility_primary": 0, "computational_performance_primary": 1})
contrast_table = contrast_table.sort_values(["model_order", "term_order"])
lines += longtable_header("p{4.5cm}p{3.9cm}rrr", "Standardized model-based probability contrasts for focal indicators.", "tab:s5-probability-contrasts", "Model & Indicator & Indicator = 0 & Indicator = 1 & Difference")
for _, row in contrast_table.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS[row['indicator']]} & {100 * row['standardized_predicted_probability_indicator_0']:.1f}\\% & {100 * row['standardized_predicted_probability_indicator_1']:.1f}\\% & {100 * row['standardized_probability_difference']:+.1f} pp " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: For each contrast, one focal indicator is set to 0 and then 1 for all publications while the remaining observed covariates are retained. These are model-based standardized contrasts, not causal effects.")
lines.append("")

# S6 dictionary composition
lines += longtable_header("p{5.0cm}p{5.4cm}rr", "Descriptive composition of the primary ethics/responsibility dictionary.", "tab:s6-ethics-composition", "Subfamily & Component terms & Publications & Share of $P$")
for _, row in subfamilies.iterrows():
    lines.append(f"{latex_escape(row['dictionary_subfamily'])} & {latex_escape(row['component_terms'])} & {int(row['n_publications_with_at_least_one_component_term']):,} & {row['pct_of_publications']:.2f}\\% " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: Subfamilies are non-exclusive descriptive groupings. They were not entered as post-hoc association predictors.")
lines.append("")

# S7 degree distribution
lines += longtable_header("rrr", "Publication-level distribution of recorded grant-link counts.", "tab:s7-grant-degree", "Recorded grant links & Publications & Share of all publications")
for _, row in grant_degree.iterrows():
    lines.append(f"{int(row['n_linked_grants'])} & {int(row['n_publications']):,} & {100 * row['share_of_all_publications']:.2f}\\% " + r"\\")
lines += longtable_footer()

# S8 hub deletion
hub_order = {"full_system": 0, "highest_contributor_removed": 1}
hub["analysis_order"] = hub["analysis"].map(hub_order)
hub["stat_order"] = hub["statistic"].map({"N_both": 0, "intensive_degree_product_sum": 1})
hub = hub.sort_values(["analysis_order", "stat_order"])
lines += longtable_header("p{4.6cm}p{3.3cm}rrrr", "Highest-contributor deletion sensitivity for local alignment.", "tab:s8-hub-deletion", "Analysis & Statistic & Observed & Null mean & Enrichment & Two-sided $p$")
for _, row in hub.iterrows():
    analysis_name = "Full system" if row["analysis"] == "full_system" else "Highest degree-product contributor removed"
    statistic_name = "$N_{\\mathrm{both}}$" if row["statistic"] == "N_both" else r"$\\sum_p d_{PG}(p)d_{PD}(p)$"
    lines.append(f"{analysis_name} & {statistic_name} & {number(row['observed'], 0)} & {number(row['null_mean'], 3)} & {number(row['enrichment'], 3)} & {p_text(row['two_sided_permutation_p'])} " + r"\\")
lines += longtable_footer()
hub_row = hub_detail.iloc[0]
lines.append(r"\\noindent\\small Note: The removed publication had $d_{PG}=" + str(int(hub_row["P_G_degree"])) + r"$, $d_{PD}=" + str(int(hub_row["P_D_degree"])) + r"$, and degree-product contribution " + str(int(hub_row["degree_product_contribution"])) + r". The publication identifier is retained in the machine-readable output but is not reported in the manuscript.")
lines.append("")

# S9 language
lines += longtable_header("p{3.5cm}p{2.7cm}p{8.0cm}", "Availability of publication-language metadata.", "tab:s9-language", "Metadata field & Status & Note")
for _, row in language.iterrows():
    lines.append(f"{latex_escape(row['field'])} & {latex_escape(row['status'])} & {latex_escape(row['note'])} " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: The matching dictionaries are English-language dictionaries. Because language metadata were not returned in the publication file, the language composition of the anchor cannot be quantified.")
lines.append("")

# S10 VIF and influence main models
vif_main = vif[vif["model_label"].isin(["grant_primary", "policy_2021_primary", "policy_2020_robustness"]) & vif["term"].isin(["ethics_responsibility_primary", "computational_performance_primary"])].copy()
vif_main["model_order"] = vif_main["model_label"].map({"grant_primary": 0, "policy_2021_primary": 1, "policy_2020_robustness": 2})
vif_main = vif_main.sort_values(["model_order", "term"])
lines += longtable_header("p{5.2cm}p{4.0cm}r", "Variance inflation factors for focal indicators in main models.", "tab:s10-vif", "Model & Indicator & VIF")
for _, row in vif_main.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS[row['term']]} & {number(row['vif'], 3)} " + r"\\")
lines += longtable_footer()

# S11: largest observed predictor correlations in principal models
correlation_rows = []
for model_label in ["grant_primary", "policy_2021_primary", "policy_2020_robustness"]:
    subset = correlations[correlations["model_label"] == model_label].sort_values("absolute_correlation", ascending=False)
    if not subset.empty:
        row = subset.iloc[0]
        correlation_rows.append({
            "model_label": model_label,
            "term_1": row["term_1"],
            "term_2": row["term_2"],
            "absolute_correlation": row["absolute_correlation"],
        })
lines += longtable_header("p{4.5cm}p{3.2cm}p{3.2cm}r", "Largest absolute Pearson correlation among predictors in each principal model.", "tab:s11-correlations", "Model & Predictor 1 & Predictor 2 & $|r|$")
for row in correlation_rows:
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {TERM_LABELS.get(row['term_1'], latex_escape(row['term_1']))} & {TERM_LABELS.get(row['term_2'], latex_escape(row['term_2']))} & {number(row['absolute_correlation'], 3)} " + r"\\")
lines += longtable_footer()

# S12: influence screening diagnostics for principal models
influence_main = influence[influence["model_label"].isin(["grant_primary", "policy_2021_primary", "policy_2020_robustness"])].copy()
influence_main["model_order"] = influence_main["model_label"].map({"grant_primary": 0, "policy_2021_primary": 1, "policy_2020_robustness": 2})
influence_main = influence_main.sort_values("model_order")
lines += longtable_header("p{5.0cm}rrr", "Influence-screening diagnostics for principal models.", "tab:s12-influence", "Model & Maximum Cook's $D$ & Cases with $D>4/n$ & Maximum $|$DFBETA$|$")
for _, row in influence_main.iterrows():
    lines.append(f"{MODEL_LABELS[row['model_label']]} & {number(row['max_cooks_distance'], 4)} & {int(row['n_cooks_distance_above_4_over_n']):,} & {number(row['max_abs_dfbetas'], 3)} " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: These values are influence-screening diagnostics rather than case-deletion effect estimates. The targeted highest-contributor deletion test for the network alignment statistic is reported separately in Supplementary Table~\\ref{tab:s8-hub-deletion}.")
lines.append("")

# S13: screening audit
screen = screening_summary.iloc[0]
lines += longtable_header("p{8.2cm}r", "Record-level audit of the two-stage publication screen.", "tab:s13-screening-audit", "Audit quantity & Value")
for label, value in [
    ("Candidate publications returned by the Dimensions query", int(screen["candidate_publications"])),
    ("Publications retained in the anchor", int(screen["retained_publications"])),
    ("Candidate publications excluded by the deterministic local screen", int(screen["excluded_publications"])),
    ("Retention rate", f"{float(screen['retained_pct']):.2f}\\%"),
    ("Candidate records with an available abstract", int(screen["records_with_available_abstract"])),
    ("Candidate records without an abstract", int(screen["records_without_abstract"])),
    ("Saved decisions reproduced from the saved matched-term fields", "Yes" if bool(screen["rule_reproduced_exactly"]) else "No"),
    ("Rule-reproduction mismatches", int(screen["rule_mismatch_n"])),
]:
    display = f"{value:,}" if isinstance(value, int) else str(value)
    lines.append(f"{label} & {display} " + r"\\")
lines += longtable_footer()
lines.append(r"\\noindent\\small Note: The audit verifies whether the saved retain/exclude decisions follow the documented literal screen when reconstructed from the saved AI/ML and core-mental-health matched-term fields. It does not establish perfect recall of all relevant research.")
lines.append("")

lines += [r"\\end{document}", ""]
OUT.write_text("\n".join(normalize_latex_commands(lines)), encoding="utf-8")
print(OUT)
