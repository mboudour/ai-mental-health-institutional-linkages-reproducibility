#!/usr/bin/env python3
"""Restarted Step 2: v6 publication text indicators and Dimensions-concept audit.

Save as:
    new_version/computations/step2_v6_operationalize_indicators.py

Run from new_version/:
    python computations/step2_v6_operationalize_indicators.py

This script uses final v6 P only. It preserves the previously approved
operationalization: the narrow computational-performance dictionary is primary,
the broader definition is sensitivity-only, and concept labels/relevance scores
are a separate descriptive audit rather than text indicators or model inputs.

Output:
    computations/outputs/step2_v6_indicators/
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PUBLICATION_FILE = PROJECT_ROOT / "data" / "linked_final_outputs_v6" / "publication_anchor_final.pkl"
STEP1_MANIFEST = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2" / "data_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step2_v6_indicators"

# ---------------------------------------------------------------------------
# FIXED, APPROVED DICTIONARIES (defined before grant/policy outcomes are read)
# ---------------------------------------------------------------------------
# Explicit ethics/responsibility language is the primary ethics measure.
ETHICS_PRIMARY = [
    ("ethic*", r"\bethic\w*\b"),
    ("fairness", r"\bfairness\b"),
    ("algorithmic bias", r"\balgorithmic\s+bias\b"),
    ("bias mitigation", r"\bbias\s+mitigation\b"),
    ("explainab*", r"\bexplainab\w*\b"),
    ("interpretab*", r"\binterpretab\w*\b"),
    ("transparen*", r"\btransparen\w*\b"),
    ("accountab*", r"\baccountab\w*\b"),
    ("responsible AI", r"\bresponsible\s+ai\b"),
    ("responsible artificial intelligence", r"\bresponsible\s+artificial\s+intelligence\b"),
    ("trustworthy AI", r"\btrustworthy\s+ai\b"),
    ("AI governance", r"\bai\s+governance\b"),
    ("algorithmic governance", r"\balgorithmic\s+governance\b"),
    ("AI regulation", r"\bai\s+regulation\b"),
    ("algorithmic regulation", r"\balgorithmic\s+regulation\b"),
    ("data privacy", r"\bdata\s+privacy\b"),
    ("privacy-preserving", r"\bprivacy[-\s]preserv\w*\b"),
    ("informed consent", r"\binformed\s+consent\b"),
    ("health equity", r"\bhealth\s+equity\b"),
    ("human rights", r"\bhuman\s+rights\b"),
]

# Narrower ethics dictionary retained as a pre-specified robustness check.
ETHICS_NARROW = [
    ("ethic*", r"\bethic\w*\b"),
    ("algorithmic bias", r"\balgorithmic\s+bias\b"),
    ("explainab*", r"\bexplainab\w*\b"),
    ("interpretab*", r"\binterpretab\w*\b"),
    ("responsible AI", r"\bresponsible\s+ai\b"),
    ("responsible artificial intelligence", r"\bresponsible\s+artificial\s+intelligence\b"),
    ("AI governance", r"\bai\s+governance\b"),
    ("AI regulation", r"\bai\s+regulation\b"),
    ("data privacy", r"\bdata\s+privacy\b"),
    ("informed consent", r"\binformed\s+consent\b"),
    ("health equity", r"\bhealth\s+equity\b"),
]

# Narrow computational-performance dictionary: PRIMARY measure.
# Generic standalone mentions of algorithm, prediction, classification,
# performance, sensitivity, specificity, precision, or recall are excluded.
COMPUTATION_PRIMARY = [
    ("model performance", r"\bmodel\s+performance\b"),
    ("predictive performance", r"\bpredictive\s+performance\b"),
    ("classification performance", r"\bclassification\s+performance\b"),
    ("predictive accuracy", r"\bpredictive\s+accuracy\b"),
    ("model accuracy", r"\bmodel\s+accuracy\b"),
    ("area under the curve", r"\barea\s+under\s+the\s+curve\b"),
    ("ROC-AUC", r"\broc[-\s]?auc\b"),
    ("receiver operating characteristic", r"\breceiver\s+operating\s+characteristic\b"),
    ("cross-validation", r"\bcross[-\s]?validation\b"),
    ("external validation", r"\bexternal\s+validation\b"),
]

# Former broad definition: expanded sensitivity check only.
COMPUTATION_EXPANDED = COMPUTATION_PRIMARY + [
    ("sensitivity", r"\bsensitivity\b"),
    ("specificity", r"\bspecificity\b"),
    ("precision", r"\bprecision\b"),
    ("recall", r"\brecall\b"),
    ("F1 score", r"\bf[-\s]?1\s+(?:score|metric)\b"),
    ("benchmark*", r"\bbenchmark\w*\b"),
]

# Concept matching is only an inventory of candidate Dimensions labels. It does
# not create text indicators and will not produce model covariates in this step.
CONCEPT_AUDIT_PATTERNS = {
    "ethics_responsibility_candidate": [
        ("ethics", r"ethic"), ("fairness", r"fair"), ("bias", r"bias"),
        ("explainability", r"explain"), ("interpretability", r"interpret"),
        ("transparency", r"transparen"), ("accountability", r"accountab"),
        ("responsibility", r"responsib"), ("trustworthiness", r"trustworth"),
        ("governance", r"governance"), ("regulation", r"regulation"),
        ("privacy", r"privacy"), ("consent", r"consent"), ("equity", r"equity"),
        ("human rights", r"human\s+right"),
    ],
    "computational_performance_candidate": [
        ("machine learning", r"machine\s+learning"), ("deep learning", r"deep\s+learning"),
        ("neural network", r"neural\s+network"), ("artificial intelligence", r"artificial\s+intelligence"),
        ("algorithm", r"algorithm"), ("prediction", r"predict"),
        ("classification", r"classif"), ("performance", r"performance"),
        ("accuracy", r"accuracy"), ("sensitivity", r"sensitivity"),
        ("specificity", r"specificity"), ("precision", r"precision"),
        ("recall", r"recall"), ("validation", r"validation"), ("benchmark", r"benchmark"),
    ],
}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def as_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
    return [text]


def matched_terms(text: str, terms: list[tuple[str, str]]) -> list[str]:
    return [label for label, pattern in terms if re.search(pattern, text, flags=re.IGNORECASE)]


def indicator_series(frame: pd.DataFrame, terms: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    title = frame["title_text"].apply(lambda x: matched_terms(x, terms))
    abstract = frame["abstract_text"].apply(lambda x: matched_terms(x, terms))
    all_matches = pd.Series(["; ".join(sorted(set(a + b))) for a, b in zip(title, abstract)], index=frame.index)
    return (all_matches.str.len().gt(0).astype(int), all_matches, title.apply(lambda x: "; ".join(x)), abstract.apply(lambda x: "; ".join(x)))


def add_indicator(out: pd.DataFrame, frame: pd.DataFrame, variable: str, terms: list[tuple[str, str]]) -> None:
    flag, all_matches, title_matches, abstract_matches = indicator_series(frame, terms)
    out[variable] = flag
    out[f"{variable}_matched_terms"] = all_matches
    out[f"{variable}_title_terms"] = title_matches
    out[f"{variable}_abstract_terms"] = abstract_matches


def term_prevalence(frame: pd.DataFrame, dictionary_label: str, indicator: str, terms: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for label, pattern in terms:
        title_match = frame["title_text"].str.contains(pattern, case=False, regex=True, na=False)
        abstract_match = frame["abstract_text"].str.contains(pattern, case=False, regex=True, na=False)
        total = title_match | abstract_match
        rows.append({
            "dictionary": dictionary_label,
            "indicator": indicator,
            "term_label": label,
            "regex": pattern,
            "n_title_matches": int(title_match.sum()),
            "n_abstract_matches": int(abstract_match.sum()),
            "n_any_text_match": int(total.sum()),
            "pct_of_P_any_text_match": round(100 * total.mean(), 3),
        })
    return rows


def overlap_table(frame: pd.DataFrame, ethics_col: str, computation_col: str, label: str) -> pd.DataFrame:
    categories = []
    for ethics, computation in zip(frame[ethics_col], frame[computation_col]):
        if ethics == 0 and computation == 0:
            categories.append("neither")
        elif ethics == 1 and computation == 0:
            categories.append("ethics_responsibility_only")
        elif ethics == 0 and computation == 1:
            categories.append("computational_performance_only")
        else:
            categories.append("both")
    order = ["neither", "ethics_responsibility_only", "computational_performance_only", "both"]
    counts = pd.Series(categories).value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({
        "definition": label,
        "overlap_group": order,
        "n_publications": counts.values,
        "pct_of_P": (100 * counts.values / len(frame)).round(3),
    })


def dictionary_comparison(frame: pd.DataFrame, name: str, primary_col: str, sensitivity_col: str) -> dict[str, Any]:
    primary = frame[primary_col].astype(bool)
    sensitivity = frame[sensitivity_col].astype(bool)
    union = primary | sensitivity
    return {
        "indicator": name,
        "n_primary": int(primary.sum()),
        "n_sensitivity": int(sensitivity.sum()),
        "n_both": int((primary & sensitivity).sum()),
        "n_primary_only": int((primary & ~sensitivity).sum()),
        "n_sensitivity_only": int((~primary & sensitivity).sum()),
        "n_neither": int((~union).sum()),
        "primary_sensitivity_jaccard": round((primary & sensitivity).sum() / union.sum(), 6) if union.sum() else None,
    }


def concept_inventory(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    concept_pubs: dict[str, set[str]] = defaultdict(set)
    concept_scores: dict[str, list[float]] = defaultdict(list)
    n_label_occurrences = 0
    n_score_records = 0
    n_bad_score_records = 0
    n_with_concepts = 0
    n_with_score_records = 0

    for pub_id, labels_value, scores_value in zip(frame["id"], frame["concepts"], frame["concepts_scores"]):
        labels = [clean_text(x) for x in as_list(labels_value)]
        labels = [x for x in labels if x]
        if labels:
            n_with_concepts += 1
        for label in labels:
            concept_pubs[label].add(str(pub_id))
            n_label_occurrences += 1

        score_items = as_list(scores_value)
        if score_items:
            n_with_score_records += 1
        for item in score_items:
            if not isinstance(item, dict):
                n_bad_score_records += 1
                continue
            label = clean_text(item.get("concept"))
            try:
                relevance = float(item.get("relevance"))
            except (TypeError, ValueError):
                n_bad_score_records += 1
                continue
            if label and pd.notna(relevance):
                concept_scores[label].append(relevance)
                n_score_records += 1
            else:
                n_bad_score_records += 1

    rows = []
    for concept, pub_ids in concept_pubs.items():
        scores = concept_scores.get(concept, [])
        rows.append({
            "concept": concept,
            "n_publications": len(pub_ids),
            "pct_of_P": round(100 * len(pub_ids) / len(frame), 3),
            "n_numeric_scores": len(scores),
            "mean_concept_score": round(sum(scores) / len(scores), 6) if scores else None,
            "median_concept_score": round(pd.Series(scores).median(), 6) if scores else None,
            "max_concept_score": round(max(scores), 6) if scores else None,
        })
    prevalence = pd.DataFrame(rows).sort_values(["n_publications", "concept"], ascending=[False, True]).reset_index(drop=True)

    candidate_rows = []
    for group, patterns in CONCEPT_AUDIT_PATTERNS.items():
        for candidate, pattern in patterns:
            matched = prevalence["concept"].str.contains(pattern, case=False, regex=True, na=False)
            for _, row in prevalence.loc[matched].iterrows():
                candidate_rows.append({
                    "audit_group": group,
                    "candidate_term": candidate,
                    "candidate_regex": pattern,
                    **row.to_dict(),
                })
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["audit_group", "n_publications", "concept"], ascending=[True, False, True]).reset_index(drop=True)

    audit = {
        "n_publications": int(len(frame)),
        "n_publications_with_at_least_one_concept": int(n_with_concepts),
        "concept_coverage_pct": round(100 * n_with_concepts / len(frame), 3),
        "n_publications_with_concept_score_records": int(n_with_score_records),
        "concept_label_occurrences": int(n_label_occurrences),
        "concept_relevance_records_parsed": int(n_score_records),
        "nonparseable_concept_score_records": int(n_bad_score_records),
        "interpretive_boundary": "Concept labels and relevance scores are Dimensions-generated publication attributes. Candidate matching is an inventory only; it does not define the text indicators or establish semantic similarity across record types.",
    }
    return prevalence, prevalence.head(200).copy(), candidates, audit


def main() -> None:
    for required in [PUBLICATION_FILE, STEP1_MANIFEST]:
        if not required.is_file():
            raise SystemExit(f"Required file is missing: {required}\nRun restarted Step 1 first and confirm this script is in new_version/computations/.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    publications = pd.read_pickle(PUBLICATION_FILE).copy()
    required_columns = {"id", "title", "abstract", "year", "times_cited", "concepts", "concepts_scores", "retain_in_final_P"}
    missing = sorted(required_columns.difference(publications.columns))
    if missing:
        raise SystemExit("Missing publication columns: " + ", ".join(missing))
    if publications["id"].astype(str).duplicated().any():
        raise SystemExit("P contains duplicate IDs. Resolve restarted Step 1 before running this script.")
    if not publications["retain_in_final_P"].astype(bool).all():
        raise SystemExit("The input includes a record not retained by the final v6 publication screen. Use publication_anchor_final.pkl unchanged.")

    publications["id"] = publications["id"].astype(str).str.strip()
    publications["title_text"] = publications["title"].apply(clean_text)
    publications["abstract_text"] = publications["abstract"].apply(clean_text)

    indicators = publications[["id", "year", "times_cited"]].copy().rename(columns={"id": "publication_id"})
    indicators["abstract_available"] = publications["abstract_text"].str.len().gt(0).astype(int)
    indicators["text_source"] = "title plus available abstract"

    add_indicator(indicators, publications, "ethics_responsibility_primary", ETHICS_PRIMARY)
    add_indicator(indicators, publications, "computational_performance_primary", COMPUTATION_PRIMARY)
    add_indicator(indicators, publications, "ethics_responsibility_narrow_robustness", ETHICS_NARROW)
    add_indicator(indicators, publications, "computational_performance_expanded_sensitivity", COMPUTATION_EXPANDED)

    codebook_rows = []
    definitions = [
        ("primary", "ethics_responsibility", ETHICS_PRIMARY, "Primary analysis"),
        ("primary", "computational_performance", COMPUTATION_PRIMARY, "Primary analysis"),
        ("ethics_narrow_robustness", "ethics_responsibility", ETHICS_NARROW, "Pre-specified narrower robustness check"),
        ("computational_expanded_sensitivity", "computational_performance", COMPUTATION_EXPANDED, "Pre-specified expanded sensitivity check"),
    ]
    for dictionary_label, indicator, terms, purpose in definitions:
        for label, regex in terms:
            codebook_rows.append({
                "dictionary": dictionary_label,
                "measurement_source": "publication title plus available abstract",
                "indicator": indicator,
                "term_label": label,
                "regex": regex,
                "purpose": purpose,
            })
    codebook = pd.DataFrame(codebook_rows)

    prevalence_rows = []
    for dictionary_label, indicator, terms, _ in definitions:
        prevalence_rows.extend(term_prevalence(publications, dictionary_label, indicator, terms))
    prevalence = pd.DataFrame(prevalence_rows)

    overlap = pd.concat([
        overlap_table(indicators, "ethics_responsibility_primary", "computational_performance_primary", "primary"),
        overlap_table(indicators, "ethics_responsibility_narrow_robustness", "computational_performance_primary", "ethics_narrow_robustness"),
        overlap_table(indicators, "ethics_responsibility_primary", "computational_performance_expanded_sensitivity", "computational_expanded_sensitivity"),
    ], ignore_index=True)

    sensitivity = pd.DataFrame([
        dictionary_comparison(indicators, "ethics_responsibility", "ethics_responsibility_primary", "ethics_responsibility_narrow_robustness"),
        dictionary_comparison(indicators, "computational_performance", "computational_performance_primary", "computational_performance_expanded_sensitivity"),
    ])

    coverage = pd.DataFrame([
        {"measurement_source": "title", "n_available": int(publications["title_text"].str.len().gt(0).sum()), "n_missing": int(publications["title_text"].str.len().eq(0).sum()), "coverage_pct": round(100 * publications["title_text"].str.len().gt(0).mean(), 3)},
        {"measurement_source": "abstract", "n_available": int(publications["abstract_text"].str.len().gt(0).sum()), "n_missing": int(publications["abstract_text"].str.len().eq(0).sum()), "coverage_pct": round(100 * publications["abstract_text"].str.len().gt(0).mean(), 3)},
        {"measurement_source": "title plus available abstract", "n_available": len(publications), "n_missing": 0, "coverage_pct": 100.0},
    ])

    concept_prevalence, top_concepts, candidate_concepts, concept_audit = concept_inventory(publications)

    codebook.to_csv(OUTPUT_DIR / "text_indicator_codebook.csv", index=False)
    indicators.to_csv(OUTPUT_DIR / "publication_text_indicators.csv", index=False)
    prevalence.to_csv(OUTPUT_DIR / "text_term_prevalence.csv", index=False)
    overlap.to_csv(OUTPUT_DIR / "text_indicator_overlap.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "dictionary_sensitivity.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "text_field_coverage.csv", index=False)
    concept_prevalence.to_csv(OUTPUT_DIR / "dimensions_concept_prevalence.csv", index=False)
    top_concepts.to_csv(OUTPUT_DIR / "dimensions_top_200_concepts.csv", index=False)
    candidate_concepts.to_csv(OUTPUT_DIR / "dimensions_concept_candidate_matches.csv", index=False)
    (OUTPUT_DIR / "dimensions_concept_parsing_audit.json").write_text(json.dumps(concept_audit, indent=2), encoding="utf-8")

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": "step2_v6_operationalize_indicators_1.0",
        "source_dataset": "linked_final_outputs_v6/publication_anchor_final.pkl",
        "source_publication_construction": "Dimensions candidate retrieval followed by the approved literal title/abstract screen",
        "step1_source_dataset": json.loads(STEP1_MANIFEST.read_text(encoding="utf-8")).get("source_dataset"),
        "step1_audit_script_version": json.loads(STEP1_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
        "unit_of_analysis": "publication in P",
        "primary_text_indicators": {
            "ethics_responsibility": "ETHICS_PRIMARY dictionary",
            "computational_performance": "COMPUTATION_PRIMARY (narrow) dictionary",
        },
        "sensitivity_text_indicators": {
            "ethics_responsibility": "ETHICS_NARROW dictionary",
            "computational_performance": "COMPUTATION_EXPANDED dictionary",
        },
        "text_measurement_rule": "case-insensitive regular-expression matches in title plus available abstract",
        "missing_abstract_rule": "Retain records without abstracts; match title text only and report coverage.",
        "concept_measurement_rule": "Parse Dimensions concepts_scores as concept/relevance records. Concept label matching is descriptive only and is separate from text-indicator construction.",
        "interpretive_boundary": "Indicators and concepts are observable publication attributes. They do not establish author intent, research quality, institutional selection, causal effects, policy impact, or semantic similarity between publications and policy documents.",
    }
    (OUTPUT_DIR / "step2_indicator_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Restarted Step 2 (v6 indicators) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Primary ethics/responsibility: {int(indicators['ethics_responsibility_primary'].sum()):,} / {len(indicators):,}")
    print(f"Primary computational-performance (narrow): {int(indicators['computational_performance_primary'].sum()):,} / {len(indicators):,}")
    print(f"Both primary indicators: {int(((indicators['ethics_responsibility_primary'] == 1) & (indicators['computational_performance_primary'] == 1)).sum()):,}")
    print(f"Concept relevance records parsed: {concept_audit['concept_relevance_records_parsed']:,}")


if __name__ == "__main__":
    main()
