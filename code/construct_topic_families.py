#!/usr/bin/env python3
"""Restarted Step 3: v6 concept processing and approved topic families.

Save as:
    new_version/computations/step3_v6_construct_topic_families_v2.py

Run from new_version/:
    python computations/step3_v6_construct_topic_families_v2.py

Inputs (read only):
    data/linked_final_outputs_v6/publication_anchor_final.pkl
    computations/outputs/step2_v6_indicators/step2_indicator_manifest.json

Outputs:
    computations/outputs/step3_v6_topic_families/

Pre-specified rules, before any grant/policy outcome is analysed:
    * Canonicalise concept labels and retain the maximum relevance score for
      each duplicate publication--concept pair.
    * Retain a primary P--concept pair only when relevance >= 0.40 and the
      label is not on the explicit generic-label stoplist.
    * Report 0.30 and 0.50 threshold sensitivity and a 1% prevalence audit.
    * Map primary non-generic edges to the four manually approved, non-exclusive
      substantive research-topic families.

Concepts are supplementary Dimensions-generated attributes of P. They are not
text indicators, policy-document topics, cross-layer semantic measures, or
causal variables.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PUBLICATION_FILE = PROJECT_ROOT / "data" / "linked_final_outputs_v6" / "publication_anchor_final.pkl"
STEP2_MANIFEST = PROJECT_ROOT / "computations" / "outputs" / "step2_v6_indicators" / "step2_indicator_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step3_v6_topic_families_v2"

PRIMARY_RELEVANCE_THRESHOLD = 0.40
SENSITIVITY_THRESHOLDS = [0.30, 0.40, 0.50]
MIN_PREVALENCE_SHARE = 0.01

# Exact canonical labels that describe generic scholarly language, broad
# methods, or document rhetoric rather than a substantive research topic.
# This list is written to the output codebook and can be reviewed directly.
GENERIC_LABEL_STOPLIST = {
    "accuracy", "algorithm", "algorithms", "analysis", "approach", "application",
    "applications", "area under the curve", "artificial", "behavior", "benchmark",
    "classification", "data", "dataset", "datasets", "deep learning", "development",
    "effect", "features", "findings", "factors", "health", "human behavior",
    "information", "intelligence", "learning", "machine", "machine learning", "method",
    "methods", "model", "models", "network", "networks", "neural network", "outcomes",
    "patterns", "performance", "potential", "precision", "prediction", "process",
    "processes", "recall", "research", "results", "sensitivity", "signal", "signals",
    "specificity", "study", "studies", "system", "systems", "technique", "techniques",
    "technology", "tools", "validation",
}

# Four final user-approved, non-exclusive substantive research-topic families.
# The fifth digital/social-setting family is absent by decision. The fourth
# family is deliberately restricted to neuropsychology, cognition, emotion,
# and affect.
CONCEPT_FAMILIES: dict[str, list[tuple[str, str]]] = {
    "mental_health_conditions_symptoms": [
        ("depression/depressive", r"\bdepress\w*\b"),
        ("anxiety", r"\banxiety\b"),
        ("schizophrenia", r"\bschizophrenia\b"),
        ("psychosis/psychotic", r"\bpsychos\w*\b"),
        ("bipolar", r"\bbipolar\b"),
        ("post-traumatic stress/PTSD", r"\b(?:post[-\s]?traumatic\s+stress|ptsd)\b"),
        ("suicide/suicidal", r"\bsuicid\w*\b"),
        ("self-harm", r"\bself[-\s]?harm\b"),
        ("eating disorder", r"\beating\s+disorders?\b"),
        ("substance use/abuse", r"\bsubstance\s+(?:use|abuse)\b"),
        ("addiction", r"\baddiction\b"),
    ],
    "clinical_assessment_diagnosis": [
        ("diagnosis/diagnostic", r"\bdiagnos\w*\b"),
        ("screening", r"\bscreening\b"),
        ("risk assessment", r"\brisk\s+assessment\b"),
        ("clinical decision-making", r"\bclinical\s+decision(?:[-\s]making|[-\s]support)?\b"),
    ],
    "treatment_intervention": [
        ("treatment", r"\btreatment\b"),
        ("intervention", r"\bintervention\b"),
        ("therapy/therapeutic", r"\btherap\w*\b"),
        ("psychotherapy", r"\bpsychotherap\w*\b"),
        ("cognitive behavioural therapy", r"\bcognitive\s+behavio(?:u)?ral\s+therapy\b"),
        ("behavioural activation", r"\bbehavio(?:u)?ral\s+activation\b"),
    ],
    "neurocognitive_affective_processes": [
        ("neuropsychology", r"\bneuropsycholog\w*\b"),
        ("cognition/cognitive", r"\bcognit\w*\b"),
        ("emotion/emotional", r"\bemotion\w*\b"),
        ("affect/affective", r"\baffect\w*\b"),
    ],
}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def canonical_label(value: Any) -> str:
    return " ".join(clean_text(value).casefold().split())


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


def stable_join(values: pd.Series) -> str:
    return "; ".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def matched_family_labels(concept: str, patterns: list[tuple[str, str]]) -> list[str]:
    return [label for label, regex in patterns if re.search(regex, concept, flags=re.IGNORECASE)]


def build_raw_concept_records(publications: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bad_score_records = 0
    labels_without_scores = 0

    for publication_id, labels_value, scores_value in zip(publications["id"], publications["concepts"], publications["concepts_scores"]):
        # Concepts_scores is the authoritative source for scored pairs.
        score_items = as_list(scores_value)
        scored_labels = set()
        for item in score_items:
            if not isinstance(item, dict):
                bad_score_records += 1
                continue
            raw_label = clean_text(item.get("concept"))
            label = canonical_label(raw_label)
            try:
                relevance = float(item.get("relevance"))
            except (TypeError, ValueError):
                bad_score_records += 1
                continue
            if not label or pd.isna(relevance):
                bad_score_records += 1
                continue
            scored_labels.add(label)
            rows.append({
                "publication_id": str(publication_id),
                "concept": label,
                "concept_display": raw_label,
                "relevance": relevance,
            })

        # Audit, rather than invent, any label without a usable relevance score.
        raw_labels = {canonical_label(label) for label in as_list(labels_value) if canonical_label(label)}
        labels_without_scores += len(raw_labels.difference(scored_labels))

    if bad_score_records:
        raise SystemExit(f"Found {bad_score_records} non-parseable concept-score records. Stopping before concept construction.")
    if not rows:
        raise SystemExit("No scored Dimensions concepts were parsed. Stopping.")

    raw = pd.DataFrame(rows)
    raw.attrs["labels_without_scores"] = labels_without_scores
    return raw


def main() -> None:
    for required in [PUBLICATION_FILE, STEP2_MANIFEST]:
        if not required.is_file():
            raise SystemExit(
                f"Required file not found: {required}\n"
                "Run restarted Step 2 first and confirm this script is in new_version/computations/."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    publications = pd.read_pickle(PUBLICATION_FILE).copy()
    required_columns = {"id", "title", "year", "concepts", "concepts_scores", "retain_in_final_P"}
    missing = sorted(required_columns.difference(publications.columns))
    if missing:
        raise SystemExit("Missing required publication field(s): " + ", ".join(missing))
    if publications["id"].astype(str).duplicated().any():
        raise SystemExit("P contains duplicate publication IDs. Resolve restarted Step 1 before continuing.")
    if not publications["retain_in_final_P"].astype(bool).all():
        raise SystemExit("The input includes a record not retained by the final v6 screen. Use publication_anchor_final.pkl unchanged.")

    n_pubs = len(publications)
    min_prevalence_n = math.ceil(n_pubs * MIN_PREVALENCE_SHARE)
    raw = build_raw_concept_records(publications)
    labels_without_scores = raw.attrs["labels_without_scores"]

    # One reproducible P--concept pair per canonical label. Duplicate raw
    # records are collapsed using max relevance, preserving the strongest
    # Dimensions-assigned relevance for that publication-label pair.
    raw = raw.sort_values(["publication_id", "concept", "relevance", "concept_display"], ascending=[True, True, False, True])
    edges = (
        raw.groupby(["publication_id", "concept"], as_index=False)
        .agg(
            concept_display=("concept_display", "first"),
            relevance=("relevance", "max"),
            n_raw_records_collapsed=("relevance", "size"),
        )
    )
    edges["is_generic_label"] = edges["concept"].isin(GENERIC_LABEL_STOPLIST)
    edges["primary_relevance_eligible"] = edges["relevance"] >= PRIMARY_RELEVANCE_THRESHOLD
    edges["primary_pair_retained"] = edges["primary_relevance_eligible"] & ~edges["is_generic_label"]

    # Concept-level summary is based on deduplicated P--concept pairs.
    concept_summary = (
        edges.groupby("concept", as_index=False)
        .agg(
            concept_display=("concept_display", "first"),
            n_publications=("publication_id", "nunique"),
            mean_relevance=("relevance", "mean"),
            median_relevance=("relevance", "median"),
            max_relevance=("relevance", "max"),
            n_duplicate_raw_records_collapsed=("n_raw_records_collapsed", lambda values: int((values - 1).sum())),
            is_generic_label=("is_generic_label", "first"),
        )
    )
    concept_summary["pct_of_P"] = 100 * concept_summary["n_publications"] / n_pubs

    primary_counts = (
        edges.loc[edges["primary_pair_retained"]]
        .groupby("concept", as_index=False)
        .agg(
            n_primary_publications=("publication_id", "nunique"),
            mean_primary_relevance=("relevance", "mean"),
            median_primary_relevance=("relevance", "median"),
        )
    )
    concept_summary = concept_summary.merge(primary_counts, on="concept", how="left", validate="one_to_one")
    concept_summary[["n_primary_publications"]] = concept_summary[["n_primary_publications"]].fillna(0).astype(int)
    concept_summary["pct_of_P_primary"] = 100 * concept_summary["n_primary_publications"] / n_pubs
    concept_summary["eligible_for_later_association"] = (
        ~concept_summary["is_generic_label"]
        & (concept_summary["n_primary_publications"] >= min_prevalence_n)
    )
    concept_summary = concept_summary.sort_values(["eligible_for_later_association", "n_primary_publications", "concept"], ascending=[False, False, True]).reset_index(drop=True)

    # Sensitivity reports use the same de-duplicated data and stoplist.
    sensitivity_rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        eligible_edges = edges[(edges["relevance"] >= threshold) & ~edges["is_generic_label"]]
        counts = eligible_edges.groupby("concept")["publication_id"].nunique()
        sensitivity_rows.append({
            "relevance_threshold": threshold,
            "n_retained_P_concept_pairs": int(len(eligible_edges)),
            "n_distinct_non_generic_concepts": int(counts.shape[0]),
            "n_concepts_at_or_above_1pct_of_P": int((counts >= min_prevalence_n).sum()),
            "min_prevalence_n": min_prevalence_n,
        })
    threshold_sensitivity = pd.DataFrame(sensitivity_rows)

    exclusion_audit = concept_summary[[
        "concept", "concept_display", "is_generic_label", "n_publications", "pct_of_P",
        "n_primary_publications", "pct_of_P_primary", "eligible_for_later_association",
    ]].copy()
    exclusion_audit["decision"] = "retained_for_later_association" 
    exclusion_audit.loc[exclusion_audit["is_generic_label"], "decision"] = "excluded_generic_label_stoplist"
    exclusion_audit.loc[
        (~exclusion_audit["is_generic_label"]) & (exclusion_audit["n_primary_publications"] < min_prevalence_n),
        "decision"
    ] = "not_eligible_below_1pct_prevalence"

    analysis_candidates = concept_summary.loc[concept_summary["eligible_for_later_association"]].copy()
    primary_edges = edges.loc[edges["primary_pair_retained"]].copy()

    # Map only primary non-generic P--concept edges to the four approved,
    # non-exclusive substantive research-topic families.
    family_frames = []
    for family, patterns in CONCEPT_FAMILIES.items():
        subset = primary_edges.copy()
        subset["matched_codebook_labels"] = subset["concept"].apply(
            lambda value: "; ".join(matched_family_labels(value, patterns))
        )
        subset = subset[subset["matched_codebook_labels"].str.len() > 0].copy()
        subset["concept_family"] = family
        family_frames.append(subset[[
            "publication_id", "concept_family", "concept", "concept_display",
            "relevance", "matched_codebook_labels",
        ]])
    family_matches = pd.concat(family_frames, ignore_index=True)
    family_matches = family_matches.sort_values(["concept_family", "publication_id", "concept"]).reset_index(drop=True)

    publication_families = publications[["id", "title", "year"]].rename(columns={"id": "publication_id"}).copy()
    for family in CONCEPT_FAMILIES:
        subset = family_matches[family_matches["concept_family"] == family]
        aggregated = (
            subset.groupby("publication_id", as_index=False)
            .agg(
                **{
                    family: ("concept", lambda values: 1),
                    f"{family}_matched_concepts": ("concept_display", stable_join),
                    f"{family}_matched_codebook_labels": ("matched_codebook_labels", stable_join),
                    f"{family}_n_matched_concepts": ("concept", "nunique"),
                    f"{family}_max_relevance": ("relevance", "max"),
                }
            )
        )
        publication_families = publication_families.merge(aggregated, on="publication_id", how="left", validate="one_to_one")
        publication_families[family] = publication_families[family].fillna(0).astype(int)
        publication_families[f"{family}_n_matched_concepts"] = publication_families[f"{family}_n_matched_concepts"].fillna(0).astype(int)

    family_columns = list(CONCEPT_FAMILIES)
    publication_families["n_concept_families"] = publication_families[family_columns].sum(axis=1)
    publication_families["any_concept_family"] = publication_families["n_concept_families"].gt(0).astype(int)
    publication_families = publication_families.sort_values("publication_id").reset_index(drop=True)

    family_summary_rows = []
    for family in family_columns:
        subset = family_matches[family_matches["concept_family"] == family]
        family_summary_rows.append({
            "concept_family": family,
            "n_publications": int(publication_families[family].sum()),
            "pct_of_P": round(100 * publication_families[family].mean(), 3),
            "n_distinct_matched_concepts": int(subset["concept"].nunique()),
            "n_P_concept_match_rows": int(len(subset)),
            "median_matched_relevance": round(float(subset["relevance"].median()), 6) if not subset.empty else None,
        })
    family_summary = pd.DataFrame(family_summary_rows)

    family_names = publication_families[family_columns].apply(
        lambda row: " + ".join([family for family in family_columns if row[family] == 1]) if row.sum() else "none",
        axis=1,
    )
    family_overlap = family_names.value_counts().rename_axis("family_combination").reset_index(name="n_publications")
    family_overlap["pct_of_P"] = (100 * family_overlap["n_publications"] / len(publication_families)).round(3)

    family_codebook = pd.DataFrame([
        {
            "concept_family": family,
            "codebook_label": label,
            "concept_label_regex": regex,
            "membership_rule": f"At least one primary non-generic Dimensions concept matches this pattern (relevance >= {PRIMARY_RELEVANCE_THRESHOLD:.2f}).",
            "family_structure": "Non-exclusive; a publication may belong to multiple families.",
        }
        for family, patterns in CONCEPT_FAMILIES.items()
        for label, regex in patterns
    ])

    family_audit_sample = (
        family_matches.merge(
            publications[["id", "title", "year"]].rename(columns={"id": "publication_id"}),
            on="publication_id", how="left", validate="many_to_one"
        )
        .sort_values(["concept_family", "relevance", "publication_id", "concept"], ascending=[True, False, True, True])
        .groupby("concept_family", group_keys=False)
        .head(25)
        .reset_index(drop=True)
    )

    family_validation = pd.DataFrame([
        {"check": "All family matches originate in primary non-generic P--concept edges", "n_failures": int((family_matches["relevance"] < PRIMARY_RELEVANCE_THRESHOLD).sum())},
        {"check": "All family-match publication IDs occur in final P", "n_failures": int((~family_matches["publication_id"].isin(set(publications["id"].astype(str)))).sum())},
        {"check": "No duplicate P--concept--family matches", "n_failures": int(family_matches.duplicated(["publication_id", "concept", "concept_family"]).sum())},
        {"check": "All four approved families have at least one matched P publication", "n_failures": int((family_summary["n_publications"] == 0).sum())},
    ])
    if family_validation["n_failures"].sum() != 0:
        raise SystemExit("Internal topic-family validation failed. No output has been certified.")

    codebook = pd.DataFrame([
        {
            "rule_component": "unit",
            "rule": "One row per publication--canonical-concept pair",
            "parameter": "publication ID plus casefolded/whitespace-normalised Dimensions concept label",
        },
        {
            "rule_component": "duplicate_pairs",
            "rule": "Collapse duplicate raw P--concept records",
            "parameter": "retain maximum relevance per P--concept pair",
        },
        {
            "rule_component": "primary_relevance_threshold",
            "rule": "Retain a non-generic P--concept pair for primary concept analysis",
            "parameter": f"relevance >= {PRIMARY_RELEVANCE_THRESHOLD:.2f}",
        },
        {
            "rule_component": "minimum_concept_prevalence",
            "rule": "Retain a concept as a later association candidate",
            "parameter": f">= {MIN_PREVALENCE_SHARE:.0%} of P (>= {min_prevalence_n} publications)",
        },
        {
            "rule_component": "generic_labels",
            "rule": "Exclude generic scholarly/method/rhetorical concept labels",
            "parameter": "; ".join(sorted(GENERIC_LABEL_STOPLIST)),
        },
        {
            "rule_component": "sensitivity_thresholds",
            "rule": "Descriptive threshold sensitivity only",
            "parameter": ", ".join(f"{threshold:.2f}" for threshold in SENSITIVITY_THRESHOLDS),
        },
        {
            "rule_component": "interpretive_boundary",
            "rule": "Dimensions concepts are supplementary publication attributes",
            "parameter": "not text indicators; not policy-document semantics; not causal measures",
        },
    ])

    construction_summary = pd.DataFrame([
        {"metric": "P publications", "value": n_pubs},
        {"metric": "raw concept-relevance records", "value": len(raw)},
        {"metric": "deduplicated P--concept pairs", "value": len(edges)},
        {"metric": "raw duplicate records collapsed", "value": int((edges["n_raw_records_collapsed"] - 1).sum())},
        {"metric": "labels without usable score", "value": labels_without_scores},
        {"metric": "distinct canonical concepts", "value": concept_summary.shape[0]},
        {"metric": "generic labels on stoplist", "value": int(concept_summary["is_generic_label"].sum())},
        {"metric": f"primary P--concept pairs (relevance >= {PRIMARY_RELEVANCE_THRESHOLD:.2f}; non-generic)", "value": len(primary_edges)},
        {"metric": f"concept candidates >= 1% of P (n >= {min_prevalence_n})", "value": len(analysis_candidates)},
        {"metric": "P--concept--family matched rows", "value": len(family_matches)},
        {"metric": "P publications with any approved topic family", "value": int(publication_families["any_concept_family"].sum())},
    ])

    # Large reusable datasets are saved as pickle files; compact audit outputs
    # are CSVs suitable for review and upload.
    edges.to_pickle(OUTPUT_DIR / "publication_concept_edges_all.pkl")
    primary_edges.to_pickle(OUTPUT_DIR / "publication_concept_edges_primary.pkl")
    family_matches.to_pickle(OUTPUT_DIR / "publication_concept_family_matches.pkl")
    publication_families.to_pickle(OUTPUT_DIR / "publication_concept_families.pkl")
    analysis_candidates.to_csv(OUTPUT_DIR / "concept_analysis_candidates.csv", index=False)
    concept_summary.to_csv(OUTPUT_DIR / "concept_prevalence_all.csv", index=False)
    exclusion_audit.to_csv(OUTPUT_DIR / "concept_exclusion_audit.csv", index=False)
    threshold_sensitivity.to_csv(OUTPUT_DIR / "concept_threshold_sensitivity.csv", index=False)
    construction_summary.to_csv(OUTPUT_DIR / "concept_construction_summary.csv", index=False)
    codebook.to_csv(OUTPUT_DIR / "concept_processing_codebook.csv", index=False)
    family_summary.to_csv(OUTPUT_DIR / "concept_family_summary.csv", index=False)
    family_overlap.to_csv(OUTPUT_DIR / "concept_family_overlap.csv", index=False)
    family_codebook.to_csv(OUTPUT_DIR / "concept_family_codebook.csv", index=False)
    family_audit_sample.to_csv(OUTPUT_DIR / "concept_family_audit_sample.csv", index=False)
    family_validation.to_csv(OUTPUT_DIR / "concept_family_validation_checks.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": "step3_v6_construct_topic_families_2.0_named_conditions_no_standalone_assessment",
        "source_dataset": "linked_final_outputs_v6/publication_anchor_final.pkl",
        "source_publication_construction": "Dimensions candidate retrieval followed by the approved literal title/abstract screen",
        "step2_indicator_timestamp_utc": json.loads(STEP2_MANIFEST.read_text(encoding="utf-8")).get("run_timestamp_utc"),
        "unit_of_analysis": "publication--Dimensions-concept pair",
        "canonicalization": "casefold and whitespace normalisation",
        "duplicate_pair_rule": "retain maximum available relevance per P--concept pair",
        "primary_relevance_threshold": PRIMARY_RELEVANCE_THRESHOLD,
        "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
        "minimum_concept_prevalence_share": MIN_PREVALENCE_SHARE,
        "minimum_concept_prevalence_n": min_prevalence_n,
        "generic_label_stoplist": sorted(GENERIC_LABEL_STOPLIST),
        "topic_family_structure": "Four manually approved, non-exclusive substantive research-topic families derived only from primary non-generic P--concept edges.",
        "families": list(CONCEPT_FAMILIES),
        "interpretive_boundary": "Concepts and topic families are supplementary Dimensions-generated attributes of P publications. They are not merged with text indicators, used to infer policy-document topical similarity, or interpreted as causal mechanisms, institutional intentions, or policy impact.",
        "output_files": [
            "publication_concept_edges_all.pkl",
            "publication_concept_edges_primary.pkl",
            "publication_concept_family_matches.pkl",
            "publication_concept_families.pkl",
            "concept_analysis_candidates.csv",
            "concept_prevalence_all.csv",
            "concept_exclusion_audit.csv",
            "concept_threshold_sensitivity.csv",
            "concept_construction_summary.csv",
            "concept_processing_codebook.csv",
            "concept_family_summary.csv",
            "concept_family_overlap.csv",
            "concept_family_codebook.csv",
            "concept_family_audit_sample.csv",
            "concept_family_validation_checks.csv",
            "step3_topic_family_manifest.json",
        ],
    }
    (OUTPUT_DIR / "step3_topic_family_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Restarted Step 3 (v6 topic families) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Deduplicated P--concept pairs: {len(edges):,}")
    print(f"Primary non-generic P--concept pairs: {len(primary_edges):,}")
    print(f"P publications with any approved topic family: {int(publication_families['any_concept_family'].sum()):,}")
    print(f"Topic-family validation failures: {int(family_validation['n_failures'].sum())}")


if __name__ == "__main__":
    main()
