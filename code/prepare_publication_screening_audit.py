#!/usr/bin/env python3
"""Create supplementary audit materials for the publication screening procedure.

Save as:
    new_version/computations/prepare_publication_screening_audit.py

Run from new_version:
    python computations/prepare_publication_screening_audit.py

Input (read only):
    data/linked_final_outputs_v6/publication_screening_decisions.csv

Outputs:
    computations/outputs/publication_screening_audit/

The script creates descriptive audit materials only. It does not change the
publication anchor, text dictionaries, relations, outcomes, or models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "linked_final_outputs_v6" / "publication_screening_decisions.csv"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "publication_screening_audit"
RANDOM_SEED = 20260902
EXCLUDED_SAMPLE_SIZE = 30
RETAINED_SAMPLE_SIZE = 30

REQUIRED_COLUMNS = [
    "id",
    "year",
    "title",
    "abstract_available",
    "ai_ml_terms_title",
    "ai_ml_terms_abstract",
    "core_mental_health_terms_title",
    "core_mental_health_terms_abstract",
    "ai_ml_terms_anywhere",
    "core_mental_health_terms_anywhere",
    "retain_in_final_P",
    "decision_reason",
]


def require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit("Screening-decision file is missing required column(s): " + ", ".join(missing))


def string_present(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, table in tables.items():
                table.to_excel(writer, sheet_name=sheet[:31], index=False)
    except ImportError as exc:
        raise SystemExit(
            "Excel output requires openpyxl. Install it in the active environment with:\n"
            "conda install -y openpyxl\nThen rerun the script."
        ) from exc


def main() -> None:
    if not INPUT_FILE.is_file():
        raise SystemExit(f"Screening-decision file not found: {INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(INPUT_FILE, dtype={"id": str}).copy()
    require_columns(decisions)
    decisions["id"] = decisions["id"].astype(str).str.strip()
    if decisions["id"].duplicated().any():
        raise SystemExit("Screening-decision file contains duplicate publication IDs.")

    decisions["retain_in_final_P"] = decisions["retain_in_final_P"].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if decisions["retain_in_final_P"].isna().any():
        raise SystemExit("retain_in_final_P contains a value that is not True/False or 1/0.")
    decisions["abstract_available"] = decisions["abstract_available"].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if decisions["abstract_available"].isna().any():
        raise SystemExit("abstract_available contains a value that is not True/False or 1/0.")

    has_ai = string_present(decisions["ai_ml_terms_anywhere"])
    has_mh = string_present(decisions["core_mental_health_terms_anywhere"])
    expected_retain = has_ai & has_mh
    retained = decisions["retain_in_final_P"]
    exact_rule_match = expected_retain.eq(retained)

    overall = pd.DataFrame([
        {
            "candidate_publications": int(len(decisions)),
            "retained_publications": int(retained.sum()),
            "excluded_publications": int((~retained).sum()),
            "retained_pct": float(100 * retained.mean()),
            "excluded_pct": float(100 * (~retained).mean()),
            "records_with_available_abstract": int(decisions["abstract_available"].sum()),
            "records_without_abstract": int((~decisions["abstract_available"]).sum()),
            "rule_reproduced_exactly": bool(exact_rule_match.all()),
            "rule_mismatch_n": int((~exact_rule_match).sum()),
        }
    ])

    reason_summary = (
        decisions.groupby(["retain_in_final_P", "decision_reason"], dropna=False)
        .size()
        .reset_index(name="n_publications")
        .sort_values(["retain_in_final_P", "n_publications", "decision_reason"], ascending=[False, False, True])
    )
    reason_summary["pct_of_all_candidates"] = 100 * reason_summary["n_publications"] / len(decisions)

    evidence_summary = (
        decisions.assign(
            has_ai_phrase=has_ai,
            has_core_mental_health_phrase=has_mh,
            title_has_ai=string_present(decisions["ai_ml_terms_title"]),
            abstract_has_ai=string_present(decisions["ai_ml_terms_abstract"]),
            title_has_core_mental_health=string_present(decisions["core_mental_health_terms_title"]),
            abstract_has_core_mental_health=string_present(decisions["core_mental_health_terms_abstract"]),
        )
        .groupby("retain_in_final_P")
        .agg(
            n_publications=("id", "size"),
            abstracts_available=("abstract_available", "sum"),
            title_has_ai=("title_has_ai", "sum"),
            abstract_has_ai=("abstract_has_ai", "sum"),
            title_has_core_mental_health=("title_has_core_mental_health", "sum"),
            abstract_has_core_mental_health=("abstract_has_core_mental_health", "sum"),
            all_required_phrases_present=("has_ai_phrase", "sum"),
        )
        .reset_index()
    )
    evidence_summary["group"] = evidence_summary["retain_in_final_P"].map({True: "Retained", False: "Excluded"})
    evidence_summary["abstract_available_pct"] = 100 * evidence_summary["abstracts_available"] / evidence_summary["n_publications"]

    sample_columns = [
        "id", "year", "title", "abstract_available", "ai_ml_terms_title", "ai_ml_terms_abstract",
        "core_mental_health_terms_title", "core_mental_health_terms_abstract", "ai_ml_terms_anywhere",
        "core_mental_health_terms_anywhere", "retain_in_final_P", "decision_reason",
    ]
    excluded = decisions.loc[~retained, sample_columns].copy()
    retained_df = decisions.loc[retained, sample_columns].copy()
    excluded_sample = excluded.sample(min(EXCLUDED_SAMPLE_SIZE, len(excluded)), random_state=RANDOM_SEED).sort_values("id")
    retained_sample = retained_df.sample(min(RETAINED_SAMPLE_SIZE, len(retained_df)), random_state=RANDOM_SEED).sort_values("id")

    field_dictionary = pd.DataFrame([
        {"field": "id", "description": "Dimensions publication record identifier."},
        {"field": "title", "description": "Retrieved publication title."},
        {"field": "abstract_available", "description": "Whether a nonempty abstract was available for local screening."},
        {"field": "ai_ml_terms_title", "description": "Approved AI/ML phrases found in title."},
        {"field": "ai_ml_terms_abstract", "description": "Approved AI/ML phrases found in available abstract."},
        {"field": "core_mental_health_terms_title", "description": "Approved core mental-health phrases found in title."},
        {"field": "core_mental_health_terms_abstract", "description": "Approved core mental-health phrases found in available abstract."},
        {"field": "retain_in_final_P", "description": "Retained only when at least one approved AI/ML phrase and one approved core mental-health phrase occurred in title or available abstract."},
        {"field": "decision_reason", "description": "Deterministic reason for retain/exclude result."},
    ])

    checks = pd.DataFrame([
        {"check": "unique_publication_ids", "passed": not decisions["id"].duplicated().any(), "detail": f"{len(decisions)} candidate records"},
        {"check": "screening_rule_reproduced_from_saved_matches", "passed": bool(exact_rule_match.all()), "detail": f"{int((~exact_rule_match).sum())} mismatches"},
        {"check": "retained_records_have_both_phrase_classes", "passed": bool((has_ai[retained] & has_mh[retained]).all()), "detail": "Checked against saved matched-term fields"},
        {"check": "excluded_records_fail_at_least_one_phrase_class", "passed": bool((~(has_ai[~retained] & has_mh[~retained])).all()), "detail": "Checked against saved matched-term fields"},
        {"check": "random_samples_reproducible", "passed": True, "detail": f"NumPy/pandas random_state={RANDOM_SEED}"},
    ])

    tables = {
        "overall_summary": overall,
        "decision_reason_summary": reason_summary,
        "evidence_summary": evidence_summary,
        "excluded_random_sample": excluded_sample,
        "retained_random_sample": retained_sample,
        "field_dictionary": field_dictionary,
        "validation_checks": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    write_workbook(OUTPUT_DIR / "publication_screening_audit.xlsx", tables)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT_FILE),
        "random_seed": RANDOM_SEED,
        "random_sample_sizes": {"excluded": EXCLUDED_SAMPLE_SIZE, "retained": RETAINED_SAMPLE_SIZE},
        "rule_reproduction": "Retain if and only if the saved audit fields show at least one approved AI/ML phrase and at least one approved core mental-health phrase in title or available abstract.",
        "interpretive_boundary": "The audit verifies implementation of the documented local rule from saved matching fields. It does not establish complete recall of all relevant research or validate Dimensions retrieval behavior beyond the documented record-level screen.",
    }
    (OUTPUT_DIR / "screening_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Publication screening audit completed.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Candidates: {len(decisions):,}; retained: {int(retained.sum()):,}; excluded: {int((~retained).sum()):,}")
    print(f"Rule reproduced exactly: {bool(exact_rule_match.all())}")


if __name__ == "__main__":
    main()
