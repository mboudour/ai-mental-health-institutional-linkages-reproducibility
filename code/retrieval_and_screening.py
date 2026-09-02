#!/usr/bin/env python3
"""Create the final screened publication anchor and retrieve its linked records.

Place this script in new_version/data/ and run from new_version/:
    python data/screen_v5_and_retrieve_linked_v6.py --key-file key.txt

Stage 1 already occurred in Dimensions: linked_final_outputs_v5/publication_anchor.pkl
is the saved candidate publication retrieval. This script performs Stage 2:
    1. Apply the approved local, literal title/abstract screen to the v5 candidate P.
    2. Save a complete inclusion/exclusion decision table with matched terms.
    3. Retrieve grants only from the retained P.supporting_grant_ids.
    4. Retrieve policy documents only where publication_ids contains a retained P ID.

It does not modify the v5 candidate files. The final output is a new v6 directory.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import dimcli
except ImportError as exc:
    raise SystemExit("Missing dependency: install dimcli in your active environment first, for example: pip install dimcli") from exc

SCRIPT_VERSION = "v6.0-two-stage-literal-screen"
YEAR_START = 2000
YEAR_END = 2025
CHUNK_SIZE = 250
AUDIT_SAMPLE_SIZE = 100
RANDOM_SEED = 42
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3

# ---------------------------------------------------------------------------
# APPROVED LOCAL SCREENING RULE
# A candidate publication is retained if at least one term from EACH list is
# visibly present (case-insensitive) in its title OR available abstract.
# No wildcard/stemming, semantic expansion, concept field, or manual decision
# is used. Every match and decision is saved.
# ---------------------------------------------------------------------------
AI_ML_TERMS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "LLM",
]

CORE_MENTAL_HEALTH_TERMS = [
    "mental health",
    "mental illness",
    "psychiatry",
    "psychiatric",
    "psychotherapy",
    "psychotherapeutic",
    "mental disorder",
    "psychiatric disorder",
]

GRANT_RETURN_FIELDS = '''
id + start_year + title + abstract + funding_usd + funder_orgs
+ research_org_countries + research_org_names
'''

POLICY_RETURN_FIELDS = '''
id + year + title + publisher_org + publisher_org_country + publication_ids
'''


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=root / "data" / "linked_final_outputs_v5",
        help="v5 candidate retrieval directory (default: new_version/data/linked_final_outputs_v5).",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=root / "key.txt",
        help="Dimensions API key file (default: new_version/key.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "linked_final_outputs_v6",
        help="Output directory (default: new_version/data/linked_final_outputs_v6).",
    )
    return parser.parse_args()


def normalise_text(value: Any) -> str:
    """Normalise only case, whitespace, and Unicode hyphen variants."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_pattern(term: str) -> re.Pattern[str]:
    """Literal phrase match with word boundaries; no stemming or wildcard."""
    escaped = re.escape(normalise_text(term)).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if phrase_pattern(term).search(text)]


def screen_candidate_publications(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["id", "title", "abstract", "supporting_grant_ids"]
    missing = [field for field in required if field not in candidate.columns]
    if missing:
        raise SystemExit(f"Candidate P is missing required field(s): {', '.join(missing)}")

    candidate = candidate.copy()
    candidate["id"] = candidate["id"].astype(str)
    if candidate["id"].duplicated().any():
        raise SystemExit("Candidate P contains duplicate Dimensions publication IDs.")

    decisions: list[dict[str, Any]] = []
    for _, row in candidate.iterrows():
        title = normalise_text(row["title"])
        abstract = normalise_text(row["abstract"])
        ai_title = matched_terms(title, AI_ML_TERMS)
        ai_abstract = matched_terms(abstract, AI_ML_TERMS)
        mh_title = matched_terms(title, CORE_MENTAL_HEALTH_TERMS)
        mh_abstract = matched_terms(abstract, CORE_MENTAL_HEALTH_TERMS)
        ai_all = sorted(set(ai_title + ai_abstract))
        mh_all = sorted(set(mh_title + mh_abstract))
        retained = bool(ai_all) and bool(mh_all)
        if retained:
            reason = "retained: explicit AI/ML and core mental-health terms in title and/or abstract"
        elif not ai_all and not mh_all:
            reason = "excluded: no approved AI/ML term and no approved core mental-health term in title/abstract"
        elif not ai_all:
            reason = "excluded: no approved AI/ML term in title/abstract"
        else:
            reason = "excluded: no approved core mental-health term in title/abstract"
        decisions.append({
            "id": row["id"],
            "year": row.get("year"),
            "title": row.get("title"),
            "abstract_available": bool(abstract),
            "ai_ml_terms_title": "; ".join(ai_title),
            "ai_ml_terms_abstract": "; ".join(ai_abstract),
            "core_mental_health_terms_title": "; ".join(mh_title),
            "core_mental_health_terms_abstract": "; ".join(mh_abstract),
            "ai_ml_terms_anywhere": "; ".join(ai_all),
            "core_mental_health_terms_anywhere": "; ".join(mh_all),
            "retain_in_final_P": retained,
            "decision_reason": reason,
        })

    decisions_df = pd.DataFrame(decisions)
    retained_ids = set(decisions_df.loc[decisions_df["retain_in_final_P"], "id"])
    final_p = candidate[candidate["id"].isin(retained_ids)].copy().reset_index(drop=True)
    final_p = final_p.merge(
        decisions_df.drop(columns=["year", "title"]),
        on="id",
        how="left",
        validate="one_to_one",
    )
    return decisions_df, final_p


def flatten_ids(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    return [text] if text and text.lower() != "nan" else []


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def dsl_list(values: list[str]) -> str:
    return ", ".join(json.dumps(value) for value in values)


def save_frame(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_pickle(output_dir / f"{stem}.pkl")
    frame.to_excel(output_dir / f"{stem}.xlsx", index=False)


def write_query_log(entries: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "linked_retrieval_query_log.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_query(dsl: Any, query: str, label: str) -> pd.DataFrame:
    delay = RETRY_DELAY_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Running {label} (attempt {attempt}/{MAX_RETRIES})")
            frame = dsl.query_iterative(query, verbose=True, force=True).as_dataframe()
            if frame.empty:
                print(f"Retrieved 0 records for {label}.")
                return pd.DataFrame(columns=["id"])
            if "id" not in frame.columns:
                raise RuntimeError(f"Non-empty Dimensions response for {label} has no id column: {list(frame.columns)}")
            frame["id"] = frame["id"].astype(str)
            frame = frame.drop_duplicates(subset="id").reset_index(drop=True)
            print(f"Retrieved {len(frame):,} unique records for {label}.")
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise SystemExit(f"Dimensions retrieval failed for {label} after {MAX_RETRIES} attempts: {exc}") from exc
            print(f"{label} failed: {exc}. Retrying in {delay} seconds.")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("Unreachable retry state")


def retrieve_linked(
    dsl: Any,
    source: str,
    relation_field: str,
    related_ids: list[str],
    return_fields: str,
    output_dir: Path,
    query_log: list[dict[str, Any]],
) -> pd.DataFrame:
    if not related_ids:
        return pd.DataFrame(columns=["id"])
    output: list[pd.DataFrame] = []
    total = (len(related_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    year_filter = f"start_year in [{YEAR_START}:{YEAR_END}]" if source == "grants" else f"year in [{YEAR_START}:{YEAR_END}]"
    for n, batch in enumerate(chunks(related_ids, CHUNK_SIZE), start=1):
        query = f'''search {source}
where {relation_field} in [{dsl_list(batch)}]
and {year_filter}
return {source}[{return_fields}]'''
        query_log.append({
            "stage": source,
            "batch": n,
            "batches_total": total,
            "relationship_field": relation_field,
            "query": query,
        })
        write_query_log(query_log, output_dir)
        output.append(run_query(dsl, query, f"{source} linked-record batch {n}/{total}"))
    combined = pd.concat(output, ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    return combined


def main() -> None:
    args = parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    key_file = args.key_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    candidate_file = candidate_dir / "publication_anchor.pkl"

    if not candidate_file.is_file():
        raise SystemExit(f"Candidate publication file not found: {candidate_file}")
    if not key_file.is_file():
        raise SystemExit(f"Dimensions key file not found: {key_file}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = pd.read_pickle(candidate_file)
    decisions, final_p = screen_candidate_publications(candidate)
    decisions.to_csv(output_dir / "publication_screening_decisions.csv", index=False)
    save_frame(final_p, output_dir, "publication_anchor_final")

    screen_codebook = pd.DataFrame(
        [{"block": "AI/ML", "term": term, "match_rule": "case-insensitive literal phrase in title or abstract"} for term in AI_ML_TERMS]
        + [{"block": "Core mental health", "term": term, "match_rule": "case-insensitive literal phrase in title or abstract"} for term in CORE_MENTAL_HEALTH_TERMS]
    )
    screen_codebook.to_csv(output_dir / "publication_screening_codebook.csv", index=False)

    candidate_query = ""
    candidate_summary_file = candidate_dir / "retrieval_summary.json"
    if candidate_summary_file.is_file():
        candidate_summary = json.loads(candidate_summary_file.read_text(encoding="utf-8"))
        candidate_query = candidate_summary.get("publication_anchor_query", "")
    if not candidate_query and (candidate_dir / "publication_anchor_query.dsl").is_file():
        candidate_query = (candidate_dir / "publication_anchor_query.dsl").read_text(encoding="utf-8")

    retained = decisions.loc[decisions["retain_in_final_P"]]
    excluded = decisions.loc[~decisions["retain_in_final_P"]]
    screening_summary = {
        "script_version": SCRIPT_VERSION,
        "candidate_source_directory": str(candidate_dir),
        "candidate_publications_n": int(len(candidate)),
        "final_screened_publications_n": int(len(final_p)),
        "excluded_candidate_publications_n": int(len(excluded)),
        "candidate_dimensions_query": candidate_query,
        "local_screen_rule": "Retain only where at least one approved AI/ML phrase AND at least one approved core mental-health phrase occurs in title or available abstract.",
        "ai_ml_terms": AI_ML_TERMS,
        "core_mental_health_terms": CORE_MENTAL_HEALTH_TERMS,
        "screening_boundary": "The local screen was added because controlled checks showed Dimensions title_abstract_only retrieval can return semantic or related matches beyond the literal terms intended. It is deterministic and stores all matched terms and exclusion decisions.",
    }
    (output_dir / "screening_manifest.json").write_text(json.dumps(screening_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Deterministic samples preserve auditability of both retained and excluded candidates.
    (retained.sample(min(AUDIT_SAMPLE_SIZE, len(retained)), random_state=RANDOM_SEED)).to_excel(
        output_dir / "audit_sample_screened_publications_retained.xlsx", index=False
    )
    (excluded.sample(min(AUDIT_SAMPLE_SIZE, len(excluded)), random_state=RANDOM_SEED)).to_excel(
        output_dir / "audit_sample_screened_publications_excluded.xlsx", index=False
    )

    if final_p.empty:
        raise SystemExit("The local screen retained zero publications. No linked retrieval was attempted.")
    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("Dimensions key file is empty.")

    run_status = {
        "script_version": SCRIPT_VERSION,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "screening_complete",
        "candidate_publications_n": int(len(candidate)),
        "final_screened_publications_n": int(len(final_p)),
    }
    (output_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")

    dimcli.login(key=api_key, endpoint="https://app.dimensions.ai")
    dsl = dimcli.Dsl()
    query_log: list[dict[str, Any]] = []

    grant_ids = sorted({grant_id for value in final_p["supporting_grant_ids"] for grant_id in flatten_ids(value)})
    grants = retrieve_linked(dsl, "grants", "id", grant_ids, GRANT_RETURN_FIELDS, output_dir, query_log)
    if not grants.empty:
        required_grant_fields = ["id", "start_year", "title", "abstract", "funding_usd", "funder_orgs", "research_org_countries", "research_org_names"]
        missing = [field for field in required_grant_fields if field not in grants.columns]
        if missing:
            raise SystemExit(f"Linked grants missing requested field(s): {', '.join(missing)}")
    save_frame(grants, output_dir, "linked_grants_final")
    run_status.update({"status": "linked_grants_saved", "linked_grants_n": int(len(grants))})
    (output_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")

    p_ids = final_p["id"].astype(str).tolist()
    policy = retrieve_linked(dsl, "policy_documents", "publication_ids", p_ids, POLICY_RETURN_FIELDS, output_dir, query_log)
    if not policy.empty:
        required_policy_fields = ["id", "year", "title", "publication_ids"]
        missing = [field for field in required_policy_fields if field not in policy.columns]
        if missing:
            raise SystemExit(f"Linked policy documents missing requested field(s): {', '.join(missing)}")
    save_frame(policy, output_dir, "linked_policy_documents_final")
    run_status.update({"status": "completed", "linked_policy_documents_n": int(len(policy)), "completed_utc": datetime.now(timezone.utc).isoformat()})
    (output_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")

    (grants.sample(min(AUDIT_SAMPLE_SIZE, len(grants)), random_state=RANDOM_SEED) if not grants.empty else grants).to_excel(
        output_dir / "audit_sample_linked_grants.xlsx", index=False
    )
    (policy.sample(min(AUDIT_SAMPLE_SIZE, len(policy)), random_state=RANDOM_SEED) if not policy.empty else policy).to_excel(
        output_dir / "audit_sample_linked_policy.xlsx", index=False
    )

    final_summary = {
        **screening_summary,
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_P_with_any_supporting_grant_ids_n": int(final_p["supporting_grant_ids"].apply(lambda x: bool(flatten_ids(x))).sum()),
        "final_P_distinct_supporting_grant_ids_n": len(grant_ids),
        "linked_grants_n": int(len(grants)),
        "linked_policy_documents_n": int(len(policy)),
        "relationship_fields": {
            "publication_to_grant": "final P.supporting_grant_ids; retrieved through grants.id",
            "policy_document_to_publication": "policy_documents.publication_ids; queried using final P IDs",
        },
        "deduplication_rule": "Candidate P is unique by Dimensions ID. Each linked Dimensions query and concatenated batch result is deduplicated by record ID, retaining the first returned row.",
        "interpretive_boundary": "Policy documents are linked through recorded publication IDs. Linkage does not establish topic similarity, policy impact, endorsement, institutional intention, or causal effects.",
    }
    (output_dir / "final_retrieval_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("TWO-STAGE PUBLICATION-ANCHORED RETRIEVAL COMPLETE")
    print("=" * 72)
    print(f"Dimensions candidate P:             {len(candidate):,}")
    print(f"Final text-screened P:              {len(final_p):,}")
    print(f"Retrieved linked grants:            {len(grants):,}")
    print(f"Retrieved linked policy documents:  {len(policy):,}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
