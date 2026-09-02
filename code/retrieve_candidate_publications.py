#!/usr/bin/env python3
"""Retrieve the final publication-anchored Dimensions dataset in one run.

Place this script in:
    new_version/data/clean_linked_retrieval_v5.py

Run from new_version/:
    python data/clean_linked_retrieval_v5.py --key-file key.txt

The script executes one end-to-end chain:
    1. Retrieve the final publication anchor P.
    2. Extract P.supporting_grant_ids and retrieve linked grants.
    3. Use P IDs to retrieve linked policy documents through publication_ids.

No ethics/responsibility, computational-performance, or concept-family
classification occurs in this script. It retrieves raw data only, preserves
the v2 field set, and writes all query metadata and audit samples.

Default output directory:
    new_version/data/linked_final_outputs_v5/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import dimcli
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install dimcli in your active environment first, for example: pip install dimcli"
    ) from exc

SCRIPT_VERSION = "v5.0-approved-no-wildcard-core-mental-health"
YEAR_START = 2000
YEAR_END = 2025
CHUNK_SIZE = 250
AUDIT_SAMPLE_SIZE = 100
RANDOM_SEED = 42
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3

# ---------------------------------------------------------------------------
# Final publication-anchor scope
# ---------------------------------------------------------------------------
# The approved P query is deliberately separate from later text indicators.
# It contains only explicit AI/ML terms AND explicit core mental-health terms.
# It uses no wildcard characters, no bare AI abbreviation, no named-condition
# branch, and no generic clinical-context terms. The saved audit sample is part
# of the retrieval output and must be reviewed before analysis.
# ---------------------------------------------------------------------------
AI_BLOCK = '''
"artificial intelligence" OR "machine learning" OR "deep learning"
OR "neural network" OR "large language model" OR LLM
'''

CORE_MENTAL_HEALTH_BLOCK = '''
"mental health" OR "mental illness" OR psychiatry OR psychiatric
OR psychotherapy OR psychotherapeutic
OR "mental disorder" OR "psychiatric disorder"
'''

MENTAL_HEALTH_BLOCK = CORE_MENTAL_HEALTH_BLOCK

PUBLICATION_RETURN_FIELDS = '''
id + year + date + title + abstract + doi + type + document_type
+ concepts + concepts_scores
+ research_org_countries + research_orgs + times_cited
+ supporting_grant_ids
'''

GRANT_RETURN_FIELDS = '''
id + start_year + title + abstract + funding_usd + funder_orgs
+ research_org_countries + research_org_names
'''

POLICY_RETURN_FIELDS = '''
id + year + title + publisher_org + publisher_org_country + publication_ids
'''

PUBLICATION_QUERY = f'''search publications
in title_abstract_only for """
({AI_BLOCK}) AND ({MENTAL_HEALTH_BLOCK})
"""
where year in [{YEAR_START}:{YEAR_END}]
and type in ["article", "review"]
return publications[
    {PUBLICATION_RETURN_FIELDS}
]'''


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=project_root / "key.txt",
        help="Path to a text file containing only the Dimensions API key (default: new_version/key.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "linked_final_outputs_v5",
        help="Output directory (default: new_version/data/linked_final_outputs_v5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwriting a prior incomplete output directory.",
    )
    return parser.parse_args()


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def dsl_list(values: list[str]) -> str:
    """Quote Dimensions IDs safely for a DSL list."""
    return ", ".join(json.dumps(value) for value in values)


def flatten_ids(value: Any) -> list[str]:
    """Normalise Dimensions relationship fields that may be lists or missing."""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    return [text] if text and text.lower() != "nan" else []


def run_query(dsl: Any, query: str, label: str) -> pd.DataFrame:
    """Run one Dimensions query with controlled retry and ID deduplication."""
    delay = RETRY_DELAY_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Running {label} (attempt {attempt}/{MAX_RETRIES})")
            response = dsl.query_iterative(query, verbose=True, force=True)
            frame = response.as_dataframe()
            # For a valid relationship batch with no matching records, dimcli
            # returns an empty DataFrame with no columns. This is a successful
            # zero-result query, not an API error.
            if frame.empty:
                print(f"Retrieved 0 records for {label}.")
                return pd.DataFrame(columns=["id"])
            if "id" not in frame.columns:
                raise RuntimeError(
                    f"Dimensions response for {label} is non-empty but has no id column. "
                    f"Returned columns: {list(frame.columns)}"
                )
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


def retrieve_related_in_batches(
    dsl: Any,
    source: str,
    relation_field: str,
    relationship_ids: list[str],
    return_fields: str,
    query_log: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Retrieve linked records in deterministic batches, retaining raw query text."""
    if not relationship_ids:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    total_batches = (len(relationship_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    year_filter = f"start_year in [{YEAR_START}:{YEAR_END}]" if source == "grants" else f"year in [{YEAR_START}:{YEAR_END}]"

    for batch_number, batch_ids in enumerate(chunks(relationship_ids, CHUNK_SIZE), start=1):
        query = f'''search {source}
where {relation_field} in [{dsl_list(batch_ids)}]
and {year_filter}
return {source}[
    {return_fields}
]'''
        label = f"{source} linkage batch {batch_number}/{total_batches}"
        query_log.append({
            "stage": source,
            "batch_number": batch_number,
            "batch_size": len(batch_ids),
            "relationship_field": relation_field,
            "query": query,
        })
        write_query_log(query_log, output_dir)
        frames.append(run_query(dsl, query, label))

    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not output.empty:
        output = output.drop_duplicates(subset="id").reset_index(drop=True)
    print(f"Retrieved {len(output):,} unique linked {source} records after cross-batch deduplication.")
    return output


def save_frame(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_pickle(output_dir / f"{stem}.pkl")
    frame.to_excel(output_dir / f"{stem}.xlsx", index=False)


def write_query_log(query_log: list[dict[str, Any]], output_dir: Path) -> None:
    """Checkpoint every issued query so a failed run remains auditable."""
    with (output_dir / "retrieval_query_log.jsonl").open("w", encoding="utf-8") as handle:
        for entry in query_log:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write(chr(10))


def assert_required_columns(frame: pd.DataFrame, label: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing requested field(s): {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    key_file = args.key_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not key_file.is_file():
        raise SystemExit(
            f"Dimensions key file not found: {key_file}\n"
            "Place key.txt in new_version/ or run with --key-file /full/path/to/key.txt"
        )
    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("The Dimensions key file is empty.")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory already contains files: {output_dir}\n"
            "Use a new --output-dir, or rerun with --overwrite only if this is an incomplete retrieval you intend to replace."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the exact P query before login/retrieval so an empty or failed run
    # still leaves an inspectable provenance record.
    (output_dir / "publication_anchor_query.dsl").write_text(PUBLICATION_QUERY + "\n", encoding="utf-8")
    run_start = {
        "script_version": SCRIPT_VERSION,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "year_window": [YEAR_START, YEAR_END],
        "publication_anchor_query": PUBLICATION_QUERY,
    }
    (output_dir / "retrieval_run_status.json").write_text(json.dumps(run_start, indent=2), encoding="utf-8")

    dimcli.login(key=api_key, endpoint="https://app.dimensions.ai")
    dsl = dimcli.Dsl()
    query_log: list[dict[str, Any]] = [{
        "stage": "publication_anchor_P",
        "batch_number": 1,
        "batch_size": None,
        "relationship_field": None,
        "query": PUBLICATION_QUERY,
    }]
    write_query_log(query_log, output_dir)

    # 1. Publication anchor P.
    publications = run_query(dsl, PUBLICATION_QUERY, "publication anchor P")
    if publications.empty:
        raise SystemExit("Publication anchor P is empty. No linked retrieval was attempted.")
    assert_required_columns(
        publications,
        "publication anchor P",
        ["id", "year", "date", "title", "abstract", "doi", "type", "document_type", "concepts", "concepts_scores", "research_org_countries", "research_orgs", "times_cited", "supporting_grant_ids"],
    )
    save_frame(publications, output_dir, "publication_anchor")
    run_start["status"] = "publication_anchor_saved"
    run_start["publication_anchor_n"] = int(len(publications))
    (output_dir / "retrieval_run_status.json").write_text(json.dumps(run_start, indent=2), encoding="utf-8")

    # 2. Linked grants, strictly from P.supporting_grant_ids.
    publication_ids = publications["id"].astype(str).tolist()
    supporting_grant_ids = sorted({
        grant_id
        for relation_value in publications["supporting_grant_ids"]
        for grant_id in flatten_ids(relation_value)
    })
    grants = retrieve_related_in_batches(
        dsl=dsl,
        source="grants",
        relation_field="id",
        relationship_ids=supporting_grant_ids,
        return_fields=GRANT_RETURN_FIELDS,
        query_log=query_log,
        output_dir=output_dir,
    )
    if not grants.empty:
        assert_required_columns(
            grants,
            "linked grants",
            ["id", "start_year", "title", "abstract", "funding_usd", "funder_orgs", "research_org_countries", "research_org_names"],
        )
    save_frame(grants, output_dir, "linked_grants")
    run_start["status"] = "linked_grants_saved"
    run_start["linked_grants_n"] = int(len(grants))
    (output_dir / "retrieval_run_status.json").write_text(json.dumps(run_start, indent=2), encoding="utf-8")

    # 3. Linked policy documents, strictly through publication_ids containing a P ID.
    policy_documents = retrieve_related_in_batches(
        dsl=dsl,
        source="policy_documents",
        relation_field="publication_ids",
        relationship_ids=publication_ids,
        return_fields=POLICY_RETURN_FIELDS,
        query_log=query_log,
        output_dir=output_dir,
    )
    if not policy_documents.empty:
        assert_required_columns(
            policy_documents,
            "linked policy documents",
            ["id", "year", "title", "publication_ids"],
        )

    # The final raw linked layer is saved before any analysis/classification.
    save_frame(policy_documents, output_dir, "linked_policy_documents")
    run_start["status"] = "all_layers_saved"
    run_start["linked_policy_documents_n"] = int(len(policy_documents))
    (output_dir / "retrieval_run_status.json").write_text(json.dumps(run_start, indent=2), encoding="utf-8")

    # Deterministic audit samples facilitate review of retrieval precision and
    # source fields. They do not change the dataset.
    publications.sample(min(AUDIT_SAMPLE_SIZE, len(publications)), random_state=RANDOM_SEED).to_excel(
        output_dir / "audit_sample_publications.xlsx", index=False
    )
    (grants.sample(min(AUDIT_SAMPLE_SIZE, len(grants)), random_state=RANDOM_SEED) if not grants.empty else grants).to_excel(
        output_dir / "audit_sample_linked_grants.xlsx", index=False
    )
    (policy_documents.sample(min(AUDIT_SAMPLE_SIZE, len(policy_documents)), random_state=RANDOM_SEED) if not policy_documents.empty else policy_documents).to_excel(
        output_dir / "audit_sample_linked_policy.xlsx", index=False
    )

    publications_with_grant_ids = int(publications["supporting_grant_ids"].apply(lambda value: bool(flatten_ids(value))).sum())
    metadata = {
        "script_version": SCRIPT_VERSION,
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "year_window": [YEAR_START, YEAR_END],
        "publication_anchor_query": PUBLICATION_QUERY,
        "publication_anchor_fields": [field.strip() for field in PUBLICATION_RETURN_FIELDS.replace("+", "\n").splitlines() if field.strip()],
        "linked_grant_fields": [field.strip() for field in GRANT_RETURN_FIELDS.replace("+", "\n").splitlines() if field.strip()],
        "linked_policy_query_template": f"search policy_documents where publication_ids in [P ID batch] and year in [{YEAR_START}:{YEAR_END}]",
        "linked_policy_fields": [field.strip() for field in POLICY_RETURN_FIELDS.replace("+", "\n").splitlines() if field.strip()],
        "relationship_fields": {
            "publication_to_grant": "publications.supporting_grant_ids; retrieved through grants.id",
            "policy_document_to_publication": "policy_documents.publication_ids; queried with P IDs",
        },
        "publication_anchor_n": int(len(publications)),
        "publication_anchor_n_with_supporting_grant_ids": publications_with_grant_ids,
        "publication_anchor_distinct_supporting_grant_ids": int(len(supporting_grant_ids)),
        "linked_grants_n": int(len(grants)),
        "linked_policy_documents_n": int(len(policy_documents)),
        "chunk_size": CHUNK_SIZE,
        "audit_sample_size_requested": AUDIT_SAMPLE_SIZE,
        "audit_random_seed": RANDOM_SEED,
        "deduplication_rule": "Deduplicate each Dimensions result and each concatenated batch result by record id, retaining the first returned row.",
        "interpretive_boundary": (
            "Grant and policy-document records are retrieved only through Dimensions-recorded identifier relationships to P. "
            "A linked policy document is not thereby a topical AI-and-mental-health policy document, and these data do not establish policy impact, endorsement, institutional intention, or causal effects."
        ),
    }
    (output_dir / "retrieval_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    run_start["status"] = "completed"
    run_start["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "retrieval_run_status.json").write_text(json.dumps(run_start, indent=2), encoding="utf-8")
    write_query_log(query_log, output_dir)

    print("\n" + "=" * 72)
    print("CLEAN PUBLICATION-ANCHORED RETRIEVAL COMPLETE")
    print("=" * 72)
    print(f"Publication anchor P:                    {len(publications):,}")
    print(f"P with supporting-grant identifiers:     {publications_with_grant_ids:,}")
    print(f"Distinct grant IDs reported by P:         {len(supporting_grant_ids):,}")
    print(f"Retrieved linked grants:                  {len(grants):,}")
    print(f"Retrieved linked policy documents:        {len(policy_documents):,}")
    print(f"\nOutputs saved to: {output_dir}")
    print("Do not begin analysis until the retrieval summary and P audit sample have been reviewed.")


if __name__ == "__main__":
    main()
