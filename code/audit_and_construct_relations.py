#!/usr/bin/env python3
"""Restarted Step 1: audit the final v6 data and construct actual edge lists.

Place this script in:
    new_version/computations/step1_v6_audit_and_edges.py

Run from new_version/:
    python computations/step1_v6_audit_and_edges.py

Inputs (created by the approved two-stage data process):
    data/linked_final_outputs_v6/publication_anchor_final.pkl
    data/linked_final_outputs_v6/linked_grants_final.pkl
    data/linked_final_outputs_v6/linked_policy_documents_final.pkl

The script does not retrieve data or make topical/semantic judgments about policy
records. It verifies the final two-stage provenance, audits data integrity, and
constructs only Dimensions-identifier-based edge lists.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_VERSION = "step1_v6_audit_and_edges_2.0_flattened_policy_issuer_fields"
YEAR_START = 2000
YEAR_END = 2025


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_list(value: Any) -> list[Any]:
    """Return list-like Dimensions values without guessing an ID from free text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        return parse_list(parsed)
    return [value]


def parse_id_list(value: Any) -> list[str]:
    return [str(item).strip() for item in parse_list(value) if item is not None and str(item).strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(frame: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    rows = []
    for field in frame.columns:
        present = frame[field].notna().sum()
        rows.append({
            "dataset": dataset,
            "field": field,
            "non_missing_n": int(present),
            "non_missing_share": round(float(present / len(frame)), 6) if len(frame) else None,
        })
    return rows


def extract_issuer(policy_row: pd.Series) -> dict[str, Any] | None:
    """Extract actual Dimensions publisher data from flattened or nested fields.

    dimcli ordinarily writes publisher_org subfields as columns such as
    publisher_org.id and publisher_org.name. The nested-object branch is only
    a fallback. No issuer identifier is fabricated when Dimensions lacks one.
    """
    identifier = policy_row.get("publisher_org.id")
    if identifier is not None and not (isinstance(identifier, float) and pd.isna(identifier)) and str(identifier).strip():
        types = policy_row.get("publisher_org.types", "")
        if isinstance(types, (list, tuple, set, dict)):
            types = json.dumps(types, ensure_ascii=False) if isinstance(types, dict) else "; ".join(str(value) for value in types)
        country = policy_row.get("publisher_org.country_name")
        if country is None or (isinstance(country, float) and pd.isna(country)) or not str(country).strip():
            country = policy_row.get("publisher_org_country.name", "")
        return {
            "issuer_id": str(identifier).strip(),
            "issuer_name": policy_row.get("publisher_org.name", ""),
            "issuer_type": types,
            "issuer_country": country,
        }

    candidates = parse_list(policy_row.get("publisher_org"))
    for item in candidates:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if identifier is None or not str(identifier).strip():
            continue
        types = item.get("types", item.get("type", ""))
        if isinstance(types, (list, tuple, set)):
            types = "; ".join(str(value) for value in types)
        country = policy_row.get("publisher_org_country", "")
        if isinstance(country, (list, tuple, set, dict)):
            country = json.dumps(country, ensure_ascii=False)
        return {
            "issuer_id": str(identifier).strip(),
            "issuer_name": item.get("name", ""),
            "issuer_type": types,
            "issuer_country": country,
        }
    return None


def require_columns(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{name} is missing required field(s): {', '.join(missing)}")


def main() -> None:
    root = project_root()
    data_dir = root / "data" / "linked_final_outputs_v6"
    output_dir = root / "computations" / "outputs" / "step1_v6_audit_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = {
        "final_publications": data_dir / "publication_anchor_final.pkl",
        "linked_grants": data_dir / "linked_grants_final.pkl",
        "linked_policy_documents": data_dir / "linked_policy_documents_final.pkl",
        "screening_manifest": data_dir / "screening_manifest.json",
        "final_retrieval_summary": data_dir / "final_retrieval_summary.json",
    }
    absent = [str(path) for path in input_files.values() if not path.is_file()]
    if absent:
        raise SystemExit("Required v6 input file(s) not found:\n" + "\n".join(absent))

    P = pd.read_pickle(input_files["final_publications"]).copy()
    G = pd.read_pickle(input_files["linked_grants"]).copy()
    D = pd.read_pickle(input_files["linked_policy_documents"]).copy()
    screening_manifest = json.loads(input_files["screening_manifest"].read_text(encoding="utf-8"))
    retrieval_summary = json.loads(input_files["final_retrieval_summary"].read_text(encoding="utf-8"))

    require_columns(P, "Final P", ["id", "year", "title", "supporting_grant_ids", "retain_in_final_P"])
    require_columns(G, "Linked grants", ["id", "start_year", "title"])
    require_columns(D, "Linked policy documents", ["id", "year", "title", "publication_ids"])

    for frame in (P, G, D):
        frame["id"] = frame["id"].astype(str).str.strip()

    duplicate_checks = pd.DataFrame([
        {"dataset": "final_publications_P", "rows": len(P), "unique_ids": P["id"].nunique(), "duplicate_id_rows": int(P["id"].duplicated().sum())},
        {"dataset": "linked_grants_G", "rows": len(G), "unique_ids": G["id"].nunique(), "duplicate_id_rows": int(G["id"].duplicated().sum())},
        {"dataset": "linked_policy_documents_D", "rows": len(D), "unique_ids": D["id"].nunique(), "duplicate_id_rows": int(D["id"].duplicated().sum())},
    ])

    year_checks = []
    for dataset, frame, field in [
        ("final_publications_P", P, "year"),
        ("linked_grants_G", G, "start_year"),
        ("linked_policy_documents_D", D, "year"),
    ]:
        years = pd.to_numeric(frame[field], errors="coerce")
        year_checks.append({
            "dataset": dataset,
            "year_field": field,
            "records": len(frame),
            "missing_year_n": int(years.isna().sum()),
            "minimum_year": int(years.min()) if years.notna().any() else None,
            "maximum_year": int(years.max()) if years.notna().any() else None,
            "outside_2000_2025_n": int(((years < YEAR_START) | (years > YEAR_END)).sum()),
        })
    year_checks = pd.DataFrame(year_checks)

    # P–grant edges: retain only source P IDs and grant IDs present in final G.
    P_ids = set(P["id"])
    G_ids = set(G["id"])
    raw_pg = []
    for _, row in P[["id", "supporting_grant_ids"]].iterrows():
        for grant_id in parse_id_list(row["supporting_grant_ids"]):
            raw_pg.append({"publication_id": row["id"], "grant_id": grant_id})
    raw_pg_df = pd.DataFrame(raw_pg, columns=["publication_id", "grant_id"])
    pg_edges = raw_pg_df[raw_pg_df["grant_id"].isin(G_ids)].drop_duplicates().reset_index(drop=True)
    pg_unretrieved = raw_pg_df[~raw_pg_df["grant_id"].isin(G_ids)].drop_duplicates().reset_index(drop=True)
    pg_edges.to_csv(output_dir / "edges_publication_grant.csv", index=False)
    pg_unretrieved.to_csv(output_dir / "publication_grant_ids_not_in_final_grant_set.csv", index=False)

    # P–policy-document edges: retain only publication IDs that are in final P.
    raw_pd = []
    for _, row in D[["id", "publication_ids"]].iterrows():
        for publication_id in parse_id_list(row["publication_ids"]):
            raw_pd.append({"policy_document_id": row["id"], "publication_id": publication_id})
    raw_pd_df = pd.DataFrame(raw_pd, columns=["policy_document_id", "publication_id"])
    pd_edges = raw_pd_df[raw_pd_df["publication_id"].isin(P_ids)].drop_duplicates().reset_index(drop=True)
    pd_non_p = raw_pd_df[~raw_pd_df["publication_id"].isin(P_ids)].drop_duplicates().reset_index(drop=True)
    pd_edges.to_csv(output_dir / "edges_publication_policy_document.csv", index=False)
    pd_non_p.to_csv(output_dir / "policy_cited_publication_ids_not_in_final_P.csv", index=False)

    # Policy issuer–P edges: derive only from retained P–policy-document edges.
    issuer_rows = []
    issuer_missing_policy_ids = []
    policy_by_id = D.set_index("id", drop=False)
    for policy_id, group in pd_edges.groupby("policy_document_id", sort=False):
        issuer = extract_issuer(policy_by_id.loc[policy_id])
        if issuer is None:
            issuer_missing_policy_ids.append(policy_id)
            continue
        for publication_id in group["publication_id"]:
            issuer_rows.append({**issuer, "policy_document_id": policy_id, "publication_id": publication_id})
    issuer_incidence = pd.DataFrame(issuer_rows, columns=["issuer_id", "issuer_name", "issuer_type", "issuer_country", "policy_document_id", "publication_id"])
    if issuer_incidence.empty:
        issuer_edges = pd.DataFrame(columns=["issuer_id", "issuer_name", "issuer_type", "issuer_country", "publication_id", "n_policy_documents", "policy_document_ids"])
    else:
        issuer_edges = (
            issuer_incidence.groupby(["issuer_id", "issuer_name", "issuer_type", "issuer_country", "publication_id"], dropna=False)
            .agg(
                n_policy_documents=("policy_document_id", "nunique"),
                policy_document_ids=("policy_document_id", lambda x: "; ".join(sorted(set(x)))),
            )
            .reset_index()
        )
    issuer_edges.to_csv(output_dir / "edges_policy_issuer_publication.csv", index=False)
    pd.DataFrame({"policy_document_id_without_valid_publisher_org_id": sorted(set(issuer_missing_policy_ids))}).to_csv(
        output_dir / "policy_documents_without_valid_publisher_org_id.csv", index=False
    )

    relation_checks = pd.DataFrame([
        {
            "relation": "publication_grant",
            "definition": "P.supporting_grant_ids intersected with final linked-grant IDs",
            "raw_pairs_before_target_intersection": len(raw_pg_df),
            "retained_unique_edges": len(pg_edges),
            "source_endpoints_not_in_final_P": int((~pg_edges["publication_id"].isin(P_ids)).sum()),
            "target_endpoints_not_in_final_G": int((~pg_edges["grant_id"].isin(G_ids)).sum()),
            "unmatched_raw_pairs": len(pg_unretrieved),
            "duplicate_pairs_removed": int(len(raw_pg_df) - raw_pg_df.drop_duplicates().shape[0]),
        },
        {
            "relation": "publication_policy_document",
            "definition": "policy_documents.publication_ids intersected with final P IDs",
            "raw_pairs_before_P_intersection": len(raw_pd_df),
            "retained_unique_edges": len(pd_edges),
            "source_endpoints_not_in_final_D": int((~pd_edges["policy_document_id"].isin(set(D["id"]))).sum()),
            "target_endpoints_not_in_final_P": int((~pd_edges["publication_id"].isin(P_ids)).sum()),
            "unmatched_raw_pairs": len(pd_non_p),
            "duplicate_pairs_removed": int(len(raw_pd_df) - raw_pd_df.drop_duplicates().shape[0]),
        },
        {
            "relation": "policy_issuer_publication",
            "definition": "Policy issuer ID joined to retained policy-document–P edges; repeated issuer–P pairs weighted by distinct policy documents",
            "raw_pairs_before_target_intersection": len(issuer_incidence),
            "retained_unique_edges": len(issuer_edges),
            "source_endpoints_not_in_final_D": 0,
            "target_endpoints_not_in_final_P": int((~issuer_edges["publication_id"].isin(P_ids)).sum()) if not issuer_edges.empty else 0,
            "unmatched_raw_pairs": len(set(issuer_missing_policy_ids)),
            "duplicate_pairs_removed": int(len(issuer_incidence) - issuer_incidence.drop_duplicates().shape[0]) if not issuer_incidence.empty else 0,
        },
    ])

    edge_summary = pd.DataFrame([
        {"network": "publication_grant", "left_node_type": "publication", "left_nodes_connected": pg_edges["publication_id"].nunique(), "right_node_type": "grant", "right_nodes_connected": pg_edges["grant_id"].nunique(), "edges": len(pg_edges)},
        {"network": "publication_policy_document", "left_node_type": "publication", "left_nodes_connected": pd_edges["publication_id"].nunique(), "right_node_type": "policy_document", "right_nodes_connected": pd_edges["policy_document_id"].nunique(), "edges": len(pd_edges)},
        {"network": "policy_issuer_publication", "left_node_type": "policy_issuer", "left_nodes_connected": issuer_edges["issuer_id"].nunique() if not issuer_edges.empty else 0, "right_node_type": "publication", "right_nodes_connected": issuer_edges["publication_id"].nunique() if not issuer_edges.empty else 0, "edges": len(issuer_edges)},
    ])

    field_coverage = pd.DataFrame(coverage(P, "final_publications_P") + coverage(G, "linked_grants_G") + coverage(D, "linked_policy_documents_D"))
    duplicate_checks.to_csv(output_dir / "duplicate_id_checks.csv", index=False)
    year_checks.to_csv(output_dir / "year_window_checks.csv", index=False)
    field_coverage.to_csv(output_dir / "field_coverage.csv", index=False)
    relation_checks.to_csv(output_dir / "relationship_checks.csv", index=False)
    edge_summary.to_csv(output_dir / "edge_summary.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "linked_final_outputs_v6",
        "two_stage_publication_construction": {
            "candidate_dimensions_publications_n": screening_manifest.get("candidate_publications_n"),
            "final_text_screened_publications_n": screening_manifest.get("final_screened_publications_n"),
            "local_screen_rule": screening_manifest.get("local_screen_rule"),
            "screening_boundary": screening_manifest.get("screening_boundary"),
        },
        "input_files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in input_files.items()},
        "record_counts": {"final_P": len(P), "linked_grants": len(G), "linked_policy_documents": len(D)},
        "relationship_fields": retrieval_summary.get("relationship_fields"),
        "edge_files": {
            "publication_grant": "edges_publication_grant.csv",
            "publication_policy_document": "edges_publication_policy_document.csv",
            "policy_issuer_publication": "edges_policy_issuer_publication.csv",
        },
        "edge_rules": {
            "publication_grant": "Each unique (P publication ID, supporting_grant_id) pair is retained only if grant ID is present in final linked-grants file.",
            "publication_policy_document": "Each unique (policy document ID, publication_id) pair is retained only if publication ID is in final P.",
            "policy_issuer_publication": "Each unique issuer–P pair derives from retained policy-document–P pairs and carries a number of distinct policy documents as its weight.",
        },
        "interpretive_boundary": "The policy-document edges are recorded identifier relationships, not evidence of topical similarity, endorsement, policy impact, institutional intention, or causal effects.",
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = []
    if duplicate_checks["duplicate_id_rows"].sum() != 0:
        failures.append("duplicate record IDs")
    if year_checks["outside_2000_2025_n"].sum() != 0:
        failures.append("date-window violations")
    if int((~pg_edges["publication_id"].isin(P_ids)).sum()) or int((~pg_edges["grant_id"].isin(G_ids)).sum()):
        failures.append("invalid publication–grant endpoints")
    if int((~pd_edges["policy_document_id"].isin(set(D["id"]))).sum()) or int((~pd_edges["publication_id"].isin(P_ids)).sum()):
        failures.append("invalid publication–policy endpoints")
    pd.DataFrame([
        {"check": "all_required_v6_inputs_present", "passed": True, "failure_notes": ""},
        {"check": "unique_record_ids", "passed": duplicate_checks["duplicate_id_rows"].sum() == 0, "failure_notes": "duplicate record IDs" if "duplicate record IDs" in failures else ""},
        {"check": "year_window_2000_2025", "passed": year_checks["outside_2000_2025_n"].sum() == 0, "failure_notes": "date-window violations" if "date-window violations" in failures else ""},
        {"check": "publication_grant_endpoints_valid", "passed": "invalid publication–grant endpoints" not in failures, "failure_notes": "invalid publication–grant endpoints" if "invalid publication–grant endpoints" in failures else ""},
        {"check": "publication_policy_endpoints_valid", "passed": "invalid publication–policy endpoints" not in failures, "failure_notes": "invalid publication–policy endpoints" if "invalid publication–policy endpoints" in failures else ""},
        {"check": "final_result", "passed": not failures, "failure_notes": "; ".join(failures)},
    ]).to_csv(output_dir / "validation_checks.csv", index=False)

    print("\n" + "=" * 72)
    print("RESTARTED STEP 1 V6 AUDIT AND EDGE CONSTRUCTION COMPLETE")
    print("=" * 72)
    print(f"Final P publications:                {len(P):,}")
    print(f"Linked grants:                       {len(G):,}")
    print(f"Linked policy documents:             {len(D):,}")
    print(f"P–grant edges:                       {len(pg_edges):,}")
    print(f"P–policy-document edges:             {len(pd_edges):,}")
    print(f"Policy-issuer–P weighted edges:      {len(issuer_edges):,}")
    print(f"Validation failures:                  {len(failures)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
