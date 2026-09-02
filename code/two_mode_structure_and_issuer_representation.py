#!/usr/bin/env python3
"""Restarted Step 6: network, organization, and country representation.

Save as:
    new_version/computations/step6_v6_networks_organizations_countries.py

Run from new_version/:
    python computations/step6_v6_networks_organizations_countries.py

Inputs are final v6 records and the validated identifier-based edge lists from
restarted Step 1. This script is descriptive. It does not infer influence,
knowledge translation, institutional intention, policy impact, endorsement, or
semantic affinity from a recorded link.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "step6_v6_networks_organizations_countries_1.0"
TOP_N = 25
CONCENTRATION_SHARES = [0.01, 0.05, 0.10]

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "linked_final_outputs_v6"
STEP1_DIR = PROJECT_ROOT / "computations" / "outputs" / "step1_v6_audit_v2"
OUTPUT_DIR = PROJECT_ROOT / "computations" / "outputs" / "step6_v6_networks_organizations_countries"

P_FILE = DATA_DIR / "publication_anchor_final.pkl"
G_FILE = DATA_DIR / "linked_grants_final.pkl"
D_FILE = DATA_DIR / "linked_policy_documents_final.pkl"
PG_FILE = STEP1_DIR / "edges_publication_grant.csv"
PD_FILE = STEP1_DIR / "edges_publication_policy_document.csv"
IP_FILE = STEP1_DIR / "edges_policy_issuer_publication.csv"
STEP1_MANIFEST = STEP1_DIR / "data_manifest.json"


def require_files(paths: Iterable[Path]) -> None:
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise SystemExit("Required input file(s) missing:\n" + "\n".join(absent))


def require_columns(frame: pd.DataFrame, label: str, columns: list[str]) -> None:
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(absent)}")


def parse_list(value: Any) -> list[Any]:
    """Parse Dimensions list-like fields without inventing values from free text."""
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
            return parse_list(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            return [text]
    return [value]


def clean_value(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def normalise_countries(value: Any) -> list[str]:
    countries: list[str] = []
    for item in parse_list(value):
        if isinstance(item, dict):
            candidate = item.get("name") or item.get("country_name") or item.get("country")
        else:
            candidate = item
        text = clean_value(candidate)
        if text:
            countries.append(text)
    return sorted(set(countries))


def gini(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not values or sum(values) == 0:
        return None
    ordered = np.sort(np.array(values, dtype=float))
    n = len(ordered)
    return float((2 * np.sum((np.arange(1, n + 1)) * ordered) / (n * ordered.sum())) - (n + 1) / n)


def degree_table(edges: pd.DataFrame, column: str, weight_column: str | None = None) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=[column, "degree", "weighted_degree"])
    degree = edges.groupby(column).size().rename("degree")
    if weight_column and weight_column in edges.columns:
        weighted = pd.to_numeric(edges[weight_column], errors="coerce").fillna(0).groupby(edges[column]).sum().rename("weighted_degree")
    else:
        weighted = degree.rename("weighted_degree")
    return pd.concat([degree, weighted], axis=1).reset_index()


def connected_components(left_ids: Iterable[str], right_ids: Iterable[str], edges: pd.DataFrame, left_col: str, right_col: str) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for _, row in edges[[left_col, right_col]].drop_duplicates().iterrows():
        left = f"L::{row[left_col]}"
        right = f"R::{row[right_col]}"
        adjacency[left].add(right)
        adjacency[right].add(left)
    # Connected nodes only: isolates outside an observed bipartite edge are described separately.
    unseen = set(adjacency)
    output = []
    component_id = 0
    while unseen:
        component_id += 1
        start = next(iter(unseen))
        queue: deque[str] = deque([start])
        seen: set[str] = {start}
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    unseen.discard(neighbor)
                    queue.append(neighbor)
        left_nodes = sum(node.startswith("L::") for node in seen)
        right_nodes = sum(node.startswith("R::") for node in seen)
        edge_count = sum(len(adjacency[node]) for node in seen) // 2
        output.append({
            "component_id": component_id,
            "left_nodes": left_nodes,
            "right_nodes": right_nodes,
            "total_nodes": len(seen),
            "edges": edge_count,
        })
    return output


def concentration_rows(network: str, partition: str, degrees: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = float(degrees["weighted_degree"].sum()) if not degrees.empty else 0.0
    n = len(degrees)
    for share in CONCENTRATION_SHARES:
        k = max(1, math.ceil(n * share)) if n else 0
        top_total = float(degrees.nlargest(k, "weighted_degree")["weighted_degree"].sum()) if k else 0.0
        rows.append({
            "network": network,
            "partition": partition,
            "node_count": n,
            "top_node_share": share,
            "top_node_count": k,
            "top_nodes_weighted_edge_share": round(top_total / total, 6) if total else None,
            "weighted_degree_gini": round(gini(degrees["weighted_degree"].tolist()), 6) if n else None,
        })
    return rows


def network_stats(network: str, left_name: str, right_name: str, edges: pd.DataFrame, left_col: str, right_col: str, left_total: int, right_total: int, weight_column: str | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dedup_edges = edges.drop_duplicates(subset=[left_col, right_col]).copy()
    left_degrees = degree_table(dedup_edges, left_col, weight_column)
    right_degrees = degree_table(dedup_edges, right_col, weight_column)
    components = pd.DataFrame(connected_components(left_degrees[left_col].astype(str), right_degrees[right_col].astype(str), dedup_edges, left_col, right_col))
    largest = components.sort_values(["total_nodes", "edges"], ascending=False).head(1) if not components.empty else pd.DataFrame()
    edge_total = int(len(dedup_edges))
    left_connected = int(len(left_degrees))
    right_connected = int(len(right_degrees))
    rows = []
    for partition, label, degrees, total_nodes in [
        ("left", left_name, left_degrees, left_total),
        ("right", right_name, right_degrees, right_total),
    ]:
        rows.append({
            "network": network,
            "partition": partition,
            "node_type": label,
            "available_nodes": int(total_nodes),
            "connected_nodes": int(len(degrees)),
            "isolated_or_unconnected_nodes": int(total_nodes - len(degrees)),
            "edges": edge_total,
            "bipartite_density_connected_nodes": round(edge_total / (left_connected * right_connected), 8) if left_connected and right_connected else None,
            "mean_unweighted_degree": round(float(degrees["degree"].mean()), 6) if not degrees.empty else None,
            "median_unweighted_degree": float(degrees["degree"].median()) if not degrees.empty else None,
            "maximum_unweighted_degree": int(degrees["degree"].max()) if not degrees.empty else None,
            "mean_weighted_degree": round(float(degrees["weighted_degree"].mean()), 6) if not degrees.empty else None,
            "median_weighted_degree": float(degrees["weighted_degree"].median()) if not degrees.empty else None,
            "maximum_weighted_degree": int(degrees["weighted_degree"].max()) if not degrees.empty else None,
            "connected_components": int(len(components)),
            "largest_component_total_nodes": int(largest.iloc[0]["total_nodes"]) if not largest.empty else 0,
            "largest_component_node_share_connected": round(float(largest.iloc[0]["total_nodes"]) / (left_connected + right_connected), 6) if not largest.empty and (left_connected + right_connected) else None,
        })
    return rows, left_degrees, right_degrees, components


def add_node_attributes(degrees: pd.DataFrame, node_col: str, nodes: pd.DataFrame, node_id_col: str, attributes: list[str], network: str, partition: str) -> pd.DataFrame:
    attributes = [attribute for attribute in attributes if attribute in nodes.columns]
    lookup = nodes[[node_id_col] + attributes].copy().rename(columns={node_id_col: node_col})
    result = degrees.merge(lookup, on=node_col, how="left", validate="one_to_one")
    result.insert(0, "network", network)
    result.insert(1, "partition", partition)
    return result.sort_values(["weighted_degree", "degree", node_col], ascending=[False, False, True]).head(TOP_N)


def country_rows(nodes: pd.DataFrame, id_col: str, country_col: str, source: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    represented_nodes = 0
    for _, row in nodes.iterrows():
        values = normalise_countries(row.get(country_col)) if country_col in nodes.columns else []
        if values:
            represented_nodes += 1
        counter.update(values)
    output = []
    for country, count in counter.most_common():
        output.append({
            "source": source,
            "country": country,
            "nodes_or_records_with_country": int(count),
            "available_nodes_or_records": int(len(nodes)),
            "records_with_at_least_one_country": int(represented_nodes),
            "percentage_of_all_nodes_or_records": round(100 * count / len(nodes), 3) if len(nodes) else None,
            "multi_country_counting_note": "A record may contribute to more than one country; percentages need not sum to 100.",
        })
    return output


def main() -> None:
    require_files([P_FILE, G_FILE, D_FILE, PG_FILE, PD_FILE, IP_FILE, STEP1_MANIFEST])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    P = pd.read_pickle(P_FILE).copy()
    G = pd.read_pickle(G_FILE).copy()
    D = pd.read_pickle(D_FILE).copy()
    pg = pd.read_csv(PG_FILE, dtype=str).copy()
    pd_edges = pd.read_csv(PD_FILE, dtype=str).copy()
    ip = pd.read_csv(IP_FILE, dtype=str).copy()

    require_columns(P, "Final P", ["id", "title", "year"])
    require_columns(G, "Linked grants", ["id", "title", "start_year"])
    require_columns(D, "Linked policy documents", ["id", "title", "year"])
    require_columns(pg, "P–grant edge list", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P–policy edge list", ["publication_id", "policy_document_id"])
    require_columns(ip, "Policy issuer–P edge list", ["issuer_id", "issuer_name", "issuer_type", "issuer_country", "publication_id", "n_policy_documents"])

    for frame, column in [(P, "id"), (G, "id"), (D, "id"), (pg, "publication_id"), (pg, "grant_id"), (pd_edges, "publication_id"), (pd_edges, "policy_document_id"), (ip, "issuer_id"), (ip, "publication_id")]:
        frame[column] = frame[column].astype(str).str.strip()
    ip["n_policy_documents"] = pd.to_numeric(ip["n_policy_documents"], errors="coerce")
    if ip["n_policy_documents"].isna().any() or (ip["n_policy_documents"] < 1).any():
        raise SystemExit("Policy issuer–P edge weights must be positive numeric counts.")

    P_ids, G_ids, D_ids, issuer_ids = set(P["id"]), set(G["id"]), set(D["id"]), set(ip["issuer_id"])
    endpoint_failures = {
        "publication_grant": int((~pg["publication_id"].isin(P_ids)).sum()) + int((~pg["grant_id"].isin(G_ids)).sum()),
        "publication_policy_document": int((~pd_edges["publication_id"].isin(P_ids)).sum()) + int((~pd_edges["policy_document_id"].isin(D_ids)).sum()),
        "policy_issuer_publication": int((~ip["publication_id"].isin(P_ids)).sum()) + int((~ip["issuer_id"].isin(issuer_ids)).sum()),
    }
    if any(endpoint_failures.values()):
        raise SystemExit("An edge endpoint is outside a final v6 node table. Re-run restarted Step 1.")

    # Networks use the unique endpoint pairs already defined in Step 1.
    pg = pg.drop_duplicates(subset=["publication_id", "grant_id"]).copy()
    pd_edges = pd_edges.drop_duplicates(subset=["publication_id", "policy_document_id"]).copy()
    ip = ip.drop_duplicates(subset=["issuer_id", "publication_id"]).copy()

    results: list[dict[str, Any]] = []
    concentration: list[dict[str, Any]] = []
    component_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []

    pg_stats, pg_left, pg_right, pg_components = network_stats("publication_grant", "publication", "grant", pg, "publication_id", "grant_id", len(P), len(G))
    results.extend(pg_stats)
    concentration.extend(concentration_rows("publication_grant", "publication", pg_left))
    concentration.extend(concentration_rows("publication_grant", "grant", pg_right))
    pg_components.insert(0, "network", "publication_grant")
    component_frames.append(pg_components)
    top_frames.extend([
        add_node_attributes(pg_left, "publication_id", P, "id", ["title", "year", "times_cited", "research_org_countries"], "publication_grant", "publication"),
        add_node_attributes(pg_right, "grant_id", G, "id", ["title", "start_year", "funding_usd", "funder_org_name", "funder_org_countries", "research_org_countries"], "publication_grant", "grant"),
    ])

    pd_stats, pd_left, pd_right, pd_components = network_stats("publication_policy_document", "publication", "policy_document", pd_edges, "publication_id", "policy_document_id", len(P), len(D))
    results.extend(pd_stats)
    concentration.extend(concentration_rows("publication_policy_document", "publication", pd_left))
    concentration.extend(concentration_rows("publication_policy_document", "policy_document", pd_right))
    pd_components.insert(0, "network", "publication_policy_document")
    component_frames.append(pd_components)
    top_frames.extend([
        add_node_attributes(pd_left, "publication_id", P, "id", ["title", "year", "times_cited", "research_org_countries"], "publication_policy_document", "publication"),
        add_node_attributes(pd_right, "policy_document_id", D, "id", ["title", "year", "publisher_org.name", "publisher_org.country_name"], "publication_policy_document", "policy_document"),
    ])

    issuer_nodes = ip[["issuer_id", "issuer_name", "issuer_type", "issuer_country"]].drop_duplicates(subset=["issuer_id"]).copy()
    ip_stats, ip_left, ip_right, ip_components = network_stats("policy_issuer_publication", "policy_issuer", "publication", ip, "issuer_id", "publication_id", len(issuer_nodes), len(P), "n_policy_documents")
    results.extend(ip_stats)
    concentration.extend(concentration_rows("policy_issuer_publication", "policy_issuer", ip_left))
    concentration.extend(concentration_rows("policy_issuer_publication", "publication", ip_right))
    ip_components.insert(0, "network", "policy_issuer_publication")
    component_frames.append(ip_components)
    top_frames.extend([
        add_node_attributes(ip_left, "issuer_id", issuer_nodes, "issuer_id", ["issuer_name", "issuer_type", "issuer_country"], "policy_issuer_publication", "policy_issuer"),
        add_node_attributes(ip_right, "publication_id", P, "id", ["title", "year", "times_cited", "research_org_countries"], "policy_issuer_publication", "publication"),
    ])

    # Issuer representation: degree counts and number of distinct policy documents behind each issuer.
    issuer_repr = (
        ip.groupby(["issuer_id", "issuer_name", "issuer_type", "issuer_country"], dropna=False)
        .agg(
            distinct_linked_publications=("publication_id", "nunique"),
            issuer_publication_pairs=("publication_id", "size"),
            distinct_policy_documents=("n_policy_documents", "sum"),
        )
        .reset_index()
        .sort_values(["distinct_linked_publications", "distinct_policy_documents", "issuer_name"], ascending=[False, False, True])
    )
    issuer_repr["issuer_rank_by_linked_publications"] = range(1, len(issuer_repr) + 1)

    # Country representations are multi-valued counts, never a fractional or exclusive allocation.
    countries: list[dict[str, Any]] = []
    countries.extend(country_rows(P, "id", "research_org_countries", "final_P_research_organization_countries"))
    countries.extend(country_rows(G, "id", "funder_org_countries", "linked_grant_funder_countries"))
    countries.extend(country_rows(G, "id", "research_org_countries", "linked_grant_recipient_organization_countries"))
    issuer_country_nodes = issuer_nodes.rename(columns={"issuer_id": "id", "issuer_country": "countries"})
    countries.extend(country_rows(issuer_country_nodes, "id", "countries", "policy_issuer_countries"))
    country_representation = pd.DataFrame(countries)

    basic_stats = pd.DataFrame(results)
    concentration_df = pd.DataFrame(concentration)
    components_df = pd.concat(component_frames, ignore_index=True)
    top_nodes = pd.concat(top_frames, ignore_index=True, sort=False)
    validation = pd.DataFrame([
        {"check": "all_final_v6_files_present", "passed": True, "detail": ""},
        {"check": "publication_grant_endpoints_valid", "passed": endpoint_failures["publication_grant"] == 0, "detail": ""},
        {"check": "publication_policy_document_endpoints_valid", "passed": endpoint_failures["publication_policy_document"] == 0, "detail": ""},
        {"check": "policy_issuer_publication_endpoints_valid", "passed": endpoint_failures["policy_issuer_publication"] == 0, "detail": ""},
        {"check": "policy_issuer_weights_positive", "passed": bool((ip["n_policy_documents"] >= 1).all()), "detail": ""},
        {"check": "all_networks_have_edges", "passed": bool((basic_stats.groupby("network")["edges"].first() > 0).all()), "detail": ""},
    ])

    basic_stats.to_csv(OUTPUT_DIR / "network_basic_statistics.csv", index=False)
    concentration_df.to_csv(OUTPUT_DIR / "network_degree_concentration.csv", index=False)
    components_df.to_csv(OUTPUT_DIR / "network_components.csv", index=False)
    top_nodes.to_csv(OUTPUT_DIR / "network_top_nodes.csv", index=False)
    issuer_repr.to_csv(OUTPUT_DIR / "policy_issuer_representation.csv", index=False)
    country_representation.to_csv(OUTPUT_DIR / "country_representation.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "network_validation_checks.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "source_dataset": "linked_final_outputs_v6",
        "unit_of_analysis": "Three undirected bipartite networks constructed from final v6 identifier-based edges.",
        "networks": {
            "publication_grant": "Unique P–grant pairs from final P.supporting_grant_ids intersected with final linked-grant IDs.",
            "publication_policy_document": "Unique policy-document–P pairs from policy_document.publication_ids intersected with final P IDs.",
            "policy_issuer_publication": "Unique issuer–P pairs derived from retained policy-document–P pairs; edge weight equals the number of distinct linked policy documents.",
        },
        "country_counting": "Countries are non-exclusive record/node counts from Dimensions attributes. Multi-country records contribute once to each recorded country, so percentages need not sum to 100.",
        "network_interpretive_boundary": "Network topology and country/issuer representation describe recorded links only. They do not establish influence, intentional selection, institutional preference, policy impact, endorsement, causal effects, or semantic affinity.",
        "step1_manifest_version": json.loads(STEP1_MANIFEST.read_text(encoding="utf-8")).get("script_version"),
    }
    (OUTPUT_DIR / "step6_network_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Restarted Step 6 (v6 networks, organizations, and countries) completed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    for network in ["publication_grant", "publication_policy_document", "policy_issuer_publication"]:
        row = basic_stats.loc[(basic_stats["network"] == network) & (basic_stats["partition"] == "left")].iloc[0]
        print(f"{network}: {int(row['edges']):,} edges; {int(row['connected_nodes']):,} left connected nodes")


if __name__ == "__main__":
    main()
