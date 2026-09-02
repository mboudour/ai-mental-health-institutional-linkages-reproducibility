#!/usr/bin/env python3
"""Step 12: Local cross-relation configuration analysis for final v6.

Run from new_version/:
    python computations/step12_v6_local_configurations.py

The inferential question is publication-centered:
    Are P-G and P-D relations disproportionately aligned on the same
    publications, relative to an appropriate cross-relation alignment null?

Raw relations used:
    P-G: publication--grant
    P-D: publication--policy document

D-I is an issuer attribution: every retained policy document has one recorded
issuer. It is used only to characterize the issuer reach of P nodes with both
P-G and P-D relations; it is not a third independently rewired relation.

Primary null:
    Permute the P endpoint of every P-D tie within publication-year strata.
    P-G is retained unchanged. The null preserves all P-G ties, the full P-D
    degree multiset within each year, every D endpoint, and all D-I issuer
    attributions, while disrupting cross-relation alignment on the same P.

Sensitivity null:
    The same P-end permutation without year stratification.

This is a descriptive structural test of database-recorded ties. It does not
identify causal effects, institutional intent, policy impact, endorsement,
semantic affinity, latent profiles, communities, or global geometry.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "step12_v6_local_configurations_1.0"
PRIMARY_NULL = "publication_year_stratified_P_end_permutation"
SENSITIVITY_NULL = "unrestricted_P_end_permutation"
N_PERMUTATIONS = 1000
RANDOM_SEED = 20260902


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required input file not found: {path}")


def require_columns(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{name} is missing required column(s): {', '.join(missing)}")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_list(value: Any) -> list[Any]:
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


def issuer_from_row(row: pd.Series) -> str | None:
    value = row.get("publisher_org.id")
    if value is not None and not (isinstance(value, float) and pd.isna(value)):
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    for item in parse_list(row.get("publisher_org")):
        if isinstance(item, dict) and item.get("id"):
            return str(item["id"]).strip()
    return None


def safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def quantile(values: np.ndarray, probability: float) -> float | None:
    return float(np.quantile(values, probability)) if len(values) else None


def two_sided_permutation_p(observed: float, null: np.ndarray) -> float:
    upper = (1 + int((null >= observed).sum())) / (1 + len(null))
    lower = (1 + int((null <= observed).sum())) / (1 + len(null))
    return float(min(1.0, 2.0 * min(upper, lower)))


def one_sided_upper_p(observed: float, null: np.ndarray) -> float:
    return float((1 + int((null >= observed).sum())) / (1 + len(null)))


def configuration_labels(pg_degree: np.ndarray, pd_degree: np.ndarray) -> np.ndarray:
    labels = np.full(len(pg_degree), "neither", dtype=object)
    labels[(pg_degree > 0) & (pd_degree == 0)] = "P_G_only"
    labels[(pg_degree == 0) & (pd_degree > 0)] = "P_D_only"
    labels[(pg_degree > 0) & (pd_degree > 0)] = "both"
    return labels


def observed_configuration_table(publications: pd.DataFrame, pg_degree: np.ndarray, pd_degree: np.ndarray, issuer_sets: list[set[str]]) -> pd.DataFrame:
    labels = configuration_labels(pg_degree, pd_degree)
    issuer_count = np.array([len(items) for items in issuer_sets], dtype=int)
    issuer_ratio = np.divide(issuer_count, pd_degree, out=np.full(len(pd_degree), np.nan), where=pd_degree > 0)
    table = publications[["publication_id", "year"]].copy()
    table["P_G_degree"] = pg_degree
    table["P_D_degree"] = pd_degree
    table["configuration"] = labels
    table["total_issuer_incidences"] = pd_degree
    table["distinct_issuer_count"] = issuer_count
    table["distinct_issuer_per_policy_document"] = issuer_ratio
    table["has_grant_link"] = (pg_degree > 0).astype(int)
    table["has_policy_document_link"] = (pd_degree > 0).astype(int)
    return table


def configuration_summary(table: pd.DataFrame) -> pd.DataFrame:
    order = ["P_G_only", "P_D_only", "both", "neither"]
    rows = []
    for label in order:
        group = table[table["configuration"] == label]
        rows.append({
            "configuration": label,
            "n_publications": int(len(group)),
            "share_of_final_P": float(len(group) / len(table)),
            "mean_P_G_degree": safe_mean(group["P_G_degree"].to_numpy(dtype=float)),
            "mean_P_D_degree": safe_mean(group["P_D_degree"].to_numpy(dtype=float)),
            "mean_distinct_issuer_count": safe_mean(group["distinct_issuer_count"].to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows)


def issuer_reach_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    both = table[table["configuration"] == "both"].copy()
    overall = pd.DataFrame([{
        "population": "P_with_both_P_G_and_P_D_relations",
        "n_publications": int(len(both)),
        "mean_total_issuer_incidences": safe_mean(both["total_issuer_incidences"].to_numpy(dtype=float)),
        "median_total_issuer_incidences": float(both["total_issuer_incidences"].median()) if len(both) else None,
        "mean_distinct_issuer_count": safe_mean(both["distinct_issuer_count"].to_numpy(dtype=float)),
        "median_distinct_issuer_count": float(both["distinct_issuer_count"].median()) if len(both) else None,
        "mean_distinct_issuer_per_policy_document": safe_mean(both["distinct_issuer_per_policy_document"].dropna().to_numpy(dtype=float)),
    }])
    bins = np.select([both["P_D_degree"].eq(1), both["P_D_degree"].eq(2), both["P_D_degree"].ge(3)], ["1", "2", "3_plus"], default="0")
    both["P_D_degree_group"] = bins
    rows = []
    for degree_group in ["1", "2", "3_plus"]:
        group = both[both["P_D_degree_group"] == degree_group]
        rows.append({
            "population": "P_with_both_P_G_and_P_D_relations",
            "P_D_degree_group": degree_group,
            "n_publications": int(len(group)),
            "mean_distinct_issuer_count": safe_mean(group["distinct_issuer_count"].to_numpy(dtype=float)),
            "median_distinct_issuer_count": float(group["distinct_issuer_count"].median()) if len(group) else None,
            "mean_distinct_issuer_per_policy_document": safe_mean(group["distinct_issuer_per_policy_document"].dropna().to_numpy(dtype=float)),
            "share_with_single_distinct_issuer": float((group["distinct_issuer_count"] == 1).mean()) if len(group) else None,
        })
    return overall, pd.DataFrame(rows)


def prepare_issuer_codes(pd_edges: pd.DataFrame, policy_documents: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    policy = policy_documents.copy()
    policy["policy_document_id"] = policy["id"].astype(str).str.strip()
    policy["issuer_id"] = policy.apply(issuer_from_row, axis=1)
    policy_to_issuer = dict(zip(policy["policy_document_id"], policy["issuer_id"]))
    output = pd_edges.copy()
    output["issuer_id"] = output["policy_document_id"].map(policy_to_issuer)
    if output["issuer_id"].isna().any():
        missing = sorted(output.loc[output["issuer_id"].isna(), "policy_document_id"].unique())
        raise SystemExit(f"Retained P-D edges have policy documents without issuer IDs: {missing[:10]}")
    output["issuer_id"] = output["issuer_id"].astype(str)
    return output, policy_to_issuer


def degrees_and_issuers(n_publications: int, pg_p_indices: np.ndarray, pd_p_indices: np.ndarray, pd_issuer_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[set[str]]]:
    pg_degree = np.bincount(pg_p_indices, minlength=n_publications)
    pd_degree = np.bincount(pd_p_indices, minlength=n_publications)
    issuer_sets: list[set[str]] = [set() for _ in range(n_publications)]
    for publication_index, issuer_id in zip(pd_p_indices, pd_issuer_codes):
        issuer_sets[int(publication_index)].add(str(issuer_id))
    return pg_degree, pd_degree, issuer_sets


def make_target_indices(years: np.ndarray, scheme: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    n = len(years)
    targets = np.arange(n, dtype=int)
    fixed_strata = 0
    permuted_strata = 0
    if scheme == PRIMARY_NULL:
        for year in sorted(pd.unique(years)):
            indices = np.flatnonzero(years == year)
            if len(indices) < 2:
                fixed_strata += 1
                continue
            targets[indices] = rng.permutation(indices)
            permuted_strata += 1
    elif scheme == SENSITIVITY_NULL:
        targets = rng.permutation(n)
        permuted_strata = 1
    else:
        raise ValueError(f"Unknown null scheme: {scheme}")
    return targets, {"permuted_strata": permuted_strata, "fixed_strata": fixed_strata}


def null_statistics(pg_degree: np.ndarray, pd_targets: np.ndarray, pd_issuer_codes: np.ndarray) -> dict[str, float]:
    n = len(pg_degree)
    pd_degree = np.bincount(pd_targets, minlength=n)
    labels = configuration_labels(pg_degree, pd_degree)
    both = labels == "both"
    issuer_sets: list[set[str]] = [set() for _ in range(n)]
    for publication_index, issuer_id in zip(pd_targets, pd_issuer_codes):
        issuer_sets[int(publication_index)].add(str(issuer_id))
    issuer_count = np.array([len(values) for values in issuer_sets], dtype=float)
    issuer_ratio = np.divide(issuer_count, pd_degree, out=np.full(n, np.nan), where=pd_degree > 0)
    both_ratio = issuer_ratio[both]
    return {
        "N_P_G_only": float((labels == "P_G_only").sum()),
        "N_P_D_only": float((labels == "P_D_only").sum()),
        "N_both": float(both.sum()),
        "N_neither": float((labels == "neither").sum()),
        "intensive_degree_product_sum": float(np.sum(pg_degree * pd_degree)),
        "total_distinct_issuers_among_both": float(issuer_count[both].sum()),
        "mean_distinct_issuer_count_among_both": safe_mean(issuer_count[both]),
        "mean_distinct_issuer_per_policy_document_among_both": safe_mean(both_ratio[~np.isnan(both_ratio)]),
    }


def null_test(observed: dict[str, float], null_frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    primary_stats = ["N_both", "intensive_degree_product_sum"]
    supplemental_stats = ["total_distinct_issuers_among_both", "mean_distinct_issuer_count_among_both", "mean_distinct_issuer_per_policy_document_among_both"]
    for statistic in primary_stats + supplemental_stats:
        values = null_frame[statistic].dropna().to_numpy(dtype=float)
        observed_value = float(observed[statistic])
        expected = safe_mean(values)
        enrichment = observed_value / expected if expected and expected != 0 else None
        direction = "alignment_enrichment" if expected is not None and observed_value > expected else "avoidance" if expected is not None and observed_value < expected else "no_difference"
        rows.append({
            "null_scheme": scheme,
            "statistic": statistic,
            "statistic_role": "primary" if statistic in primary_stats else "supplementary_issuer_characterization",
            "observed": observed_value,
            "null_mean": expected,
            "null_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "null_interval_2_5": quantile(values, 0.025),
            "null_interval_97_5": quantile(values, 0.975),
            "observed_to_null_mean_enrichment": enrichment,
            "permutation_p_two_sided": two_sided_permutation_p(observed_value, values),
            "permutation_p_enrichment_upper_tail": one_sided_upper_p(observed_value, values),
            "direction_relative_to_null": direction,
            "n_permutations": int(len(values)),
        })
    return pd.DataFrame(rows)


def indicator_characterization(config_table: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    require_columns(indicators, "Step 2 text indicators", ["publication_id", "ethics_responsibility_primary", "computational_performance_primary"])
    merged = config_table.merge(
        indicators[["publication_id", "ethics_responsibility_primary", "computational_performance_primary"]],
        on="publication_id", how="left", validate="one_to_one",
    )
    rows = []
    for configuration in ["P_G_only", "P_D_only", "both", "neither"]:
        group = merged[merged["configuration"] == configuration]
        for indicator in ["ethics_responsibility_primary", "computational_performance_primary"]:
            rows.append({
                "configuration": configuration,
                "supplementary_indicator": indicator,
                "n_publications": int(len(group)),
                "n_indicator_positive": int(group[indicator].fillna(0).astype(int).sum()),
                "indicator_share": float(group[indicator].fillna(0).astype(int).mean()) if len(group) else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    edge_dir = root / "computations" / "outputs" / "step1_v6_audit_v2"
    data_dir = root / "data" / "linked_final_outputs_v6"
    indicator_path = root / "computations" / "outputs" / "step2_v6_indicators" / "publication_text_indicators.csv"
    out = root / "computations" / "outputs" / "step12_v6_local_configurations"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "P_G_edges": edge_dir / "edges_publication_grant.csv",
        "P_D_edges": edge_dir / "edges_publication_policy_document.csv",
        "final_publications": data_dir / "publication_anchor_final.pkl",
        "policy_documents": data_dir / "linked_policy_documents_final.pkl",
        "text_indicators": indicator_path,
    }
    for path in paths.values():
        require(path)

    publications = pd.read_pickle(paths["final_publications"]).copy()
    require_columns(publications, "Final P", ["id", "year", "retain_in_final_P"])
    if not publications["retain_in_final_P"].astype(bool).all():
        raise SystemExit("Final publication file includes a record not retained by the approved literal screen")
    publications = publications[["id", "year"]].copy().rename(columns={"id": "publication_id"})
    publications["publication_id"] = publications["publication_id"].astype(str).str.strip()
    publications["year"] = pd.to_numeric(publications["year"], errors="coerce")
    if publications["publication_id"].duplicated().any():
        raise SystemExit("Final P has duplicate publication IDs")
    if publications["year"].isna().any():
        raise SystemExit("Final P has missing publication year(s); stratified primary null cannot be run")
    publications["year"] = publications["year"].astype(int)

    pg = pd.read_csv(paths["P_G_edges"], dtype=str)
    pd_edges = pd.read_csv(paths["P_D_edges"], dtype=str)
    require_columns(pg, "P-G edges", ["publication_id", "grant_id"])
    require_columns(pd_edges, "P-D edges", ["publication_id", "policy_document_id"])
    pg = pg[["publication_id", "grant_id"]].dropna().astype(str).drop_duplicates().reset_index(drop=True)
    pd_edges = pd_edges[["publication_id", "policy_document_id"]].dropna().astype(str).drop_duplicates().reset_index(drop=True)
    policy_documents = pd.read_pickle(paths["policy_documents"]).copy()
    require_columns(policy_documents, "Linked policy documents", ["id"])
    pd_edges, _ = prepare_issuer_codes(pd_edges, policy_documents)
    indicators = pd.read_csv(paths["text_indicators"], dtype={"publication_id": str})

    p_ids = publications["publication_id"].to_numpy(dtype=str)
    p_index = {value: index for index, value in enumerate(p_ids)}
    if not set(pg["publication_id"]).issubset(p_index):
        raise SystemExit("P-G edges include a P endpoint absent from final P")
    if not set(pd_edges["publication_id"]).issubset(p_index):
        raise SystemExit("P-D edges include a P endpoint absent from final P")
    pg_p_indices = pg["publication_id"].map(p_index).to_numpy(dtype=int)
    pd_p_indices = pd_edges["publication_id"].map(p_index).to_numpy(dtype=int)
    pd_issuer_codes = pd_edges["issuer_id"].to_numpy(dtype=str)
    pg_degree, pd_degree, issuer_sets = degrees_and_issuers(len(publications), pg_p_indices, pd_p_indices, pd_issuer_codes)
    config_table = observed_configuration_table(publications, pg_degree, pd_degree, issuer_sets)
    config_summary = configuration_summary(config_table)
    issuer_overall, issuer_by_pd_degree = issuer_reach_summary(config_table)
    indicator_summary = indicator_characterization(config_table, indicators)

    observed = {
        "N_both": float((config_table["configuration"] == "both").sum()),
        "intensive_degree_product_sum": float(np.sum(pg_degree * pd_degree)),
        "total_distinct_issuers_among_both": float(config_table.loc[config_table["configuration"] == "both", "distinct_issuer_count"].sum()),
        "mean_distinct_issuer_count_among_both": safe_mean(config_table.loc[config_table["configuration"] == "both", "distinct_issuer_count"].to_numpy(dtype=float)),
        "mean_distinct_issuer_per_policy_document_among_both": safe_mean(config_table.loc[config_table["configuration"] == "both", "distinct_issuer_per_policy_document"].dropna().to_numpy(dtype=float)),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    years = publications["year"].to_numpy(dtype=int)
    all_null_tables = []
    stratum_rows = []
    for scheme in [PRIMARY_NULL, SENSITIVITY_NULL]:
        statistics = []
        for replicate in range(1, N_PERMUTATIONS + 1):
            targets, meta = make_target_indices(years, scheme, rng)
            pd_targets = targets[pd_p_indices]
            values = null_statistics(pg_degree, pd_targets, pd_issuer_codes)
            values.update({"null_scheme": scheme, "replicate": replicate})
            statistics.append(values)
            if replicate == 1:
                stratum_rows.append({"null_scheme": scheme, **meta})
        all_null_tables.append(pd.DataFrame(statistics))
    null_replicates = pd.concat(all_null_tables, ignore_index=True)
    null_tests = pd.concat([
        null_test(observed, null_replicates[null_replicates["null_scheme"] == scheme], scheme)
        for scheme in [PRIMARY_NULL, SENSITIVITY_NULL]
    ], ignore_index=True)

    year_counts = publications.groupby("year", as_index=False).agg(n_final_P=("publication_id", "size"))
    year_counts["stratum_permutable"] = year_counts["n_final_P"] >= 2
    checks = pd.DataFrame([
        {"check": "final_P_unique_ids", "passed": not publications["publication_id"].duplicated().any(), "detail": f"{len(publications)} final P records"},
        {"check": "all_final_P_have_year", "passed": not publications["year"].isna().any(), "detail": "Required for primary year-stratified null"},
        {"check": "P_G_endpoints_in_final_P", "passed": set(pg["publication_id"]).issubset(p_index), "detail": f"{len(pg)} unique P-G ties"},
        {"check": "P_D_endpoints_in_final_P", "passed": set(pd_edges["publication_id"]).issubset(p_index), "detail": f"{len(pd_edges)} unique P-D ties"},
        {"check": "all_retained_P_D_documents_have_issuer", "passed": pd_edges["issuer_id"].notna().all(), "detail": f"{pd_edges['policy_document_id'].nunique()} policy documents"},
        {"check": "primary_null_preserves_year_strata", "passed": True, "detail": "P-D publication endpoints permuted only within P publication-year strata"},
        {"check": "text_indicators_excluded_from_null", "passed": True, "detail": "Indicators are supplementary configuration characterizations only"},
    ])

    pg_only = int((config_table["configuration"] == "P_G_only").sum())
    pd_only = int((config_table["configuration"] == "P_D_only").sum())
    both = int((config_table["configuration"] == "both").sum())
    neither = int((config_table["configuration"] == "neither").sum())
    config_table.to_csv(out / "publication_cross_relation_configurations.csv", index=False)
    config_summary.to_csv(out / "configuration_observed_summary.csv", index=False)
    issuer_overall.to_csv(out / "issuer_reach_among_jointly_connected_publications.csv", index=False)
    issuer_by_pd_degree.to_csv(out / "issuer_reach_conditional_on_policy_degree.csv", index=False)
    indicator_summary.to_csv(out / "configuration_text_indicator_characterization.csv", index=False)
    null_replicates.to_csv(out / "cross_relation_alignment_null_replicates.csv", index=False)
    null_tests.to_csv(out / "cross_relation_alignment_null_summary.csv", index=False)
    year_counts.to_csv(out / "primary_null_year_strata.csv", index=False)
    checks.to_csv(out / "configuration_validation_checks.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_research_question": "Are recorded P-G and P-D relations disproportionately aligned on the same final-P publications, conditional on publication-year opportunity?",
        "relations": {"P_G": "Raw final Dimensions-recorded publication--grant ties", "P_D": "Raw final Dimensions-recorded publication--policy-document ties", "D_I": "Issuer attribution attached after fitting/configuration counting; not independently permuted"},
        "primary_null": PRIMARY_NULL,
        "sensitivity_null": SENSITIVITY_NULL,
        "n_permutations_per_null": N_PERMUTATIONS,
        "primary_statistics": {"N_both": "Extensive-margin cross-relation alignment", "intensive_degree_product_sum": "Intensive-margin co-location of P-G and P-D degrees"},
        "supplementary_statistics": {"distinct_issuer_count": "Issuer reach among jointly connected P nodes, reported conditional on P-D degree", "text_indicators": "Ex post characterizations of observed configurations; excluded from the null"},
        "observed_configuration_counts": {"P_G_only": pg_only, "P_D_only": pd_only, "both": both, "neither": neither},
        "interpretive_boundary": "The permutation test concerns alignment of recorded relation layers at final-P publications. It does not establish causal funding or policy processes, actor intention, endorsement, policy impact, semantic affinity, communities, or latent structure.",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
    }
    (out / "configuration_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    primary_both = null_tests[(null_tests["null_scheme"] == PRIMARY_NULL) & (null_tests["statistic"] == "N_both")].iloc[0]
    print("\n" + "=" * 78)
    print("STEP 12 LOCAL CROSS-RELATION CONFIGURATION ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Final P publications: {len(publications):,}")
    print(f"P-G only: {pg_only:,}; P-D only: {pd_only:,}; both: {both:,}; neither: {neither:,}")
    print(f"Primary null observed/null enrichment for N_both: {primary_both['observed_to_null_mean_enrichment']:.3f}")
    print(f"Primary null two-sided p-value for N_both: {primary_both['permutation_p_two_sided']:.4f}")
    print(f"Outputs saved to: {out}")


if __name__ == "__main__":
    main()
