#!/usr/bin/env python3
"""Generate data-driven figures for the manuscript.

Save in new_version/computations/ and run from new_version/:
    python computations/generate_manuscript_figures.py

Figures are written directly to:
    manuscript/figures/

The script uses the validated outputs of the publication-anchored analysis.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

SCRIPT_VERSION = "generate_manuscript_figures_1.0"
DPI = 400
TEXT_COLOR = "#1D2633"
MUTED = "#6B7280"
NAVY = "#153E59"
TEAL = "#087E8B"
GOLD = "#E0A458"
RED = "#C44E52"
PURPLE = "#7054A6"
LIGHT_BLUE = "#DCEEF7"
LIGHT_TEAL = "#D7F0F0"
LIGHT_GOLD = "#F8E8C4"
LIGHT_RED = "#F5D8D7"
FIGURE_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#AAB2BD",
    "grid.color": "#D8DEE6",
    "grid.linewidth": 0.7,
}


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required validated output is missing: {path}")


def save_figure(fig: plt.Figure, out: Path, basename: str, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(out / f"{basename}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def term_label(term: str) -> str:
    labels = {
        "ethics_responsibility_primary": "Ethics/responsibility\nindicator",
        "computational_performance_primary": "Computational-performance\nindicator",
        "mental_health_conditions_symptoms": "Conditions/symptoms",
        "clinical_assessment_diagnosis": "Assessment/diagnosis",
        "treatment_intervention": "Treatment/intervention",
        "neurocognitive_affective_processes": "Neurocognitive/affective",
    }
    return labels.get(term, term.replace("_", " ").title())


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, face: str, edge: str = NAVY, fontsize: float = 10) -> None:
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.3, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT_COLOR, wrap=True)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.35, color=NAVY))


def figure_1_design(manifest: dict[str, Any], edge_summary: pd.DataFrame, out: Path) -> None:
    p_n = manifest["record_counts"]["final_P"]
    g_n = manifest["record_counts"]["linked_grants"]
    d_n = manifest["record_counts"]["linked_policy_documents"]
    pg_edges = int(edge_summary.loc[edge_summary["network"] == "publication_grant", "edges"].iloc[0])
    pd_edges = int(edge_summary.loc[edge_summary["network"] == "publication_policy_document", "edges"].iloc[0])
    issuer_row = edge_summary.loc[edge_summary["network"] == "policy_issuer_publication"].iloc[0]
    issuer_n = int(issuer_row["left_nodes_connected"])

    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Two-stage construction and recorded relational system", loc="left", pad=12)

    draw_box(ax, (0.35, 4.95), 2.15, 1.15, "Dimensions candidate retrieval\n13,574 publications", LIGHT_BLUE)
    draw_box(ax, (3.15, 4.95), 2.35, 1.15, "Literal title/abstract screen\nexplicit AI/ML AND mental-health terms", LIGHT_GOLD)
    draw_box(ax, (6.2, 4.95), 2.15, 1.15, f"Publication anchor P\n{p_n:,} publications", LIGHT_TEAL)
    arrow(ax, (2.52, 5.53), (3.12, 5.53))
    arrow(ax, (5.52, 5.53), (6.17, 5.53))

    draw_box(ax, (2.05, 1.2), 2.25, 1.1, f"Linked grants G\n{g_n:,} records", "#E4ECF4")
    draw_box(ax, (6.0, 1.2), 2.55, 1.1, f"Policy documents D\n{d_n:,} records", "#E4ECF4")
    draw_box(ax, (9.55, 1.2), 1.95, 1.1, f"Issuers I\n{issuer_n} organizations", "#E4ECF4")
    arrow(ax, (7.05, 4.92), (3.35, 2.34))
    arrow(ax, (7.28, 4.92), (7.28, 2.34))
    arrow(ax, (8.58, 1.75), (9.52, 1.75))

    ax.text(4.7, 3.55, f"P–G: {pg_edges:,} recorded ties", ha="center", va="center", fontsize=9, color=NAVY, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B4C4D1"))
    ax.text(8.15, 3.55, f"P–D: {pd_edges:,} recorded ties", ha="center", va="center", fontsize=9, color=NAVY, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B4C4D1"))
    ax.text(10.42, 2.65, "D–I: issuer attribution\n(one recorded issuer per D)", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.text(0.35, 0.25, "All P–G and P–D ties use Dimensions-recorded identifiers. Policy-document links are citation/linkage records, not evidence of topical similarity, endorsement, or policy impact.", fontsize=8.5, color=MUTED)
    save_figure(fig, out, "figure_1_study_design")


def figure_2_indicators(overlap: pd.DataFrame, out: Path) -> None:
    data = overlap[overlap["definition"] == "primary"].copy()
    order = ["neither", "ethics_responsibility_only", "computational_performance_only", "both"]
    colors = ["#C9D3DD", TEAL, GOLD, PURPLE]
    labels = ["Neither", "Ethics/\nresponsibility only", "Computational-\nperformance only", "Both"]
    data = data.set_index("overlap_group").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    bars = ax.bar(labels, data["pct_of_P"], color=colors, width=0.68)
    ax.set_ylabel("Share of publications in P (%)")
    ax.set_ylim(0, max(data["pct_of_P"]) * 1.18)
    ax.set_title("Non-exclusive publication text indicators")
    ax.grid(axis="y")
    for bar, n, pct in zip(bars, data["n_publications"], data["pct_of_P"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{n:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.text(0.01, -0.20, "Indicators were measured from publication titles plus available abstracts. The narrow computational-performance dictionary is the primary specification.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    save_figure(fig, out, "figure_2_text_indicators")


def figure_3_models(grant_models: pd.DataFrame, policy_models: pd.DataFrame, out: Path) -> None:
    """Create one readable grouped forest plot rather than three compressed panels."""
    terms = ["ethics_responsibility_primary", "computational_performance_primary"]
    specifications = [
        (grant_models[grant_models["outcome"] == "has_grant_link"].copy(), "Grant", NAVY),
        (policy_models[policy_models["analysis_sample"] == "publications_through_2021_primary"].copy(), "Policy 2021", TEAL),
        (policy_models[policy_models["analysis_sample"] == "publications_through_2020_window_robustness"].copy(), "Policy 2020", PURPLE),
    ]
    positions = [5, 4, 3, 2, 1, 0]
    rows = []
    for (frame, group_label, color), group_positions in zip(specifications, [positions[:2], positions[2:4], positions[4:]]):
        selected = frame[frame["term"].isin(terms)].copy()
        selected["term"] = pd.Categorical(selected["term"], categories=terms, ordered=True)
        selected = selected.sort_values("term")
        if len(selected) != 2:
            raise ValueError(f"Figure 3 requires both focal indicators for: {group_label}")
        for y, (_, row) in zip(group_positions, selected.iterrows()):
            rows.append((y, row, group_label, color))

    fig, ax = plt.subplots(figsize=(8.7, 5.25))
    for y, row, _, color in rows:
        ax.errorbar(
            row["odds_ratio"],
            y,
            xerr=np.array([[row["odds_ratio"] - row["ci_95_odds_ratio_low"]], [row["ci_95_odds_ratio_high"] - row["odds_ratio"]]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3.5,
            markersize=6.5,
            zorder=3,
        )
        # The estimate column lies outside the plotting region, so no value text
        # crosses a confidence interval or the reference line.
        ax.text(
            1.03,
            y,
            f"{row['odds_ratio']:.2f} [{row['ci_95_odds_ratio_low']:.2f}, {row['ci_95_odds_ratio_high']:.2f}]",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.4,
            clip_on=False,
        )

    ax.axvline(1, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.axhline(3.5, color="#D5DCE2", linewidth=0.8, zorder=0)
    ax.axhline(1.5, color="#D5DCE2", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.35, 4.4)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_ylim(-0.72, 5.72)
    ax.set_yticks(positions)
    row_labels = [
        f"{group_label}\n{term_label(str(row['term'])).replace(chr(10), ' ')}"
        for _, row, group_label, _ in rows
    ]
    ax.set_yticklabels(row_labels, fontsize=7.7)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_xlabel("Adjusted odds ratio (log scale)", labelpad=10)
    fig.suptitle("Adjusted associations of primary publication indicators with recorded links", x=0.01, y=0.975, ha="left", fontsize=11, fontweight="bold")
    ax.text(1.03, 1.005, "OR [95% CI]", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4, fontweight="bold", clip_on=False)
    fig.text(0.01, 0.035, "Points are adjusted odds ratios; bars are 95% confidence intervals. All models adjust for publication year, log citations, and supplementary topic-family descriptors. Policy models use the primary through-2021 and more restrictive through-2020 publication-age eligibility samples; estimates are associational.", fontsize=7.1, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.30, right=0.82, bottom=0.29, top=0.85)
    save_figure(fig, out, "figure_3_adjusted_associations", use_tight_layout=False)


def figure_4_temporal(timing: pd.DataFrame, out: Path) -> None:
    timing = timing.copy()
    timing["relation_label"] = timing["relation"].map({"publication_grant": "P–G", "publication_policy_document": "P–D"})
    counts = timing.set_index("relation_label")[["n_negative_year_differences", "n_zero_year_differences", "n_positive_year_differences"]].reindex(["P–G", "P–D"])
    shares = counts.div(counts.sum(axis=1), axis=0).mul(100)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    bottom = np.zeros(len(shares))
    colors = [RED, "#B7C1CC", TEAL]
    labels = ["Earlier linked-record year", "Same year", "Later linked-record year"]
    for column, color, label in zip(shares.columns, colors, labels):
        values = shares[column].to_numpy(dtype=float)
        ax.bar(shares.index, values, bottom=bottom, color=color, label=label, width=0.62)
        bottom += values
    ax.set_ylabel("Share of recorded ties (%)")
    ax.set_ylim(0, 114)
    ax.set_title("Record-year ordering of recorded relations", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    for x, relation in enumerate(shares.index):
        row = timing[timing["relation_label"] == relation].iloc[0]
        total = int(counts.loc[relation].sum())
        ax.text(x, 102.5, f"n = {total:,}\nMedian: {row['median_year_difference_all']:+.0f} years", ha="center", va="bottom", fontsize=8.5)
    fig.text(0.01, -0.16, "Each bar is normalized to 100%. Record-year differences describe database record dates only; they are not causal lags or evidence of policy impact.", fontsize=8.5, color=MUTED)
    save_figure(fig, out, "figure_4_temporal_ordering")


def plot_null_hist(
    ax: plt.Axes,
    null_values: pd.Series,
    observed: float,
    title: str,
    x_label: str,
    enrichment: float,
    p_two: float,
    offscale_observed: bool = False,
) -> None:
    """Plot a permutation null without wasting the axis on an off-scale observation."""
    values = pd.Series(null_values, dtype=float).dropna()
    ax.hist(values, bins=min(30, max(10, int(np.sqrt(len(values))))), color=LIGHT_BLUE, edgecolor="#8AA9BF")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Permutations")

    if offscale_observed:
        # The observed statistic is far beyond the null range. Showing it as a
        # vertical line forces an uninformative empty span across the panel.
        # Display the complete null distribution and report the off-scale value
        # explicitly in a red-bordered annotation instead.
        span = max(values.max() - values.min(), 1.0)
        ax.set_xlim(values.min() - 0.04 * span, values.max() + 0.08 * span)
        annotation = (
            f"Observed = {observed:.0f}\n"
            f"Null maximum = {values.max():.0f}\n"
            f"Enrichment = {enrichment:.2f}\n"
            f"Two-sided p = {p_two:.3f}"
        )
        ax.text(
            0.98,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor=RED, linewidth=1.1),
        )
    else:
        ax.axvline(observed, color=RED, linewidth=2.2, label=f"Observed = {observed:.0f}")
        ax.legend(frameon=False, fontsize=8)
        ax.text(
            0.98,
            0.96,
            f"Enrichment = {enrichment:.2f}\nTwo-sided p = {p_two:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B4C4D1"),
        )


def figure_5_alignment(config: pd.DataFrame, null_summary: pd.DataFrame, null_replicates: pd.DataFrame, out: Path) -> None:
    order = ["P_G_only", "P_D_only", "both", "neither"]
    labels = ["P–G only", "P–D only", "Both", "Neither"]
    colors = [NAVY, TEAL, GOLD, "#C9D3DD"]
    data = config.set_index("configuration").reindex(order).reset_index()
    primary = null_summary[null_summary["null_scheme"] == "publication_year_stratified_P_end_permutation"].set_index("statistic")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]})
    bars = axes[0].bar(labels, data["n_publications"], color=colors, width=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Publications in P (log scale)")
    axes[0].set_title("A. Observed local configurations", loc="left", fontsize=10)
    for bar, value, share in zip(bars, data["n_publications"], data["share_of_final_P"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:,}\n({share:.2%})", ha="center", va="bottom", fontsize=8)
    both_row = primary.loc["N_both"]
    intensity_row = primary.loc["intensive_degree_product_sum"]
    null_primary = null_replicates[null_replicates["null_scheme"] == "publication_year_stratified_P_end_permutation"]
    plot_null_hist(axes[1], null_primary["N_both"], float(both_row["observed"]), "B. Extensive-margin alignment", "P publications with both P–G and P–D ties", float(both_row["observed_to_null_mean_enrichment"]), float(both_row["permutation_p_two_sided"]))
    plot_null_hist(axes[2], null_primary["intensive_degree_product_sum"], float(intensity_row["observed"]), "C. Intensive-margin concentration", r"$\sum_p d_{PG}(p)\,d_{PD}(p)$ under the primary null", float(intensity_row["observed_to_null_mean_enrichment"]), float(intensity_row["permutation_p_two_sided"]), offscale_observed=True)
    fig.suptitle("Local alignment of recorded grant and policy-document relations", x=0.01, y=1.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, "Primary null: complete P–D incidence profiles are reassigned within publication-year strata while P–G ties, policy-document endpoints, and issuer attributions are retained. This tests alignment of recorded ties, not causal pathways.", fontsize=8.3, color=MUTED)
    save_figure(fig, out, "figure_5_cross_relation_alignment")


def figure_6_issuers(issuers: pd.DataFrame, out: Path) -> None:
    data = issuers.sort_values("distinct_linked_publications", ascending=False).head(10).copy()
    labels = ["\n".join(textwrap.wrap(str(value), width=24)) for value in data["issuer_name"]]
    color_map = {"Government": NAVY, "Nonprofit": TEAL, "Facility": GOLD, "Other": PURPLE}
    colors = [color_map.get(str(value), "#8C97A3") for value in data["issuer_type"]]
    fig, ax = plt.subplots(figsize=(10.6, 6.3))
    bars = ax.barh(np.arange(len(data))[::-1], data["distinct_linked_publications"], color=colors)
    ax.set_yticks(np.arange(len(data))[::-1])
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_xlabel("Distinct publications in P linked through issuer policy documents")
    ax.set_title("Policy-issuer representation in recorded P–D–I paths", loc="left")
    for bar, value, country in zip(bars, data["distinct_linked_publications"], data["issuer_country"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value}  ({country})", va="center", fontsize=8.2)
    handles = [plt.Line2D([0], [0], color=color, lw=6, label=label) for label, color in color_map.items() if label in set(data["issuer_type"].astype(str))]
    ax.legend(handles=handles, title="Issuer type", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    ax.text(0.01, -0.15, "Issuer attributes identify organizations recorded as publishers of policy documents. They do not indicate endorsement, intention, or influence over the linked publications.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    save_figure(fig, out, "figure_6_policy_issuer_representation")


def main() -> None:
    root = root_dir()
    out = root / "manuscript" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "step1_manifest": root / "computations" / "outputs" / "step1_v6_audit_v2" / "data_manifest.json",
        "edge_summary": root / "computations" / "outputs" / "step1_v6_audit_v2" / "edge_summary.csv",
        "indicator_overlap": root / "computations" / "outputs" / "step2_v6_indicators" / "text_indicator_overlap.csv",
        "grant_models": root / "computations" / "outputs" / "step4_v6_linkage_associations" / "association_model_coefficients.csv",
        "policy_models": root / "computations" / "outputs" / "step5_v6_policy_window_and_timing_v2" / "policy_window_model_coefficients.csv",
        "timing": root / "computations" / "outputs" / "step5_v6_policy_window_and_timing_v2" / "record_year_difference_summary.csv",
        "issuers": root / "computations" / "outputs" / "step6_v6_networks_organizations_countries" / "policy_issuer_representation.csv",
        "config": root / "computations" / "outputs" / "step12_v6_local_configurations" / "configuration_observed_summary.csv",
        "null_summary": root / "computations" / "outputs" / "step12_v6_local_configurations" / "cross_relation_alignment_null_summary.csv",
        "null_replicates": root / "computations" / "outputs" / "step12_v6_local_configurations" / "cross_relation_alignment_null_replicates.csv",
    }
    for path in paths.values():
        require(path)
    with plt.rc_context(FIGURE_STYLE):
        manifest = read_json(paths["step1_manifest"])
        edge_summary = pd.read_csv(paths["edge_summary"])
        figure_1_design(manifest, edge_summary, out)
        figure_2_indicators(pd.read_csv(paths["indicator_overlap"]), out)
        figure_3_models(pd.read_csv(paths["grant_models"]), pd.read_csv(paths["policy_models"]), out)
        figure_4_temporal(pd.read_csv(paths["timing"]), out)
        figure_5_alignment(pd.read_csv(paths["config"]), pd.read_csv(paths["null_summary"]), pd.read_csv(paths["null_replicates"]), out)
        figure_6_issuers(pd.read_csv(paths["issuers"]), out)
    provenance = {
        "script_version": SCRIPT_VERSION,
        "figure_files": [
            "figure_1_study_design", "figure_2_text_indicators", "figure_3_adjusted_associations",
            "figure_4_temporal_ordering", "figure_5_cross_relation_alignment", "figure_6_policy_issuer_representation",
        ],
        "input_sources": ["record-system audit", "publication-indicator analysis", "adjusted association models", "record-year analysis", "issuer representation", "local configuration permutation analysis"],
        "interpretive_boundary": "Figures use observed record links and validated association/permutation outputs. They do not establish causal effects, institutional intention, endorsement, policy impact, or topical affinity.",
    }
    (out / "figure_data_sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Generated manuscript figures:")
    for name in provenance["figure_files"]:
        print(f"  {out / (name + '.png')}")
    print(f"  {out / 'figure_data_sources.json'}")


if __name__ == "__main__":
    main()
