# Reproducibility Materials for *Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research*

## Purpose

This repository should document the complete construction and analysis of the publication-anchored Dimensions record system while respecting the Dimensions data-use agreement. The study is reproducible for users with authorized Dimensions access through the exact retrieval specification, local screening code, linkage-construction code, analytic scripts, and derived non-confidential outputs.

## Release Contents

| Material | Repository status | Notes |
| --- | --- | --- |
| Exact Dimensions publication query | Include | Deposit the executable DSL query and the query log saved during retrieval. |
| Retrieval date and API-run timestamps | Include | Deposit `run_status.json`, `final_retrieval_summary.json`, and `linked_retrieval_query_log.jsonl`. |
| Requested fields and pagination procedure | Include | Documented in the retrieval script and query log. Linked records were retrieved in batches of 250 IDs and deduplicated by Dimensions record identifier. |
| Local screening code | Include | Deposit the retrieval-and-screening script that defines the approved AI/ML and core mental-health phrase lists. |
| Screening dictionary and decisions | Include, subject to record-title permission | Deposit `publication_screening_codebook.csv` and `publication_screening_decisions.csv`. If titles cannot be shared, provide record identifiers, decision status, matched-term fields, and decision reason only. |
| Indicator dictionaries and code | Include | Deposit the text-indicator script and its dictionary/term-prevalence outputs. |
| Topic-family code and mapping | Include | Deposit the topic-family script, codebook, and derived publication-level topic-family indicators where permitted. |
| Record-linkage edge lists | Include only if permitted | Deposit publication--grant and publication--policy-document identifier pairs if the Dimensions agreement permits redistribution; otherwise deposit an ID-restricted or reconstruction version. |
| Derived aggregate outputs | Include | Deposit coefficient tables, diagnostics, permutation summaries, configuration summaries, figure data/provenance, and manuscript tables. |
| Raw Dimensions records | Do not redistribute unless explicitly permitted | Users with authorized Dimensions access should reconstruct these records using the supplied queries and identifiers. |

## Required Repository Structure

```text
README.md
LICENSE_OR_DATA_USE_NOTICE.md
queries/
  publication_anchor_query.dsl
  linked_retrieval_query_log.jsonl
code/
  retrieval_and_screening.py
  step1_relation_audit.py
  step2_text_indicators.py
  step3_topic_families.py
  step4_linkage_associations.py
  step5_policy_window_and_timing.py
  step6_two_mode_structure_and_issuers.py
  step7_indicator_robustness.py
  step12_local_configurations.py
  generate_manuscript_figures.py
codebooks/
  publication_screening_codebook.csv
  text_indicator_dictionary.csv
  topic_family_codebook.csv
metadata/
  run_status.json
  final_retrieval_summary.json
  screening_manifest.json
  data_manifest.json
derived_data/
  publication_screening_decisions.csv
  publication_text_indicators.csv
  publication_concept_families.csv
  association_model_coefficients.csv
  association_model_diagnostics.csv
  policy_window_model_coefficients.csv
  policy_window_model_diagnostics.csv
  configuration_observed_summary.csv
  cross_relation_alignment_null_summary.csv
  record_year_difference_summary.csv
  network_basic_statistics.csv
  network_degree_concentration.csv
  policy_issuer_representation.csv
figures/
  figure_data_sources.json
```

## Methods That Must Be Reconstructible

The repository must make the following procedures directly inspectable.

1. **Candidate publication retrieval.** The Dimensions query must be deposited as executable text, including entity type, search fields, date range, document types, return fields, and any language restriction or lack of language restriction.
2. **Local screening.** A candidate record is retained only when at least one approved AI/ML phrase and at least one approved core mental-health phrase occur in its title or available abstract. Matching is case-insensitive and literal, uses Unicode NFKC normalization, normalizes Unicode hyphen variants, permits whitespace variation within phrases, uses word-boundary safeguards, and uses no stemming, wildcard expansion, manual inclusion decisions, or outcome information.
3. **Record linkage.** Grants are retrieved only from publication `supporting_grant_ids`; policy documents are retrieved only where their recorded `publication_ids` contain an anchor-publication identifier. All retrieved entities are deduplicated by Dimensions record identifier. The raw relations are publication--grant, publication--policy document, and policy document--issuer.
4. **Publication indicators and covariates.** The repository must state the exact primary and sensitivity dictionaries, specify title-plus-available-abstract matching, describe missing-abstract handling, and document the supplementary topic-family rules.
5. **Models and permutation test.** The repository must provide exact model terms, sample definitions, robust-standard-error specification, diagnostic/robustness outputs, the random seed, the number of permutations, and the empirical $p$-value formula.

## Pre-submission Release Checklist

Before submission, replace the manuscript's provisional Data Availability statement with the repository DOI, public URL, or blinded review link. Confirm that all materials above comply with the Dimensions license. The manuscript should not claim that raw Dimensions records are public if redistribution is prohibited.

## Suggested Data Availability Wording

Use this text only after inserting the actual repository link or DOI and confirming the licensing position:

> Reproducibility materials are available at **[repository DOI or blinded review URL]**. The repository contains the exact retrieval specification, run metadata, local screening decisions and codebook, text-indicator and topic-family dictionaries, relation-construction and analysis code, derived aggregate outputs, and instructions for reconstruction by authorized Dimensions users. Raw Dimensions records are not redistributed where restricted by the Dimensions data-use agreement.

This wording distinguishes reproducible procedures and shareable derived materials from licensed raw records.
