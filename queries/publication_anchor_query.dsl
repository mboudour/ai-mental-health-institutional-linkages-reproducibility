search publications
in title_abstract_only for """
(
  "artificial intelligence" OR "machine learning" OR "deep learning"
  OR "neural network" OR "large language model" OR LLM
)
AND
(
  "mental health" OR "mental illness" OR psychiatry OR psychiatric
  OR psychotherapy OR psychotherapeutic
  OR "mental disorder" OR "psychiatric disorder"
)
"""
where year in [2000:2025]
and type in ["article", "review"]
return publications[
  id + year + date + title + abstract + doi + type + document_type
  + concepts + concepts_scores
  + research_org_countries + research_orgs + times_cited
  + supporting_grant_ids
]
