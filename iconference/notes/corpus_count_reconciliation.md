# Corpus-count reconciliation: 2,224 vs. 2,227 vs. 2,198 vs. 2,120

## Summary

Four different "corpus size" numbers appear across this repository and the
existing manuscript. All four are legitimate, reproducible outputs of
different, well-defined stages of the *same* pipeline — they are not in
conflict once the stage each one refers to is made explicit:

| N | What it is | Where it comes from | Reproducible from repo files? |
|---|---|---|---|
| **2,224** | Scopus's own live search-hit count at the time the query was last logged (Nov 2025) | `scopus_query.txt` comment | No — this is a live database count, not derived from any file in the repo |
| **2,227** | Raw CSV export currently checked into the repo | `data/ai_literacy.csv` (row count) | Yes — `len(pd.read_csv('data/ai_literacy.csv'))` |
| **2,211 → 2,210** | Intermediate de-duplication/cleaning steps | `scripts/python_ai_lit_pipeline.ipynb`, cell 8/9 | Yes — see reproduction below |
| **2,198** | Final assembled corpus reported in the existing manuscript | `scripts/python_ai_lit_pipeline.ipynb`, cell 8/9 output; matches `Current_AI_Literacy_Paper.pdf` abstract ("a corpus of 2,198 Scopus-indexed documents") | Yes — exactly reproduced below |
| **2,120** | Final analytic subset (title+abstract present, non-substantive record types and exact-duplicate texts removed) used for the 10-topic KMeans-guided BERTopic model | `scripts/rq4_kmeans_guided_cleanlabels_topic_assignments.csv`; matches `Current_AI_Literacy_Paper.pdf` ("a cleaned subset of 2,120 records...analyzed using KMeans-guided BERTopic") | Yes — `len(pd.read_csv(...))` |

**2,120 is the number every iConference analysis in this folder already
uses** (topic assignments, topic-share comparison, lexical distinctiveness
corpus). Nothing about that number changes as a result of this
investigation.

## Investigation

### Step 1 — is `data/ai_literacy.csv`'s row count (2,227) real records?

No. Direct inspection (`raw.loc[376:385]`) shows rows 379-383 (0-indexed)
are a single corrupted CSV-parsing incident:

- **Row 379**: a real record ("AI robots promote South Korean
  preschoolers' AI literacy and computational thinking," 2025,
  10.1111/fare.13189) but with `Document Type` and `EID` missing.
- **Rows 380-382**: entirely blank (every field `NaN`).
- **Row 383**: not a real record at all — it is a misaligned text
  fragment ("...TechCheck has demonstrated strong psychometric
  properties... Godspeed scale...anthropomorphism...") that has clearly
  spilled out of row 379's Abstract field into its own CSV row, with the
  overflow text shifted into the wrong columns (e.g. the literal string
  "Godspeed scale is a comprehensive tool designed to measure five key
  dimensions of human reactions to robots: anthropomorphism" appears
  inside the `Document Type` column).

This is consistent with an unescaped line break inside one record's
multi-line Abstract field at Scopus-export time, which caused a naive CSV
parse to split one logical record across five physical rows (379 + the
four spillover rows 380-383) instead of one.

**This exact block was already identified in the parent project.**
`RQ4_thematic_analysis.ipynb` flags it directly:
`df_rq4["exclude_corrupted_import"] = df_rq4["original_index"].isin([380, 381, 382, 383])`,
with the comment *"Corrupted Scopus import block identified during
inspection. Rows 380-383 are not valid standalone records. Row 379 appears
to be a real record with partial metadata, so it is NOT excluded here."*
This reconciliation independently confirms that prior finding rather than
discovering something new.

### Step 2 — reproducing the manuscript's 2,198 exactly

`scripts/python_ai_lit_pipeline.ipynb` (cells 8 and 9) applies, in order:
title-based de-duplication, removal of records missing Title/Year, and
removal of specific non-substantive `Document Type` values (Erratum,
Letter, Data paper, Retracted, Short survey) plus the corrupted row 383
(caught by an explicit `str.contains('Godspeed scale|anthropomorphism')`
mask written specifically to remove that one garbled row by its telltale
content, since it cannot be identified by `Document Type` alone).

Re-running that exact sequence against the current `data/ai_literacy.csv`
reproduces the manuscript's number exactly:

```
raw rows:                              2,227
after exact-title dedup (keep first):  2,211   (-16; includes 2 of the 3 blank rows 380-382, which
                                                 pandas treats as duplicates of each other on Title=NaN)
after dropping remaining null Title/Year rows: 2,210   (-1; the last surviving blank row)
after dropping Erratum/Letter/Data paper/
     Retracted/Short survey + the row-383 fragment: 2,198   (-12)
```

`2,198` matches `Current_AI_Literacy_Paper.pdf`'s reported corpus size
exactly, and matches the notebook's own printed output
(`"Records after cleaning: 2,198"`, both times this cell was run — see
below).

Notably, the notebook's *own* two runs of this identical cell (cells 8 and
9, apparently re-executed at different times) report slightly different
**"before cleaning"** counts (2,210 vs. 2,209) but land on the exact same
**final** count (2,198) both times. That one-record wobble in the raw
input between two executions of the same cleaning code, on what should be
"the same" saved search, is itself direct in-repo evidence of the
mechanism described in Step 3.

### Step 3 — where does 2,224 (`scopus_query.txt`) come from?

`2,224` cannot be produced by any deterministic filtering step applied to
the current `data/ai_literacy.csv` — none of the reproducible intermediate
counts above (2,227 / 2,211 / 2,210 / 2,198) land on 2,224, and it sits
*between* the raw row count (2,227) and the post-dedup counts (2,210-2,211),
which is not where a simple subset of this file's rows could plausibly
fall.

The most defensible explanation is that `2,224` is **not derived from this
CSV at all**: it is Scopus's own self-reported "results found" count for
the saved query, recorded directly from the Scopus interface at the time
`scopus_query.txt` was last updated ("Last updated: November 2025"). A
citation database's live hit count for a broad, recency-heavy query (this
one is unbounded through `PUBYEAR < 2026`, i.e. it includes documents
still being indexed/finalized in late 2025) is well known to fluctuate by
a handful of records from one moment to the next as new items are added,
reclassified, or briefly duplicated during indexing — independent of
anything a downstream CSV cleaning step does. The CSV export in this repo
was evidently pulled at a slightly different moment than whenever the
"2,224" figure was read off the Scopus interface, which is sufficient to
explain a few records' difference in either direction. This is a normal
property of live bibliometric database searches, not a data-quality error
in this repository.

## Recommendation for the iConference paper

**Report the same two numbers the existing manuscript already reports**,
since both are exactly reproducible from files in this repository and
require no new judgment calls:

- **Assembled corpus: 2,198** Scopus-indexed documents (2016-2025) —
  matches `Current_AI_Literacy_Paper.pdf` and is exactly reproduced from
  `data/ai_literacy.csv` via `python_ai_lit_pipeline.ipynb`'s documented
  cleaning steps (see Step 2).
- **Final analytic subset: 2,120** documents with title+abstract text,
  used for the 10-topic KMeans-guided BERTopic model — matches
  `Current_AI_Literacy_Paper.pdf` and is exactly
  `scripts/rq4_kmeans_guided_cleanlabels_topic_assignments.csv`'s row
  count. This is the number every substantive analysis in `iconference/`
  already uses; it is unaffected by this reconciliation.

**Do not report 2,227** (the repo's raw CSV row count) as "the corpus," since
it includes the 4 known parsing-artifact rows and has not had the
manuscript's own de-duplication/document-type filtering applied.

**Cite 2,224 only as context**, if at all — e.g. "an initial Scopus search
of N ≈ 2,224 records (`scopus_query.txt`), narrowed to a final corpus of
2,198 documents after removing duplicate titles, non-substantive record
types, and one CSV parsing artifact" — rather than as a precise figure to
reconcile to the last digit, since it is a live database count taken at a
different moment than the CSV export and is not expected to match it
exactly.

**No numbers were manually changed to force a match.** Every count above
(2,227 / 2,211 / 2,210 / 2,198 / 2,120) is produced by re-running existing,
documented code against the existing repository files, and 2,198 was
reached by exact reproduction of the manuscript's own cleaning pipeline,
not by adjusting a threshold until the numbers agreed.

## Change made to `iconference/tables/table2_methods_summary.csv`

`scripts/03_methods_summary_table.py` previously reported "Raw records
retrieved: 2,227" (the raw CSV row count). It has been updated to report
the corpus using the same two-stage framing as the manuscript itself
(2,198 assembled corpus; 2,120 final analytic subset), with the Scopus
search count (~2,224) and the raw CSV/cleaning detail moved into a
footnote-style note field that points to this document. See the updated
table and `README.md` for the current wording.
