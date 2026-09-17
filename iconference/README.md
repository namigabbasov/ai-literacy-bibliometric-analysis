# iConference paper — analysis subfolder

This folder contains a self-contained, additional analysis built on top of
the parent repository's AI literacy bibliometric project, for a separate
~3,000-word iConference paper with the narrower research question:

> How has the rise of generative AI reconfigured the thematic and
> conceptual orientation of AI literacy research?

**Argument:** AI literacy is shifting from primarily *understanding* AI
toward *interacting with*, *evaluating*, and exercising *judgment* with AI
in situated contexts.

Nothing in the parent repository (`../scripts/`, `../data/`, `../outputs/`)
was modified. Everything here reuses the parent project's existing cleaned
data and topic-modeling results read-only, and writes only into this
`iconference/` subfolder.

---

## 1. What was reused from the parent project

| Parent file | Used for |
|---|---|
| `../scripts/rq4_kmeans_guided_cleanlabels_topic_assignments.csv` | Document-level 10-topic assignments (topic 0-9), `Year`, `Title`, `Abstract`, and `topic_text` (title+abstract, minimally cleaned) for **every** analysis in this folder |
| `../scripts/rq4_topic_label_lookup.csv` | Manual long-form and short-form topic labels |
| `../scripts/rq4_topic_share_change_table.csv` | Used only to **verify** that this folder's independently recomputed pre-2023 vs. 2024-2025 shares match the parent project's published numbers (they do, within rounding) |
| `../data/ai_literacy.csv`, `../scopus_query.txt` | Raw record count and Scopus search string, for Table 2 only |
| `../scripts/python_ai_lit_pipeline.ipynb` | Cleaning steps (title de-duplication, document-type filtering) reproduced in `03_methods_summary_table.py` to derive the "assembled corpus" count (2,198) reported in the existing manuscript; see `notes/corpus_count_reconciliation.md` |

**Corpus-count note:** this repository contains four different "corpus
size" numbers (2,224 in `scopus_query.txt`; 2,227 raw rows in
`data/ai_literacy.csv`; 2,198 in `Current_AI_Literacy_Paper.pdf`; 2,120 in
the RQ4 analytic subset). These are not in conflict — each refers to a
different, well-defined pipeline stage. Table 2 and this project report
**2,198** (assembled corpus) and **2,120** (final analytic subset), both
matching the existing manuscript exactly. See
`notes/corpus_count_reconciliation.md` for the full, reproducible
derivation and why the other two numbers appear.

The 10-topic solution itself (KMeans-guided BERTopic, `all-MiniLM-L6-v2`
embeddings, k=10, domain-adjusted c-TF-IDF vocabulary) is **not**
re-estimated anywhere in this folder — per the task scope, no new topic
model was run.

`topic_text` (used as the "titles + abstracts" text for the new lexical
analysis) is the parent project's own minimally-cleaned concatenation of
`Title` + `Abstract`, with HTML entities decoded, Scopus copyright
boilerplate stripped, and whitespace normalized — see
`RQ4_thematic_analysis.ipynb`, cell "df_rq4 = df.copy()...". Reusing it
avoids re-deriving text-cleaning logic the parent project already
validated, and keeps the lexical analysis on exactly the same document set
used for the topic-prevalence comparison.

---

## 2. Analyses performed

### 2.1 Temporal topic-prevalence comparison (reproduced + redesigned figure)
**Scripts:** `scripts/01_topic_prevalence_change.py` (point estimates and
tables) and `scripts/04_topic_share_bootstrap.py` (bootstrap uncertainty
and Figure 1 itself)

- `01_topic_prevalence_change.py` recomputes within-period topic shares
  (not raw counts) directly from the document-level topic assignments for
  three periods:
  - **Pre-2023**: `Year <= 2022` (n = 106)
  - **2023 (transition)**: `Year == 2023` (n = 147)
  - **2024-2025**: `Year` in {2024, 2025} (n = 1,867; the corpus does not
    extend past 2025)
- The main comparison (and Figure 1) emphasizes **pre-2023 vs. 2024-2025**;
  2023 is shown/retained in the long-format share table but excluded from
  the main change figure and table, consistent with treating it as a
  transition year.
- The script asserts that its independently recomputed change table
  matches the parent project's saved
  `rq4_topic_share_change_table.csv` (max difference 0.044 percentage
  points, i.e., rounding noise from the parent table's 1-decimal storage).
- **Figure 1** (built by `04_topic_share_bootstrap.py` from this change
  table) redesigns the parent project's horizontal bar chart of share
  change by color-coding each topic as **increased**, **relatively
  stable**, or **decreased**, using a ±2.0-percentage-point band (chosen
  because it falls in a clear gap in the observed distribution of changes:
  the two smallest-magnitude changes are -0.2 and -1.3 pp; the next
  smallest is 4.6 pp) — this is a descriptive/visualization threshold, not
  a statistical significance test, documented here rather than silently
  assumed — and adds 95% bootstrap CI error bars around each topic's
  observed change (10,000 replicates, resampling with replacement
  independently within each period, quantifying sampling uncertainty given
  the small pre-2023 sample, n=106).
- Outputs: `figures/fig1_topic_share_change.png` / `.pdf` (Figure 1),
  `tables/table_topic_share_change_bootstrap.csv` (observed change + 95%
  percentile CI for all 10 topics), `notes/topic_share_uncertainty.md`.
- Result: 8 of 10 topics' 95% CIs exclude zero; the 2 that do not
  ("Healthcare/medical AI literacy," "Human decision-making/trust") are
  exactly the 2 topics already classified "relatively stable" by the ±2 pp
  descriptive band. **That band remains a visualization threshold, not a
  significance test** — this note explicitly says so — but the two
  happen to agree here.

### 2.2 Lexical-distinctiveness analysis (new)
**Script:** `scripts/02_lexical_distinctiveness.py`

Compares the vocabulary of **pre-2023** documents (n=106) against
**2024-2025** documents (n=1,867), using `topic_text` (title + abstract).
**2023 is excluded** from this contrast (transition year).

- **Method:** weighted log-odds-ratio with an informative Dirichlet prior
  (Monroe, Colaresi & Quinn, 2008, *"Fightin' Words: Lexical Feature
  Selection and Evaluation for Identifying the Content of Political
  Conflict,"* Political Analysis). This is the standard method for
  identifying group/period-distinctive vocabulary while accounting for
  very different corpus sizes (implemented, e.g., in the `scattertext` and
  `convokit` "fighting words" modules — neither package is available in
  this project's conda environment (`ai-literacy`), so the statistic is
  implemented directly from the published formula in
  `02_lexical_distinctiveness.py::monroe_log_odds`).
- **Why this method** (documented per task instructions): raw frequency or
  simple log-odds differences are unstable for rare terms and do not
  account for the large period-size imbalance (106 vs. 1,867 documents)
  present here; the Dirichlet-smoothed z-score handles both.
- **Prior specification:** no independent/external background corpus is
  available in this repository. The prior's *shape* (relative term
  frequency) is taken from the combined pre-2023 + 2024-2025 sub-corpus.
  Its *total mass* (alpha0) is capped at the token count of the smaller
  period (pre-2023, ~9,100 tokens after preprocessing) rather than the
  full combined corpus. This was a deliberate correction: an initial
  version that set alpha0 to the full combined-corpus token count produced
  a degenerate prior (because the prior's per-term mass scales with the
  same counts being tested, the prior and data are not independent, which
  mechanically compresses z-scores toward zero regardless of how skewed a
  term actually is — e.g. "generative" (5 vs. 1,307 occurrences) produced
  |z| < 2). Capping alpha0 at the smaller period's size follows Monroe et
  al.'s own guidance not to let the prior's mass swamp the data, and its
  effect is checked via the min_df sensitivity analysis (Section 3 below;
  full detail in `notes/robustness_checks.md`).
- **Preprocessing:** lowercase; regex tokenization on alphabetic runs;
  removal of standard English stopwords (`sklearn.ENGLISH_STOP_WORDS`);
  removal of the corpus's own non-discriminating terms — "ai",
  "artificial intelligence", "literacy", and related forms — reusing
  verbatim the `domain_stop_words` set already defined and validated in
  the parent project's KMeans-guided BERTopic cell; and removal of
  residual bibliographic/publisher noise tokens (elsevier, springer,
  copyright, doi, ieee, acm, publisher, proceedings, issn, isbn, vol,
  etc.), identified by direct inspection of `topic_text` for boilerplate
  residue.
- **N-grams:** unigrams and bigrams, built from the filtered token stream
  (not sklearn's default stop-word handling — see the "false adjacency"
  caveat documented in the script and in `notes/robustness_checks.md`).
- **Minimum document-frequency threshold:** `min_df = 5` (a term/phrase
  must appear in at least 5 of the 1,973 combined pre-2023 + 2024-2025
  documents), chosen so results are not driven by rare/idiosyncratic
  phrases. Sensitivity to this threshold (min_df in {3, 5, 10}) is
  reported in `notes/robustness_checks.md`.
- **No terms were manually forced into the results.** The top-15-per-side
  ranked list in `tables/table_lexical_distinctive_terms.csv` is exactly
  the script's z-score ranking; nothing was reordered, substituted, or
  added by hand. Figure 2 itself (built by a separate, display-only
  script) groups a subset of these ranked terms for readability — see
  2.2b.

### 2.2a Prior-strength sensitivity (added in the methodological refinement pass)
**Script:** `scripts/05_lexical_prior_sensitivity.py`

Re-runs the identical lexical analysis (same preprocessing, vocabulary,
min_df=5) at four Dirichlet prior total-mass (alpha0) settings — 0.25x,
0.50x, 1.00x (current), 2.00x the smaller period's token count — and
compares each to the current specification via top-15 overlap and Spearman
rank correlation across the full shared vocabulary (6,322 terms).

- Outputs: `derived/lexical_prior_sensitivity.csv` (per-term z-score at
  every setting), `tables/table_lexical_prior_sensitivity_summary.csv`,
  `notes/lexical_prior_sensitivity.md`.
- **Result: robust.** Spearman rho ≥ 0.999 vs. the current specification
  at every tested setting; top-15 overlap per side ≥ 93%. The current
  alpha0 choice does not appear to be doing unacknowledged work in the
  ranking.

### 2.2b Document prevalence, literal bigram check, and Figure 2
(added in the methodological refinement pass)

- **Document prevalence** — **Script:** `scripts/06_lexical_docprevalence.py`.
  For every term in the top-30 table, adds the number and percentage of
  each period's documents that contain the term at least once (not just
  token counts), as a robustness/interpretability supplement — it does not
  re-rank anything. Output: `tables/table_lexical_distinctive_terms_with_docprevalence.csv`.
  Most top terms are widespread; a handful of pre-2023 terms are driven by
  very few documents (flagged automatically at <5% document prevalence,
  not silently dropped).
- **Literal contiguous bigram check** — **Script:** `scripts/07_literal_bigram_check.py`.
  Directly tests the main method's "false adjacency" caveat by re-deriving
  bigrams as literal adjacent word pairs from the lightly-cleaned original
  text (stopword-heavy pairs are filtered only *after* bigram
  construction, not before). Outputs:
  `derived/lexical_literal_bigram_check.csv`, `notes/literal_bigram_check.md`.
  All three bigrams in the main top-30 table are confirmed as genuine
  literal adjacent phrases (ranking 1st-2nd by this construction).
- **Figure 2** — **Script:** `scripts/08_figure2.py`. Builds Figure 2 from
  the same ranked results as a display-only step (no z-score, rank, count,
  or preprocessing choice is touched). Shows 9 concepts per side, built by
  (a) collapsing obvious lexical variants of one concept into a single bar
  — e.g. "generative"/"genai"/"gai"/"generative genai" → **Generative AI**;
  "critical"/"critical thinking" → **Critical thinking**, preferring the
  literal-bigram-confirmed phrase — and (b) deprioritizing terms resting on
  very few supporting documents (fewer than 5, reusing this project's own
  min_df=5 convention) in favor of more widespread, still-top-ranked
  alternatives. No term is skipped merely for sounding generic. Every
  grouping decision, and every skipped term, is documented with a full
  audit trail in `notes/figure2_concept_grouping.md`. Output:
  `figures/fig2_lexical_distinctiveness.png` / `.pdf`.

### 2.3 Interpretive synthesis of the existing 10 topics (new; no new topic model)
**Files:** `tables/table1_conceptual_synthesis.csv` / `.md`

A hand-authored (not statistically derived) mapping of the existing 10
topics into 5 broader conceptual orientations — Understanding AI,
Interacting with AI, Adopting AI, Evaluating/judging AI, and
Situated/professional/institutional AI literacy — based on each topic's
manual label, representative terms, and (for cross-checking) its temporal
behavior from Section 2.1. Three topics (0, 2, 3) are explicitly
cross-listed under two orientations rather than forced into a single
exclusive category; the justification column explains why in each case.
**This table is a researcher interpretation**, presented as such, not the
output of a clustering algorithm.

### 2.4 Methods/data summary table (new)
**Script:** `scripts/03_methods_summary_table.py`
**Files:** `tables/table2_methods_summary.csv` / `.md`

A compact table (corpus size, date range, source database, text fields,
topic-modeling approach, number of topics, period definitions, lexical
method) with all countable fields (raw record count, final corpus size,
date range, per-period document counts) computed directly from the raw
and assignment CSVs rather than typed by hand, so they cannot drift out of
sync with the data.

---

## 3. Generated outputs

### Figures
| File | Description |
|---|---|
| `figures/fig1_topic_share_change.pdf` / `.png` (300 dpi) | **Figure 1.** Change in topic prevalence, pre-2023 → 2024-2025, color-coded increased/stable/decreased, with 95% bootstrap CI error bars |
| `figures/fig2_lexical_distinctiveness.pdf` / `.png` (300 dpi) | **Figure 2.** Diverging bar chart of 9 concept-grouped terms per period by weighted log-odds z-score, built via a documented, mechanical display rule |

### Tables (paper-ready)
| File | Description |
|---|---|
| `tables/table_topic_share_change.csv` | Per-topic pre-2023 %, 2024-2025 %, change (pp), direction label — underlies Figure 1 |
| `tables/table_topic_share_change_bootstrap.csv` | Adds observed change, bootstrap mean/SE, and 95% CI (10,000 replicates) for all 10 topics |
| `tables/table_lexical_distinctive_terms.csv` | Top 15 terms per period with z-score, log-odds delta, raw counts, and document frequencies per period — Figure 2 is a concept-grouped subset of these (see `notes/figure2_concept_grouping.md`) |
| `tables/table_lexical_distinctive_terms_with_docprevalence.csv` | Same top-30 table, plus per-period document count/percentage containing each term |
| `tables/table_lexical_prior_sensitivity_summary.csv` | Top-15 overlap and Spearman rho vs. the current prior specification, at 4 alternative alpha0 settings |
| `tables/table1_conceptual_synthesis.csv` / `.md` | **Table 1.** Conceptual synthesis of the 10 topics into 5 broader orientations |
| `tables/table2_methods_summary.csv` / `.md` | **Table 2.** Compact corpus/analytical-design summary (now reporting the resolved 2,198 / 2,120 corpus counts) |

### Derived/working outputs (`derived/`)
`topic_period_shares_long.csv`, `topic_share_change_table.csv` (full
version with both long/short labels and raw shares),
`lexical_period_corpus_stats.csv`, `lexical_distinctive_terms_full.csv`
(all 6,322 terms passing min_df=5, not just the top 15/side),
`lexical_sensitivity_mindf.csv`, `lexical_sensitivity_top15_overlap.csv`,
`lexical_tech_term_audit.csv` / `_summary.csv`,
`lexical_example_documents.csv`, `lexical_prior_sensitivity.csv`
(per-term z-score at 4 alpha0 settings), `lexical_literal_bigram_check.csv`.

### Notes
`notes/robustness_checks.md` — full detail on the sensitivity check,
tech-name-domination check, and document-inspection check requested for
the lexical analysis, plus a documented-limitations section (updated with
pointers to the four notes below).
`notes/corpus_count_reconciliation.md` — resolution of the 2,224 / 2,227 /
2,198 / 2,120 corpus-count discrepancy.
`notes/topic_share_uncertainty.md` — bootstrap methodology and per-topic
95% CIs for the topic-share change.
`notes/lexical_prior_sensitivity.md` — prior-strength sensitivity write-up
and robustness verdict.
`notes/literal_bigram_check.md` — literal contiguous bigram construction
and comparison against the main analysis's bigrams.
`notes/figure2_concept_grouping.md` — the exact, predetermined concept-family
groupings and document-prevalence rule behind Figure 2, with a full audit
trail of every merged and every skipped term.

---

## 4. Main empirical findings (see `analysis_summary.md` for the
paper-ready version, with empirical findings kept explicitly separate from
interpretive claims)

- Topic prevalence shifted substantially between pre-2023 and 2024-2025:
  the three largest **declines** are School curriculum/STEM (-21.9 pp),
  Educator readiness/ethics (-12.3 pp), and Children/creative learning
  (-10.2 pp); the largest **increases** are GenAI instructional integration
  (+12.3 pp), ChatGPT/writing/prompting (+10.1 pp), and Higher-ed
  integration/ethics (+10.1 pp).
- The strongest pre-2023-distinctive lexical terms are conceptual/curricular
  and child/K-12-oriented ("machine", "children", "curriculum", "learn",
  "smart", "book", "e-learning", "game", "conversational"); the strongest
  2024-2025-distinctive terms combine generative-AI tool names with
  institutional/evaluative vocabulary ("generative", "genai", "chatgpt",
  "academic", "integration", "critical thinking", "acceptance").
- Only 3 of the top 15 2024-2025 terms are bare technology names,
  indicating the shift is not merely "ChatGPT gets mentioned more" but
  also reflects a shift in surrounding institutional and evaluative
  vocabulary (see `notes/robustness_checks.md`, Section 2).
- 8 of 10 topics' pre-2023 → 2024-2025 changes have a 95% bootstrap CI
  that excludes zero (`notes/topic_share_uncertainty.md`); the prior
  choice for the lexical z-score is robust to alpha0 ranging from 0.25x to
  2.00x the current specification (`notes/lexical_prior_sensitivity.md`).

## 5. Limitations and interpretive cautions

- The lexical-distinctiveness Dirichlet prior is estimated from the same
  two periods being compared (no independent background corpus was
  available in this repository); see the prior-specification note above
  and in `notes/robustness_checks.md`. This choice was stress-tested for
  prior *strength* (Section 2.2a) and found robust; its *shape* (drawn
  from the combined two-period corpus rather than an external reference)
  was not separately varied.
- Pre-2023 is a comparatively small sample (106 documents) relative to
  2024-2025 (1,867 documents); the z-score statistic accounts for this,
  but readers should weight pre-2023 findings accordingly. This is now
  quantified for the topic-share comparison via the bootstrap CIs
  (Section 2.1); 2 of 10 topics' CIs include zero.
- The ±2 pp "relatively stable" band in Figure 1, and the conceptual
  mapping in Table 1, are both researcher judgment calls, documented as
  such rather than presented as statistically derived. **This band is a
  descriptive visualization threshold, not a significance test** — treat
  the bootstrap CIs, not the band, as the significance-relevant quantity.
- Table 1's orientation mapping is an interpretive synthesis for framing
  the paper's argument, not a re-clustering of the topic model; several
  topics legitimately span more than one orientation, which the table
  notes rather than forcing an artificial single assignment.
- Figure 2's 9-concept-per-side display rule (Section 2.2b) rests on a
  predetermined but still subjective judgment about which terms are
  obvious lexical variants of one concept (e.g. "generative"/"genai"/"gai"
  → Generative AI) and which representative term/statistic to plot for
  each group; a different, equally defensible judgment could group a few
  borderline terms differently. The full unfiltered ranked results are
  unaffected and remain available for independent inspection in
  `derived/lexical_distinctive_terms_full.csv` and
  `tables/table_lexical_distinctive_terms.csv`.
- The topic-share bootstrap (Section 2.1) quantifies sampling uncertainty
  given the existing document-to-topic assignments; it does not capture
  uncertainty from the topic model itself (the fixed k=10 KMeans-guided
  BERTopic solution is treated as given).

---

## 6. Reproducing the outputs

Environment: this repository's existing `ai-literacy` conda environment
(see `../environment.yml` / `../requirements.txt`). No new packages are
required beyond what that environment already provides (pandas, numpy,
matplotlib, scikit-learn).

From the repository root:

```bash
# Activate the existing project environment (adjust path if needed)
conda activate ai-literacy
# or, if conda activate is not on PATH in your shell:
# /opt/anaconda3/envs/ai-literacy/bin/python3 <script>

cd iconference

python scripts/01_topic_prevalence_change.py     # Topic-share point-estimate tables
python scripts/02_lexical_distinctiveness.py     # Lexical ranked results + robustness CSVs
python scripts/03_methods_summary_table.py       # Table 2
python scripts/04_topic_share_bootstrap.py       # Figure 1 (with bootstrap CIs) + bootstrap table
python scripts/05_lexical_prior_sensitivity.py   # Prior-strength sensitivity
python scripts/06_lexical_docprevalence.py       # Document-prevalence supplement
python scripts/07_literal_bigram_check.py        # Literal contiguous bigram check
python scripts/08_figure2.py                     # Figure 2 (concept-grouped display)
```

Scripts 04-08 depend on outputs from 01 and 02 (they read
`derived/topic_share_change_table.csv` and
`derived/lexical_distinctive_terms_full.csv` respectively, and 05/07
import 02 directly as a module), so run 01 and 02 first if starting from
a clean checkout. All eight scripts have been verified to run end-to-end
in this order without manual intervention. Figure 1 is produced only by
script 04 (script 01 only builds the point-estimate tables it reads), and
Figure 2 is produced only by script 08 (script 02 only builds the ranked
results table it reads) — this avoids two scripts writing competing
versions of the same figure file.

Table 1 (`tables/table1_conceptual_synthesis.csv` / `.md`) is
hand-authored (a researcher interpretation, per the task's own
instruction) and is not regenerated by a script.

Random seeds: `01`-`03` and `06`-`08` are fully deterministic (no
randomness beyond fixed closed-form statistics and CountVectorizer term
ordering). `04_topic_share_bootstrap.py` uses `RANDOM_SEED = 42`
(`np.random.default_rng(42)`) for its 10,000-replicate bootstrap — set
explicitly so the exact CIs reported here are reproducible.
`02_lexical_distinctiveness.py` sets `RANDOM_SEED = 42` for completeness
even though nothing in it currently consumes it.
