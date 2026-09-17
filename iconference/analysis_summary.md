# Analysis summary (paper-ready)

Research question: How has the rise of generative AI reconfigured the
thematic and conceptual orientation of AI literacy research?

Argument: AI literacy is shifting from primarily *understanding* AI toward
*interacting with*, *evaluating*, and exercising *judgment* with AI in
situated contexts.

This document separates **empirical findings** (directly computed from the
data, in `derived/` and `tables/`) from **interpretive claims** (the
researcher's conceptual synthesis, marked as such).

**Corpus note:** this analysis uses the same corpus and counts already
reported in the existing manuscript (`Current_AI_Literacy_Paper.pdf`):
an assembled corpus of 2,198 Scopus-indexed documents (2016-2025), of
which 2,120 have title+abstract text and were used for the 10-topic
KMeans-guided BERTopic model that every analysis below draws on. A
separate reconciliation of why other counts (2,224 in `scopus_query.txt`;
2,227 raw CSV rows) also appear in this repository, and why they are not
in conflict, is in `notes/corpus_count_reconciliation.md`.

---

## A. Empirical findings

### A.1 Topic-prevalence shift, pre-2023 → 2024-2025
(Source: `tables/table_topic_share_change.csv`, Figure 1; n=106 pre-2023,
n=1,867 2024-2025 documents; shares are within-period, not raw counts.
Uncertainty: `tables/table_topic_share_change_bootstrap.csv`,
`notes/topic_share_uncertainty.md` — 10,000-replicate document-level
bootstrap, resampled independently within each period; Figure 1 itself
already includes these 95% CIs as error bars.)

**Increased (≥ +2 pp observed; all 5 topics' 95% bootstrap CIs exclude zero):**
| Topic | Pre-2023 % | 2024-2025 % | Change (pp) | 95% CI (pp) |
|---|---|---|---|---|
| GenAI instructional integration (T3) | 0.0 | 12.3 | +12.3 | [+10.9, +13.9] |
| ChatGPT / writing / prompting (T5) | 0.9 | 11.1 | +10.1 | [+7.5, +12.2] |
| Higher-ed integration / ethics (T2) | 2.8 | 13.0 | +10.1 | [+6.3, +13.3] |
| Self-efficacy / adoption (T1) | 4.7 | 13.3 | +8.6 | [+4.0, +12.6] |
| Libraries / information services (T9) | 0.0 | 4.6 | +4.6 | [+3.7, +5.6] |

**Relatively stable (±2 pp observed; both topics' 95% CIs include zero):**
| Topic | Pre-2023 % | 2024-2025 % | Change (pp) | 95% CI (pp) |
|---|---|---|---|---|
| Healthcare / medical AI literacy (T8) | 8.5 | 8.3 | -0.2 | [-5.9, +5.0] |
| Human decision-making / trust (T6) | 10.4 | 9.1 | -1.3 | [-7.7, +4.2] |

**Decreased (≤ -2 pp observed; all 3 topics' 95% CIs exclude zero):**
| Topic | Pre-2023 % | 2024-2025 % | Change (pp) | 95% CI (pp) |
|---|---|---|---|---|
| Children / creative learning (T7) | 17.9 | 7.8 | -10.2 | [-17.8, -3.1] |
| Educator readiness / ethics (T0) | 23.6 | 11.3 | -12.3 | [-20.8, -4.4] |
| School curriculum / STEM (T4) | 31.1 | 9.3 | -21.9 | [-30.8, -13.2] |

8 of the 10 topics' bootstrap 95% CIs exclude zero — only the two
"relatively stable" topics (Healthcare; Human decision-making/trust) have
CIs straddling zero, which is consistent with (though does not by itself
prove) the ±2 pp descriptive band used to color-code Figure 1.
**That ±2 pp band remains a descriptive visualization threshold chosen for
readability, not a significance test** — the bootstrap CIs above are the
significance-relevant quantity, and the two should not be conflated (see
`notes/topic_share_uncertainty.md`).

Pre-2023, the corpus was dominated by three topics — School curriculum/STEM,
Educator readiness/ethics, and Children/creative learning — which together
accounted for 72.6% of pre-2023 documents but only 28.3% of 2024-2025
documents. Two topical concentrations (GenAI instructional integration;
Libraries/information services) are essentially absent in the pre-2023
corpus (0.0% each) and reach 12.3% and 4.6% of 2024-2025 documents,
respectively.

### A.2 Lexical distinctiveness, pre-2023 vs. 2024-2025 (2023 excluded)
(Source: `tables/table_lexical_distinctive_terms.csv`, Figure 2; method:
weighted log-odds-ratio with informative Dirichlet prior, Monroe et al.
2008; min_df=5 across n=1,973 combined documents.)

**Top terms distinctive of pre-2023** (z-score, count pre-2023 / 2024-2025):
machine (z=9.84, 56/177), children (z=7.86, 64/343), curriculum (z=7.30,
60/341), learn (z=6.99, 29/94), smart (z=6.63, 19/41), book (z=6.53,
16/27), health care (z=6.46, 20/50), people (z=6.42, 29/112), e-learning
(z=5.91, 10/6), design (z=5.69, 88/819), intention learn (z=5.52, 9/3),
concepts (z=5.49, 39/243), citizens (z=5.30, 15/43), game (z=5.27, 18/64),
conversational (z=5.25, 15/44).

**Top terms distinctive of 2024-2025** (z-score, count pre-2023 /
2024-2025): generative (z=-5.01, 5/1,307), genai (z=-4.71, 0/980), chatgpt
(z=-4.28, 0/808), academic (z=-4.22, 0/787), integration (z=-3.94, 4/848),
higher (z=-3.11, 10/803), language (z=-2.94, 5/577), engagement (z=-2.79,
4/502), critical (z=-2.61, 23/1,071), critical thinking (z=-2.60, 1/341),
nursing (z=-2.58, 0/295), tools (z=-2.56, 30/1,260), integrating (z=-2.46,
3/387), thinking (z=-2.31, 9/559), acceptance (z=-2.29, 0/232).

**Robustness (full detail in `notes/robustness_checks.md`):** the
2024-2025 top-15 list is identical across min_df in {3, 5, 10}; the
pre-2023 top-15 list is 87-93% stable across the same range. Only 3 of the
15 top 2024-2025 terms are bare technology names (generative, genai,
chatgpt); the remainder describe institutional, pedagogical, or evaluative
concerns. Spot-checked source documents for the top 5 terms per side
confirm in-context, non-spurious usage.

**Additional robustness checks (this refinement pass):**
- **Prior-strength sensitivity** (`notes/lexical_prior_sensitivity.md`,
  `tables/table_lexical_prior_sensitivity_summary.csv`): re-running the
  identical analysis at 0.25x, 0.50x, 1.00x (current), and 2.00x the
  Dirichlet prior's total mass changes almost nothing — Spearman rank
  correlation with the current specification is ≥0.999 at every setting,
  and top-15 overlap per side is ≥93%. The prior-mass choice is robust.
- **Document prevalence** (`tables/table_lexical_distinctive_terms_with_docprevalence.csv`):
  most top-ranked terms are genuinely widespread, not driven by a handful
  of documents — e.g. "machine" appears in 25.5% of pre-2023 documents,
  "generative" in 31.8% of 2024-2025 documents. A few pre-2023 terms are
  driven by a small number of documents (e.g. "book," "e-learning,"
  "intention learn" each appear in ≤3 of the 106 pre-2023 documents; "game"
  and "conversational" in 4 each) — these should be read as suggestive
  rather than as evidence of a widespread pre-2023 pattern.
- **Literal contiguous bigram check** (`notes/literal_bigram_check.md`,
  `derived/lexical_literal_bigram_check.csv`): re-deriving bigrams as
  literal adjacent word pairs (rather than the main method's
  filter-then-ngram construction, which can create "false adjacency")
  confirms all three bigrams in the top-30 table ("health care,"
  "intention learn," "critical thinking") are genuine literal adjacent
  phrases, ranking 1st or 2nd by this alternative construction on their
  respective side. This check also surfaces additional interpretable
  multiword expressions not in the main top-15 (e.g. "conversational
  agents," "computer science," "ethical issues," "digital
  entrepreneurship" on the pre-2023 side; "ethical considerations,"
  "academic integrity," "prompt engineering," "genai tools" on the
  2024-2025 side), which are substantively consistent with the main
  analysis's story.

**Figure 2** (`figures/fig2_lexical_distinctiveness.png`/`.pdf`,
`notes/figure2_concept_grouping.md`): a 9-concepts-per-side version built
by a predetermined, mechanical display rule — (a) collapse obvious lexical
variants of the same underlying concept into a single bar (e.g.
"generative"/"genai"/"gai"/"generative genai" → **Generative AI**;
"critical"/"critical thinking" → **Critical thinking**, preferring the
literal-bigram-confirmed phrase), and (b) deprioritize terms resting on
fewer than 5 supporting documents in favor of more widespread,
still-top-ranked terms — rather than manual curation, and without skipping
any term merely for sounding generic. Displayed pre-2023 concepts:
Machine, Children, Curriculum, Learn, Smart, People, Design, Concepts,
Citizens. Displayed 2024-2025 concepts: Generative AI, ChatGPT, Academic,
Integration, Higher, Language, Engagement, Critical thinking, Nursing. The
full unfiltered ranked results and every grouping/omission decision (which
terms were merged, which representative term's z-score is plotted, which
low-prevalence terms were omitted and why) remain fully documented in
`notes/figure2_concept_grouping.md` and inspectable in
`derived/lexical_distinctive_terms_full.csv` /
`tables/table_lexical_distinctive_terms.csv`.

---

## B. Interpretive claims (researcher synthesis, not statistically derived)

### B.1 Conceptual mapping of the 10 topics (full table: `tables/table1_conceptual_synthesis.csv`)

- **Understanding AI**: T4 (School curriculum/STEM), T7 (Children/creative
  learning), T0 (Educator readiness/ethics, also touches Evaluating).
- **Interacting with AI**: T5 (ChatGPT/writing/prompting), T3 (GenAI
  instructional integration, also touches Adopting).
- **Adopting AI**: T1 (Self-efficacy/adoption), T3 (also touches
  Interacting).
- **Evaluating/judging AI**: T6 (Human decision-making/trust), T2
  (Higher-ed integration/ethics, also touches Situated), T0 (also touches
  Understanding).
- **Situated/professional/institutional AI literacy**: T8 (Healthcare), T9
  (Libraries), T2 (also touches Evaluating).

### B.2 Interpretive synthesis linking A.1 and B.1

The topics that declined most sharply pre-2023 → 2024-2025 (T4, T0, T7) map
predominantly onto **Understanding AI** — conceptual/curricular, K-12/child
oriented literacy. The topics that increased most (T3, T5, T2, T1, T9) map
onto **Interacting with AI**, **Adopting AI**, **Evaluating/judging AI**,
and **Situated/institutional AI literacy** — hands-on generative-AI tool
use, adoption psychology, higher-education/ethics integration, and
domain-specific (library) professional practice. The lexical analysis
(A.2) is consistent with this same reading: pre-2023 vocabulary is
conceptual and K-12-oriented ("children", "curriculum", "learn", "smart",
"game"), while 2024-2025 vocabulary combines generative-tool naming with
institutional and evaluative language ("integration", "academic",
"critical thinking", "acceptance").

**This convergence between the complementary topic-prevalence shift (A.1)
and lexical-distinctiveness result (A.2) — complementary in that they use
different procedures (topic assignment vs. lexical statistics) applied to
the same underlying corpus, not independent samples — read through the
researcher-authored conceptual mapping (B.1), is
the paper's central interpretive claim: it is offered as a plausible,
well-evidenced reading of the data, not as a statistical proof that AI
literacy research has "moved from understanding to interacting/adopting/
evaluating/situating."** Alternative readings — e.g., that the shift
partly reflects a generic scaling-up of the field's total output (2024-2025
has ~17.6x as many documents as pre-2023) rather than a pure compositional
reorientation — are not ruled out by within-period share normalization
alone, though normalizing to within-period shares (rather than raw counts)
is precisely the step taken to separate compositional change from sheer
volume growth.

### B.3 Caveats on the interpretive claims

- Table 1's category assignments are the authors' judgment, cross-checked
  against topic labels, representative terms, and temporal behavior, but
  not independently validated by a second coder or statistical test.
- Three of ten topics (T0, T2, T3) are explicitly multi-orientation; this
  is treated as evidence of genuine conceptual overlap in the literature,
  not a flaw to be resolved.
- The pre-2023 period's small sample size (n=106) means both its topic
  shares and its lexical-distinctiveness estimates are less stable than
  the 2024-2025 estimates. For the topic-share comparison, this is now
  quantified directly via the bootstrap CIs in A.1 rather than only noted
  qualitatively; the lexical analysis's prior-strength sensitivity check
  (A.2) shows the ranking is not an artifact of a specific smoothing
  choice, but does not by itself quantify sampling uncertainty in the same
  way the topic-share bootstrap does.
- The topic-share bootstrap resamples documents within each already-fixed
  period and topic assignment; it does not capture uncertainty from the
  topic model itself (the fixed k=10 KMeans-guided BERTopic solution is
  treated as given, not re-estimated).
