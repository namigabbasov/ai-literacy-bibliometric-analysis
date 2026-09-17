# Figure 2 concept grouping

**This is a presentation-only display decision, not a re-analysis.** No z-score, rank, count, preprocessing step, prior specification, or robustness result was changed. The full, unmodified ranked results remain at `derived/lexical_distinctive_terms_full.csv` and `tables/table_lexical_distinctive_terms.csv`; nothing there was altered by this script. This note documents exactly which display decisions were made and why.

## Why this figure is built this way

Simply showing the top-ranked terms per side devotes several 2024-2025 display slots to near-duplicate technology labels for the same underlying concept (generative, genai, gai, generative genai), crowding out the institutional / evaluative / professional vocabulary that is central to the paper's argument. This figure fixes that by (a) collapsing obvious lexical variants of one concept into a single bar, and (b) deprioritizing terms resting on very few supporting documents in favor of more widespread, still-top-ranked terms -- without skipping any term merely for sounding generic; 'academic', 'higher', 'tools'-type terms are kept whenever they are reached by rank.

## Rule (applied identically to both sides)


1. Start from the same ranked results used everywhere else in this project (`derived/lexical_distinctive_terms_full.csv`, min_df=5, unchanged).
2. Walk down the ranking, most-distinctive term first.
3. If the term belongs to a predetermined concept family (below), register that family once, using its designated representative term's statistic -- the highest-ranked family member by default, or a literal-bigram-confirmed multiword expression where the task explicitly calls for that preference. Skip other family members when later encountered (already represented).
4. If the representative term's document count in its own period is below 5 (this project's existing min_df=5 convention, reused here as a display-prevalence rule rather than a new threshold), skip the whole concept for the main display -- it remains fully visible in the underlying tables.
5. Continue until 9 concepts are kept per side.

No concept was added out of rank order, and no concept was dropped for supporting or complicating the paper's argument -- only for matching a family-membership rule or the document-count rule above, both fixed before the displayed term lists were assembled.

## Predetermined concept families

| Family | Members | Representative term (plotted) | Why |
|---|---|---|---|
| Generative AI | generative, genai, gai, generative genai | generative | Four surface forms (full word, standard abbreviation, alternate abbreviation, adjective+abbreviation bigram) of the same referent, generative AI. |
| Integration | integration, integrating | integration | Same lemma, noun vs. gerund form. |
| Critical thinking | critical, critical thinking | critical thinking | Per task instruction: prefer the literal-bigram-confirmed multiword expression ('critical thinking' ranks 1st among literal contiguous bigrams on the 2024-2025 side; see notes/literal_bigram_check.md) over the bare adjective, even though 'critical' alone ranks marginally higher (z=-2.614 vs. -2.601). |
| LLMs | llms, llm | llms | Plural/singular of the same abbreviation. |
| Libraries | libraries, library, librarians | libraries | Plural noun, singular noun, and agent-noun of the same institutional referent. |
| Nursing | nursing, nurse, nurses | nursing | Same professional-domain referent; 'nursing' is already the highest-ranked and most prevalent form. |
| Concepts | concepts, concept | concepts | Plural/singular of the same lemma. |
| Conversational agents | conversational, conversational agents | conversational agents | Analogous to critical/critical thinking: the bare adjective 'conversational' is, in this corpus, overwhelmingly used to modify 'agents'; 'conversational agents' is confirmed as a literal contiguous bigram (rank 3 on the pre-2023 side; see notes/literal_bigram_check.md). |
| Citizens | citizens, citizen | citizens | Plural/singular of the same lemma (citizen appears far down the ranking, rank 238, and does not affect display selection either way). |
| Game | game, games | game | Singular/plural of the same lemma (games appears far down the ranking, rank 287, and does not affect display selection either way). |

**Note on 'higher education':** no `higher education` bigram passed min_df=5 (it does not appear in the ranked results at all). This is because "education" is itself one of the parent project's domain stopwords (removed before bigram construction; see `DOMAIN_STOP_WORDS_FROM_PARENT` in `02_lexical_distinctiveness.py`), so "higher" + "education" can never form as a bigram in this vocabulary. The displayed label is therefore the bare ranked unigram "higher" (Title-cased to "Higher"), not an invented "higher education" bigram.

## Displayed concepts

### Pre-2023 (9 concepts)

| Display label | Plotted term | z-score | Doc. count (%) | Original rank(s) |
|---|---|---|---|---|
| Machine | machine | 9.839 | 27 (25.5%) | 1 |
| Children | children | 7.855 | 22 (20.8%) | 2 |
| Curriculum | curriculum | 7.301 | 26 (24.5%) | 3 |
| Learn | learn | 6.994 | 17 (16.0%) | 4 |
| Smart | smart | 6.629 | 7 (6.6%) | 5 |
| People | people | 6.419 | 14 (13.2%) | 8 |
| Design | design | 5.687 | 42 (39.6%) | 10 |
| Concepts (from: concepts, concept) | concepts | 5.494 | 25 (23.6%) | 12 |
| Citizens (from: citizens, citizen) | citizens | 5.303 | 12 (11.3%) | 13 |

### 2024-2025 (9 concepts)

| Display label | Plotted term | z-score | Doc. count (%) | Original rank(s) |
|---|---|---|---|---|
| Generative AI (from: generative, genai, gai, generative genai) | generative | -5.006 | 594 (31.8%) | 1 |
| ChatGPT | chatgpt | -4.277 | 287 (15.4%) | 3 |
| Academic | academic | -4.221 | 372 (19.9%) | 4 |
| Integration (from: integration, integrating) | integration | -3.941 | 586 (31.4%) | 5 |
| Higher | higher | -3.114 | 419 (22.4%) | 6 |
| Language | language | -2.938 | 294 (15.7%) | 7 |
| Engagement | engagement | -2.791 | 331 (17.7%) | 8 |
| Critical thinking (from: critical, critical thinking) | critical thinking | -2.601 | 206 (11.0%) | 10 |
| Nursing (from: nursing, nurse, nurses) | nursing | -2.580 | 55 (2.9%) | 11 |

## Low-document-prevalence terms omitted from the main display (< 5 supporting documents in their period)

### Pre-2023 side

| Display label (if shown) | Plotted term | z-score | Doc. count (%) | Rank |
|---|---|---|---|---|
| Book | book | 6.533 | 3 (2.8%) | 6 |
| Health care | health care | 6.462 | 3 (2.8%) | 7 |
| E-learning | e-learning | 5.914 | 1 (0.9%) | 9 |
| Intention learn | intention learn | 5.516 | 3 (2.8%) | 11 |

### 2024-2025 side

(none met the low-prevalence skip condition within the display window)

These terms remain fully visible, ranked, and unmodified in `derived/lexical_distinctive_terms_full.csv` and `tables/table_lexical_distinctive_terms.csv` / `tables/table_lexical_distinctive_terms_with_docprevalence.csv`. They are omitted only from this main-text display figure because each rests on very few actual documents relative to what a higher-prevalence, still-top-ranked alternative offers -- exactly the concern the task raised about not letting rare terms dominate the figure.

## Why "nursing" (2.9% of 2024-2025 documents) is kept while pre-2023 terms at similar or higher percentages are not

The document-count rule above uses an **absolute** document count (< 5 documents), not a period-relative percentage. "nursing" rests on 55 real 2024-2025 documents -- comparable in absolute terms to "machine" or "curriculum" on the pre-2023 side -- and only *looks* rare (2.9%) because the 2024-2025 period itself is ~17.6x larger than pre-2023 (1,867 vs. 106 documents). By contrast, the pre-2023 terms omitted above ("book," "health care," "e-learning," "intention learn," "game," "conversational"/"conversational agents") rest on only 1-4 actual documents each -- genuinely thin evidence, not an artifact of period size. Applying a single relative-percentage threshold across two periods of such different size would misclassify "nursing" as rare when it is not; the absolute-count rule avoids that.

## Confirmation

- The underlying weighted log-odds z-scores, ranks, counts, document frequencies, preprocessing, vocabulary (min_df=5), prior specification, and every prior robustness result (`notes/robustness_checks.md`, `notes/lexical_prior_sensitivity.md`, `notes/literal_bigram_check.md`) are **unchanged** by this script.
- `derived/lexical_distinctive_terms_full.csv` and `tables/table_lexical_distinctive_terms.csv` are untouched.
- This script only reads those files and decides how to group and label already-existing ranked terms for the one Figure 2 used in the paper (`figures/fig2_lexical_distinctiveness.png` / `.pdf`). Earlier draft variants of this figure (an unmerged top-15 version and a generic-word-filtered top-9 version) were superseded by this concept-grouped version and are no longer part of the output set.

