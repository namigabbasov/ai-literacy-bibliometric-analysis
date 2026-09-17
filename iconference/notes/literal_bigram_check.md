# Literal contiguous bigram check

1,538 literal contiguous bigrams (word pairs truly adjacent in the original topic_text, with stopword-heavy pairs removed after bigram construction) pass min_df=5, using the same weighted log-odds statistic, prior specification, and period definitions as the main analysis.

## Top literal contiguous bigrams, pre-2023

| Bigram | z-score | count pre / post |
|---|---|---|
| health care | 7.30 | 20 / 50 |
| intention learn | 5.85 | 9 / 3 |
| conversational agents | 5.40 | 8 / 10 |
| computer science | 5.07 | 14 / 56 |
| ethical issues | 4.93 | 13 / 51 |
| design considerations | 4.91 | 6 / 4 |
| guide future | 4.46 | 5 / 4 |
| public understanding | 4.38 | 5 / 5 |
| design process | 4.34 | 6 / 11 |
| years old | 4.00 | 4 / 3 |
| digital entrepreneurship | 3.87 | 5 / 10 |
| social good | 3.87 | 5 / 10 |
| augmented reality | 3.69 | 4 / 6 |
| basic concepts | 3.67 | 5 / 12 |
| everyday lives | 3.49 | 5 / 14 |

## Top literal contiguous bigrams, 2024-2025

| Bigram | z-score | count pre / post |
|---|---|---|
| critical thinking | -2.29 | 1 / 341 |
| ethical considerations | -1.61 | 0 / 145 |
| large language | -1.55 | 0 / 135 |
| genai tools | -1.42 | 0 / 113 |
| academic integrity | -1.40 | 0 / 111 |
| academic writing | -1.36 | 0 / 104 |
| systematic review | -1.22 | 0 / 84 |
| generative tools | -1.20 | 0 / 81 |
| prompt engineering | -1.18 | 0 / 79 |
| generative genai | -1.06 | 0 / 63 |
| technology acceptance | -1.02 | 0 / 59 |
| ethical awareness | -1.01 | 0 / 58 |
| practical implications | -0.99 | 0 / 56 |
| academic libraries | -0.99 | 0 / 55 |
| foreign language | -0.96 | 0 / 52 |

## Comparison with the main analysis's bigrams

| term | main_analysis_z | found_as_literal_contiguous_bigram | literal_bigram_z | literal_bigram_rank |
|---|---|---|---|---|
| health care | 6.46 | True | 7.3 | 1 |
| intention learn | 5.52 | True | 5.85 | 2 |
| critical thinking | -2.6 | True | -2.29 | 1 |

## Interpretation
- The strongest literal contiguous bigrams on each side are substantively consistent with the main (filter-then-ngram) analysis's unigram-level story: pre-2023 multiword expressions cluster around conceptual/curricular framing, 2024-2025 multiword expressions cluster around generative-AI tool use, institutional integration, and critical evaluation.
- Where a main-analysis bigram is NOT found among the literal contiguous bigrams (see comparison table), that specific two-word combination did not occur as literally adjacent text at min_df=5 -- i.e., the main analysis's filter-then-ngram construction produced it from two words that were not next to each other in the original sentence often enough to also pass min_df=5 as a literal pair. This confirms the false-adjacency caveat is a real, checkable phenomenon for specific phrases, but it does not change the paper's overall lexical-distinctiveness narrative, since the unigram-level evidence (the bulk of both top-15 lists) is unaffected by bigram construction at all.
- Recommendation: report bigrams in the paper only when they also survive this literal-contiguous check (or note explicitly when a bigram is filter-then-ngram only).
