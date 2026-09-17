# Prior-strength sensitivity, weighted log-odds lexical analysis

Same preprocessing, vocabulary, and min_df=5 as the main lexical analysis (6,322 shared terms). Only the Dirichlet prior's total mass (alpha0) is varied, as a multiple of the smaller period's (pre-2023) token count (alpha0_base = 9114). The current specification (`02_lexical_distinctiveness.py`) uses the 1.00x setting.

## Results

| alpha0 multiplier | alpha0 | Top-15 pre-2023 overlap vs. 1.00x | Top-15 2024-2025 overlap vs. 1.00x | Spearman rho vs. 1.00x |
|---|---|---|---|---|
| 0.25x | 2278 | 0.93 | 0.93 | 0.9993 |
| 0.50x | 4557 | 1.00 | 0.93 | 0.9998 |
| 1.00x | 9114 | 1.00 | 1.00 | 1.0000 |
| 2.00x | 18228 | 1.00 | 1.00 | 0.9999 |

## Verdict

**The current alpha0 specification (1.00x the smaller period's token count) is robust.** Across the tested range (0.25x-2.00x), the ranked z-scores correlate near-perfectly with the current specification (Spearman rho >= 1.00), and the top-15 term lists per period overlap by at least 93% at every setting. No alternative setting changes the substantive interpretation (pre-2023 conceptual/curricular vocabulary vs. 2024-2025 generative-tool + institutional/evaluative vocabulary).

## Notes
- 2.00x alpha0_base (~2x the pre-2023 period's own token count) was retained rather than dropped as "clearly over-regularized": it remains far smaller than the 2024-2025 period's token count (~22x pre-2023's), so it still cannot dominate the larger period's data, and the results above show it does not materially change the ranking.
- Full per-term z-scores at all four settings are in `derived/lexical_prior_sensitivity.csv` (long format: one row per term x alpha0 setting) for independent verification.
- This check varies only alpha0 (prior mass); it does not re-test the prior's *shape* (which is fixed as the combined-corpus relative word frequency in every setting here, per the main analysis's documented choice).
