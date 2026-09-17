# Bootstrap uncertainty for topic-share change (pre-2023 vs. 2024-2025)

Document-level bootstrap, 10,000 replicates, resampling with replacement independently within each period (pre-2023: n=106; 2024-2025: n=1867), random seed 42.

**The existing ±2.0 percentage-point "relatively stable" band used to color-code Figure 1 is a descriptive visualization threshold chosen for readability (see 01_topic_prevalence_change.py), not a statistical significance test. The bootstrap CIs below are the significance-relevant quantity; the two should not be conflated.**

## Results (full table: tables/table_topic_share_change_bootstrap.csv)

| Topic | Label | Observed change (pp) | 95% CI (pp) | CI excludes 0? |
|---|---|---|---|---|
| 3 | GenAI instructional integration | +12.3 | [+10.9, +13.9] | Yes |
| 5 | ChatGPT / writing / prompting | +10.1 | [+7.5, +12.2] | Yes |
| 2 | Higher-ed integration / ethics | +10.1 | [+6.3, +13.2] | Yes |
| 1 | Self-efficacy / adoption | +8.6 | [+4.0, +12.6] | Yes |
| 9 | Libraries / information services | +4.6 | [+3.7, +5.6] | Yes |
| 8 | Healthcare / medical AI literacy | -0.2 | [-5.8, +5.0] | No |
| 6 | Human decision-making / trust | -1.3 | [-7.7, +4.2] | No |
| 7 | Children / creative learning | -10.2 | [-17.8, -3.1] | Yes |
| 0 | Educator readiness / ethics | -12.3 | [-20.8, -4.4] | Yes |
| 4 | School curriculum / STEM | -21.9 | [-30.8, -13.2] | Yes |

**8 of 10 topics have a 95% bootstrap CI that excludes zero** (i.e., the change is not attributable to sampling noise at the 95% level under this resampling scheme).

Topics whose CI does **not** exclude zero: Healthcare / medical AI literacy (-0.2 pp, 95% CI [-5.8, +5.0]), Human decision-making / trust (-1.3 pp, 95% CI [-7.7, +4.2]). These are the smallest-magnitude observed changes and should be described in the paper as directionally suggestive rather than confidently established.

## Interpretation notes
- The bootstrap resamples documents *within* each already-observed period; it quantifies sampling uncertainty in the topic-share estimate given the assignments this project already has, not uncertainty in the underlying topic assignments themselves (i.e., it does not re-run BERTopic/KMeans clustering, and does not capture uncertainty from the fixed k=10 topic solution or from document-to-topic assignment).
- Because pre-2023 has only 106 documents, its bootstrap resamples have more granular/discrete possible shares (multiples of 1/106) than 2024-2025's (multiples of 1/1,867), which is reflected in wider CIs for topics whose prevalence is concentrated in the pre-2023 period.
- This bootstrap does not change any point estimate already reported in Figure 1 / `tables/table_topic_share_change.csv`; `change_pp_observed` here is identical to that table's `Change (pp)` column.
