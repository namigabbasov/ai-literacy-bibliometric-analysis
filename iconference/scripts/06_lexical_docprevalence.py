"""
iConference paper — methodological refinement: document-prevalence
supplement for the top-ranked lexical-distinctiveness table.

Token-count log-odds (the main statistic in 02_lexical_distinctiveness.py)
can in principle be driven by a term repeated many times within a small
number of documents rather than by the term being widespread across many
documents. This script adds a document-prevalence view -- for every term
in the current top-ranked table (tables/table_lexical_distinctive_terms.csv,
i.e. the top 15 terms per period), it reports how many/what percentage of
each period's documents actually contain the term at least once.

This is a robustness/interpretability supplement, not a replacement
statistic: ranking is still by the weighted log-odds z-score from the main
analysis; nothing here is re-ranked by document prevalence.

Output:
  tables/table_lexical_distinctive_terms_with_docprevalence.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ICONF = ROOT / "iconference"
TABLES = ICONF / "tables"
DERIVED = ICONF / "derived"

MAIN_TABLE_PATH = TABLES / "table_lexical_distinctive_terms.csv"
CORPUS_STATS_PATH = DERIVED / "lexical_period_corpus_stats.csv"


def main():
    main_table = pd.read_csv(MAIN_TABLE_PATH)
    corpus_stats = pd.read_csv(CORPUS_STATS_PATH).set_index("period")

    n_pre = int(corpus_stats.loc["Pre-2023", "n_documents"])
    n_post = int(corpus_stats.loc["2024-2025", "n_documents"])

    out = main_table.copy()
    out["n_pre_2023_docs"] = n_pre
    out["n_2024_2025_docs"] = n_post
    out["pct_pre_2023_docs_containing_term"] = (
        out["doc_freq_pre_2023"] / n_pre * 100
    ).round(1)
    out["pct_2024_2025_docs_containing_term"] = (
        out["doc_freq_2024_2025"] / n_post * 100
    ).round(1)

    ordered_cols = [
        "term", "associated_period", "z_score", "log_odds_delta", "n_grams",
        "count_pre_2023", "count_2024_2025",
        "doc_freq_pre_2023", "n_pre_2023_docs", "pct_pre_2023_docs_containing_term",
        "doc_freq_2024_2025", "n_2024_2025_docs", "pct_2024_2025_docs_containing_term",
    ]
    out = out[ordered_cols]

    out_path = TABLES / "table_lexical_distinctive_terms_with_docprevalence.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))

    # Quick sanity flag: terms whose document-level prevalence looks weak
    # relative to their token-count z-score (e.g., concentrated in very few
    # documents), for manual inspection -- not an automatic exclusion.
    low_prevalence = out[
        ((out["associated_period"] == "Pre-2023") & (out["pct_pre_2023_docs_containing_term"] < 5))
        | ((out["associated_period"] == "2024-2025") & (out["pct_2024_2025_docs_containing_term"] < 5))
    ]
    print(f"\n{len(low_prevalence)} of {len(out)} top-ranked terms appear in <5% of "
          "their associated period's documents (worth a manual look, not "
          "automatically excluded):")
    if len(low_prevalence) > 0:
        print(low_prevalence[["term", "associated_period", "doc_freq_pre_2023", "doc_freq_2024_2025"]].to_string(index=False))

    print("\nSaved:")
    print(" -", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
