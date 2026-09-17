"""
iConference paper — Table 2: compact corpus and analytical-design summary.

Pulls factual counts directly from the parent project's raw corpus and
final RQ4 analytic corpus so the numbers in Table 2 cannot drift out of
sync with the actual data. The qualitative/methodological description
fields (source database, modeling approach, etc.) are entered directly
since they describe design choices already documented in the parent
project's RQ4_thematic_analysis.ipynb rather than being computed here.

Corpus-size reporting (updated after the count reconciliation in
notes/corpus_count_reconciliation.md): this script reproduces the exact
de-duplication/cleaning steps used in the parent project's
python_ai_lit_pipeline.ipynb (title dedup -> drop missing Title/Year ->
drop non-substantive Document Types + the one known corrupted-parsing
row) to derive the "assembled corpus" count (2,198), which matches
Current_AI_Literacy_Paper.pdf exactly. The raw CSV row count (2,227) is
NOT reported as "the corpus" because it includes 4 rows that are a known
single-record CSV parsing artifact (see the reconciliation note); it is
reported only as a labeled intermediate count.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_SCRIPTS = ROOT / "scripts"
ICONF = ROOT / "iconference"
TABLES = ICONF / "tables"

RAW_DATA_PATH = ROOT / "data" / "ai_literacy.csv"
ASSIGNMENTS_PATH = PARENT_SCRIPTS / "rq4_kmeans_guided_cleanlabels_topic_assignments.csv"

# Exact cleaning sequence reproduced from python_ai_lit_pipeline.ipynb
# (cells 8-9), used only to derive the "assembled corpus" count (2,198)
# reported in Current_AI_Literacy_Paper.pdf. See
# notes/corpus_count_reconciliation.md for the full derivation and the
# rationale for reproducing it here rather than hardcoding 2,198.
_KEEP_COLS = [
    "Authors", "Title", "Year", "Source title", "Author Keywords", "Index Keywords",
    "Abstract", "Cited by", "Affiliations", "DOI", "Document Type", "Conference name",
    "Publisher", "Language of Original Document", "Open Access", "EID",
]
_DROP_DOC_TYPES = ["Erratum", "Letter", "Data paper", "Retracted", "Short survey"]


def compute_assembled_corpus_count(raw):
    df = raw[_KEEP_COLS].copy()
    df.drop_duplicates(subset="Title", keep="first", inplace=True)
    df = df[df["Title"].notna() & df["Year"].notna()].copy()
    df["Document Type"] = df["Document Type"].astype(str).str.strip()
    mask = df["Document Type"].isin(_DROP_DOC_TYPES) | df["Document Type"].str.contains(
        "Godspeed scale|anthropomorphism", case=False, na=False
    )
    return len(df[~mask])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    assign = pd.read_csv(ASSIGNMENTS_PATH)
    assign["Year"] = pd.to_numeric(assign["Year"], errors="coerce")

    n_raw = len(raw)
    n_assembled = compute_assembled_corpus_count(raw)
    n_final = len(assign)
    year_min = int(assign["Year"].min())
    year_max = int(assign["Year"].max())

    n_pre = int((assign["Year"] <= 2022).sum())
    n_2023 = int((assign["Year"] == 2023).sum())
    n_post = int(((assign["Year"] >= 2024) & (assign["Year"] <= 2025)).sum())

    rows = [
        ("Source database", "Scopus (Elsevier); query in scopus_query.txt"),
        ("Search date range", "PUBYEAR > 2009 AND PUBYEAR < 2026 (initial Scopus search: ~2,224 records logged November 2025, scopus_query.txt)"),
        ("Assembled corpus (matches manuscript)", f"{n_assembled:,} documents, {year_min}-{year_max} (raw export: {n_raw:,} rows, incl. 4 rows from one known CSV parsing artifact; see notes/corpus_count_reconciliation.md)"),
        ("Final analytic corpus (this paper)", f"{n_final:,} documents, {year_min}-{year_max}"),
        ("Text fields analyzed", "Title + Abstract (topic_text: minimally cleaned, boilerplate-stripped concatenation)"),
        ("Topic-modeling approach (reused, not re-run)", "BERTopic representation over KMeans-guided clusters (sentence-transformer embeddings, all-MiniLM-L6-v2; fixed k=10 clusters; domain-adjusted c-TF-IDF vocabulary)"),
        ("Number of topics", "10 (topics 0-9)"),
        ("Period definitions", f"Pre-2023 (Year<=2022, n={n_pre}); 2023 transition year (n={n_2023}, excluded from main lexical contrast); 2024-2025 (Year in 2024-2025, n={n_post})"),
        ("New lexical-analysis method (this paper)", "Weighted log-odds-ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn 2008); unigrams + bigrams; min_df=5; alpha0 capped at smaller period's token count"),
        ("Lexical contrast corpus", f"Pre-2023 vs. 2024-2025 only (n={n_pre + n_post}); 2023 excluded"),
    ]

    assert n_assembled == 2198, (
        f"Reproduced assembled-corpus count ({n_assembled}) no longer matches the "
        "manuscript's reported 2,198 -- data/ai_literacy.csv may have changed; "
        "see notes/corpus_count_reconciliation.md."
    )

    table2 = pd.DataFrame(rows, columns=["Item", "Value"])
    table2.to_csv(TABLES / "table2_methods_summary.csv", index=False)

    with open(TABLES / "table2_methods_summary.md", "w") as f:
        f.write("# Table 2. Corpus and analytical-design summary\n\n")
        f.write("| Item | Value |\n|---|---|\n")
        for item, value in rows:
            f.write(f"| {item} | {value} |\n")

    print(table2.to_string(index=False))
    print("\nSaved:")
    print(" -", (TABLES / "table2_methods_summary.csv").relative_to(ROOT))
    print(" -", (TABLES / "table2_methods_summary.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
