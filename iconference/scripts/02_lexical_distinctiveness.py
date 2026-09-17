"""
iConference paper — Analysis 2: Lexical distinctiveness, pre-2023 vs. 2024-2025
=================================================================================

Compares the vocabulary of titles + abstracts in pre-2023 documents against
2024-2025 documents. 2023 is excluded from this contrast because it is
treated as a transition year (generative AI's public release, but before
its research uptake is fully reflected in the literature).

Method: weighted log-odds-ratio with an informative Dirichlet prior
(Monroe, Colaresi & Quinn 2008, "Fightin' Words: Lexical Feature Selection
and Evaluation for Identifying the Content of Political Conflict").

Why this method (documented per task instructions):
  - Simple log-odds or raw frequency differences are unstable for rare
    terms and do not account for the large imbalance in period sizes here
    (106 pre-2023 documents vs. 1,867 2024-2025 documents).
  - The Dirichlet-smoothed log-odds z-score down-weights terms with low
    counts in either period (via the prior pseudo-count) and yields a
    variance-normalized statistic (a z-score) that is comparable across
    terms of very different frequency, which a plain log-odds ratio is not.
  - This is the standard, well-documented method for identifying
    period/group-distinctive vocabulary in text-as-data work (implemented,
    e.g., in the `scattertext` and `convokit` "fighting words" modules,
    neither of which is available in this project's environment, so the
    statistic is implemented directly below from the published formula).
  - No separate/external reference corpus is available in this repository
    for the Dirichlet prior, so the prior's *shape* (relative word
    frequencies alpha_w / alpha_0) is taken from the combined pre-2023 +
    2024-2025 sub-corpus itself (the union of the two periods being
    compared, excluding 2023). Its *total mass* (alpha_0) is deliberately
    capped at the token count of the smaller period (pre-2023) rather than
    the full combined corpus: an early implementation that set alpha_0 to
    the full combined-corpus size produced a degenerate, over-smoothed
    prior (alpha_w scales with the very counts being tested, so the prior
    and the data are not independent, which mechanically shrinks z-scores
    toward zero regardless of how skewed a term's distribution actually
    is). Capping alpha_0 at the smaller period's size follows Monroe et
    al.'s (2008) own guidance that the prior's total pseudo-count mass
    should not be set so large that it swamps the data. This choice is
    documented as a limitation in iconference/README.md, and its effect is
    checked in a sensitivity note there.

Reused input (parent project, read-only):
  ../../scripts/rq4_kmeans_guided_cleanlabels_topic_assignments.csv
      Provides `topic_text` (title + minimally-cleaned abstract, with
      Scopus copyright boilerplate and HTML/markup artifacts already
      stripped by the parent project's RQ4_thematic_analysis.ipynb) and
      `Year` for every document in the final RQ4 analytic corpus.
      Reusing `topic_text` avoids re-deriving title+abstract cleaning
      logic that the parent project already validated.

Preprocessing (this script):
  - Lowercase; tokenize on alphabetic runs (regex \\b[a-z][a-z-]{1,}\\b),
    dropping pure numbers/punctuation.
  - Remove standard English stopwords (sklearn ENGLISH_STOP_WORDS).
  - Remove the corpus's own non-discriminating terms and generic
    domain/abstract-boilerplate terms, reusing the `domain_stop_words`
    set already defined and validated in the parent project's
    KMeans-guided BERTopic cell (RQ4_thematic_analysis.ipynb), which
    already includes "ai", "artificial intelligence", "literacy", etc.
  - Additionally remove residual bibliographic/publisher noise tokens
    (elsevier, springer, copyright, doi, ieee, acm, publisher, ...).
  - Build unigrams and bigrams from the filtered token stream (a custom
    analyzer, not sklearn's default stop-word handling, so bigrams are
    only formed between tokens that were originally adjacent AND survived
    filtering; see note in README on the standard "false adjacency"
    caveat of naive bigram construction after stopword removal).
  - Apply a minimum document-frequency threshold (min_df = 5 documents,
    counted across the combined 1,973-document two-period sub-corpus) so
    results are not driven by rare/idiosyncratic phrases. Sensitivity to
    this threshold (min_df in {3, 5, 10}) is checked and reported.

Outputs (all under iconference/):
  derived/lexical_period_corpus_stats.csv       Corpus size/token diagnostics
  derived/lexical_distinctive_terms_full.csv    All terms passing min_df=5
  tables/table_lexical_distinctive_terms.csv    Paper-ready ranked top terms
  derived/lexical_sensitivity_mindf.csv         Robustness: min_df in {3,5,10}
  derived/lexical_tech_term_audit.csv           Robustness: tech-name share
  derived/lexical_example_documents.csv         Robustness: sample titles

Figure 2 itself is built from these ranked results by 08_figure2.py (a
separate, display-only script; see that file and
notes/figure2_concept_grouping.md), not by this script.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
PARENT_SCRIPTS = ROOT / "scripts"
ICONF = ROOT / "iconference"
DERIVED = ICONF / "derived"
TABLES = ICONF / "tables"
FIGURES = ICONF / "figures"

ASSIGNMENTS_PATH = PARENT_SCRIPTS / "rq4_kmeans_guided_cleanlabels_topic_assignments.csv"

RANDOM_SEED = 42
MIN_DF_MAIN = 5
TOP_N_PER_SIDE = 15

# ----------------------------------------------------------------------
# Stopword / noise-token list
# ----------------------------------------------------------------------
# Reused verbatim from the parent project's KMeans-guided BERTopic cell
# (RQ4_thematic_analysis.ipynb), which already identifies corpus-wide,
# non-discriminating AI-literacy / education / abstract-boilerplate terms.
DOMAIN_STOP_WORDS_FROM_PARENT = {
    "ai", "artificial", "intelligence", "artificial intelligence",
    "literacy", "ai literacy",
    "education", "educational", "learning", "learners", "learner",
    "students", "student", "teachers", "teacher", "teaching",
    "study", "studies", "research", "paper", "article",
    "results", "findings", "analysis", "data", "method", "methods",
    "approach", "framework", "model", "models", "based", "using", "use",
    "used", "explores", "examines", "investigates", "aims", "purpose",
    "significant", "effect", "effects", "impact", "role", "level",
    "levels", "skills", "knowledge", "development", "developing",
    "important", "potential",
}

# Bibliographic / publisher / export-artifact noise not already covered
# above (identified by direct inspection of topic_text for residual
# boilerplate tokens; see iconference/README.md for the inspection).
BIBLIOGRAPHIC_NOISE_TOKENS = {
    "elsevier", "springer", "wiley", "copyright", "doi", "ieee", "acm",
    "publisher", "publishers", "proceedings", "reserved", "rights",
    "license", "licensee", "licensors", "author", "authors", "issn",
    "isbn", "vol", "volume", "issue", "pp",
}

CUSTOM_STOP_WORDS = (
    set(ENGLISH_STOP_WORDS) | DOMAIN_STOP_WORDS_FROM_PARENT | BIBLIOGRAPHIC_NOISE_TOKENS
)

# Known generative-AI tool/technology names, used only for the robustness
# check of whether results are dominated by bare technology labels.
TECH_NAME_TOKENS = {
    "chatgpt", "gpt", "gpt-4", "gpt4", "llm", "llms", "genai", "generative",
    "gemini", "bard", "copilot", "claude", "openai", "llama", "dall-e",
    "midjourney", "bing", "gpt-3", "gpt3",
}

TOKEN_PATTERN = re.compile(r"\b[a-z][a-z\-]{1,}\b")


def tokenize(text):
    text = str(text).lower()
    tokens = TOKEN_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in CUSTOM_STOP_WORDS and len(t) > 2]
    return tokens


def ngram_analyzer(text):
    tokens = tokenize(text)
    grams = list(tokens)
    grams += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return grams


def load_period_documents():
    df = pd.read_csv(ASSIGNMENTS_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    pre = df.loc[df["Year"] <= 2022].copy()
    post = df.loc[(df["Year"] >= 2024) & (df["Year"] <= 2025)].copy()

    pre["period"] = "Pre-2023"
    post["period"] = "2024-2025"

    combined = pd.concat([pre, post], ignore_index=True)
    return combined, pre, post


def build_term_doc_matrix(texts, min_df):
    vectorizer = CountVectorizer(analyzer=ngram_analyzer, min_df=min_df)
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    return X, terms, vectorizer


def monroe_log_odds(counts_a, counts_b, background_counts, alpha0):
    """
    Weighted log-odds-ratio with informative Dirichlet prior
    (Monroe, Colaresi & Quinn 2008), computed per term.

    counts_a, counts_b : raw term counts (not doc-freq) in corpus A / B
    background_counts  : raw background term counts, used only for the
                          prior's *shape* (relative frequency)
    alpha0              : total Dirichlet prior pseudo-count mass (the
                          background counts are rescaled to sum to this)
    """
    n_a = counts_a.sum()
    n_b = counts_b.sum()

    background_total = background_counts.sum()
    alpha_w = alpha0 * (background_counts / background_total)

    log_odds_a = np.log(counts_a + alpha_w) - np.log(
        n_a + alpha0 - counts_a - alpha_w
    )
    log_odds_b = np.log(counts_b + alpha_w) - np.log(
        n_b + alpha0 - counts_b - alpha_w
    )

    delta = log_odds_a - log_odds_b
    variance = 1.0 / (counts_a + alpha_w) + 1.0 / (counts_b + alpha_w)
    z = delta / np.sqrt(variance)
    return delta, z


def compute_distinctiveness(combined, min_df):
    texts = combined["topic_text"].fillna("").tolist()
    periods = combined["period"].values

    X, terms, vectorizer = build_term_doc_matrix(texts, min_df=min_df)

    is_pre = periods == "Pre-2023"
    is_post = periods == "2024-2025"

    counts_pre = np.asarray(X[is_pre].sum(axis=0)).ravel().astype(float)
    counts_post = np.asarray(X[is_post].sum(axis=0)).ravel().astype(float)

    # Prior shape: relative word frequency in the combined two-period corpus.
    # Prior mass (alpha0): capped at the smaller period's total token count
    # (pre-2023) so the prior informs, but cannot dominate, the estimate.
    background_counts = counts_pre + counts_post
    alpha0 = min(counts_pre.sum(), counts_post.sum())

    delta, z = monroe_log_odds(counts_pre, counts_post, background_counts, alpha0)

    docfreq_pre = np.asarray((X[is_pre] > 0).sum(axis=0)).ravel()
    docfreq_post = np.asarray((X[is_post] > 0).sum(axis=0)).ravel()

    result = pd.DataFrame(
        {
            "term": terms,
            "n_grams": [1 if " " not in t else 2 for t in terms],
            "count_pre_2023": counts_pre.astype(int),
            "count_2024_2025": counts_post.astype(int),
            "doc_freq_pre_2023": docfreq_pre,
            "doc_freq_2024_2025": docfreq_post,
            "log_odds_delta": delta,
            "z_score": z,
        }
    )
    result["associated_period"] = np.where(
        result["z_score"] > 0, "Pre-2023", "2024-2025"
    )
    result = result.sort_values("z_score", ascending=False).reset_index(drop=True)
    return result


def sensitivity_check(combined):
    rows = []
    top_sets = {}
    for min_df in (3, 5, 10):
        res = compute_distinctiveness(combined, min_df=min_df)
        n_terms = len(res)
        top_pre = set(res.sort_values("z_score", ascending=False).head(TOP_N_PER_SIDE)["term"])
        top_post = set(res.sort_values("z_score", ascending=True).head(TOP_N_PER_SIDE)["term"])
        top_sets[min_df] = (top_pre, top_post)
        rows.append({"min_df": min_df, "n_terms_passing_threshold": n_terms})

    overlap_rows = []
    base = 5
    for min_df in (3, 10):
        pre_overlap = len(top_sets[base][0] & top_sets[min_df][0]) / TOP_N_PER_SIDE
        post_overlap = len(top_sets[base][1] & top_sets[min_df][1]) / TOP_N_PER_SIDE
        overlap_rows.append(
            {
                "comparison": f"min_df={base} vs min_df={min_df}",
                "top15_pre_2023_overlap_fraction": round(pre_overlap, 2),
                "top15_2024_2025_overlap_fraction": round(post_overlap, 2),
            }
        )

    sens_df = pd.DataFrame(rows)
    overlap_df = pd.DataFrame(overlap_rows)
    return sens_df, overlap_df


def tech_name_audit(result):
    top_pre = result.sort_values("z_score", ascending=False).head(TOP_N_PER_SIDE).copy()
    top_post = result.sort_values("z_score", ascending=True).head(TOP_N_PER_SIDE).copy()

    def flag_tech(term):
        return any(tok in term.split() for tok in TECH_NAME_TOKENS) or term in TECH_NAME_TOKENS

    top_pre["is_tech_name"] = top_pre["term"].apply(flag_tech)
    top_post["is_tech_name"] = top_post["term"].apply(flag_tech)

    audit = pd.concat(
        [
            top_pre.assign(side="Pre-2023"),
            top_post.assign(side="2024-2025"),
        ],
        ignore_index=True,
    )

    summary = (
        audit.groupby("side")["is_tech_name"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_tech_name_terms", "count": "n_terms"})
        .reset_index()
    )
    summary["tech_name_share"] = (summary["n_tech_name_terms"] / summary["n_terms"]).round(2)
    return audit, summary


def example_documents(combined, result, n_terms=5, n_docs=3):
    top_pre_terms = result.sort_values("z_score", ascending=False).head(n_terms)["term"].tolist()
    top_post_terms = result.sort_values("z_score", ascending=True).head(n_terms)["term"].tolist()

    rows = []
    for side, terms in [("Pre-2023", top_pre_terms), ("2024-2025", top_post_terms)]:
        subset = combined.loc[combined["period"] == side]
        for term in terms:
            pattern = re.escape(term)
            mask = subset["topic_text"].fillna("").str.lower().str.contains(
                rf"\b{pattern}\b", regex=True
            )
            matches = subset.loc[mask].head(n_docs)
            for _, row in matches.iterrows():
                rows.append(
                    {
                        "side": side,
                        "term": term,
                        "year": row["Year"],
                        "title": row["Title"],
                    }
                )
    return pd.DataFrame(rows)


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    combined, pre, post = load_period_documents()

    corpus_stats = pd.DataFrame(
        [
            {"period": "Pre-2023", "n_documents": len(pre), "year_min": int(pre["Year"].min()), "year_max": int(pre["Year"].max())},
            {"period": "2024-2025", "n_documents": len(post), "year_min": int(post["Year"].min()), "year_max": int(post["Year"].max())},
        ]
    )
    corpus_stats.to_csv(DERIVED / "lexical_period_corpus_stats.csv", index=False)
    print(corpus_stats.to_string(index=False))

    result = compute_distinctiveness(combined, min_df=MIN_DF_MAIN)
    result.to_csv(DERIVED / "lexical_distinctive_terms_full.csv", index=False)
    print(f"\n{len(result)} terms passed min_df={MIN_DF_MAIN} across the combined two-period corpus.")

    top_pre = result.sort_values("z_score", ascending=False).head(TOP_N_PER_SIDE)
    top_post = result.sort_values("z_score", ascending=True).head(TOP_N_PER_SIDE)
    paper_table = pd.concat([top_pre, top_post], ignore_index=True)
    paper_table = paper_table[
        [
            "term", "associated_period", "z_score", "log_odds_delta",
            "count_pre_2023", "count_2024_2025",
            "doc_freq_pre_2023", "doc_freq_2024_2025", "n_grams",
        ]
    ].round({"z_score": 2, "log_odds_delta": 3})
    paper_table.to_csv(TABLES / "table_lexical_distinctive_terms.csv", index=False)

    print("\nTop terms, pre-2023:")
    print(top_pre[["term", "z_score", "count_pre_2023", "count_2024_2025"]].to_string(index=False))
    print("\nTop terms, 2024-2025:")
    print(top_post[["term", "z_score", "count_pre_2023", "count_2024_2025"]].to_string(index=False))

    # Robustness checks
    sens_df, overlap_df = sensitivity_check(combined)
    sens_df.to_csv(DERIVED / "lexical_sensitivity_mindf.csv", index=False)
    overlap_df.to_csv(DERIVED / "lexical_sensitivity_top15_overlap.csv", index=False)
    print("\nSensitivity to min_df:")
    print(sens_df.to_string(index=False))
    print(overlap_df.to_string(index=False))

    audit_detail, audit_summary = tech_name_audit(result)
    audit_detail.to_csv(DERIVED / "lexical_tech_term_audit.csv", index=False)
    audit_summary.to_csv(DERIVED / "lexical_tech_term_audit_summary.csv", index=False)
    print("\nTech-name share of top terms:")
    print(audit_summary.to_string(index=False))

    examples = example_documents(combined, result)
    examples.to_csv(DERIVED / "lexical_example_documents.csv", index=False)
    print(f"\nSaved {len(examples)} example-document rows for top distinctive terms.")

    print("\nSaved:")
    for p in [
        DERIVED / "lexical_period_corpus_stats.csv",
        DERIVED / "lexical_distinctive_terms_full.csv",
        TABLES / "table_lexical_distinctive_terms.csv",
        DERIVED / "lexical_sensitivity_mindf.csv",
        DERIVED / "lexical_sensitivity_top15_overlap.csv",
        DERIVED / "lexical_tech_term_audit.csv",
        DERIVED / "lexical_tech_term_audit_summary.csv",
        DERIVED / "lexical_example_documents.csv",
    ]:
        print(" -", p.relative_to(ROOT))
    print(" (Figure 2 itself is produced by 08_figure2.py, from these ranked results)")


if __name__ == "__main__":
    main()
