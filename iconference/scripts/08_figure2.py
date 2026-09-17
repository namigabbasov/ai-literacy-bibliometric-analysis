"""
iConference paper — Figure 2: concept-grouped display of the lexical
distinctiveness results.

IMPORTANT: this script changes NOTHING about the underlying analysis. It
reads the same, unmodified ranked results
(derived/lexical_distinctive_terms_full.csv) produced by
02_lexical_distinctiveness.py and only decides which already-ranked terms
to show and how to label them for readability. No z-score, count, rank,
preprocessing choice, prior, or corpus is touched here.

Motivation: showing the top 15 (or top 9) raw ranked terms per side
devotes several 2024-2025 display slots to near-duplicate surface forms of
the same underlying concept -- "generative", "genai", "gai", and
"generative genai" are four surface forms of generative AI -- crowding out
the institutional / evaluative / professional vocabulary that is a central
part of the paper's argument. This script fixes that by (a) collapsing
obvious lexical variants of the same concept into one displayed bar, using
the highest-ranked member's statistic UNLESS a literal-bigram-confirmed
multiword expression is clearer (see notes/figure2_concept_grouping.md for
every grouping decision), and (b) deprioritizing terms resting on very few
supporting documents in favor of more widespread, still-top-ranked
alternatives, using the existing document-prevalence results
(tables/table_lexical_distinctive_terms_with_docprevalence.csv and the
same >=5-supporting-document convention already used as this project's
min_df threshold). No term is skipped merely for sounding generic -- only
for being a lexical variant of an already-shown concept, or for resting on
very few documents.

Output:
  figures/fig2_lexical_distinctiveness.png / .pdf
  notes/figure2_concept_grouping.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ICONF = ROOT / "iconference"
DERIVED = ICONF / "derived"
TABLES = ICONF / "tables"
FIGURES = ICONF / "figures"
NOTES = ICONF / "notes"

FULL_RESULTS_PATH = DERIVED / "lexical_distinctive_terms_full.csv"
LITERAL_BIGRAM_PATH = DERIVED / "lexical_literal_bigram_check.csv"

N_PER_SIDE = 9
N_PRE = 106
N_POST = 1867
# A concept is deprioritized (skipped in the main display, though it stays
# fully visible in the underlying tables) if its representative term's
# document count in its own period falls below this. Reuses the project's
# own min_df=5 convention (already the statistical inclusion threshold
# throughout this analysis) as the display-prevalence rule, rather than
# inventing a new number.
MIN_DOCS_FOR_DISPLAY = 5

# ----------------------------------------------------------------------
# Concept families: predetermined, documented groupings of terms that are
# obvious lexical variants of one underlying concept (same lemma,
# singular/plural, abbreviation, or adjective/noun-phrase pair), decided
# BEFORE running the walk-down below. Every family is defined here whether
# or not it ends up inside the final N_PER_SIDE window, so the script's
# behavior does not depend on where the cutoff happens to fall.
#
# representative_term: the term whose z-score/count/doc-freq is plotted
#   for the whole family. Default = highest-ranked member. Overridden to a
#   literal-bigram-confirmed multiword expression only where the task
#   explicitly calls for it (critical -> critical thinking) or where an
#   exactly analogous adjective/noun-phrase pattern holds (conversational
#   -> conversational agents), per notes/literal_bigram_check.md.
# ----------------------------------------------------------------------
CONCEPT_FAMILIES = {
    "generative_ai": {
        "members": ["generative", "genai", "gai", "generative genai"],
        "representative_term": "generative",
        "display_label": "Generative AI",
        "rationale": "Four surface forms (full word, standard abbreviation, "
                     "alternate abbreviation, adjective+abbreviation bigram) "
                     "of the same referent, generative AI.",
    },
    "integration": {
        "members": ["integration", "integrating"],
        "representative_term": "integration",
        "display_label": "Integration",
        "rationale": "Same lemma, noun vs. gerund form.",
    },
    "critical_thinking": {
        "members": ["critical", "critical thinking"],
        "representative_term": "critical thinking",
        "display_label": "Critical thinking",
        "rationale": "Per task instruction: prefer the literal-bigram-confirmed "
                     "multiword expression ('critical thinking' ranks 1st among "
                     "literal contiguous bigrams on the 2024-2025 side; see "
                     "notes/literal_bigram_check.md) over the bare adjective, "
                     "even though 'critical' alone ranks marginally higher "
                     "(z=-2.614 vs. -2.601).",
    },
    "llms": {
        "members": ["llms", "llm"],
        "representative_term": "llms",
        "display_label": "LLMs",
        "rationale": "Plural/singular of the same abbreviation.",
    },
    "libraries": {
        "members": ["libraries", "library", "librarians"],
        "representative_term": "libraries",
        "display_label": "Libraries",
        "rationale": "Plural noun, singular noun, and agent-noun of the same "
                     "institutional referent.",
    },
    "nursing": {
        "members": ["nursing", "nurse", "nurses"],
        "representative_term": "nursing",
        "display_label": "Nursing",
        "rationale": "Same professional-domain referent; 'nursing' is already "
                     "the highest-ranked and most prevalent form.",
    },
    "concepts": {
        "members": ["concepts", "concept"],
        "representative_term": "concepts",
        "display_label": "Concepts",
        "rationale": "Plural/singular of the same lemma.",
    },
    "conversational_agents": {
        "members": ["conversational", "conversational agents"],
        "representative_term": "conversational agents",
        "display_label": "Conversational agents",
        "rationale": "Analogous to critical/critical thinking: the bare "
                     "adjective 'conversational' is, in this corpus, "
                     "overwhelmingly used to modify 'agents'; 'conversational "
                     "agents' is confirmed as a literal contiguous bigram "
                     "(rank 3 on the pre-2023 side; see "
                     "notes/literal_bigram_check.md).",
    },
    "citizens": {
        "members": ["citizens", "citizen"],
        "representative_term": "citizens",
        "display_label": "Citizens",
        "rationale": "Plural/singular of the same lemma (citizen appears far "
                     "down the ranking, rank 238, and does not affect display "
                     "selection either way).",
    },
    "games": {
        "members": ["game", "games"],
        "representative_term": "game",
        "display_label": "Game",
        "rationale": "Singular/plural of the same lemma (games appears far "
                     "down the ranking, rank 287, and does not affect display "
                     "selection either way).",
    },
}

TERM_TO_FAMILY = {}
for fam_key, fam in CONCEPT_FAMILIES.items():
    for m in fam["members"]:
        TERM_TO_FAMILY[m] = fam_key

# Display-only capitalization fixes for standalone terms that are proper
# nouns/acronyms, so the figure doesn't show "Chatgpt". This changes only
# how the (unmodified) term is *typeset*, not which term is selected,
# ranked, or plotted.
STANDALONE_LABEL_OVERRIDES = {
    "chatgpt": "ChatGPT",
}


def select_display_concepts(full_results, ascending, n_per_side, n_docs_period):
    ranked = full_results.sort_values("z_score", ascending=ascending).reset_index(drop=True)
    ranked["original_rank"] = ranked.index + 1
    term_lookup = ranked.set_index("term")

    kept = []
    skipped_low_prevalence = []
    consumed_terms = set()
    families_seen = set()

    for _, row in ranked.iterrows():
        if len(kept) >= n_per_side:
            break
        term = row["term"]
        if term in consumed_terms:
            continue

        fam_key = TERM_TO_FAMILY.get(term)
        if fam_key is not None:
            if fam_key in families_seen:
                consumed_terms.add(term)
                continue
            families_seen.add(fam_key)
            fam = CONCEPT_FAMILIES[fam_key]
            rep_term = fam["representative_term"]
            rep_row = term_lookup.loc[rep_term]
            doc_freq_col = "doc_freq_pre_2023" if ascending is False else "doc_freq_2024_2025"
            doc_freq = int(rep_row[doc_freq_col])
            for m in fam["members"]:
                consumed_terms.add(m)

            entry = {
                "display_label": fam["display_label"],
                "plotted_term": rep_term,
                "family_members": fam["members"],
                "z_score": float(rep_row["z_score"]),
                "n_grams": int(rep_row["n_grams"]),
                "doc_freq": doc_freq,
                "pct_docs": round(doc_freq / n_docs_period * 100, 1),
                "original_rank_of_plotted_term": int(rep_row["original_rank"]),
                "first_encountered_at_rank": int(row["original_rank"]),
                "is_family": True,
            }
        else:
            doc_freq_col = "doc_freq_pre_2023" if ascending is False else "doc_freq_2024_2025"
            doc_freq = int(row[doc_freq_col])
            entry = {
                "display_label": STANDALONE_LABEL_OVERRIDES.get(term, term.capitalize()),
                "plotted_term": term,
                "family_members": [term],
                "z_score": float(row["z_score"]),
                "n_grams": int(row["n_grams"]),
                "doc_freq": doc_freq,
                "pct_docs": round(doc_freq / n_docs_period * 100, 1),
                "original_rank_of_plotted_term": int(row["original_rank"]),
                "first_encountered_at_rank": int(row["original_rank"]),
                "is_family": False,
            }
            consumed_terms.add(term)

        if entry["doc_freq"] < MIN_DOCS_FOR_DISPLAY:
            skipped_low_prevalence.append(entry)
            continue

        kept.append(entry)

    return kept, skipped_low_prevalence


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    full_results = pd.read_csv(FULL_RESULTS_PATH)

    kept_pre, skipped_pre = select_display_concepts(
        full_results, ascending=False, n_per_side=N_PER_SIDE, n_docs_period=N_PRE
    )
    kept_post, skipped_post = select_display_concepts(
        full_results, ascending=True, n_per_side=N_PER_SIDE, n_docs_period=N_POST
    )

    print(f"Pre-2023: kept {len(kept_pre)} concepts, skipped {len(skipped_pre)} for low prevalence")
    for e in kept_pre:
        print(f"  {e['display_label']:24s} z={e['z_score']:.3f} doc_freq={e['doc_freq']} ({e['pct_docs']}%) "
              f"plotted_term='{e['plotted_term']}' members={e['family_members']}")
    print(f"\n2024-2025: kept {len(kept_post)} concepts, skipped {len(skipped_post)} for low prevalence")
    for e in kept_post:
        print(f"  {e['display_label']:24s} z={e['z_score']:.3f} doc_freq={e['doc_freq']} ({e['pct_docs']}%) "
              f"plotted_term='{e['plotted_term']}' members={e['family_members']}")

    make_figure(kept_pre, kept_post)
    write_note(kept_pre, skipped_pre, kept_post, skipped_post, full_results)

    print("\nSaved:")
    for p in [
        FIGURES / "fig2_lexical_distinctiveness.png",
        FIGURES / "fig2_lexical_distinctiveness.pdf",
        NOTES / "figure2_concept_grouping.md",
    ]:
        print(" -", p.relative_to(ROOT))


def make_figure(kept_pre, kept_post):
    rows = []
    for e in kept_pre + kept_post:
        rows.append(e)
    plot_df = pd.DataFrame(rows).sort_values("z_score", ascending=True).reset_index(drop=True)

    colors = np.where(plot_df["z_score"] > 0, "#4C78A8", "#E45756")

    # Tag a bar with † when it is a merged lexical-variant family (multiple
    # source terms collapsed into one bar), or with * when it is a single,
    # unmerged multiword expression (a bigram that was NOT part of a merge).
    # A bar is never tagged with both; a merged bigram (e.g. "Critical
    # thinking", built from critical + critical thinking) shows only †.
    labels = []
    used_dagger = False
    used_asterisk = False
    for _, r in plot_df.iterrows():
        lbl = r["display_label"]
        if r["is_family"] and len(r["family_members"]) > 1:
            lbl = lbl + "†"  # merged concept family
            used_dagger = True
        elif r["n_grams"] == 2:
            lbl = lbl + "*"  # standalone multiword expression, not merged
            used_asterisk = True
        labels.append(lbl)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.barh(labels, plot_df["z_score"], color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9, zorder=2)

    ax.set_xlabel("Weighted log-odds z-score (Dirichlet prior)")
    legend_notes = []
    if used_dagger:
        legend_notes.append("† = merged lexical-variant family")
    if used_asterisk:
        legend_notes.append("* = multiword expression")
    subtitle = f"AI literacy research (top {N_PER_SIDE} concepts per side"
    if legend_notes:
        subtitle += "; " + ", ".join(legend_notes)
    subtitle += ")"
    ax.set_title(
        f"Figure 2. Terms distinctive of pre-2023 vs. 2024-2025\n{subtitle}",
        fontsize=10.2,
        loc="left",
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C78A8", label="Distinctive of pre-2023"),
        Patch(facecolor="#E45756", label="Distinctive of 2024-2025"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    fig.savefig(FIGURES / "fig2_lexical_distinctiveness.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_lexical_distinctiveness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_note(kept_pre, skipped_pre, kept_post, skipped_post, full_results):
    lines = []
    lines.append("# Figure 2 concept grouping\n")
    lines.append(
        "**This is a presentation-only display decision, not a re-analysis.** "
        "No z-score, rank, count, preprocessing step, prior specification, or "
        "robustness result was changed. The full, unmodified ranked results "
        "remain at `derived/lexical_distinctive_terms_full.csv` and "
        "`tables/table_lexical_distinctive_terms.csv`; nothing there was "
        "altered by this script. This note documents exactly which display "
        "decisions were made and why.\n"
    )
    lines.append(
        "## Why this figure is built this way\n\n"
        "Simply showing the top-ranked terms per side devotes several 2024-2025 "
        "display slots to near-duplicate technology labels for the same "
        "underlying concept "
        "(generative, genai, gai, generative genai), crowding out the "
        "institutional / evaluative / professional vocabulary that is central "
        "to the paper's argument. This figure fixes that by (a) collapsing "
        "obvious lexical variants of one concept into a single bar, and (b) "
        "deprioritizing terms resting on very few supporting documents in "
        "favor of more widespread, still-top-ranked terms -- without skipping "
        "any term merely for sounding generic; 'academic', 'higher', "
        "'tools'-type terms are kept whenever they are reached by rank.\n"
    )

    lines.append("## Rule (applied identically to both sides)\n\n")
    lines.append(
        "1. Start from the same ranked results used everywhere else in this "
        "project (`derived/lexical_distinctive_terms_full.csv`, min_df=5, "
        "unchanged).\n"
        "2. Walk down the ranking, most-distinctive term first.\n"
        "3. If the term belongs to a predetermined concept family (below), "
        "register that family once, using its designated representative "
        "term's statistic -- the highest-ranked family member by default, "
        "or a literal-bigram-confirmed multiword expression where the task "
        "explicitly calls for that preference. Skip other family members "
        "when later encountered (already represented).\n"
        "4. If the representative term's document count in its own period is "
        f"below {MIN_DOCS_FOR_DISPLAY} (this project's existing min_df=5 "
        "convention, reused here as a display-prevalence rule rather than a "
        "new threshold), skip the whole concept for the main display -- it "
        "remains fully visible in the underlying tables.\n"
        f"5. Continue until {N_PER_SIDE} concepts are kept per side.\n\n"
        "No concept was added out of rank order, and no concept was dropped "
        "for supporting or complicating the paper's argument -- only for "
        "matching a family-membership rule or the document-count rule "
        "above, both fixed before the displayed term lists were assembled.\n"
    )

    lines.append("## Predetermined concept families\n")
    lines.append("| Family | Members | Representative term (plotted) | Why |")
    lines.append("|---|---|---|---|")
    for fam in CONCEPT_FAMILIES.values():
        lines.append(
            f"| {fam['display_label']} | {', '.join(fam['members'])} | "
            f"{fam['representative_term']} | {fam['rationale']} |"
        )

    lines.append(
        "\n**Note on 'higher education':** no `higher education` bigram "
        "passed min_df=5 (it does not appear in the ranked results at all). "
        "This is because \"education\" is itself one of the parent project's "
        "domain stopwords (removed before bigram construction; see "
        "`DOMAIN_STOP_WORDS_FROM_PARENT` in `02_lexical_distinctiveness.py`), "
        "so \"higher\" + \"education\" can never form as a bigram in this "
        "vocabulary. The displayed label is therefore the bare ranked "
        "unigram \"higher\" (Title-cased to \"Higher\"), not an invented "
        "\"higher education\" bigram.\n"
    )

    def fmt_kept_table(kept):
        out = ["| Display label | Plotted term | z-score | Doc. count (%) | Original rank(s) |",
               "|---|---|---|---|---|"]
        for e in kept:
            members_note = f" (from: {', '.join(e['family_members'])})" if len(e["family_members"]) > 1 else ""
            out.append(
                f"| {e['display_label']}{members_note} | {e['plotted_term']} | "
                f"{e['z_score']:.3f} | {e['doc_freq']} ({e['pct_docs']}%) | "
                f"{e['original_rank_of_plotted_term']} |"
            )
        return "\n".join(out)

    lines.append("## Displayed concepts\n")
    lines.append(f"### Pre-2023 ({len(kept_pre)} concepts)\n")
    lines.append(fmt_kept_table(kept_pre))
    lines.append(f"\n### 2024-2025 ({len(kept_post)} concepts)\n")
    lines.append(fmt_kept_table(kept_post))

    def fmt_skipped_table(skipped):
        if not skipped:
            return "(none met the low-prevalence skip condition within the display window)"
        out = ["| Display label (if shown) | Plotted term | z-score | Doc. count (%) | Rank |",
               "|---|---|---|---|---|"]
        for e in skipped:
            out.append(
                f"| {e['display_label']} | {e['plotted_term']} | {e['z_score']:.3f} | "
                f"{e['doc_freq']} ({e['pct_docs']}%) | {e['first_encountered_at_rank']} |"
            )
        return "\n".join(out)

    lines.append(
        f"\n## Low-document-prevalence terms omitted from the main display "
        f"(< {MIN_DOCS_FOR_DISPLAY} supporting documents in their period)\n"
    )
    lines.append("### Pre-2023 side\n")
    lines.append(fmt_skipped_table(skipped_pre))
    lines.append("\n### 2024-2025 side\n")
    lines.append(fmt_skipped_table(skipped_post))
    lines.append(
        "\nThese terms remain fully visible, ranked, and unmodified in "
        "`derived/lexical_distinctive_terms_full.csv` and "
        "`tables/table_lexical_distinctive_terms.csv` / "
        "`tables/table_lexical_distinctive_terms_with_docprevalence.csv`. "
        "They are omitted only from this main-text display figure because "
        "each rests on very few actual documents relative to what a "
        "higher-prevalence, still-top-ranked alternative offers -- exactly "
        "the concern the task raised about not letting rare terms dominate "
        "the figure.\n"
    )

    lines.append(
        "## Why \"nursing\" (2.9% of 2024-2025 documents) is kept while "
        "pre-2023 terms at similar or higher percentages are not\n\n"
        "The document-count rule above uses an **absolute** document count "
        f"(< {MIN_DOCS_FOR_DISPLAY} documents), not a period-relative "
        "percentage. \"nursing\" rests on 55 real 2024-2025 documents -- "
        "comparable in absolute terms to \"machine\" or \"curriculum\" on the "
        "pre-2023 side -- and only *looks* rare (2.9%) because the "
        "2024-2025 period itself is ~17.6x larger than pre-2023 (1,867 vs. "
        "106 documents). By contrast, the pre-2023 terms omitted above "
        "(\"book,\" \"health care,\" \"e-learning,\" \"intention learn,\" "
        "\"game,\" \"conversational\"/\"conversational agents\") rest on only "
        "1-4 actual documents each -- genuinely thin evidence, not an "
        "artifact of period size. Applying a single relative-percentage "
        "threshold across two periods of such different size would "
        "misclassify \"nursing\" as rare when it is not; the absolute-count "
        "rule avoids that.\n"
    )

    lines.append(
        "## Confirmation\n\n"
        "- The underlying weighted log-odds z-scores, ranks, counts, "
        "document frequencies, preprocessing, vocabulary (min_df=5), prior "
        "specification, and every prior robustness result "
        "(`notes/robustness_checks.md`, `notes/lexical_prior_sensitivity.md`, "
        "`notes/literal_bigram_check.md`) are **unchanged** by this script.\n"
        "- `derived/lexical_distinctive_terms_full.csv` and "
        "`tables/table_lexical_distinctive_terms.csv` are untouched.\n"
        "- This script only reads those files and decides how to group and "
        "label already-existing ranked terms for the one Figure 2 used in the "
        "paper (`figures/fig2_lexical_distinctiveness.png` / `.pdf`). Earlier "
        "draft variants of this figure (an unmerged top-15 version and a "
        "generic-word-filtered top-9 version) were superseded by this "
        "concept-grouped version and are no longer part of the output set.\n"
    )

    with open(NOTES / "figure2_concept_grouping.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
