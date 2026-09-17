"""
iConference paper — Figure 1: topic-share change with bootstrap
uncertainty, and Table 1's underlying bootstrap results.

This script builds the single Figure 1 used in the paper (topic-share
change from pre-2023 to 2024-2025, with 95% bootstrap CI error bars). The
point estimates (topic shares, percentage-point change) come from
01_topic_prevalence_change.py and are not re-estimated here; this script
only adds an uncertainty quantification around them and plots both
together.

Motivation: the pre-2023 group is small (n=106 documents) relative to
2024-2025 (n=1,867), so point-estimate topic shares for pre-2023 could be
noisy. This script adds a document-level bootstrap to quantify that
uncertainty, without changing the point estimates already reported in
01_topic_prevalence_change.py.

Procedure:
  - Resample documents WITH REPLACEMENT, independently within each period
    (i.e., a pre-2023 bootstrap sample always has exactly 106 draws from
    the 106 pre-2023 documents; a 2024-2025 bootstrap sample always has
    exactly 1,867 draws from the 1,867 2024-2025 documents). This holds
    each period's sample size fixed and mirrors the fact that the two
    periods are independent, non-overlapping document sets.
  - For each of 10,000 replicates, recompute each topic's within-period
    share in the resampled pre-2023 and resampled 2024-2025 sets, and the
    resulting percentage-point difference (post minus pre).
  - Report the OBSERVED (non-bootstrapped) change alongside the 2.5th and
    97.5th percentiles of the bootstrap distribution of that change (a
    percentile bootstrap 95% CI), for all 10 topics.

This is an uncertainty quantification for the existing point estimates,
not a re-estimation of them: the observed change values here are identical
to those in derived/topic_share_change_table.csv (up to floating point).

Outputs:
  tables/table_topic_share_change_bootstrap.csv
  notes/topic_share_uncertainty.md
  figures/fig1_topic_share_change.png / .pdf   (Figure 1, the paper's only
                                                 Figure 1 — point estimates
                                                 from script 01 plus this
                                                 script's bootstrap CIs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_SCRIPTS = ROOT / "scripts"
ICONF = ROOT / "iconference"
DERIVED = ICONF / "derived"
TABLES = ICONF / "tables"
FIGURES = ICONF / "figures"
NOTES = ICONF / "notes"

ASSIGNMENTS_PATH = PARENT_SCRIPTS / "rq4_kmeans_guided_cleanlabels_topic_assignments.csv"
LABEL_LOOKUP_PATH = PARENT_SCRIPTS / "rq4_topic_label_lookup.csv"
OBSERVED_CHANGE_PATH = DERIVED / "topic_share_change_table.csv"

N_BOOT = 10_000
RANDOM_SEED = 42
STABLE_BAND_PP = 2.0  # same descriptive band as Figure 1; NOT a significance threshold
N_TOPICS = 10


def load_period_topics():
    df = pd.read_csv(ASSIGNMENTS_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    pre = df.loc[df["Year"] <= 2022, "topic"].to_numpy()
    post = df.loc[(df["Year"] >= 2024) & (df["Year"] <= 2025), "topic"].to_numpy()
    return pre, post


def bootstrap_shares(topic_array, n_boot, n_topics, rng):
    """Return an (n_boot, n_topics) array of within-sample topic shares."""
    n = len(topic_array)
    draws = rng.integers(0, n, size=(n_boot, n))
    sampled_topics = topic_array[draws]  # (n_boot, n)
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_topics), axis=1, arr=sampled_topics
    )
    shares = counts / n
    return shares


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(LABEL_LOOKUP_PATH)
    pre_topics, post_topics = load_period_topics()
    n_pre, n_post = len(pre_topics), len(post_topics)
    print(f"Pre-2023 documents: {n_pre}; 2024-2025 documents: {n_post}")
    print(f"Bootstrap replicates: {N_BOOT}")

    rng = np.random.default_rng(RANDOM_SEED)

    pre_boot_shares = bootstrap_shares(pre_topics, N_BOOT, N_TOPICS, rng)
    post_boot_shares = bootstrap_shares(post_topics, N_BOOT, N_TOPICS, rng)

    diff_pp = (post_boot_shares - pre_boot_shares) * 100  # (N_BOOT, N_TOPICS)

    ci_lower = np.percentile(diff_pp, 2.5, axis=0)
    ci_upper = np.percentile(diff_pp, 97.5, axis=0)
    boot_mean = diff_pp.mean(axis=0)
    boot_se = diff_pp.std(axis=0, ddof=1)

    # Observed (non-bootstrapped) shares and change, for comparison
    pre_counts = np.bincount(pre_topics, minlength=N_TOPICS)
    post_counts = np.bincount(post_topics, minlength=N_TOPICS)
    pre_pct_obs = pre_counts / n_pre * 100
    post_pct_obs = post_counts / n_post * 100
    change_obs = post_pct_obs - pre_pct_obs

    result = labels.copy()
    result["pre_2023_pct_observed"] = pre_pct_obs
    result["post_2024_2025_pct_observed"] = post_pct_obs
    result["change_pp_observed"] = change_obs
    result["boot_mean_change_pp"] = boot_mean
    result["boot_se_pp"] = boot_se
    result["ci95_lower_pp"] = ci_lower
    result["ci95_upper_pp"] = ci_upper
    result["ci95_excludes_zero"] = (ci_lower > 0) | (ci_upper < 0)

    def classify(pp):
        if pp >= STABLE_BAND_PP:
            return "Increased"
        elif pp <= -STABLE_BAND_PP:
            return "Decreased"
        else:
            return "Relatively stable"

    result["direction_descriptive_band"] = result["change_pp_observed"].apply(classify)

    result = result.sort_values("change_pp_observed", ascending=False).reset_index(drop=True)

    # Sanity check against the point-estimate table from script 01 (full precision,
    # before this table's own display rounding below)
    if OBSERVED_CHANGE_PATH.exists():
        prior = pd.read_csv(OBSERVED_CHANGE_PATH)[["topic", "change_pct_points"]]
        merged = result.merge(prior, on="topic")
        max_diff = (merged["change_pp_observed"] - merged["change_pct_points"]).abs().max()
        print(f"\nMax abs. diff vs. script 01's point-estimate change table: {max_diff:.6f} pp")
        assert max_diff < 1e-6, "Observed change here should exactly match script 01's output."
        print("Verified: observed change matches script 01 exactly.")

    for col in [
        "pre_2023_pct_observed", "post_2024_2025_pct_observed", "change_pp_observed",
        "boot_mean_change_pp", "boot_se_pp", "ci95_lower_pp", "ci95_upper_pp",
    ]:
        result[col] = result[col].round(2)

    out_cols = [
        "topic", "manual_topic_label", "short_topic_label",
        "pre_2023_pct_observed", "post_2024_2025_pct_observed", "change_pp_observed",
        "boot_mean_change_pp", "boot_se_pp", "ci95_lower_pp", "ci95_upper_pp",
        "ci95_excludes_zero", "direction_descriptive_band",
    ]
    result = result[out_cols]
    result.to_csv(TABLES / "table_topic_share_change_bootstrap.csv", index=False)
    print(result.to_string(index=False))

    make_figure(result)
    write_note(result, n_pre, n_post)

    print("\nSaved:")
    for p in [
        TABLES / "table_topic_share_change_bootstrap.csv",
        NOTES / "topic_share_uncertainty.md",
        FIGURES / "fig1_topic_share_change.png",
        FIGURES / "fig1_topic_share_change.pdf",
    ]:
        print(" -", p.relative_to(ROOT))


def make_figure(result):
    plot_df = result.sort_values("change_pp_observed", ascending=True).copy()

    color_map = {
        "Increased": "#2E7D32",
        "Relatively stable": "#9E9E9E",
        "Decreased": "#C62828",
    }
    colors = plot_df["direction_descriptive_band"].map(color_map)

    err_lower = plot_df["change_pp_observed"] - plot_df["ci95_lower_pp"]
    err_upper = plot_df["ci95_upper_pp"] - plot_df["change_pp_observed"]

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    ax.barh(
        plot_df["short_topic_label"],
        plot_df["change_pp_observed"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.errorbar(
        plot_df["change_pp_observed"],
        plot_df["short_topic_label"],
        xerr=[err_lower, err_upper],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )

    ax.axvline(0, color="black", linewidth=0.9, zorder=2)
    ax.axvspan(-STABLE_BAND_PP, STABLE_BAND_PP, color="#9E9E9E", alpha=0.10, zorder=1)

    ax.set_xlabel(
        "Change in topic share, percentage points (pre-2023 → 2024-2025)\n"
        "error bars: 95% bootstrap CI (document-level resampling, 10,000 replicates)"
    )
    ax.set_title(
        "Figure 1. Change in topic prevalence from pre-2023 to 2024-2025",
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(plot_df["ci95_lower_pp"].min() - 3, plot_df["ci95_upper_pp"].max() + 3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map["Increased"], label=f"Increased (≥ +{STABLE_BAND_PP:.0f} pp, observed)"),
        Patch(facecolor=color_map["Relatively stable"], label=f"Relatively stable (±{STABLE_BAND_PP:.0f} pp, observed)"),
        Patch(facecolor=color_map["Decreased"], label=f"Decreased (≤ -{STABLE_BAND_PP:.0f} pp, observed)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    fig.savefig(FIGURES / "fig1_topic_share_change.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_topic_share_change.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_note(result, n_pre, n_post):
    n_excl_zero = int(result["ci95_excludes_zero"].sum())
    lines = []
    lines.append("# Bootstrap uncertainty for topic-share change (pre-2023 vs. 2024-2025)\n")
    lines.append(
        f"Document-level bootstrap, {N_BOOT:,} replicates, resampling with "
        f"replacement independently within each period (pre-2023: n={n_pre}; "
        f"2024-2025: n={n_post}), random seed {RANDOM_SEED}.\n"
    )
    lines.append(
        "**The existing ±2.0 percentage-point \"relatively stable\" band used to "
        "color-code Figure 1 is a descriptive visualization threshold chosen for "
        "readability (see 01_topic_prevalence_change.py), not a statistical "
        "significance test. The bootstrap CIs below are the significance-relevant "
        "quantity; the two should not be conflated.**\n"
    )
    lines.append("## Results (full table: tables/table_topic_share_change_bootstrap.csv)\n")
    lines.append(
        "| Topic | Label | Observed change (pp) | 95% CI (pp) | CI excludes 0? |"
    )
    lines.append("|---|---|---|---|---|")
    for _, row in result.iterrows():
        lines.append(
            f"| {row['topic']} | {row['short_topic_label']} | "
            f"{row['change_pp_observed']:+.1f} | "
            f"[{row['ci95_lower_pp']:+.1f}, {row['ci95_upper_pp']:+.1f}] | "
            f"{'Yes' if row['ci95_excludes_zero'] else 'No'} |"
        )
    lines.append("")
    lines.append(
        f"**{n_excl_zero} of 10 topics have a 95% bootstrap CI that excludes zero** "
        "(i.e., the change is not attributable to sampling noise at the 95% level "
        "under this resampling scheme)."
    )
    excl = result.loc[~result["ci95_excludes_zero"]]
    if len(excl) > 0:
        excl_list = ", ".join(
            f"{r['short_topic_label']} ({r['change_pp_observed']:+.1f} pp, "
            f"95% CI [{r['ci95_lower_pp']:+.1f}, {r['ci95_upper_pp']:+.1f}])"
            for _, r in excl.iterrows()
        )
        lines.append(
            f"\nTopics whose CI does **not** exclude zero: {excl_list}. These are "
            "the smallest-magnitude observed changes and should be described in the "
            "paper as directionally suggestive rather than confidently established."
        )
    lines.append(
        "\n## Interpretation notes\n"
        "- The bootstrap resamples documents *within* each already-observed period; "
        "it quantifies sampling uncertainty in the topic-share estimate given the "
        "assignments this project already has, not uncertainty in the underlying "
        "topic assignments themselves (i.e., it does not re-run BERTopic/KMeans "
        "clustering, and does not capture uncertainty from the fixed k=10 topic "
        "solution or from document-to-topic assignment).\n"
        "- Because pre-2023 has only 106 documents, its bootstrap resamples have "
        "more granular/discrete possible shares (multiples of 1/106) than "
        "2024-2025's (multiples of 1/1,867), which is reflected in wider CIs for "
        "topics whose prevalence is concentrated in the pre-2023 period.\n"
        "- This bootstrap does not change any point estimate already reported in "
        "Figure 1 / `tables/table_topic_share_change.csv`; `change_pp_observed` "
        "here is identical to that table's `Change (pp)` column."
    )

    with open(NOTES / "topic_share_uncertainty.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
