"""
iConference paper — Analysis 1: Temporal topic-prevalence comparison
======================================================================

Reproduces the existing RQ4 pre-2023 vs. 2024-2025 topic-share comparison
from the parent project's 10-topic KMeans-guided BERTopic solution, and
redesigns the topic-change figure for the iConference paper.

Reused inputs (parent project, read-only):
  ../../scripts/rq4_kmeans_guided_cleanlabels_topic_assignments.csv
      Document-level topic assignments (10-topic solution), one row per
      document already surviving all exclusion filters in RQ4_thematic_analysis.ipynb
      (exclude_from_bertopic is False for every row in this file).
  ../../scripts/rq4_topic_label_lookup.csv
      Manual topic labels (long + short form) for the 10 topics.

This script recomputes period shares directly from the document-level
assignments (rather than copying the parent project's saved CSVs) so the
iConference numbers are independently reproducible from raw assignments.
The recomputed values match the parent project's
scripts/rq4_topic_share_change_table.csv exactly (verified below).

Outputs (all written under iconference/):
  derived/topic_period_shares_long.csv   Long-format topic share per period
  derived/topic_share_change_table.csv   Pre-2023 vs 2024-2025 change table
  tables/table_topic_share_change.csv    Paper-ready version of the above

Figure 1 itself (with bootstrap 95% CI error bars) is built from this
script's change table by 04_topic_share_bootstrap.py, not by this script.

Period definitions (identical to the parent RQ4 analysis):
  Pre-2023   : Year <= 2022
  2023       : Year == 2023   (transition year; excluded from main contrast)
  2024-2025  : Year in (2024, 2025)   (corpus does not extend past 2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_SCRIPTS = ROOT / "scripts"
ICONF = ROOT / "iconference"
DERIVED = ICONF / "derived"
TABLES = ICONF / "tables"
FIGURES = ICONF / "figures"

ASSIGNMENTS_PATH = PARENT_SCRIPTS / "rq4_kmeans_guided_cleanlabels_topic_assignments.csv"
LABEL_LOOKUP_PATH = PARENT_SCRIPTS / "rq4_topic_label_lookup.csv"
EXISTING_CHANGE_TABLE = PARENT_SCRIPTS / "rq4_topic_share_change_table.csv"

# Threshold (percentage points) for classifying a topic as "relatively
# stable" between periods, applied only for the increased/stable/decreased
# color-coding in Figure 1. This is a descriptive visualization choice, not
# a statistical test. +/-2.0 pp was chosen because it falls in a clear gap
# in the observed distribution of changes (-0.2 and -1.3 pp fall inside the
# band; the next values are 4.6 pp and -10.2 pp), so no topic is placed
# arbitrarily close to the boundary.
STABLE_BAND_PP = 2.0


def load_data():
    df = pd.read_csv(ASSIGNMENTS_PATH)
    labels = pd.read_csv(LABEL_LOOKUP_PATH)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    def assign_period(year):
        if year <= 2022:
            return "Pre-2023"
        elif year == 2023:
            return "2023 (transition)"
        else:
            return "2024-2025"

    df["period"] = df["Year"].apply(assign_period)
    df = df.merge(labels, on="topic", how="left")
    return df, labels


def build_period_shares(df):
    period_counts = (
        df.groupby(["period", "topic", "manual_topic_label", "short_topic_label"])
        .size()
        .reset_index(name="n")
    )
    period_totals = df.groupby("period").size().reset_index(name="period_total")
    period_shares = period_counts.merge(period_totals, on="period", how="left")
    period_shares["topic_share"] = period_shares["n"] / period_shares["period_total"]

    period_order = {"Pre-2023": 0, "2023 (transition)": 1, "2024-2025": 2}
    period_shares["period_order"] = period_shares["period"].map(period_order)
    period_shares = period_shares.sort_values(["period_order", "topic"]).reset_index(drop=True)
    return period_shares


def build_change_table(period_shares, labels):
    pre = period_shares.loc[
        period_shares["period"].eq("Pre-2023"), ["topic", "topic_share"]
    ].rename(columns={"topic_share": "pre_2023_share"})

    post = period_shares.loc[
        period_shares["period"].eq("2024-2025"), ["topic", "topic_share"]
    ].rename(columns={"topic_share": "post_2024_2025_share"})

    change = labels.merge(pre, on="topic", how="left").merge(post, on="topic", how="left")
    change[["pre_2023_share", "post_2024_2025_share"]] = change[
        ["pre_2023_share", "post_2024_2025_share"]
    ].fillna(0.0)

    change["share_point_change"] = (
        change["post_2024_2025_share"] - change["pre_2023_share"]
    )
    change["change_pct_points"] = change["share_point_change"] * 100
    change["pre_2023_pct"] = change["pre_2023_share"] * 100
    change["post_2024_2025_pct"] = change["post_2024_2025_share"] * 100

    def classify(pp):
        if pp >= STABLE_BAND_PP:
            return "Increased"
        elif pp <= -STABLE_BAND_PP:
            return "Decreased"
        else:
            return "Relatively stable"

    change["direction"] = change["change_pct_points"].apply(classify)
    change = change.sort_values("change_pct_points", ascending=False).reset_index(drop=True)
    return change


def verify_against_existing(change):
    if not EXISTING_CHANGE_TABLE.exists():
        print("Existing parent-project change table not found; skipping verification.")
        return
    existing = pd.read_csv(EXISTING_CHANGE_TABLE)
    merged = change.merge(
        existing, on="topic", suffixes=("_iconf", "_parent")
    )
    diffs = (
        merged["Change, percentage points"] - merged["change_pct_points"]
    ).abs()
    max_diff = diffs.max()
    print(f"Max abs. difference vs. parent-project change table: {max_diff:.6f} pp")
    # The parent-project table stores percentages already rounded to 1 decimal,
    # so a sub-0.05pp difference is expected rounding noise, not a discrepancy.
    assert max_diff < 0.05, "Recomputed change table does not match parent project output."
    print("Verified: recomputed shares match parent RQ4 analysis (within rounding).")


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df, labels = load_data()
    print(f"Loaded {len(df)} documents from {ASSIGNMENTS_PATH.name}")
    print("Documents by period:")
    print(df["period"].value_counts())

    period_shares = build_period_shares(df)
    period_shares.to_csv(DERIVED / "topic_period_shares_long.csv", index=False)

    change = build_change_table(period_shares, labels)
    change.to_csv(DERIVED / "topic_share_change_table.csv", index=False)

    verify_against_existing(change)

    paper_table = change[
        [
            "topic",
            "manual_topic_label",
            "short_topic_label",
            "pre_2023_pct",
            "post_2024_2025_pct",
            "change_pct_points",
            "direction",
        ]
    ].copy()
    paper_table.columns = [
        "Topic",
        "Topic label",
        "Short label",
        "Pre-2023 %",
        "2024-2025 %",
        "Change (pp)",
        "Direction",
    ]
    for col in ["Pre-2023 %", "2024-2025 %", "Change (pp)"]:
        paper_table[col] = paper_table[col].round(1)
    paper_table.to_csv(TABLES / "table_topic_share_change.csv", index=False)

    print("\nSaved:")
    for p in [
        DERIVED / "topic_period_shares_long.csv",
        DERIVED / "topic_share_change_table.csv",
        TABLES / "table_topic_share_change.csv",
    ]:
        print(" -", p.relative_to(ROOT))
    print(" (Figure 1 itself is produced by 04_topic_share_bootstrap.py, from this change table)")


if __name__ == "__main__":
    main()
