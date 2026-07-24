"""Generate every Section 3 visualization as an individual image.

Unlike ``generate_figures.py``, which bundles related panels into combined
multi-panel figures, this script renders each chart separately so the paper
can place, pair, or scale them independently (e.g. two side-by-side
``subfigure`` environments). The underlying data, screening rules, and
aggregate assertions are imported unchanged from ``generate_figures.py``.

Outputs (PDF + PNG each) in ``study/paper/figures/individual/``

Run from the repository root:

    python study/reproducibility/generate_individual_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from generate_figures import (
    AUDIT,
    COVERAGE_COLUMNS,
    MODEL_COLORS,
    PROMPT_MARKERS,
    build_failure_table,
    build_phase_table,
    build_run_table,
)

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "individual"
FIGURES.mkdir(parents=True, exist_ok=True)

# Preregistered P2/P3 Mini phase. Tables produced by
# ``study/audit_preregistered_nodes.py``; sits beside the exploratory audit.
PREREG_AUDIT = AUDIT.parent / "audit_preregistered"

MODEL_SHORT = {"GPT-5 Nano": "Nano", "GPT-5.4 Mini": "Mini", "GPT-5.4": "GPT-5.4"}


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 3.1: Original study and replication outcomes
# ---------------------------------------------------------------------------

def plot_phase_comparison(phases: pd.DataFrame) -> None:
    # Compact single-column layout: the raw counts live in the row labels so
    # the plot area needs no in-axis text annotations.
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    evidence_colors = {
        "Original reported": "#CC79A7",
        "Historical reported": "#999999",
        "Artifact verified": "#0072B2",
    }
    evidence_markers = {
        "Original reported": "D",
        "Historical reported": "o",
        "Artifact verified": "s",
    }
    short_names = {
        "Original study": "Original study",
        "Historical detailed prompt": "Hist. detailed",
        "Historical simplified prompt": "Hist. simplified",
        "Retained v1.0.0 follow-up": "v1.0.0 follow-up",
    }

    positions = np.arange(len(phases))[::-1]
    for y, row in zip(positions, phases.itertuples(index=False), strict=True):
        low_error = row.rate_pct - row.wilson_low_pct
        high_error = row.wilson_high_pct - row.rate_pct
        ax.errorbar(
            row.rate_pct,
            y,
            xerr=np.array([[low_error], [high_error]]),
            fmt=evidence_markers[row.evidence],
            color=evidence_colors[row.evidence],
            ecolor=evidence_colors[row.evidence],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=6,
            capsize=2.5,
            linewidth=1.2,
            zorder=3,
        )

    phase_labels = [
        f"{short_names[row.phase]}\n{row.successes}/{row.runs} ({row.rate_pct:.1f}%)"
        for row in phases.itertuples(index=False)
    ]
    ax.set_yticks(positions, phase_labels)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Run-level completion rate (%)", fontsize=7)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)

    legend = [
        Line2D(
            [0],
            [0],
            marker=evidence_markers[label],
            color="none",
            markerfacecolor=evidence_colors[label],
            markeredgecolor="black",
            markersize=5,
            label=label,
        )
        for label in evidence_colors
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=3,
        fontsize=6.1,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.36)
    save_figure(fig, "phase_comparison")


# ---------------------------------------------------------------------------
# Section 3.2: Artifact-verified completion and partial progress
# ---------------------------------------------------------------------------

def stage_axis(
    ax: plt.Axes,
    labels: list[str],
    counts: list[int],
    denominator: int,
    color: str,
) -> None:
    x = np.arange(len(counts))
    sizes = 80 + 520 * np.asarray(counts) / max(denominator, 1)
    ax.plot(x, counts, color=color, linewidth=1.8, zorder=1)
    ax.scatter(x, counts, s=sizes, color=color, edgecolor="white", linewidth=1.0, zorder=2)
    for xx, value in zip(x, counts, strict=True):
        ax.text(xx, value, str(value), ha="center", va="center", color="white", weight="bold")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.set_ylim(-max(denominator * 0.08, 3), denominator * 1.12)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def _node_key(path: str) -> str:
    # checkpoint/<node>/... -> <node>; anything else (workspace) -> WORKSPACE
    parts = path.replace("\\", "/").split("/")
    if "checkpoint" in parts:
        return parts[parts.index("checkpoint") + 1]
    return "WORKSPACE"


def _merged_funnel_node_counts(runs: pd.DataFrame) -> None:
    """Verify the audit-derived node-level counts used by the merged funnel.

    Search-level counts (19, 17, 14, 12, 2, 0) and the published node counts
    (188, 78, 54) are asserted in generate_figures.build_run_table. The added
    numbers are recomputed here from the audit tables: searches with at least
    one exception-free node without mock/fallback markers (16), checkpoint
    nodes containing at least one native results.json (51), checkpoint nodes
    with at least one requested finite metric entry (45), and nodes containing
    a complete 90-entry MovieLens block (2). One workspace-level result
    (P0/Mini/R03) contributes at search level but is not a checkpoint node.
    """
    production = set(runs["run"])
    code = pd.read_csv(AUDIT / "generated_code_assessment.csv")
    code = code[code["run"].isin(production)].copy()
    assert len(code) == 188
    code["node"] = code["relative_path"].apply(_node_key)
    code["no_exception"] = ~(code["out_traceback"] | code["out_program_crashed"])
    clean = code["no_exception"] & ~code["uses_mock_or_fallback"]
    assert code.groupby("run")["no_exception"].any().sum() == 17
    assert clean.groupby(code["run"]).any().sum() == 16

    native = pd.read_csv(AUDIT / "native_result_files.csv")
    native = native[native["run"].isin(production)].copy()
    native["node"] = native["relative_path"].apply(_node_key)
    native_nodes = {
        (run, node)
        for run, node in zip(native["run"], native["node"], strict=True)
        if node != "WORKSPACE"
    }
    assert len(native_nodes) == 51

    metrics = pd.read_csv(AUDIT / "native_metrics.csv")
    metrics = metrics[metrics["run"].isin(production)].copy()
    metrics["node"] = metrics["relative_path"].apply(_node_key)
    requested = metrics[
        metrics["metric"].str.lower().str.contains("ndcg|precision")
        & metrics["k"].isin([1, 5, 10])
        & metrics["finite"]
    ].copy()
    requested_nodes = {
        (run, node)
        for run, node in zip(requested["run"], requested["node"], strict=True)
        if node != "WORKSPACE"
    }
    assert len(requested_nodes) == 45
    assert requested["run"].nunique() == 12

    movielens = requested[
        requested["dataset_id"].str.lower().str.contains("movielens|ml", na=False)
        & (requested["node"] != "WORKSPACE")
    ]
    entries = movielens.drop_duplicates(
        subset=["run", "node", "dataset_id", "algorithm_id", "seed", "metric", "k"]
    ).groupby(["run", "node"]).size()
    assert (entries == 90).sum() == 2


def plot_merged_funnel(runs: pd.DataFrame) -> None:
    # Single-axes merge of both funnels. Counts are normalized to percent of
    # each level's denominator (188 nodes / 19 searches) so the two lines share
    # one axis; raw counts label each point. Criteria are evaluated at both
    # levels: a search counts when at least one of its nodes provides the
    # evidence. See _merged_funnel_node_counts for the audit derivation.
    _merged_funnel_node_counts(runs)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    stage_labels = [
        "Generated\ncode",
        "No recorded\nexception",
        "No mock/\nfallback",
        "Native\nresults.json",
        "Requested\nmetric",
        "90-entry\nML",
        "Complete\n270",
    ]
    node_x = np.arange(7)
    node_counts = [188, 78, 54, 51, 45, 2, 0]
    node_pct = 100 * np.array(node_counts) / 188
    search_x = np.arange(7)
    search_counts = [19, 17, 16, 14, 12, 2, 0]
    search_pct = 100 * np.array(search_counts) / 19

    # Nudge the coinciding 100% starting and 0% final points apart.
    node_plot_x = node_x.astype(float)
    search_plot_x = search_x.astype(float)
    node_plot_x[0] -= 0.12
    search_plot_x[0] += 0.12
    node_plot_x[-1] -= 0.12
    search_plot_x[-1] += 0.12

    for x, pct, counts, color, label_offset, z in (
        (node_plot_x, node_pct, node_counts, "#0072B2", (-2, -10), 2),
        (search_plot_x, search_pct, search_counts, "#009E73", (2, 8), 3),
    ):
        ax.plot(x, pct, color=color, linewidth=1.6, zorder=z)
        ax.scatter(x, pct, s=42, color=color, edgecolor="white", linewidth=0.9, zorder=z + 2)
        for xx, yy, value in zip(x, pct, counts, strict=True):
            ax.annotate(
                str(value),
                (xx, yy),
                xytext=label_offset,
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color="#333333",
                zorder=z + 3,
            )

    ax.set_xticks(np.arange(len(stage_labels)), stage_labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=5.6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(-6, 108)
    ax.set_ylabel("Share of level denominator (%)", fontsize=7)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)

    legend = [
        Line2D([0], [0], marker="o", color="#0072B2", markerfacecolor="#0072B2",
               markersize=5, linewidth=1.4, label="Nodes (of 188)"),
        Line2D([0], [0], marker="o", color="#009E73", markerfacecolor="#009E73",
               markersize=5, linewidth=1.4, label="Searches (of 19)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=6, frameon=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.24)
    save_figure(fig, "completion_funnel_merged")


def plot_coverage_heatmap(runs: pd.DataFrame) -> None:
    # Single-column layout: rotated algorithm labels with dataset group
    # headers above, and a horizontal colorbar below the matrix.
    fig, ax = plt.subplots(figsize=(3.45, 4.15))
    matrix = runs[COVERAGE_COLUMNS].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#F2F2F2", "#9ECAE1", "#0072B2"]
    )
    heatmap = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=30, aspect="auto")
    algorithm_labels = ["ALS", "ItemKNN", "Pop"] * 3
    ax.set_xticks(np.arange(len(algorithm_labels)), algorithm_labels, rotation=90)
    ax.tick_params(axis="x", labelsize=5.6)
    for center, dataset in zip((1, 4, 7), ("MovieLens", "Amazon", "Last.FM"), strict=True):
        ax.text(
            center,
            -3.4,
            dataset,
            ha="center",
            va="bottom",
            fontsize=6.2,
            weight="bold",
        )
    run_labels = [
        (
            f"{row.prompt}/{MODEL_SHORT[row.model]}/{row.repetition}"
            + (" †" if row.known_semantic_case else "")
        )
        for row in runs.itertuples(index=False)
    ]
    ax.set_yticks(np.arange(len(runs)), run_labels)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(len(runs)):
        for j, value in enumerate(matrix[i]):
            color = "white" if value >= 17 else "#222222"
            ax.text(
                j, i, f"{int(value)}", ha="center", va="center", fontsize=5.2, color=color
            )

    ax.set_xticks(np.arange(-0.5, len(algorithm_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(runs), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", top=False, bottom=False, left=False)
    # Heavier separators between the three dataset blocks.
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="white", linewidth=2.4)

    colorbar = fig.colorbar(
        heatmap, ax=ax, orientation="horizontal", fraction=0.035, pad=0.03, aspect=38
    )
    colorbar.set_label("Requested entries (maximum 30 per cell)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.5)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.86, bottom=0.08)
    save_figure(fig, "coverage_heatmap")


# ---------------------------------------------------------------------------
# Sections 3.4 / 3.5: reviewer score and cost versus coherent coverage
# ---------------------------------------------------------------------------

def condition_legend(fig: plt.Figure) -> None:
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6,
            label=model,
        )
        for model, color in MODEL_COLORS.items()
    ]
    prompt_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor="black",
            markersize=6,
            label=prompt,
        )
        for prompt, marker in PROMPT_MARKERS.items()
    ]
    fig.legend(
        handles=model_handles + prompt_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
    )


def scatter_runs(ax: plt.Axes, runs: pd.DataFrame, x_field: str, y_field: str) -> None:
    for row in runs.itertuples(index=False):
        ax.scatter(
            getattr(row, x_field),
            getattr(row, y_field),
            marker=PROMPT_MARKERS[row.prompt],
            s=44,
            facecolor=MODEL_COLORS[row.model],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )


def plot_reviewer_vs_coverage(runs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    scatter_runs(ax, runs, "coverage_pct", "best_reviewer_score_pct")
    ax.plot(
        [0, 100],
        [0, 100],
        "--",
        color="#888888",
        linewidth=0.9,
        zorder=1,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Coherent requested coverage (%)")
    ax.set_ylabel("Best reviewer score (%)")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)

    annotations = {
        "V100_P0_GPT54_I05_R01": ("P0/GPT-5.4/R01\n90-entry ML partial", (9, 5)),
        "V100_P0_MINI_I05_R02": ("P0/Mini/R02\n90-entry ML partial", (9, -18)),
        "V100_P2_GPT54_I05_R02": ("P2/GPT-5.4/R02\n81.8%, 2/270", (-4, 13)),
    }
    for row in runs.itertuples(index=False):
        if row.run not in annotations:
            continue
        label, offset = annotations[row.run]
        ax.annotate(
            label,
            (row.coverage_pct, row.best_reviewer_score_pct),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )

    condition_legend(fig)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.13)
    save_figure(fig, "reviewer_vs_coverage")


def plot_cost_vs_coverage(runs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    scatter_runs(ax, runs, "cost_usd", "coverage_pct")
    ax.set_xlim(-0.15, 5.8)
    ax.set_ylim(-1, 36)
    ax.set_xlabel("Recorded API cost (USD)")
    ax.set_ylabel("Coherent requested coverage (%)")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)

    annotations = {
        "V100_P0_GPT54_I05_R01": ("P0/GPT-5.4/R01", (-4, 7)),
        "V100_P0_MINI_I05_R02": ("P0/Mini/R02", (6, 6)),
        "V100_P2_GPT54_I05_R02": ("P2/GPT-5.4/R02", (6, 6)),
    }
    for row in runs.itertuples(index=False):
        if row.run not in annotations:
            continue
        label, offset = annotations[row.run]
        ax.annotate(
            label,
            (row.cost_usd, row.coverage_pct),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )

    condition_legend(fig)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.13)
    save_figure(fig, "cost_vs_coverage")


# ---------------------------------------------------------------------------
# Section 3: Preregistered intervention (P2 vs P3 on GPT-5.4 Mini)
# ---------------------------------------------------------------------------

def plot_p2_p3_native_custom_coverage() -> None:
    """Per-run coverage by source for the ten preregistered runs.

    Shows the central behavioural result: the P3 contract condition produced no
    native framework metrics and only self-reported custom coverage, while
    neither condition reached the 270-entry valid target. Reads the
    preregistered-phase audit table; skipped if that phase is not present.
    """
    table = PREREG_AUDIT / "run_coverage.csv"
    if not table.exists():
        return
    df = pd.read_csv(table)
    df["rep"] = df["run"].str.extract(r"R0?(\d+)").astype(int)
    df = df.sort_values(["condition", "rep"]).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 2.3))

    native_color = MODEL_COLORS["GPT-5 Nano"]   # colour-blind-safe blue
    custom_color = MODEL_COLORS["GPT-5.4"]      # colour-blind-safe orange
    b_native = ax.bar(x - width / 2, df["best_native_coverage"], width,
                      label="Native (framework results.json)", color=native_color)
    b_custom = ax.bar(x + width / 2, df["best_custom_coverage"], width,
                      label="Custom (self-reported table)", color=custom_color)
    ax.bar_label(b_native, labels=[str(v) if v else "" for v in df["best_native_coverage"]],
                 fontsize=6, padding=1)
    ax.bar_label(b_custom, labels=[str(v) if v else "" for v in df["best_custom_coverage"]],
                 fontsize=6, padding=1)

    n_p2 = int((df["condition"] == "P2/Mini").sum())
    ax.axvline(n_p2 - 0.5, color="0.6", lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}" for r in df["rep"]], fontsize=7)
    ax.set_ylabel("Max coherent coverage\n(entries, of 270)", fontsize=8)
    top = max(95, float(df[["best_native_coverage", "best_custom_coverage"]].to_numpy().max()) * 1.18)
    ax.set_ylim(0, top)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    xt = ax.get_xaxis_transform()
    ax.text((n_p2 - 1) / 2, -0.20, "P2/Mini", ha="center", va="top",
            transform=xt, fontsize=8, fontweight="bold")
    ax.text(n_p2 + (n - n_p2 - 1) / 2, -0.20, "P3/Mini", ha="center", va="top",
            transform=xt, fontsize=8, fontweight="bold")

    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.20)
    save_figure(fig, "p2_p3_native_custom_coverage")


def main() -> None:
    runs = build_run_table()
    phases = build_phase_table()
    build_failure_table()
    plot_phase_comparison(phases)
    plot_merged_funnel(runs)
    plot_coverage_heatmap(runs)
    plot_reviewer_vs_coverage(runs)
    plot_cost_vs_coverage(runs)
    plot_p2_p3_native_custom_coverage()
    print(f"Wrote individual figures to {FIGURES}")


if __name__ == "__main__":
    main()
