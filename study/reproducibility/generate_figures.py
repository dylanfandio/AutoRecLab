from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]  # package root (patched)
AUDIT = ROOT / "audit"
DERIVED = ROOT / "reproducibility" / "derived"
FIGURES = ROOT / "figures"

DERIVED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Okabe-Ito-inspired, colorblind-safe model colors.
MODEL_COLORS = {
    "GPT-5 Nano": "#0072B2",
    "GPT-5.4 Mini": "#009E73",
    "GPT-5.4": "#D55E00",
}
PROMPT_MARKERS = {"P0": "o", "P1": "s", "P2": "^"}

# Artifact-screened maximum requested native coverage for a coherent node or
# continued execution. GPT-5.4/P0/R01 is the documented continued execution
# whose five seed-specific native files jointly contain 90 entries. Mini/P0/R03
# has 108 native rows, but only its 54 NDCG rows use a requested metric name;
# its Recall rows were later copied and relabelled as Precision.
COHERENT_REQUESTED_COVERAGE = {
    "V100_P0_GPT54_I05_R01": 90,
    "V100_P0_GPT54_I05_R02": 1,
    "V100_P0_MINI_I05_R01": 18,
    "V100_P0_MINI_I05_R02": 90,
    "V100_P0_MINI_I05_R03": 54,
    "V100_P0_MINI_I05_R04": 10,
    "V100_P0_MINI_I05_R05": 2,
    "V100_P0_NANO_I05_R01": 18,
    "V100_P0_NANO_I05_R02": 0,
    "V100_P0_NANO_I05_R03": 0,
    "V100_P0_NANO_I05_R04": 0,
    "V100_P0_NANO_I05_R05": 5,
    "V100_P1_NANO_I05_R01": 5,
    "V100_P1_NANO_I05_R02": 0,
    "V100_P1_NANO_I05_R03": 0,
    "V100_P1_NANO_I05_R04": 0,
    "V100_P1_NANO_I05_R05": 1,
    "V100_P2_GPT54_I05_R01": 0,
    "V100_P2_GPT54_I05_R02": 2,
}

COVERAGE_BASIS = {
    "V100_P0_GPT54_I05_R01": "five seed-specific native files from one continued MovieLens execution",
    "V100_P0_MINI_I05_R02": "single 90-row native MovieLens result",
    "V100_P0_MINI_I05_R03": "54 requested NDCG rows; Recall rows excluded",
}

KNOWN_SEMANTIC_CASE = {
    "V100_P0_GPT54_I05_R01": "threshold and split deviation",
    "V100_P0_MINI_I05_R02": "threshold and split deviation",
    "V100_P0_MINI_I05_R03": "Recall relabelled as Precision",
    "V100_P0_NANO_I05_R05": "mock recommender output",
}

# Requested native entries in the strongest coherent node/continued execution,
# split by logical dataset and requested algorithm. Each cell has at most
# 5 seeds x 3 cutoffs x 2 metrics = 30 entries. Recall rows in Mini/P0/R03 are
# excluded because Recall was not requested.
COVERAGE_CELLS = {
    "V100_P0_GPT54_I05_R01": [30, 30, 30, 0, 0, 0, 0, 0, 0],
    "V100_P0_GPT54_I05_R02": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "V100_P0_MINI_I05_R01": [6, 6, 6, 0, 0, 0, 0, 0, 0],
    "V100_P0_MINI_I05_R02": [30, 30, 30, 0, 0, 0, 0, 0, 0],
    "V100_P0_MINI_I05_R03": [15, 15, 15, 3, 3, 3, 0, 0, 0],
    "V100_P0_MINI_I05_R04": [0, 10, 0, 0, 0, 0, 0, 0, 0],
    "V100_P0_MINI_I05_R05": [0, 0, 2, 0, 0, 0, 0, 0, 0],
    "V100_P0_NANO_I05_R01": [6, 6, 6, 0, 0, 0, 0, 0, 0],
    "V100_P0_NANO_I05_R02": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P0_NANO_I05_R03": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P0_NANO_I05_R04": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P0_NANO_I05_R05": [5, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P1_NANO_I05_R01": [5, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P1_NANO_I05_R02": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P1_NANO_I05_R03": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P1_NANO_I05_R04": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P1_NANO_I05_R05": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P2_GPT54_I05_R01": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V100_P2_GPT54_I05_R02": [0, 0, 2, 0, 0, 0, 0, 0, 0],
}

COVERAGE_COLUMNS = [
    "ml_als", "ml_itemknn", "ml_pop",
    "amazon_als", "amazon_itemknn", "amazon_pop",
    "lastfm_als", "lastfm_itemknn", "lastfm_pop",
]


def parse_condition(run: str) -> tuple[str, str, str]:
    parts = run.split("_")
    prompt = parts[1]
    model_code = parts[2]
    repetition = parts[-1]
    model = {
        "NANO": "GPT-5 Nano",
        "MINI": "GPT-5.4 Mini",
        "GPT54": "GPT-5.4",
    }[model_code]
    return prompt, model, repetition


def build_run_table() -> pd.DataFrame:
    summary = pd.read_csv(AUDIT / "run_summary.csv")
    production = summary[
        summary["run"].str.match(r"^V100_P[012]_(NANO|MINI|GPT54)_I05_R\d\d$")
        & (summary["cost_usd"] > 0)
    ].copy()

    records: list[dict] = []
    for row in production.itertuples(index=False):
        prompt, model, repetition = parse_condition(row.run)
        coverage = COHERENT_REQUESTED_COVERAGE[row.run]
        record = {
                "run": row.run,
                "short_label": f"{prompt}-{model.replace('GPT-5', '').strip()}-{repetition}",
                "prompt": prompt,
                "model": model,
                "repetition": repetition,
                "api_calls": int(row.api_calls),
                "cost_usd": float(row.cost_usd),
                "cost_log_minutes": float(row.elapsed_minutes_from_cost_log),
                "best_reviewer_score_pct": float(row.best_reviewer_score_pct),
                "final_requirement_count": int(row.final_requirements),
                "generated_nodes": int(row.generated_code_files),
                "nodes_without_recorded_exception": int(row.execution_success_count),
                "nodes_with_recorded_exception": int(row.execution_failure_count),
                "has_exception_free_node": int(row.execution_success_count) > 0,
                "has_native_results_file": int(row.results_json_files) > 0,
                "requested_native_entries": coverage,
                "coverage_pct": 100.0 * coverage / 270.0,
                "has_requested_native_metric": coverage > 0,
                "has_90_entry_movielens_block": row.run
                in {"V100_P0_GPT54_I05_R01", "V100_P0_MINI_I05_R02"},
                "complete_270": False,
                "is_satisfactory": int(row.satisfactory_true_count) > 0,
                "known_semantic_case": KNOWN_SEMANTIC_CASE.get(row.run, ""),
                "coverage_basis": COVERAGE_BASIS.get(
                    row.run, "maximum screened requested coverage in one coherent native artifact"
                ),
            }
        record.update(dict(zip(COVERAGE_COLUMNS, COVERAGE_CELLS[row.run], strict=True)))
        assert sum(COVERAGE_CELLS[row.run]) == coverage
        records.append(record)

    result = pd.DataFrame(records).sort_values(
        ["prompt", "model", "repetition"], kind="stable"
    )
    assert len(result) == 19
    assert result["generated_nodes"].sum() == 188
    assert result["nodes_without_recorded_exception"].sum() == 78
    assert result["nodes_with_recorded_exception"].sum() == 110
    assert result["has_exception_free_node"].sum() == 17
    assert result["has_native_results_file"].sum() == 14
    assert result["has_requested_native_metric"].sum() == 12
    assert result["has_90_entry_movielens_block"].sum() == 2
    assert result["complete_270"].sum() == 0
    assert result["is_satisfactory"].sum() == 0
    assert result["api_calls"].sum() == 2832
    assert math.isclose(result["cost_usd"].sum(), 23.573121, abs_tol=1e-6)

    result.to_csv(DERIVED / "run_level_results.csv", index=False)
    reviewer_coverage_r = result[
        ["best_reviewer_score_pct", "requested_native_entries"]
    ].corr().iloc[0, 1]
    assert math.isclose(reviewer_coverage_r, 0.3371002534, abs_tol=1e-9)
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return 100 * (center - half), 100 * (center + half)


def build_phase_table() -> pd.DataFrame:
    rows = [
        {
            "phase": "Original study",
            "successes": 3,
            "runs": 4,
            "evidence": "Original reported",
            "success_definition": "Original run-level completion",
        },
        {
            "phase": "Historical detailed prompt",
            "successes": 0,
            "runs": 33,
            "evidence": "Historical reported",
            "success_definition": "Recorded run success",
        },
        {
            "phase": "Historical simplified prompt",
            "successes": 1,
            "runs": 3,
            "evidence": "Historical reported",
            "success_definition": "Recorded run success",
        },
        {
            "phase": "Retained v1.0.0 follow-up",
            "successes": 0,
            "runs": 19,
            "evidence": "Artifact verified",
            "success_definition": "Complete semantic validity",
        },
    ]
    result = pd.DataFrame(rows)
    result["rate_pct"] = 100 * result["successes"] / result["runs"]
    intervals = [
        wilson_interval(int(row.successes), int(row.runs))
        for row in result.itertuples(index=False)
    ]
    result["wilson_low_pct"] = [item[0] for item in intervals]
    result["wilson_high_pct"] = [item[1] for item in intervals]
    result.to_csv(DERIVED / "phase_comparison.csv", index=False)
    return result


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_comparison(phases: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
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
            markersize=7,
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
        label_x = min(row.rate_pct + high_error + 2.2, 96)
        ax.text(
            label_x,
            y,
            f"{row.successes}/{row.runs} ({row.rate_pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    ax.set_yticks(positions, phases["phase"])
    evidence_suffix = {
        "Original reported": "original report",
        "Historical reported": "historical record",
        "Artifact verified": "artifact verified",
    }
    phase_labels = [
        f"{row.phase}\n({evidence_suffix[row.evidence]})"
        for row in phases.itertuples(index=False)
    ]
    ax.set_yticks(positions, phase_labels)
    ax.set_xlim(-2, 108)
    ax.set_xlabel("Run-level completion rate (%)")
    ax.set_title("Reported and artifact-verified replication outcomes", loc="left")
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
            markersize=6,
            label=label,
        )
        for label in evidence_colors
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.62, -0.16),
        frameon=False,
        ncol=3,
    )
    fig.text(
        0.01,
        -0.01,
        "Points show observed rates; horizontal lines show 95% Wilson intervals. "
        "Definitions differ by evidence tier and must not be interpreted as a controlled comparison. "
        "Four develop-branch team runs are omitted because equivalent completion evidence was unavailable.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.29, right=0.98, top=0.88, bottom=0.34)
    save_figure(fig, "phase_comparison")


def stage_axis(
    ax: plt.Axes,
    labels: list[str],
    counts: list[int],
    denominator: int,
    color: str,
    title: str,
) -> None:
    x = np.arange(len(counts))
    sizes = 80 + 520 * np.asarray(counts) / max(denominator, 1)
    ax.plot(x, counts, color=color, linewidth=1.8, zorder=1)
    ax.scatter(x, counts, s=sizes, color=color, edgecolor="white", linewidth=1.0, zorder=2)
    for xx, value in zip(x, counts, strict=True):
        ax.text(xx, value, str(value), ha="center", va="center", color="white", weight="bold")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylim(-max(denominator * 0.08, 3), denominator * 1.12)
    ax.set_ylabel("Count")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_completion_dashboard(runs: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 6.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 2.1],
        width_ratios=[0.78, 1.22],
        hspace=0.62,
        wspace=0.32,
    )
    node_ax = fig.add_subplot(grid[0, 0])
    search_ax = fig.add_subplot(grid[0, 1])
    heat_ax = fig.add_subplot(grid[1, :])

    stage_axis(
        node_ax,
        ["Generated", "No recorded\nexception", "No mock/fallback\nmarker"],
        [188, 78, 54],
        188,
        "#0072B2",
        "(a) Generated-node stages",
    )
    stage_axis(
        search_ax,
        ["Code", "No-exception\nnode", "Native\nartifact", "Requested\nmetric", "90-entry\nML", "Complete\n270"],
        [19, 17, 14, 12, 2, 0],
        19,
        "#009E73",
        "(b) Search-level stages",
    )

    matrix = runs[COVERAGE_COLUMNS].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#F2F2F2", "#9ECAE1", "#0072B2"]
    )
    heatmap = heat_ax.imshow(matrix, cmap=cmap, vmin=0, vmax=30, aspect="auto")
    column_labels = [
        "ML\nALS", "ML\nItemKNN", "ML\nPop",
        "Amazon\nALS", "Amazon\nItemKNN", "Amazon\nPop",
        "Last.FM\nALS", "Last.FM\nItemKNN", "Last.FM\nPop",
    ]
    heat_ax.set_xticks(np.arange(len(column_labels)), column_labels)
    run_labels = [
        (
            f"{row.prompt}/{row.model.replace('GPT-5', '').strip()}/{row.repetition}"
            + (" †" if row.known_semantic_case else "")
        )
        for row in runs.itertuples(index=False)
    ]
    heat_ax.set_yticks(np.arange(len(runs)), run_labels)
    heat_ax.set_title(
        "(c) Requested native coverage in the strongest coherent node", loc="left"
    )
    heat_ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    for i, row in enumerate(runs.itertuples(index=False)):
        for j, value in enumerate(matrix[i]):
            color = "white" if value >= 17 else "#222222"
            heat_ax.text(
                j, i, f"{int(value)}", ha="center", va="center", fontsize=6.2, color=color
            )

    heat_ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(runs), 1), minor=True)
    heat_ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    heat_ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(heatmap, ax=heat_ax, fraction=0.018, pad=0.018)
    colorbar.set_label("Requested entries (maximum 30 per cell)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)

    fig.text(
        0.01,
        0.01,
        "Node and search counts use different denominators. † marks runs with a documented semantic defect; "
        "unmarked coverage is not automatically semantically valid. The two 90-entry MovieLens blocks failed "
        "protocol checks and later datasets.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.08)
    save_figure(fig, "completion_and_coverage")


def plot_reviewer_cost(runs: pd.DataFrame) -> None:
    fig, (review_ax, cost_ax) = plt.subplots(1, 2, figsize=(7.25, 3.25))

    for row in runs.itertuples(index=False):
        common = {
            "marker": PROMPT_MARKERS[row.prompt],
            "s": 44,
            "facecolor": MODEL_COLORS[row.model],
            "edgecolor": "black",
            "linewidth": 0.55,
            "alpha": 0.9,
            "zorder": 3,
        }
        review_ax.scatter(row.coverage_pct, row.best_reviewer_score_pct, **common)
        cost_ax.scatter(row.cost_usd, row.coverage_pct, **common)

    review_ax.plot([0, 100], [0, 100], "--", color="#888888", linewidth=0.9, label="Equal score and coverage")
    review_ax.set_xlim(-2, 102)
    review_ax.set_ylim(0, 100)
    review_ax.set_xlabel("Coherent requested coverage (%)")
    review_ax.set_ylabel("Best reviewer score (%)")
    review_ax.set_title("(a) Self-review versus artifact coverage", loc="left")
    review_ax.grid(color="#E5E5E5", linewidth=0.6)

    cost_ax.set_xlim(-0.15, 5.8)
    cost_ax.set_ylim(-1, 36)
    cost_ax.set_xlabel("Recorded API cost (USD)")
    cost_ax.set_ylabel("Coherent requested coverage (%)")
    cost_ax.set_title("(b) Cost versus artifact coverage", loc="left")
    cost_ax.grid(color="#E5E5E5", linewidth=0.6)

    annotations = {
        "V100_P0_GPT54_I05_R01": ("P0/GPT-5.4/R01\n90-entry ML partial", (-8, -25)),
        "V100_P0_MINI_I05_R02": ("P0/Mini/R02\n90-entry ML partial", (7, 7)),
        "V100_P2_GPT54_I05_R02": ("P2/GPT-5.4/R02\n81.8%, 2/270", (7, -8)),
    }
    for row in runs.itertuples(index=False):
        if row.run not in annotations:
            continue
        label, offset = annotations[row.run]
        review_ax.annotate(
            label,
            (row.coverage_pct, row.best_reviewer_score_pct),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.4,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )
        cost_ax.annotate(
            label.split("\n")[0],
            (row.cost_usd, row.coverage_pct),
            xytext=(7, 7) if row.run == "V100_P2_GPT54_I05_R02" else offset,
            textcoords="offset points",
            fontsize=6.4,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )

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
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=False,
    )
    fig.text(
        0.01,
        0.005,
        "Coverage counts only requested native entries from one coherent node or continued execution. "
        "Semantic validity is assessed separately; no point represents a complete valid run.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.22, wspace=0.32)
    save_figure(fig, "reviewer_cost_coverage")


def build_failure_table() -> pd.DataFrame:
    rows = [
        ["Technical", "Split/validation mismatch", "Crash or changed 80/20 protocol", "P0/Mini/R02", "Split contract"],
        ["Technical", "Schema/loader mismatch", "Dataset could not be loaded", "P0/Mini/R04", "Schema preflight"],
        ["Technical", "Framework API mismatch", "Import or method failure", "P0/Nano/R02", "API smoke test"],
        ["Technical", "Infeasible prediction allocation", "Approximately 3.64 GiB allocation failure", "Excluded smoke test", "Resource-bound preflight"],
        ["Semantic", "rating >= 3 instead of > 3", "Different interaction data", "P0/Mini/R02", "Count invariant"],
        ["Semantic", "Recall relabelled Precision", "Incorrect metric identity", "P0/Mini/R03", "Metric provenance"],
        ["Semantic", "Mock recommender/fallback", "Invalid algorithm comparison", "P0/Nano/R05", "Class identity"],
        ["Orchestration", "Weaker final-node selection", "Better partial evidence omitted", "Multiple searches", "Evidence-based selection"],
    ]
    result = pd.DataFrame(
        rows,
        columns=["class", "mechanism", "consequence", "representative_case", "proposed_check"],
    )
    result.to_csv(DERIVED / "failure_cases.csv", index=False)
    return result


def main() -> None:
    runs = build_run_table()
    phases = build_phase_table()
    build_failure_table()
    plot_phase_comparison(phases)
    plot_completion_dashboard(runs)
    plot_reviewer_cost(runs)
    print(f"Wrote derived tables to {DERIVED}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
