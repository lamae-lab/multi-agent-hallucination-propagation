"""Statistical analysis for the hallucination resilience study.

Wilcoxon signed-rank tests, Cohen's d, Holm-Bonferroni correction,
bootstrap confidence intervals, and publication figure generation.
"""

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def wilcoxon_pairwise(family_metrics: dict[str, list[float]]) -> dict:
    """Pairwise Wilcoxon signed-rank tests between families.

    Args:
        family_metrics: {family_id: [metric_values]} — paired by index.

    Returns:
        Dict of {(fam1, fam2): {"statistic": W, "p_value": p, "n": n}}
    """
    results = {}
    families = sorted(family_metrics.keys())

    for f1, f2 in combinations(families, 2):
        vals1 = np.array(family_metrics[f1])
        vals2 = np.array(family_metrics[f2])
        n = min(len(vals1), len(vals2))
        vals1, vals2 = vals1[:n], vals2[:n]

        if n < 5 or np.all(vals1 == vals2):
            results[f"{f1}_vs_{f2}"] = {
                "statistic": 0,
                "p_value": 1.0,
                "n": n,
            }
            continue

        try:
            stat, p = stats.wilcoxon(vals1, vals2)
            results[f"{f1}_vs_{f2}"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "n": n,
            }
        except Exception:
            results[f"{f1}_vs_{f2}"] = {
                "statistic": 0,
                "p_value": 1.0,
                "n": n,
            }

    return results


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    a1, a2 = np.array(group1), np.array(group2)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a1, ddof=1) + (n2 - 1) * np.var(a2, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a1) - np.mean(a2)) / pooled_std)


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni multiple testing correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        corrected_p = p * (n - rank)
        corrected[orig_idx] = min(corrected_p, 1.0)

    # Enforce monotonicity
    for rank in range(len(indexed)):
        orig_idx = indexed[rank][0]
        if rank > 0:
            prev_idx = indexed[rank - 1][0]
            corrected[orig_idx] = max(corrected[orig_idx], corrected[prev_idx])

    return corrected


def bootstrap_ci(
    data: list[float], confidence: float = 0.95, n_boot: int = 1000
) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    if not data:
        return (0.0, 0.0)
    arr = np.array(data)
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    alpha = 1 - confidence
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def generate_figures(
    family_summaries: dict, output_dir: str, pairwise_results: dict | None = None
):
    """Generate publication figures.

    Args:
        family_summaries: {family_id: {mean_survival, mean_distance, mean_amplification, ...}}
        output_dir: Directory to save PNGs.
        pairwise_results: Optional Wilcoxon results for annotation.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    families = sorted(family_summaries.keys())
    if not families:
        return

    # 1. Survival x Amplification scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for fid in families:
        s = family_summaries[fid]
        ax.scatter(
            s.get("mean_survival", 0),
            s.get("mean_amplification", 0),
            s=120,
            label=fid.upper(),
            zorder=5,
        )
        ax.annotate(
            fid.upper(),
            (s.get("mean_survival", 0), s.get("mean_amplification", 0)),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
        )
    ax.set_xlabel("Mean Survival Rate")
    ax.set_ylabel("Mean Amplification Factor")
    ax.set_title("Survival Rate vs Amplification Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(output_dir, "scatter_survival_amplification.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Propagation Distance bars
    fig, ax = plt.subplots(figsize=(8, 5))
    distances = [family_summaries[f].get("mean_distance", 0) for f in families]
    ax.bar(
        [f.upper() for f in families],
        distances,
        color=plt.cm.Set2.colors[: len(families)],
    )
    ax.set_ylabel("Mean Propagation Distance")
    ax.set_title("Propagation Distance per Family")
    fig.savefig(
        os.path.join(output_dir, "bar_propagation_distance.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Family markers
    fig, ax = plt.subplots(figsize=(8, 5))
    marker_vals = [family_summaries[f].get("family_marker", 0) for f in families]
    marker_names = [family_summaries[f].get("family_marker_name", "") for f in families]
    bars = ax.bar(
        [f.upper() for f in families],
        marker_vals,
        color=plt.cm.Set3.colors[: len(families)],
    )
    for bar, name in zip(bars, marker_names):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            name,
            ha="center",
            fontsize=7,
            rotation=15,
        )
    ax.set_ylabel("Marker Value")
    ax.set_title("Family-Specific Behavioral Markers")
    fig.savefig(
        os.path.join(output_dir, "bar_family_markers.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 4. Pareto: resilience vs cost
    fig, ax = plt.subplots(figsize=(8, 6))
    for fid in families:
        s = family_summaries[fid]
        resilience = 1 - s.get("mean_survival", 0)  # lower survival = more resilient
        cost = s.get("mean_latency", s.get("mean_distance", 0))
        ax.scatter(cost, resilience, s=120, label=fid.upper(), zorder=5)
        ax.annotate(
            fid.upper(),
            (cost, resilience),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
        )
    ax.set_xlabel("Operational Cost (latency/distance)")
    ax.set_ylabel("Hallucination Resilience (1 - survival)")
    ax.set_title("Pareto: Resilience vs Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(output_dir, "pareto_resilience_cost.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
