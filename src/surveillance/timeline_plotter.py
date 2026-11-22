"""Surveillance Timeline Analytics and Plotter.

Generates temporal progression graphs:
- Top panel: Calibrated disease diagnosis & confidence over surveillance time.
- Bottom panel: Detected Tuta absoluta pest count over surveillance time.
"""

from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.surveillance.frame_analyzer import FrameAnalysisResult


def plot_surveillance_timeline(
    results: Union[List[FrameAnalysisResult], pd.DataFrame],
    save_path: Path | str = "artifacts/figures/video/disease_timeline.png",
) -> None:
    """Generate dual-axis surveillance timeline plot styled for dark dashboard theme."""
    if isinstance(results, list):
        data = []
        for r in results:
            data.append({
                "frame_idx": r.frame_idx,
                "timestamp_seconds": r.timestamp_seconds,
                "disease_label": r.disease_label,
                "disease_confidence": r.disease_confidence,
                "pest_count": r.pest_count,
            })
        df = pd.DataFrame(data)
    else:
        df = results.copy()

    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#151D18")

    times = df["timestamp_seconds"].values
    confs = df["disease_confidence"].values * 100.0
    pests = df["pest_count"].values

    # Panel 1: Disease Confidence Timeline
    ax1.set_facecolor("#101713")
    ax1.plot(times, confs, color="#35C979", linewidth=2.0, marker="o", markersize=3.5, label="Calibrated Confidence")
    ax1.axhline(50.0, color="#FF6464", linestyle="--", alpha=0.75, label="50% Confidence Threshold")
    ax1.set_ylabel("Confidence (%)", fontsize=11, fontweight="bold", color="#F1F5F2")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.45, color="#29372F")
    ax1.set_title("Tomato Surveillance: Disease Classification & Pest Detection Over Time", fontsize=13, fontweight="bold", color="#F1F5F2", pad=12)
    ax1.tick_params(colors="#A5B3A9")
    for spine in ax1.spines.values():
        spine.set_color("#29372F")

    # Annotate unique diagnoses along the timeline
    unique_labels = df["disease_label"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    label_color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    for lbl in unique_labels:
        mask = df["disease_label"] == lbl
        ax1.scatter(
            times[mask],
            confs[mask],
            color=label_color_map[lbl],
            label=f"Diagnosis: {lbl}",
            s=45,
            zorder=4,
            edgecolors="#151D18",
            linewidths=0.5,
        )

    ax1.legend(loc="upper right", framealpha=0.92, facecolor="#151D18", edgecolor="#29372F", labelcolor="#F1F5F2", fontsize=9)

    # Panel 2: Pest Count Timeline
    ax2.set_facecolor("#101713")
    ax2.step(times, pests, where="mid", color="#F2B84B", linewidth=2.2, label="Detected Tuta absoluta Count")
    ax2.fill_between(times, pests, step="mid", color="#F2B84B", alpha=0.20)
    ax2.set_xlabel("Surveillance Time (seconds)", fontsize=11, fontweight="bold", color="#F1F5F2")
    ax2.set_ylabel("Pest Lesion Count", fontsize=11, fontweight="bold", color="#F1F5F2")
    ax2.set_ylim(-0.5, max(max(pests) + 1.5, 3))
    ax2.grid(True, linestyle=":", alpha=0.45, color="#29372F")
    ax2.tick_params(colors="#A5B3A9")
    for spine in ax2.spines.values():
        spine.set_color("#29372F")
    ax2.legend(loc="upper right", framealpha=0.92, facecolor="#151D18", edgecolor="#29372F", labelcolor="#F1F5F2", fontsize=9)

    plt.tight_layout()
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
