"""Redraw Epoch AI's published training-power chart for the slide frame.

Run with Python + matplotlib from any working directory. The local source
snapshot preserves Epoch's point coordinates, frontier groups, and fitted line.
No classification or regression is recomputed. Source and license: see README.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/epoch_training_power_chart.json"
OUTPUT = ROOT / "public/figures/frontier_training_power.svg"
WIDTH_PX = 610  # Match .training-power-split in style.css.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=370,
                        help="SVG height in slide pixels (default: 370); fonts stay fixed.")
    parser.add_argument("--top", type=float, default=0.93,
                        help="Plot top as a fraction of canvas height (default: 0.93).")
    parser.add_argument("--bottom", type=float, default=0.12,
                        help="Plot bottom as a fraction of canvas height (default: 0.12).")
    parser.add_argument("--font-scale", type=float, default=1.15,
                        help="Multiply chart font sizes (default: 1.15).")
    args = parser.parse_args()
    if args.height <= 0 or args.font_scale <= 0 or not 0 < args.bottom < args.top < 1:
        parser.error("height and font-scale must be positive and 0 < bottom < top < 1")
    font = lambda size: size * args.font_scale
    chart = json.loads(SOURCE.read_text())
    frontier, remainder, trend = chart["objects"][:3]
    points = frontier["points"] + remainder["points"]
    with (ROOT / "data/epoch_training_power.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Check that the exact chart coordinates correspond to the CSV download.
    assert len(frontier["points"]) == 61
    assert len(points) == len(rows) == 495
    for point, row in zip(points, rows):
        assert " ".join(point["tooltipData"]["Model"].split()) == " ".join(row["Model"].split())
        assert point["tooltipData"]["Publication date"] == row["Publication date"]
        assert point["tooltipData"]["Training power draw (W)"] == row["Training power draw (W)"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font(8.5),
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#595959",
        "ytick.color": "#595959",
        "svg.fonttype": "none",
        "svg.hashsalt": "epoch-training-power",
    })
    # Width matches the CSS grid. The slide follows the SVG's intrinsic height.
    # Redraw at the chosen height rather than stretching text in the browser.
    fig, ax = plt.subplots(figsize=(WIDTH_PX / 100, args.height / 100), dpi=100)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=args.bottom, top=args.top)
    ax.set_yscale("log")
    ax.set_xlim(*chart["xAxis"]["lim"])
    ax.set_ylim(*chart["yAxis"]["lim"])
    ax.set_xticks(range(2012, 2027, 2))
    ax.set_yticks([10 ** exponent for exponent in range(2, 9)])
    ax.set_yticklabels(["100", "1k", "10k", "100k", "1M", "10M", "100M"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Publication date", labelpad=7)
    ax.set_ylabel("Training power draw (W)", labelpad=8)
    ax.tick_params(length=0, pad=5, labelsize=font(8))
    ax.set_axisbelow(True)
    ax.grid(color="#e5e5e5", linewidth=0.65)
    for spine in ax.spines.values():
        spine.set_color("#b9b9b9")
        spine.set_linewidth(0.7)

    # Draw the background population first; frontier points remain visible.
    background = ax.scatter(
        [p["x"] for p in remainder["points"]],
        [p["y"] for p in remainder["points"]],
        s=20, facecolor="#b9c3d5", edgecolor="#9faec7", alpha=0.48,
        linewidth=0.5, zorder=2, label="Other models",
    )
    foreground = ax.scatter(
        [p["x"] for p in frontier["points"]],
        [p["y"] for p in frontier["points"]],
        s=23, facecolor="#7696ba", edgecolor="#315fc1", alpha=0.9,
        linewidth=0.65, zorder=3, label="Frontier models (61)",
    )
    ax.plot(
        [p["x"] for p in trend["points"]],
        [p["y"] for p in trend["points"]],
        color="#b31b1b", linewidth=1.6, zorder=4,
    )
    ax.legend(
        handles=[foreground, background], loc="lower left",
        bbox_to_anchor=(0.02, 1.025), ncol=2, frameon=False,
        borderaxespad=0, handletextpad=0.5, columnspacing=1.8, fontsize=font(8),
    )
    by_name = {p["tooltipData"]["Model"].strip(): p for p in points}
    labels = [
        ("GPT-3 175B (davinci)", "GPT-3", (-12, 5)),
        ("GPT-4", "GPT-4", (-15, 9)),
        ("Llama 3.1-405B", "Llama 3.1", (-15, 26)),
        ("Grok 3", "Grok 3", (-10, 21)),
    ]
    for name, label, offset in labels:
        point = by_name[name]
        ax.annotate(
            label, xy=(point["x"], point["y"]), xytext=offset,
            textcoords="offset points", ha="right", va="center",
            fontsize=font(8), zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
            arrowprops={"arrowstyle": "-", "color": "#595959", "lw": 0.65},
        )
    source_annotation = chart["objects"][6]
    ax.annotate(
        "2.1× / year", xy=(source_annotation["targetX"], source_annotation["targetY"][0]),
        xytext=(2019.5, 2500), color="#b31b1b", fontsize=font(10), fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
        arrowprops={"arrowstyle": "->", "color": "#b31b1b", "lw": 1,
                    "connectionstyle": "arc3,rad=-0.25"}, zorder=6,
    )
    fig.savefig(OUTPUT, metadata={
        "Date": None,
        "Creator": "plot_training_power.py (matplotlib)",
        "Description": "Adapted from Epoch AI, Luke Emberson and Robi Rahman, CC BY. "
                       "https://epoch.ai/data-insights/power-usage-trend. "
                       "Original coordinates, frontier groups, and trend line preserved.",
    })
    plt.close(fig)
    print(f"Saved {OUTPUT} at {WIDTH_PX} × {args.height:g}px; "
          f"verified {len(points)} points and original trend coordinates.")


if __name__ == "__main__":
    main()
