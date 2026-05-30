"""
Visualize CVD Simulation
========================
Shows how the deuteranomaly simulation (cvd_simulation.py) transforms colors,
to verify it behaves correctly and to produce figures for the report/slides.

Produces two figures:
  1. cvd_severity_progression.png - a synthetic traffic scene seen at
     increasing CVD severity (0.0 normal -> 1.0 deuteranopia).
  2. cvd_color_patches.png        - pure color swatches (red/green/amber/etc.)
     before vs. after simulation, with the red-green confusion made obvious.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from cvd_simulation import CVDSimulation


def make_traffic_scene(size=256):
    """Build a simple synthetic street scene as a [3,H,W] tensor in [0,1].
    Sky, road, grass, a traffic-light box with a red lamp, and a green sign."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    # sky (light blue)
    img[: int(size * 0.55)] = [0.75, 0.85, 0.95]
    # grass strip (green)
    img[int(size * 0.55): int(size * 0.72)] = [0.45, 0.65, 0.30]
    # road (gray)
    img[int(size * 0.72):] = [0.45, 0.45, 0.47]

    # traffic-light housing (dark) on the left
    img[int(size*0.18):int(size*0.50), int(size*0.16):int(size*0.30)] = [0.15, 0.15, 0.17]
    # red lamp (lit)
    cy, cx, r = int(size*0.26), int(size*0.23), int(size*0.045)
    yy, xx = np.ogrid[:size, :size]
    red_lamp = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[red_lamp] = [0.95, 0.10, 0.10]

    # green road sign on the right
    img[int(size*0.20):int(size*0.40), int(size*0.62):int(size*0.84)] = [0.10, 0.65, 0.25]

    # a red car on the road (a distractor red, like the paper's billboard/brake light)
    img[int(size*0.78):int(size*0.92), int(size*0.55):int(size*0.80)] = [0.80, 0.12, 0.12]

    return torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]


def to_numpy(img_chw):
    return img_chw.clamp(0, 1).permute(1, 2, 0).numpy()


def figure_severity_progression(out_path="cvd_severity_progression.png"):
    scene = make_traffic_scene().unsqueeze(0)  # [1,3,H,W]
    severities = [0.0, 0.3, 0.6, 1.0]
    titles = ["Normal (0.0)", "Mild (0.3)", "Moderate (0.6)", "Deuteranopia (1.0)"]

    fig, axes = plt.subplots(1, len(severities), figsize=(4 * len(severities), 4.3))
    for ax, sev, title in zip(axes, severities, titles):
        sim = CVDSimulation(severity=sev, in01=True)
        with torch.no_grad():
            out = sim(scene)[0]
        ax.imshow(to_numpy(out))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(
        "Deuteranomaly simulation — traffic scene at increasing severity\n"
        "Note how the red lamp, red car, and green sign drift toward similar muddy tones",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def figure_color_patches(out_path="cvd_color_patches.png"):
    colors = {
        "Red":   [0.90, 0.10, 0.10],
        "Green": [0.10, 0.70, 0.20],
        "Amber": [0.95, 0.70, 0.05],
        "Blue":  [0.10, 0.30, 0.85],
    }
    sim = CVDSimulation(severity=1.0, in01=True)

    fig, axes = plt.subplots(2, len(colors), figsize=(3 * len(colors), 5.5))
    for j, (name, rgb) in enumerate(colors.items()):
        patch = torch.tensor(rgb).view(1, 3, 1, 1).expand(1, 3, 80, 80)
        with torch.no_grad():
            sim_patch = sim(patch)[0]
        axes[0, j].imshow(to_numpy(patch[0])); axes[0, j].axis("off")
        axes[0, j].set_title(f"{name}\n(normal)", fontsize=11)
        axes[1, j].imshow(to_numpy(sim_patch)); axes[1, j].axis("off")
        axes[1, j].set_title("deuteranopia", fontsize=11)

    fig.suptitle(
        "Pure colors: normal vision (top) vs deuteranopia (bottom)\n"
        "Red and green collapse toward the same yellow-ish tone — the core CVD problem",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    figure_severity_progression("/home/claude/cvd_severity_progression.png")
    figure_color_patches("/home/claude/cvd_color_patches.png")
    print("done")
