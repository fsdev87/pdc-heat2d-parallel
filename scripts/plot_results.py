"""
plot_results.py — Generate all performance charts for PDC report
CS-3006 FAST-NUCES Spring 2026
Run: python3 scripts/plot_results.py
Outputs: results/plots/*.png (300 DPI, ready for LaTeX)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

# ============================================================
# RAW DATA (from benchmark logs)
# ============================================================
T_SEQ = 1613.468692   # V0 sequential baseline

data = {
    "v1": {1: 1895.363040, 2: 1014.575701, 4: 570.752275,  8: 345.157653},
    "v2": {1: 1841.980108, 2: 993.781398,  4: 556.368231,  8: 332.523773},
    "v3": {                                4: 542.065251,  8: 322.506545},
    "v4": {
        "4p2t": 728.664691,   # NP=4 THREADS=2 (8 total)
        "4p4t": 888.367550,   # NP=4 THREADS=4 (16 total)
        "8p2t": 493.620821,   # NP=8 THREADS=2 (16 total)
    }
}

def speedup(t): return T_SEQ / t
def efficiency(t, p): return speedup(t) / p

# ============================================================
# STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "v1": "#2196F3",   # blue
    "v2": "#4CAF50",   # green
    "v3": "#FF9800",   # orange
    "v4": "#9C27B0",   # purple
    "ideal": "#F44336" # red
}

# ============================================================
# GRAPH 1 — Speedup vs Process Count (V1, V2, V3 + Ideal)
# ============================================================
procs_v1v2 = [1, 2, 4, 8]
procs_v3   = [4, 8]

sp_v1 = [speedup(data["v1"][p]) for p in procs_v1v2]
sp_v2 = [speedup(data["v2"][p]) for p in procs_v1v2]
sp_v3 = [speedup(data["v3"][p]) for p in procs_v3]
sp_ideal = [1, 2, 4, 8]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(procs_v1v2, sp_v1,   "o-",  color=COLORS["v1"],    linewidth=2, markersize=7, label="V1: MPI Blocking (1D)")
ax.plot(procs_v1v2, sp_v2,   "s-",  color=COLORS["v2"],    linewidth=2, markersize=7, label="V2: MPI Non-Blocking (1D)")
ax.plot(procs_v3,   sp_v3,   "^-",  color=COLORS["v3"],    linewidth=2, markersize=7, label="V3: MPI 2D Cartesian")
ax.plot(procs_v1v2, sp_ideal, "--", color=COLORS["ideal"],  linewidth=1.5, alpha=0.7,  label="Ideal Linear Speedup")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Speedup (T$_{seq}$ / T$_{parallel}$)")
ax.set_title("Speedup vs. Process Count — N=1024, 2,000,000 Iterations")
ax.set_xticks([1, 2, 4, 8])
ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xlim(0.5, 9)
ax.set_ylim(0, 9)
ax.legend(loc="upper left")

# Annotate actual values
for p, s in zip(procs_v1v2, sp_v1):
    ax.annotate(f"{s:.2f}x", (p, s), textcoords="offset points", xytext=(5, -14), fontsize=9, color=COLORS["v1"])
for p, s in zip(procs_v1v2, sp_v2):
    ax.annotate(f"{s:.2f}x", (p, s), textcoords="offset points", xytext=(5, 4),   fontsize=9, color=COLORS["v2"])
for p, s in zip(procs_v3, sp_v3):
    ax.annotate(f"{s:.2f}x", (p, s), textcoords="offset points", xytext=(5, 4),   fontsize=9, color=COLORS["v3"])

fig.tight_layout()
fig.savefig("results/plots/graph1_speedup.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: graph1_speedup.png")

# ============================================================
# GRAPH 2 — Execution Time Comparison (grouped bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

configs = ["V0\n(seq)", "V1\nNP=1", "V1\nNP=2", "V1\nNP=4", "V1\nNP=8",
           "V2\nNP=2", "V2\nNP=4", "V2\nNP=8",
           "V3\nNP=4", "V3\nNP=8"]

times = [
    T_SEQ,
    data["v1"][1], data["v1"][2], data["v1"][4], data["v1"][8],
    data["v2"][2],  data["v2"][4],  data["v2"][8],
    data["v3"][4],  data["v3"][8],
]

bar_colors = (
    ["#757575"] +                          # V0 gray
    [COLORS["v1"]] * 4 +                   # V1 blue
    [COLORS["v2"]] * 3 +                   # V2 green
    [COLORS["v3"]] * 2                     # V3 orange
)

bars = ax.bar(configs, times, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)

# Value labels on bars
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f"{t:.0f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_ylabel("Wall Clock Time (seconds)")
ax.set_title("Execution Time — All Versions | N=1024, 2,000,000 Iterations")
ax.set_ylim(0, max(times) * 1.15)

legend_patches = [
    mpatches.Patch(color="#757575",      label="V0: Sequential"),
    mpatches.Patch(color=COLORS["v1"],   label="V1: MPI Blocking"),
    mpatches.Patch(color=COLORS["v2"],   label="V2: MPI Non-Blocking"),
    mpatches.Patch(color=COLORS["v3"],   label="V3: MPI 2D Cartesian"),
]
ax.legend(handles=legend_patches, loc="upper right")

fig.tight_layout()
fig.savefig("results/plots/graph2_execution_time.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: graph2_execution_time.png")

# ============================================================
# GRAPH 3 — Parallel Efficiency vs Process Count
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

eff_v1 = [efficiency(data["v1"][p], p) for p in procs_v1v2]
eff_v2 = [efficiency(data["v2"][p], p) for p in procs_v1v2]
eff_v3 = [efficiency(data["v3"][p], p) for p in procs_v3]
eff_ideal = [1.0, 1.0, 1.0, 1.0]

ax.plot(procs_v1v2, eff_v1,   "o-",  color=COLORS["v1"],   linewidth=2, markersize=7, label="V1: MPI Blocking (1D)")
ax.plot(procs_v1v2, eff_v2,   "s-",  color=COLORS["v2"],   linewidth=2, markersize=7, label="V2: MPI Non-Blocking (1D)")
ax.plot(procs_v3,   eff_v3,   "^-",  color=COLORS["v3"],   linewidth=2, markersize=7, label="V3: MPI 2D Cartesian")
ax.plot(procs_v1v2, eff_ideal, "--", color=COLORS["ideal"], linewidth=1.5, alpha=0.7,  label="Ideal Efficiency (1.0)")

ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(8.1, 0.51, "50%", fontsize=9, color="gray")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Parallel Efficiency (Speedup / P)")
ax.set_title("Parallel Efficiency vs. Process Count — N=1024")
ax.set_xticks([1, 2, 4, 8])
ax.set_ylim(0, 1.2)
ax.set_xlim(0.5, 9)
ax.legend(loc="upper right")

for p, e in zip(procs_v1v2, eff_v1):
    ax.annotate(f"{e:.2f}", (p, e), textcoords="offset points", xytext=(5, -14), fontsize=9, color=COLORS["v1"])
for p, e in zip(procs_v1v2, eff_v2):
    ax.annotate(f"{e:.2f}", (p, e), textcoords="offset points", xytext=(5, 4),   fontsize=9, color=COLORS["v2"])

fig.tight_layout()
fig.savefig("results/plots/graph3_efficiency.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: graph3_efficiency.png")

# ============================================================
# GRAPH 4 — V1 vs V2: Blocking vs Non-Blocking (latency hiding)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

width = 0.35
x = np.array([1, 2, 4, 8])
x_pos = np.arange(len(x))

t_v1 = [data["v1"][p] for p in x]
t_v2 = [data["v2"][p] for p in x]
improvement = [(t1 - t2) / t1 * 100 for t1, t2 in zip(t_v1, t_v2)]

bars1 = ax.bar(x_pos - width/2, t_v1, width, label="V1: Blocking MPI_Sendrecv",    color=COLORS["v1"], alpha=0.85)
bars2 = ax.bar(x_pos + width/2, t_v2, width, label="V2: Non-Blocking MPI_Isend/Irecv", color=COLORS["v2"], alpha=0.85)

# Improvement percentage arrow annotations
for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvement)):
    ax.annotate(f"−{imp:.1f}%",
                xy=(b2.get_x() + b2.get_width()/2, b2.get_height()),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, color="#2E7D32", fontweight="bold")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Wall Clock Time (seconds)")
ax.set_title("V1 vs V2: Blocking vs Non-Blocking Communication\nN=1024, 2,000,000 Iterations")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"NP={p}" for p in x])
ax.legend()

fig.tight_layout()
fig.savefig("results/plots/graph4_v1_vs_v2.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: graph4_v1_vs_v2.png")

# ============================================================
# GRAPH 5 — V4 Hybrid Configurations vs Pure MPI
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

labels = [
    "V3\nNP=8 T=1\n(best pure MPI)",
    "V4\nNP=4 T=2\n(8 total)",
    "V4\nNP=8 T=2\n(16 total)",
    "V4\nNP=4 T=4\n(16 total)",
]
times_v4 = [
    data["v3"][8],
    data["v4"]["4p2t"],
    data["v4"]["8p2t"],
    data["v4"]["4p4t"],
]
bar_cols = [COLORS["v3"], COLORS["v4"], COLORS["v4"], COLORS["v4"]]
alphas   = [0.9, 0.9, 0.7, 0.5]

bars = []
for i, (lbl, t, c, a) in enumerate(zip(labels, times_v4, bar_cols, alphas)):
    b = ax.bar(i, t, color=c, alpha=a, edgecolor="white", linewidth=0.5, width=0.5)
    bars.append(b)

for i, (bar_group, t) in enumerate(zip(bars, times_v4)):
    b = bar_group[0]
    sp = speedup(t)
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
            f"{t:.0f}s\n({sp:.2f}x)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Wall Clock Time (seconds)")
ax.set_title("V4 Hybrid MPI+OpenMP Configurations vs Best Pure MPI\nN=1024, 2,000,000 Iterations")
ax.set_ylim(0, max(times_v4) * 1.2)

legend_patches = [
    mpatches.Patch(color=COLORS["v3"], label="V3: Pure MPI 2D (reference)"),
    mpatches.Patch(color=COLORS["v4"], label="V4: Hybrid MPI+OpenMP"),
]
ax.legend(handles=legend_patches)

# Add note about oversubscription
ax.text(0.98, 0.05,
        "Note: All runs on 4-core machine.\nNP×T > 4 = oversubscribed (context switching).",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

fig.tight_layout()
fig.savefig("results/plots/graph5_v4_hybrid.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: graph5_v4_hybrid.png")

# ============================================================
# SUMMARY TABLE — print to console for verification
# ============================================================
print("\n" + "="*65)
print(f"{'Config':<22} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
print("="*65)
print(f"{'V0 Sequential':<22} {T_SEQ:>10.2f} {'1.00x':>10} {'1.000':>12}")
for v, pdict in [("v1", data["v1"]), ("v2", data["v2"])]:
    for p, t in sorted(pdict.items()):
        sp = speedup(t)
        ef = efficiency(t, p)
        print(f"{'V'+v[1:]+' NP='+str(p):<22} {t:>10.2f} {sp:>9.2f}x {ef:>12.3f}")
for p, t in sorted(data["v3"].items()):
    sp = speedup(t)
    ef = efficiency(t, p)
    print(f"{'V3 NP='+str(p):<22} {t:>10.2f} {sp:>9.2f}x {ef:>12.3f}")
for k, t in data["v4"].items():
    lbl = {"4p2t":"V4 NP=4 T=2","4p4t":"V4 NP=4 T=4","8p2t":"V4 NP=8 T=2"}[k]
    sp = speedup(t)
    print(f"{lbl:<22} {t:>10.2f} {sp:>9.2f}x {'N/A':>12}")
print("="*65)
print("\nAll graphs saved to results/plots/")
