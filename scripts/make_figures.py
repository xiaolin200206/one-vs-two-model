#!/usr/bin/env python3
"""
make_figures.py — regenerate every figure in the paper and supplement from results/.

Run from the repository root:

    python scripts/make_figures.py

No value is hard-coded; all four figures are computed from the released trial records.
Requires: pandas, numpy, matplotlib.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

D = "results/cachebench_{}.csv"
cfg = ["combined_640", "separate_640", "combined_1280", "separate_1280"]
NICE = {"combined_640": "Unified\n@640", "separate_640": "Separate\n@640",
        "combined_1280": "Unified\n@1280", "separate_1280": "Separate\n@1280"}
LINE = {"combined_640": "Unified @640", "separate_640": "Separate @640",
        "combined_1280": "Unified @1280", "separate_1280": "Separate @1280"}
COL = {"combined_640": "#3B8FD4", "separate_640": "#E8862A",
       "combined_1280": "#1B4F86", "separate_1280": "#C1441A"}
f = {k: pd.read_csv(D.format(k)) for k in cfg}
labels = [NICE[k] for k in cfg]
colors = [COL[k] for k in cfg]

def mu(k, c, s=1.0): return (f[k][c]*s).mean()
def sd(k, c, s=1.0): return (f[k][c]*s).std(ddof=1)

# ---------------- FIG 1: system cost matrix ----------------
panels = [("(a) Inference latency", "lat_mean", 1.0, "Latency (ms / image)", "{:.1f}"),
          ("(b) Last-level cache read misses", "ll_cache_miss_rd", 1e-9, "LL cache read misses ($\\times10^9$)", "{:.2f}"),
          ("(c) Energy per image", "energy_per_img_j", 1.0, "Gross energy (J / image)", "{:.2f}"),
          ("(d) Peak CPU temperature", "peak_temp", 1.0, "Peak temperature (\u00b0C)", "{:.1f}")]
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
for ax, (title, c, s, ylab, fmt) in zip(axes.ravel(), panels):
    m = [mu(k, c, s) for k in cfg]; e = [sd(k, c, s) for k in cfg]
    bars = ax.bar(labels, m, yerr=e, capsize=4, color=colors, edgecolor="black", linewidth=0.8,
                  error_kw=dict(ecolor="black", elinewidth=1.1, capthick=1.1))
    for b, v, u in zip(bars, m, e):
        ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, v+u), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    if c == "peak_temp":
        hot = max(f[k].peak_temp.max() for k in cfg)
        ax.axhspan(80, 85, color="#D62728", alpha=0.10, zorder=0)
        ax.axhline(80, color="#D62728", ls="--", lw=1.3, label="Throttling onset (80 \u00b0C)")
        ax.axhline(hot, color="#555555", ls=":", lw=1.2, label=f"Hottest single trial ({hot:.1f} \u00b0C)")
        ax.set_ylim(70, 86); ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.set_ylim(0, max(np.array(m)+np.array(e))*1.18)
fig.suptitle("System cost matrix: unified vs. separate $\\times$ 640 vs. 1280\n"
             "(mean $\\pm$ SD over 8 trials; Raspberry Pi 5, performance governor, on battery)",
             fontsize=12, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig_system_cost_matrix.png"); plt.close(fig)

# ---------------- FIG 2: savings summary ----------------
groups = [("Latency\nReduction", "lat_mean", 1.0),
          ("LL Cache Read-Miss\nReduction", "ll_cache_miss_rd", 1e-9),
          ("Energy\nReduction", "energy_per_img_j", 1.0)]
def red(c, s, r):
    u, sp = mu(f"combined_{r}", c, s), mu(f"separate_{r}", c, s)
    return (sp-u)/sp*100
x = np.arange(3); w = 0.34
v6  = [red(c, s, "640")  for _, c, s in groups]
v12 = [red(c, s, "1280") for _, c, s in groups]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for off, v, col, lab in [(-w/2, v6, "#3B8FD4", "@640"), (w/2, v12, "#1B4F86", "@1280")]:
    bars = ax.bar(x+off, v, w, label=lab, color=col, edgecolor="black", linewidth=0.8)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([g for g, _, _ in groups])
ax.set_ylabel("Reduction vs. separate configuration (%)"); ax.set_ylim(0, 36)
ax.set_title("System cost reduction of the unified architecture\n(8 trials per configuration; Raspberry Pi 5, on battery)",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
fig.savefig("figures/fig_savings_summary.png"); plt.close(fig)

# ---------------- FIG 3: per-trial traces ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
t = np.arange(1, 9)
for k in cfg:
    a1.plot(t, f[k].lat_mean, "o-", color=COL[k], lw=1.6, ms=5, label=LINE[k])
a1.set_xlabel("Trial"); a1.set_ylabel("Mean latency (ms / image)")
a1.set_title("(a) Per-trial mean latency", fontsize=10.5, fontweight="bold")
a1.set_xticks(t); a1.set_yscale("log")
a1.set_yticks([400, 600, 1000, 1500, 2500])
a1.get_yaxis().set_major_formatter(mtick.ScalarFormatter())
a1.grid(ls=":", lw=0.6, alpha=0.6); a1.set_axisbelow(True)
a1.legend(frameon=False, fontsize=8.5, ncol=2, loc="center left")

for k in cfg:
    m = f[k].lat_mean.mean()
    dev = (f[k].lat_mean - m)/m*100
    a2.plot(t, dev, "o-", color=COL[k], lw=1.6, ms=5,
            label=f"{LINE[k]}  (SD = {f[k].lat_mean.std(ddof=1)/m*100:.2f}%)")
a2.axhline(0, color="black", lw=0.8)
a2.axhspan(-0.6, 0.6, color="#999999", alpha=0.13, zorder=0)
a2.set_xlabel("Trial"); a2.set_ylabel("Deviation from configuration mean (%)")
a2.set_title("(b) Trial-to-trial dispersion", fontsize=10.5, fontweight="bold")
a2.set_xticks(t); a2.set_ylim(-1.7, 1.7)
a2.grid(ls=":", lw=0.6, alpha=0.6); a2.set_axisbelow(True)
a2.legend(frameon=False, fontsize=7.6, loc="lower right")
fig.suptitle("Per-trial latency traces: reproducibility under the thermal-aware protocol\n"
             "(8 independent trials per configuration; device cooled below 55 \u00b0C before each)",
             fontsize=12, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig("figures/fig_trial_traces.png"); plt.close(fig)

# ---------------- FIG 4: battery endurance ----------------
frames = [72*3600/mu(k, "energy_per_img_j") for k in cfg]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.bar(labels, frames, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
for b, v in zip(bars, frames):
    ax.annotate(f"{v:,.0f}", (b.get_x()+b.get_width()/2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
base = frames[0]
for b, v, k in zip(bars, frames, cfg):
    if k != "combined_640":
        ax.annotate(f"{v/base:.2f}\u00d7", (b.get_x()+b.get_width()/2, v/2),
                    ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
ax.set_ylabel("Images per 72 Wh charge")
ax.set_ylim(0, max(frames)*1.16)
ax.set_title("Battery endurance from measured per-image energy\n"
             "(72 Wh pack; arithmetic conversion \u2014 an upper bound)",
             fontsize=11, fontweight="bold", pad=10)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
fig.savefig("figures/fig_battery_endurance.png"); plt.close(fig)

print("Wrote 4 figures to figures/ :")
for n in ["fig_system_cost_matrix", "fig_savings_summary", "fig_trial_traces", "fig_battery_endurance"]:
    print(f"  figures/{n}.png")
print("\nEvery value is computed from results/ ; nothing is hard-coded.")
