"""One visual system for every figure in the paper: palette, typography, axes and diagram primitives.
Charts (make_assets_v2.py) and diagrams (diagrams.py) both import this, so a change here restyles everything."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# palette: one colour per entity, fixed across figures
BLUE = "#2467A7"      # the token-id store / anything "ours"
BLUE_L = "#7FA7CF"    # smaller stores, secondary store marks
BLUE_XL = "#C5DBEE"   # store fills in diagrams
DARK = "#263444"      # dense models, text
GREY = "#8995A1"      # fitted frontier, secondary text
GREY_L = "#E1E7ED"    # GPU / backbone fills in diagrams
LIGHT = "#EAF2F9"     # store-pool fills
PANEL = "#F7F9FB"     # card background
TEAL = "#238476"      # conventional router
ORANGE = "#C66A35"    # previous-step router, frontier MoE points
PURPLE = "#7257A3"    # multi-expert store
RED = "#B3402E"       # the violated requirement mark
GREEN = "#2E8B57"     # the satisfied requirement mark
GRID = "#E3E8ED"
EDGE = "#B9CAD9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#AAB4BE", "axes.linewidth": 0.65,
    "axes.grid": False, "axes.axisbelow": True, "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.7, "lines.markersize": 4.5, "legend.frameon": False, "legend.handlelength": 1.6, "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def ptitle(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", pad=8)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(length=3, width=0.6)


def save(fig, path_stem):
    fig.savefig(path_stem + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path_stem + ".png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ------------------------------------------------------------------ diagram primitives (units: 0-100 on both axes)
def canvas(width_in, height_in):
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    return fig, ax


def txt(ax, x, y, s, size=8, color=DARK, ha="left", va="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **kw)


def box(ax, x, y, w, h, s="", fc=GREY_L, ec=EDGE, size=8, color=DARK, weight="normal", r=1.2, lw=0.8, zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    if s:
        txt(ax, x + w / 2, y + h / 2, s, size=size, color=color, ha="center", weight=weight, zorder=zorder + 1)
    return p


def arrow(ax, p, q, color=GREY, lw=1.0, head=8, ls="-", zorder=3):
    a = FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.55,head_width=0.28", mutation_scale=head, color=color, lw=lw, ls=ls, zorder=zorder)
    ax.add_patch(a)
    return a


def line(ax, pts, color=GREY, lw=1.0, ls="-", zorder=3):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)


def oplus(ax, x, y, r=1.6, lw=0.8):
    ax.add_patch(Circle((x, y), r, fc="white", ec=DARK, lw=lw, zorder=4))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=DARK, lw=lw, zorder=5)
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=DARK, lw=lw, zorder=5)
