"""Methodology overview figure (fig:methodology-overview).

All numbers come from pipeline outputs: collected PRs from
results/collection_gaps.csv, AI and control PRs from data/ai_prs.csv and
data/human_prs.csv, repositories with a threshold from
results/rq1_threshold_origin_per_repository.csv and valid survey answers from
results_survey/survey_response_rate.csv. Icons from Font Awesome Free
(fonts/, SIL OFL 1.1, see fonts/FONTAWESOME_LICENSE.txt).
"""

import glob

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import os

os.makedirs("figures", exist_ok=True)

# numbers
N_REPOS = len(glob.glob("data/raw_prs/*.json"))
gaps = pd.read_csv("results/collection_gaps.csv")
N_PRS = int(gaps.loc[gaps.repo == "TOTAL", "collected_450"].iloc[0])
N_AI = len(pd.read_csv("data/ai_prs.csv", usecols=["PR_Number"], low_memory=False))
N_HUM = len(pd.read_csv("data/human_prs.csv", usecols=["PR_Number"], low_memory=False))
N_THR = len(pd.read_csv("results/rq1_threshold_origin_per_repository.csv"))
sv = pd.read_csv("results_survey/survey_response_rate.csv").set_index("group")
N_RESP, N_INV = int(sv.loc["Total", "valid_answers"]), int(sv.loc["Total", "invitations"])
fmt = lambda n: f"{n:,}"

# style
SOLID = FontProperties(fname="fonts/fa-solid-900.ttf")
BRANDS = FontProperties(fname="fonts/fa-brands-400.ttf")
SHADE = "#262626"
COLORS = {  # (fill, edge)
    "collection": ("#fff2cc", "#d6a72a"),
    "detection": ("#fde3cc", "#d9822b"),
    "threshold": ("#dcedd5", "#6aa84f"),
    "rq1": ("#dae8fc", "#5b84c4"),
    "rq2": ("#f8cecc", "#b85450"),
    "rq3": ("#e6dcef", "#8e6bb0"),
    "def": ("#f2f2f2", "#7f7f7f"),
}
WINDOWS = [
    ("baseline", "#c2d3bb", SHADE),
    ("before", "#5672b3", "white"),
    ("after", "#b35672", "white"),
]
FS_TTL, FS_TXT, FS_NUM, FS_WIN = 8.8, 7.5, 7.5, 7.0  # font sizes (pt)
FS_LAT = 8.0
X_ICON, X_TXT = 0.18, 0.36
LW, RADIUS = 0.8, 0.07
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_TXT})

# grid (inches)
W = 7.0
M = 0.03  # outer margin
GAPS = [0.66, 0.18, 0.52]  # gap between columns (arrows)
COLS = [1.22, 1.27, 1.5]
COLS.append(W - 2 * M - sum(GAPS) - sum(COLS))
XS = [M + sum(COLS[:i]) + sum(GAPS[:i]) for i in range(4)]

PAD = 0.09  # inner padding
LH = 0.13
Y_TTL = PAD + 0.065
Y_ICON = Y_TTL + 0.26
Y_DESC = (Y_ICON + 0.24, Y_ICON + 0.24 + LH)
Y_RES = (Y_DESC[1] + 0.19, Y_DESC[1] + 0.19 + LH)
HH = Y_RES[1] + LH / 2 + PAD
Y0 = M
TOP = Y0 + HH
MIDDLE = Y0 + HH / 2
DH = 0.42
DY = TOP + 0.17
H = DY + DH + M

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, color):
    f, b = COLORS[color]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={RADIUS}",
            facecolor=f,
            edgecolor=b,
            linewidth=LW,
        )
    )


def icon(x, y, glyph, color, size=15, font=SOLID):
    ax.text(
        x,
        y,
        glyph,
        fontproperties=font,
        fontsize=size,
        ha="center",
        va="center",
        color=COLORS[color][1],
    )


def text(x, y, s, ha="center", **kw):
    kw.setdefault("fontsize", FS_TXT)
    ax.text(x, y, s, ha=ha, va="center", color=SHADE, **kw)


def arrow(x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=LW,
            color=SHADE,
            shrinkA=0,
            shrinkB=0,
        )
    )


def line(xs, ys, cap="butt"):
    ax.plot(xs, ys, color=SHADE, lw=LW, solid_capstyle=cap, solid_joinstyle="miter")


def label_arrow(x, number, unit):
    """Bold number above the arrow, with the unit right below."""
    text(x, MIDDLE + 0.095, number, fontweight="bold", fontsize=FS_NUM)
    text(x, MIDDLE - 0.095, unit, fontweight="bold", fontsize=FS_NUM)


def stage(i, color, title, glyph, desc, res, font=SOLID):
    """Main box with title, icon, two description lines and two result lines."""
    x, w = XS[i], COLS[i]
    cx = x + w / 2
    box(x, Y0, w, HH, color)
    text(cx, TOP - Y_TTL, title, fontsize=FS_TTL, fontweight="bold")
    icon(cx, TOP - Y_ICON, glyph, color, font=font)
    for y, s in zip(Y_DESC, desc):
        text(cx, TOP - y, s)
    for y, (s, weight) in zip(Y_RES, res):
        text(cx, TOP - y, s, fontweight=weight, fontsize=FS_NUM)


# 1. collection
stage(
    0,
    "collection",
    "Data Collection",
    "",
    ["Top 30 repos in", "15 languages"],
    [(f"{N_REPOS} repos", "bold"), ("PRs since 2022", "normal")],
    font=BRANDS,
)

# 2. detection
arrow(XS[0] + COLS[0], MIDDLE, XS[1], MIDDLE)
label_arrow(XS[1] - GAPS[0] / 2, fmt(N_PRS), "PRs")
stage(
    1,
    "detection",
    "AI Detection",
    "",
    ["Heuristics of", "Robbes et al."],
    [("AI Agent · Human AI", "bold"), ("Human control", "bold")],
)

# 3. threshold, with the baseline | before | after mini timeline
arrow(XS[1] + COLS[1], MIDDLE, XS[2], MIDDLE)
stage(2, "threshold", "Adoption Threshold", "", [], [])
x, w = XS[2], COLS[2]
bx, bw, bh = x + 0.08, w - 0.16, 0.24
by = TOP - (Y_DESC[0] + Y_RES[1]) / 2 - bh / 2
fr, sep = [0.4, 0.3, 0.3], 0.012
x0 = bx
for f, (name, color, shade) in zip(fr, WINDOWS):
    ax.add_patch(Rectangle((x0, by), bw * f - sep, bh, facecolor=color, lw=0))
    ax.text(
        x0 + (bw * f - sep) / 2,
        by + bh / 2,
        name,
        ha="center",
        va="center",
        fontsize=FS_WIN,
        color=shade,
    )
    x0 += bw * f
xt = bx + bw * (fr[0] + fr[1]) - sep / 2
line([xt, xt], [by - 0.05, by + bh + 0.05])

# top box (newcomer definition) and the arrow down to the threshold
dx, dw = XS[1], 1.85
box(dx, DY, dw, DH, "def")
icon(dx + X_ICON, DY + DH / 2, "", "def", size=14)
tx = dx + X_TXT
text(tx, DY + DH / 2 + 0.075, "Newcomer Definition", ha="left", fontsize=FS_LAT, fontweight="bold")
text(tx, DY + DH / 2 - 0.085, "Steinmacher et al.", ha="left")
cx3 = XS[2] + COLS[2] / 2
line([dx + dw, cx3, cx3], [DY + DH / 2, DY + DH / 2, TOP + 0.06])
arrow(cx3, TOP + 0.07, cx3, TOP)

# 4. stacked RQs, fed by a bus from the threshold
rx, rw = XS[3], COLS[3]
gap_rq = 0.06
alt = (HH - 2 * gap_rq) / 3
xb = rx - 0.09
line([XS[2] + COLS[2], xb], [MIDDLE, MIDDLE])
label_arrow((XS[2] + COLS[2] + xb) / 2, str(N_THR), "repos")
rqs = [
    ("rq1", "RQ1", "Newcomer Flow", "", None),
    ("rq2", "RQ2", "Contribution Process", "", None),
    ("rq3", "RQ3", "Survey", "", None),
]
ys = [TOP - alt - i * (alt + gap_rq) for i in range(3)]
line([xb, xb], [ys[2] + alt / 2, ys[0] + alt / 2], cap="projecting")
for (color, rq, ttl, gl, desc), y in zip(rqs, ys):
    box(rx, y, rw, alt, color)
    arrow(xb, y + alt / 2, rx, y + alt / 2)
    icon(rx + X_ICON, y + alt / 2, gl, color, size=14)
    text(rx + X_TXT, y + alt / 2 + 0.07, rq, ha="left", fontsize=FS_LAT, fontweight="bold")
    text(rx + X_TXT, y + alt / 2 - 0.07, ttl, ha="left", fontsize=FS_LAT)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/methodology.{ext}", dpi=300)
print(N_REPOS, N_PRS, N_AI, N_HUM, N_THR, N_RESP, N_INV)
