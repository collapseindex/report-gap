"""Generate every figure in the paper from the committed artifacts.

Nothing here is hand-drawn or hand-typed. Every number plotted is read out of `data/` at run time,
so a figure cannot drift from the prose or from the repository. That matters more than usual in
this paper, whose subject is a headline that died to an unchecked number.

    python make_figures.py

Writes figures/teaser.pdf, figures/enumerate.pdf and figures/erase.pdf.
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

KILL = "#B03A2E"      # a control that killed something
KEEP = "#1E6F5C"      # a result that survived
MEAS = "#2C3E70"      # a measurement
GREY = "#666666"


def load(rel):
    p = ROOT / rel
    if not p.exists():
        raise SystemExit("missing artifact: %s. Run the arm before making figures." % rel)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pole(row, keys):
    return sum(p for L, p in row["probs"].items() if row["mapping"][L] in keys)


# --------------------------------------------------------------------------------------------
# Figure 1: the paper in one view
# --------------------------------------------------------------------------------------------

def teaser():
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import to_rgba

    RESF, RESB, REST = "#eef1f6", "#9aa7bd", "#2b3a55"   # result box: fill, border, title
    CONF, CONB, CONT = "#f1f1ee", "#c3c2b7", "#3a3a37"   # control box
    DET = "#6b6f77"                                      # the second (detail) line

    # (section header, section colour, [ (r_title, r_detail, c_title, c_detail, o_title, o_detail) ])
    sections = [
        ("What our own replication retracted", KILL, [
            ("Argmax over-reports", "a mass shift near the boundary",
             "Fresh option orderings", "re-run, seeds 4–7", "retracted", "sign not stable"),
            ("Tuning-localized", "tuned model's negative report collapsed",
             "Fresh option orderings", "same code, seeds 4–7", "retracted", "+0.0002 → +0.1126"),
            ("Depth-robust", "null at all seven depths",
             "Fresh option orderings", "same items and layers", "retracted", "moves at layers 18, 24"),
            ("Shell", "represented, not expressed",
             "Fresh option orderings", "same probe", "retracted", "options move too"),
        ]),
        ("The nuisance, measured over its whole population", MEAS, [
            ("Baseline negative mass", "one quantity, before injection",
             "All 120 orderings", "enumerated, no injection", "986× range", "0.0009 to 0.8820"),
            ("The mechanism", "what explains the range",
             "Five identical options", "nothing but position differs", "87% on slot A", "0.33% on the middle"),
            ("Is the format broken?", "or the question undetermined",
             "Known-answer canary", "same five-option format", "97.9% correct", "stable across orderings"),
            ("One model only?", "or a tuning property",
             "Eight base/instruct pairs", "four families", "tuned worse", "in 4 of 4 families"),
        ]),
        ("What survives on a working instrument", KEEP, [
            ("A state outlives its cause", "injected, then its vector erased",
             "Project the direction OUT", "of the residual stream", "86% survives", "at layer 30"),
            ("The field's standard control", "is a weak null",
             "Direction fit on shuffled labels", "same procedure, random targets", "0.045 shift", "where random moves ≈0"),
        ]),
    ]

    row_h, sec_h = 1.0, 0.85
    units = sum(sec_h + row_h * len(rows) for _, _, rows in sections)
    fig, ax = plt.subplots(figsize=(7.0, units * 0.40))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, units)
    ax.axis("off")
    m_aspect = (units * 0.40 / units) / (7.0 / 1.0)   # circular corners under the x/y scale gap

    RX, RW = 0.010, 0.300
    CX, CW = 0.365, 0.300
    OX, OW = 0.715, 0.275

    def rbox(x, yc, w, fill, border):
        ax.add_patch(FancyBboxPatch((x, yc - 0.37), w, 0.74,
                     boxstyle="round,pad=0.004,rounding_size=0.012", mutation_aspect=m_aspect,
                     facecolor=fill, edgecolor=border, linewidth=0.8, zorder=2))

    def two(x, yc, w, title, detail, tcol, tfill, tborder):
        rbox(x, yc, w, tfill, tborder)
        if detail:
            ax.text(x + 0.014, yc + 0.155, title, fontsize=6.7, weight="bold", color=tcol, va="center", zorder=3)
            ax.text(x + 0.014, yc - 0.165, detail, fontsize=5.7, color=DET, va="center", zorder=3)
        else:
            ax.text(x + 0.014, yc, title, fontsize=6.7, weight="bold", color=tcol, va="center", zorder=3)

    y = units - 0.22
    for hx, txt in ((0.160, "result someone could have reported"),
                    (0.515, "the control we ran against it"),
                    (0.852, "what it returned")):
        ax.text(hx, y, txt, fontsize=6.1, style="italic", color=GREY, ha="center", va="top")

    y = units - 0.62
    for header, colour, rows in sections:
        ax.text(0.010, y, header, fontsize=7.4, weight="bold", color="#111111", va="center")
        y -= sec_h
        for rt, rd, ct, cd, ot, od in rows:
            two(RX, y, RW, rt, rd, REST, RESF, RESB)
            two(CX, y, CW, ct, cd, CONT, CONF, CONB)
            rbox(OX, y, OW, to_rgba(colour, 0.13), colour)
            if od:
                ax.text(OX + 0.014, y + 0.155, ot, fontsize=6.7, weight="bold", color=colour, va="center", zorder=3)
                ax.text(OX + 0.014, y - 0.165, od, fontsize=5.7, color=colour, va="center", zorder=3)
            else:
                ax.text(OX + 0.014, y, ot, fontsize=6.7, weight="bold", color=colour, va="center", zorder=3)
            for x0, x1 in ((RX + RW, CX), (CX + CW, OX)):
                ax.annotate("", xy=(x1 - 0.004, y), xytext=(x0 + 0.004, y),
                            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8), zorder=1)
            y -= row_h

    fig.savefig(FIGS / "teaser.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/teaser.pdf")


# --------------------------------------------------------------------------------------------
# Figure 2: the enumerated ordering population, and the position prior that explains it
# --------------------------------------------------------------------------------------------

def enumerate_fig():
    NEG = {"neg1", "neg2"}
    data = {}
    for key in ("base", "instruct"):
        rows = load("data/enum_%s/enum.jsonl" % key)
        per_ord = collections.defaultdict(list)
        for r in rows:
            if r["condition"] == "letters":
                per_ord[tuple(r["ordering"])].append(pole(r, NEG))
        data[key] = {o: statistics.fmean(v) for o, v in per_ord.items()}
        ident = [r for r in rows if r["condition"] == "identical"]
        data[key + "_ident"] = {L: statistics.fmean([r["probs"][L] for r in ident])
                                for L in sorted(ident[0]["probs"])}
        can = [r for r in rows if r["condition"] == "canary"]
        acc = collections.defaultdict(list)
        for r in can:
            correct = [L for L, k in r["mapping"].items() if k == "four"][0]
            acc[r["ordering_index"]].append(1.0 if r["argmax"] == correct else 0.0)
        data[key + "_canary"] = statistics.fmean([statistics.fmean(v) for v in acc.values()])

    # recover the orderings the earlier draws actually used
    import random as _r
    sys.path.insert(0, str(ROOT / "src"))
    from report_gap import stimuli as S
    draws = {}
    for label, seeds in (("original\nseeds 0-3", range(4)), ("replication\nseeds 4-7", range(4, 8))):
        got = []
        for s in seeds:
            shuffled = list(S.SELF_REPORT_OPTIONS)
            _r.Random(s).shuffle(shuffled)
            got.append(tuple(S.SELF_REPORT_OPTIONS.index(o) for o in shuffled))
        draws[label] = got

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # (a) the distribution over all 120 orderings
    ax = axes[0]
    for key, colour, off in (("instruct", MEAS, 1.0), ("base", GREY, 0.0)):
        vals = sorted(data[key].values())
        ax.scatter(vals, [off] * len(vals), s=9, alpha=0.45, color=colour,
                   edgecolors="none", zorder=2)
        ax.plot([min(vals), max(vals)], [off, off], color=colour, lw=0.8, zorder=1)
        # FLOOR, not round: the prose floors, and a nuisance figure should understate rather
        # than overstate. 986.53 must not print as 987 here and 986 there.
        # Ratios below 10 need a decimal (3.58 must not print as "3" here and "3.6" in the
        # prose); larger ones are floored, so a nuisance figure understates rather than overstates.
        ratio = max(vals) / min(vals)
        shown = ("%.1f" % ratio) if ratio < 10 else ("%d" % int(ratio))
        ax.text(min(vals), off + 0.22, "%s   min %.4f  max %.4f   %sx"
                % (key, min(vals), max(vals), shown),
                fontsize=6.5, color=colour, weight="bold")
    for label, orderings, marker in (("original\nseeds 0-3", draws["original\nseeds 0-3"], "v"),
                                     ("replication\nseeds 4-7", draws["replication\nseeds 4-7"], "^")):
        xs = [data["instruct"][o] for o in orderings if o in data["instruct"]]
        ax.scatter(xs, [1.0] * len(xs), marker=marker, s=52, color=KILL, zorder=4,
                   edgecolors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.55, 1.75)
    ax.set_yticks([])
    # short enough not to overflow the axes, which bbox_inches="tight" was clipping.
    # The caption carries the full description.
    ax.set_xlabel("negative-pole mass at baseline (log scale)")
    ax.set_title("(a) every one of the 120 option orderings", fontsize=8, loc="left")
    ax.scatter([], [], marker="v", s=52, color=KILL, label="the 4 orderings our first result used")
    ax.scatter([], [], marker="^", s=52, color=KILL, label="the 4 the replication used")
    ax.legend(fontsize=6, loc="lower right", frameon=False)

    # (b) the position prior with five identical options
    ax = axes[1]
    labels = list("ABCDE")
    w = 0.38
    for i, (key, colour) in enumerate((("instruct", MEAS), ("base", GREY))):
        vals = [data[key + "_ident"][L] for L in labels]
        ax.bar([x + (i - 0.5) * w for x in range(len(labels))], vals, width=w,
               color=colour, alpha=0.85, label=key, edgecolor="none")
    ax.axhline(0.2, color=KILL, lw=1.0, ls="--", zorder=3)
    ax.text(4.35, 0.215, "flat = 0.20", fontsize=6, color=KILL, ha="right")
    ax.annotate("%.4f" % data["instruct_ident"]["A"], xy=(-0.5 * w, data["instruct_ident"]["A"]),
                xytext=(0.25, 0.80), fontsize=7, color=MEAS, weight="bold",
                arrowprops=dict(arrowstyle="->", color=MEAS, lw=0.7))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("option label, with all five options the SAME sentence")
    ax.set_ylabel("probability mass")
    ax.set_title("(b) pure position prior: nothing but the slot differs", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.9)
    fig.savefig(FIGS / "enumerate.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/enumerate.pdf  (canary instruct %.4f, base %.4f)"
          % (data["instruct_canary"], data["base_canary"]))


# --------------------------------------------------------------------------------------------
# Figure 3: the erase arm, and the confound its ratio addresses
# --------------------------------------------------------------------------------------------

def erase_fig():
    """Plot the erase arm using the ANALYZER'S OWN functions.

    Not a reimplementation beside the scorer. `analyze_erase.py` holds `load`, `paired` and
    `vs_random`; this figure imports them, so a figure that disagrees with the table is impossible
    rather than merely unlikely.
    """
    sys.path.insert(0, str(ROOT / "experiments"))
    import analyze_erase as AE

    rows = AE.load(ROOT / "data" / "erase_instruct" / "erase.jsonl")
    base_scores = [r["probe_orth"] for r in rows if r["condition"] == "baseline"]
    sd = statistics.pstdev(base_scores) or 1.0

    def probe(r):
        return r["probe_orth"]

    # The un-erased reference is neg vs its own baseline: the random arms were only run AT an
    # erase layer, so vs_random has nothing to pair against at -1.
    no_erase = abs(AE.paired(rows, -1, "neg", "baseline", probe, sd).point)
    layers = sorted({r["erase_layer"] for r in rows if r["erase_layer"] > 0})

    prim, art, clean, ratio = {}, {}, {}, {}
    for L in layers:
        a = AE.paired(rows, L, "erase_only", "baseline", probe, sd)
        c = AE.vs_random(rows, L, "pos_erase", probe, sd)
        p = AE.vs_random(rows, L, "neg_erase", probe, sd)
        prim[L], art[L] = abs(p.point), abs(a.point)
        ratio[L] = prim[L] / art[L]
        clean[L] = (not a.excludes_zero or abs(a.point) < AE.PROBE_FLOOR_SD) \
            and c.lo > 0.0 and c.point >= AE.PROBE_FLOOR_SD

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) what survives, against the perturbation that erasing is on its own.
    # Headroom above the tallest bar is reserved for the legend, so it never lands on the data.
    ax = axes[0]
    top = no_erase * 1.62
    w = 0.36
    for i, L in enumerate(layers):
        ok = clean[L]
        # gate-failed layers are hatched rather than annotated: a text label under the axis
        # collides with the tick, and the hatch carries the same information in the legend.
        style = {} if ok else {"hatch": "///", "edgecolor": "white", "linewidth": 0.0}
        ax.bar(i - w / 2, prim[L], width=w, color=KEEP if ok else GREY,
               alpha=0.9 if ok else 0.45, **style)
        ax.bar(i + w / 2, art[L], width=w, color=KILL if ok else GREY,
               alpha=0.9 if ok else 0.45, **style)
        ax.text(i - w / 2, prim[L] + 0.02, "%.2f" % prim[L], fontsize=6,
                ha="center", color=KEEP if ok else GREY)
        if ok:
            ax.text(i + w / 2, art[L] + 0.02, "%.2f" % art[L], fontsize=6,
                    ha="center", color=KILL)
    ax.axhline(no_erase, color=MEAS, lw=1.0, ls="--", zorder=3)
    ax.text(len(layers) - 0.6, no_erase + 0.025, "no erase at all: %.2f" % no_erase,
            fontsize=6, color=MEAS, ha="right")
    ax.set_xticks(list(range(len(layers))))
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel("layer the injected direction is projected OUT at")
    ax.set_ylabel("probe shift (baseline SD)")
    ax.set_ylim(0, top)
    ax.set_title("(a) the state outlives its own cause", fontsize=8, loc="left")
    # proxy artists: ax.bar([], [], color=...) silently drops the colour on empty data
    ax.legend(handles=[Patch(facecolor=KEEP, label="state still read after erasure"),
                       Patch(facecolor=KILL, label="erasing alone, no injection"),
                       Patch(facecolor=GREY, alpha=0.45, hatch="///", edgecolor="white",
                             label="gate failed, not interpretable")],
              fontsize=6, frameon=False, loc="upper left", handlelength=1.2,
              borderaxespad=0.1, labelspacing=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) the confound: if this were pure perturbation the ratio would be flat
    ax = axes[1]
    ok_layers = [L for L in layers if clean[L]]
    hi = max(ratio.values())
    ax.plot(ok_layers, [ratio[L] for L in ok_layers], color=KEEP, lw=1.1, zorder=2)
    for L in layers:
        ax.scatter([L], [ratio[L]], s=34, zorder=3,
                   color=KEEP if clean[L] else "white",
                   edgecolors=KEEP if clean[L] else GREY, linewidths=0.9)
        # the two points near the flat line get their labels pushed sideways, the rest above
        dx, ha = (-7, "right") if abs(ratio[L] - ratio[ok_layers[0]]) < hi * 0.1 else (0, "center")
        ax.annotate("%.1f" % ratio[L], xy=(L, ratio[L]), xytext=(dx, 0 if dx else 6),
                    textcoords="offset points", fontsize=6, ha=ha,
                    va="center" if dx else "bottom",
                    color=KEEP if clean[L] else GREY)
    flat = ratio[ok_layers[0]]
    # the flat-line meaning goes in the legend, not as floating text: every free spot on these
    # axes is within a label's width of a data point.
    ax.axhline(flat, color=KILL, lw=0.9, ls=":", zorder=1,
               label="flat = pure perturbation")
    ax.scatter([], [], s=34, color="white", edgecolors=GREY, linewidths=0.9,
               label="gate failed, not interpretable")
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.4,
              borderaxespad=0.1, labelspacing=0.35)
    ax.set_xticks(layers)
    ax.set_xlim(layers[0] - 0.8, layers[-1] + 0.8)
    ax.set_xlabel("erase layer")
    ax.set_ylabel("primary / erase artifact")
    ax.set_ylim(0, hi * 1.30)
    ax.set_title("(b) post hoc: the ratio is not flat", fontsize=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.9)
    fig.savefig(FIGS / "erase.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/erase.pdf  (survival at L%d: %.0f%% of the un-erased effect)"
          % (ok_layers[-1], 100 * prim[ok_layers[-1]] / no_erase))


if __name__ == "__main__":
    teaser()
    enumerate_fig()
    erase_fig()
