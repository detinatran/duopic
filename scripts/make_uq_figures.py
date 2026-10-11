"""Figures for the uncertainty study (run after the UQ scripts).

python scripts/make_uq_figures.py --data-root ../HemoPIC/data/ISLES2017_Training [--configs baseline full full_aif]
Writes docs/figures/uq_*.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, INK, INK2  # noqa: E402  (shared palette and styling)

FIG = ROOT / "docs" / "figures"
LABEL = {"baseline": "HemoPIC original", "full": "HemoPIC+ full", "full_aif": "HemoPIC+ full_aif"}
PARAMS = [("log_R", "R"), ("log_tau", "τ"), ("log_V", "V"), ("log_F", "F"), ("log_Cin", "C_in"), ("delay", "delay"),
          ("log_disp", "dispersion")]
SEQ = ["#2a78d6", "#8fb8ea", "#d7e5f6", "#b9b8b3"]   # learnt, partly, prior-dominated, pinned


def rows_of(out, cfg):
    rows = []
    for f in sorted((out / cfg).glob("training_*/uq/roi_posterior.csv")):
        with open(f, encoding="utf8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def fig_identifiability(out, cfgs):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.0 * len(cfgs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cfg in zip(axes, cfgs):
        rows = rows_of(out, cfg)
        names = [p for p in PARAMS if f"{p[0]}_post_sd" in rows[0]]
        y = np.arange(len(names))[::-1]
        for j, (p, lab) in enumerate(names):
            pinned = np.array([r[f"{p}_pinned"] == "True" for r in rows])
            ratio = np.array([float(r[f"{p}_post_sd"]) / max(float(r[f"{p}_prior_sd"]), 1e-30) for r in rows])
            fr = [np.mean(~pinned & (ratio < 0.5)), np.mean(~pinned & (ratio >= 0.5) & (ratio <= 0.9)),
                  np.mean(~pinned & (ratio > 0.9)), np.mean(pinned)]
            left = 0.0
            for k, v in enumerate(fr):
                ax.barh(y[j], v, left=left, color=SEQ[k], height=0.7, edgecolor="#fcfcfb", linewidth=1.5)
                if v > 0.12:
                    ax.text(left + v / 2, y[j], f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                            color="#fcfcfb" if k == 0 else INK)
                left += v
        ax.set_yticks(y, [lab for _, lab in names])
        ax.set_xlim(0, 1)
        ax.set_title(LABEL.get(cfg, cfg), loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ]
    fig.legend(handles, ["learnt from data (sd ratio < 0.5)", "partly (0.5–0.9)", "prior decides (> 0.9)", "stuck at a bound"],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("What does the data actually determine? Posterior sd / prior sd per ROI parameter", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "uq_identifiability.png", dpi=160)
    plt.close(fig)


def fig_calibration(out, cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot([0.4, 1], [0.4, 1], color=INK2, lw=0.8, ls="--")
    for i, cfg in enumerate(cfgs):
        f = out / cfg / "eval_uq" / "heldout.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf8") as fh:
            rows = list(csv.DictReader(fh))
        levels = sorted({float(r["level"]) for r in rows})
        cov = []
        for lev in levels:
            sel = [r for r in rows if float(r["level"]) == lev]
            n = np.array([float(r["n"]) for r in sel])
            cov.append(np.sum(n * np.array([float(r["covered"]) for r in sel])) / n.sum())
        ax.plot(levels, cov, "o-", color=C[i], lw=2, ms=6, label=LABEL.get(cfg, cfg))
    ax.set_xlabel("nominal interval")
    ax.set_ylabel("hidden frames covered")
    ax.set_title("Check 1: hidden time frames", loc="left", color=INK)
    ax.legend(fontsize=8)
    ax = axes[1]
    qty = ["CBF", "V", "MTT", "tau"]
    w = 0.25
    for i, cfg in enumerate(cfgs):
        f = out / cfg / "eval_uq" / "semisynth.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf8") as fh:
            rows = list(csv.DictReader(fh))
        cov = [np.mean([r[f"{q}_covered"] == "True" for r in rows]) for q in qty]
        x = np.arange(len(qty)) + (i - (len(cfgs) - 1) / 2) * w
        ax.bar(x, cov, w * 0.92, color=C[i], label=LABEL.get(cfg, cfg))
        for xx, v in zip(x, cov):
            ax.text(xx, v + 0.01, f"{v:.0%}", ha="center", fontsize=7, color=INK)
    ax.axhline(0.95, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(qty)), ["CBF", "CBV (V)", "MTT", "τ"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("truth inside the 95% interval")
    ax.set_title("Check 3: noise added to the fitted curves", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "uq_calibration.png", dpi=160)
    plt.close(fig)


def fig_selective(out, cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, m in zip(axes, ("CBF", "MTT")):
        for i, cfg in enumerate(cfgs):
            f = out / cfg / "eval_uq" / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text(encoding="utf8"))["selective"].get(m)
            if not s:
                continue
            fr = np.array(s["fractions"]) * 100
            ax.plot(fr, s["auc_uncertain_removed"], "o-", color=C[i], lw=2, ms=5, label=f"{LABEL.get(cfg, cfg)}: most uncertain removed")
            ax.plot(fr, s["auc_random_removed"], ":", color=C[i], lw=1.5, label=f"{LABEL.get(cfg, cfg)}: random removed")
        ax.set_xlabel("% of voxels removed")
        ax.set_ylabel("lesion ROC AUC")
        ax.set_title(f"Check 2: {m}", loc="left", color=INK)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "uq_selective.png", dpi=160)
    plt.close(fig)


def fig_tau(out, cfgs):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, cfg in enumerate(cfgs):
        f = out / cfg / "eval_uq" / "tau_diff_draws.npy"
        if not f.exists():
            continue
        d = np.load(f)
        s = json.loads((out / cfg / "eval_uq" / "summary.json").read_text(encoding="utf8"))["tau_lesion_minus_contra"]
        ax.hist(d, bins=40, color=C[i], alpha=0.55, label=f"{LABEL.get(cfg, cfg)}: 95% CI [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}] s")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("τ(lesion) − τ(contralateral), seconds   (< 0: lesion relaxes faster, the paper's claim)")
    ax.set_ylabel("posterior draws")
    ax.legend(fontsize=8)
    ax.set_title("The paper's τ claim with uncertainty", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "uq_tau.png", dpi=160)
    plt.close(fig)


def fig_maps(out, data_root, cfgs, pid=22):
    import nibabel as nib

    def load(p):
        return nib.as_closest_canonical(nib.load(str(p))).get_fdata()

    base = Path(data_root) / f"training_{pid}"
    seg = load(base / "hemopic_seg4" / "seg4_label_ref24.nii.gz") > 0
    ot = load(base / "OT.nii") > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    cols = [("CBF", "CBF / brain median", None), ("CBF", "CBF: posterior 95% width / value", "uq"),
            ("CBF", "CBF: pipeline sd / value", "ens"), ("MTT", "MTT: posterior 95% width / value", "uq")]
    fig, axes = plt.subplots(len(cfgs), len(cols), figsize=(3.0 * len(cols), 2.9 * len(cfgs)), squeeze=False)
    for i, cfg in enumerate(cfgs):
        fd = out / cfg / f"training_{pid}"
        for j, (m, title, kind) in enumerate(cols):
            ax = axes[i, j]
            v = load(fd / f"HemoPIC_{m}.nii.gz")
            ok = seg & np.isfinite(v) & (v > 0)
            if kind is None:
                img = np.where(ok, v / np.median(v[ok]), np.nan)
                vmax, cmap = 2.5, "Blues"
            elif kind == "uq":
                lo, hi = load(fd / "uq" / f"HemoPIC_{m}_lo.nii.gz"), load(fd / "uq" / f"HemoPIC_{m}_hi.nii.gz")
                img = np.where(ok, (hi - lo) / v, np.nan)
                vmax, cmap = 1.5, "Oranges"
            else:
                sd = load(fd / "ens" / f"HemoPIC_{m}_ens_sd.nii.gz")
                img = np.where(ok, sd / v, np.nan)
                vmax, cmap = 0.5, "Oranges"
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#fcfcfb")
            im = ax.imshow(img[:, :, z][crop].T, origin="lower", cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(ot[:, :, z][crop].T, levels=[0.5], colors=[INK], linewidths=0.8, origin="lower")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(LABEL.get(cfg, cfg), fontsize=9, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"Uncertainty maps, training_{pid} slice {z} (black contour = lesion)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "uq_maps_example.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--configs", nargs="*", default=["baseline", "full", "full_aif"])
    a = p.parse_args()
    out = Path(a.out_root)
    FIG.mkdir(parents=True, exist_ok=True)
    fig_identifiability(out, a.configs)
    fig_calibration(out, a.configs)
    fig_selective(out, a.configs)
    fig_tau(out, a.configs)
    fig_maps(out, a.data_root, [c for c in a.configs if c != "full_aif"])
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
