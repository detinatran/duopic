"""HemoPIC versus DuoPIC-vox: maps of one patient and per-patient lesion AUC
(run after run_voxel_pde.py and eval_voxel_maps.py).

python scripts/make_hemopic_comparison.py [--data-root ...] [--out-root outputs] [--patient 22]

Left: CBF, CBV and MTT relative to the brain median for HemoPIC (the original
pipeline, outputs/baseline) and DuoPIC-vox (indicator-dilution CBV, voxel MTT,
CBF = 60 CBV / MTT), with the final infarct and each map's lesion AUC for this
patient. Right: lesion AUC of every patient with oSVD maps, HemoPIC on x and
DuoPIC-vox on y (above the diagonal: DuoPIC-vox better). Scored like
compare_osvd.py. Sized for the full text width of the paper (7 in).
Writes docs/figures/hemopic_vs_duopic.png and docs/results/vox_pde/hemopic_vs_duopic.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, INK, INK2  # noqa: E402

from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results" / "vox_pde"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6})


def duopic_vox(out, sub):
    cbv = nib.load(str(out / "full_aif_idvox_osvd_nc" / sub / "HemoPIC_CBV.nii.gz")).get_fdata()
    mtt = nib.load(str(out / "full_aif__vox_tank" / sub / "VOX_MTT.nii.gz")).get_fdata()
    cbf = np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / np.maximum(mtt, 1e-6), np.nan)
    return {"CBF": cbf, "CBV": cbv, "MTT": mtt}


def hemopic(out, sub):
    return {m: nib.load(str(out / "baseline" / sub / f"HemoPIC_{m}.nii.gz")).get_fdata() for m, _ in MAPS}


def aucs(vols, brain, ot):
    res = {}
    for m, direction in MAPS:
        valid, score = _lesion_scores(vols[m], brain, direction)
        res[m] = roc_auc(ot[valid], score) if score is not None else np.nan
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--patient", type=int, default=22)
    a = p.parse_args()
    out, data_root = Path(a.out_root), Path(a.data_root)
    rows = list(csv.DictReader(open(RES / "central_volume.csv", encoding="utf8")))
    vox = {int(r["patient"]): {m: float(r[f"DuoPIC-vox tank {m}"]) for m, _ in MAPS} for r in rows}
    hem = {}
    for pid in vox:
        base = data_root / f"training_{pid}"
        brain = load_nifti(base / SEG4_REL)[1] > 0
        ot = load_nifti(find_nii(base, "OT"))[1] > 0
        hem[pid] = aucs(hemopic(out, f"training_{pid}"), brain, ot)
    stats = {"n": len(vox)}
    for m, _ in MAPS:
        d = np.array([vox[p][m] - hem[p][m] for p in vox])
        stats[m] = {"HemoPIC": float(np.mean([hem[p][m] for p in vox])), "DuoPIC-vox": float(np.mean([vox[p][m] for p in vox])),
                    "diff": float(d.mean()), "p": float(wilcoxon(d).pvalue), "n_better": int((d > 0).sum())}
    dm = np.array([np.mean([vox[p][m] for m, _ in MAPS]) - np.mean([hem[p][m] for m, _ in MAPS]) for p in vox])
    stats["mean"] = {"diff": float(dm.mean()), "p": float(wilcoxon(dm).pvalue), "n_better": int((dm > 0).sum())}
    (RES / "hemopic_vs_duopic.json").write_text(json.dumps(stats, indent=2), encoding="utf8")

    # maps of one patient (display orientation as the other map figures)
    sub = f"training_{a.patient}"
    base = data_root / sub
    canon = lambda v, ref: nib.as_closest_canonical(nib.Nifti1Image(v, nib.load(str(ref)).affine)).get_fdata()  # noqa: E731
    ref = base / "hemopic_seg4" / "seg4_label_ref24.nii.gz"
    seg = canon(nib.load(str(ref)).get_fdata(), ref) > 0
    ot = canon(load_nifti(find_nii(base, "OT"))[1].astype(float), ref) > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    brain_sl = seg[:, :, z][crop].T
    o2 = ot[:, :, z][crop].T.astype(float)
    groups = [("HemoPIC [4]", hemopic(out, sub), hem[a.patient]), ("DuoPIC-vox", duopic_vox(out, sub), vox[a.patient])]

    fig = plt.figure(figsize=(7.0, 1.3))
    gs = fig.add_gridspec(1, 9, width_ratios=[1, 1, 1, 0.3, 1, 1, 1, 0.42, 1.6], wspace=0.05, left=0.005, right=0.99, bottom=0.21, top=0.76)
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad("white")
    im = None
    for g, (gname, vols, auc) in enumerate(groups):
        axs = []
        for j, (m, _) in enumerate(MAPS):
            ax = fig.add_subplot(gs[0, 4 * g + j])
            v = canon(vols[m], ref)[:, :, z][crop].T
            ok = brain_sl & np.isfinite(v) & (v > 0)
            img = np.where(ok, v / np.median(v[ok]), np.nan)
            im = ax.imshow(img, origin="lower", cmap=cm, vmin=0, vmax=2.5, interpolation="nearest")
            ax.contour(o2, levels=[0.5], colors=["#c8372d"], linewidths=0.6, origin="lower")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(m, pad=2, color=INK)
            ax.text(0.5, -0.04, f"AUC {auc[m]:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=6, color=INK2)
            axs.append(ax)
        fig.canvas.draw()
        x0, x1 = axs[0].get_position().x0, axs[-1].get_position().x1
        fig.text(0.5 * (x0 + x1), 0.995, gname, ha="center", va="top", fontsize=7.5, fontweight="bold", color=INK)
    gap = fig.add_subplot(gs[0, 3])
    pos = gap.get_position()
    gap.remove()
    cax = fig.add_axes([pos.x0 + 0.35 * pos.width, pos.y0 + 0.15 * pos.height, 0.25 * pos.width, 0.7 * pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 2])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    ax = fig.add_subplot(gs[0, 8])
    ax.plot([0.3, 1.0], [0.3, 1.0], color=INK2, lw=0.6, ls="--")
    for k, (m, _) in enumerate(MAPS):
        ax.scatter([hem[p][m] for p in vox], [vox[p][m] for p in vox], s=5, color=C[k], lw=0, alpha=0.85, label=m)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("AUC, HemoPIC", labelpad=1)
    ax.set_ylabel("AUC, DuoPIC-vox", labelpad=1)
    ax.tick_params(length=2, pad=1)
    ax.legend(loc="lower right", handletextpad=0.05, borderaxespad=0.15, labelspacing=0.1, markerscale=1.6, frameon=False)
    fig.savefig(FIG / "hemopic_vs_duopic.png", dpi=400)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    print("figure ->", FIG / "hemopic_vs_duopic.png")


if __name__ == "__main__":
    main()
