"""Neural PDE: learn the coefficient fields of the transport PDE across patients,
with patient-level cross-validation.

python scripts/train_npde.py configs/full.yaml --data-root ../HemoPIC/data/ISLES2017_Training [--folds 5]

Needs run_cohort.py and run_pde.py outputs (the plain PDE fit is the start).
For each fold the shared network is trained jointly with the per-ROI physics
of the training patients; on the held-out patients the network is frozen and
only the per-ROI physics is fitted. The plain PDE is refitted on the same
held-out patients with the same optimiser, so the two differ only by the
learned fields. Writes
  <out_root>/<name>__npde/      fit folders of the neural PDE (held-out fits)
  <out_root>/<name>__pdeadam/   fit folders of the plain PDE (same optimiser)
  <out_root>/<name>__npde/eval/{summary.json, roi_compare.csv, fold_*.pt}
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde import neural as NN  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import derived  # noqa: E402
from run_pde import evaluate, write_variant  # noqa: E402


def run_fold(args):
    (k, fit_dirs, pde_root, train_idx, test_idx, steps, test_steps, cells, threads, out_dir, shape_penalty) = args
    torch.set_num_threads(threads)
    t0 = time.time()
    co = NN.load_cohort(fit_dirs, pde_root)
    tr, te = co.take(train_idx), co.take(test_idx)
    net = NN.Closure(co.z.shape[1])
    _, _, d_tr, _, hist = NN.fit(tr, net, train_closure=True, steps=steps, n_cells=cells, shape_penalty=shape_penalty,
                                 log_every=max(steps // 5, 1))
    for p in net.parameters():
        p.requires_grad_(False)
    th_n, _, d_n, c_n, _ = NN.fit(te, net, steps=test_steps, n_cells=cells)
    th_p, _, d_p, c_p, _ = NN.fit(te, None, steps=test_steps, n_cells=cells)
    with torch.no_grad():
        nu, phi = net(te.z)
        cv_n = TP.transit_cv(th_n[:, 7], cells, (nu, phi))
        cv_p = TP.transit_cv(th_p[:, 7], cells)
    torch.save({"state": net.state_dict(), "test_idx": test_idx, "train_idx": train_idx, "hist": hist,
                "nu": nu, "phi": phi, "theta_npde": th_n, "theta_pde": th_p}, out_dir / f"fold_{k}.pt")
    print(f"  fold {k}: {len(train_idx)} train / {len(test_idx)} test ROIs, train data {float(d_tr.median()):.4f}, "
          f"test data neural {float(d_n.median()):.4f} vs plain {float(d_p.median()):.4f}, {time.time() - t0:.0f}s", flush=True)
    return (k, th_n.numpy(), d_n.numpy(), c_n.numpy(), th_p.numpy(), d_p.numpy(), c_p.numpy(), nu.numpy(), phi.numpy(),
            cv_n.numpy(), cv_p.numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--cells", type=int, default=24)
    p.add_argument("--shape-penalty", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--patients", type=int, nargs="*")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    root = cfg.resolve("out_root") / cfg.name
    pde_root = root.parent / f"{cfg.name}__pde"
    pids = sorted(pid for pid in (a.patients or list_patients(root)) if (root / subject_id(pid) / "fit_curves.npz").exists())
    fit_dirs = [root / subject_id(pid) for pid in pids]
    out_dir = root.parent / f"{cfg.name}__npde" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    co = NN.load_cohort(fit_dirs, pde_root)
    rng = np.random.default_rng(0)
    fold_of = {pid: f for pid, f in zip(rng.permutation(pids), np.arange(len(pids)) % a.folds)}
    roi_fold = np.array([fold_of[s] for s in co.subject])
    jobs = [(k, fit_dirs, pde_root, np.where(roi_fold != k)[0], np.where(roi_fold == k)[0], a.steps, a.test_steps, a.cells,
             a.threads, out_dir, a.shape_penalty) for k in range(a.folds)]
    print(f"[{cfg.name}] neural PDE: {len(pids)} patients, {len(co.roi_ids)} ROIs, {a.folds} folds, {a.cells} cells")
    t0 = time.time()
    B = len(co.roi_ids)
    res = {}
    with ProcessPoolExecutor(max_workers=a.folds) as ex:
        for r in ex.map(run_fold, jobs):
            res[r[0]] = r
    th = {"npde": np.zeros((B, 8)), "pdeadam": np.zeros((B, 8))}
    dat = {"npde": np.zeros(B), "pdeadam": np.zeros(B)}
    cur = {"npde": np.zeros(tuple(co.c_obs.shape)), "pdeadam": np.zeros(tuple(co.c_obs.shape))}
    fields = {"nu": np.zeros((B, 2)), "phi": np.zeros((B, 2)), "cv_npde": np.zeros(B), "cv_pde": np.zeros(B)}
    for k, (_, thn, dn, cn, thp, dp, cp, nu, phi, cvn, cvp) in res.items():
        idx = jobs[k][4]
        th["npde"][idx], dat["npde"][idx], cur["npde"][idx] = thn, dn, cn
        th["pdeadam"][idx], dat["pdeadam"][idx], cur["pdeadam"][idx] = thp, dp, cp
        fields["nu"][idx], fields["phi"][idx], fields["cv_npde"][idx], fields["cv_pde"][idx] = nu, phi, cvn, cvp
    print(f"[{cfg.name}] all folds done in {time.time() - t0:.0f}s")

    # fit folders for evaluation
    rows = []
    for pid, fd in zip(pids, fit_dirs):
        b = load_subject(fd)
        sel = np.where(co.subject == pid)[0]
        assert list(co.roi_ids[sel]) == list(b.roi_ids)
        T = b.t.shape[0]
        with torch.no_grad():
            ref = {q: v.numpy() for q, v in derived(b, b.theta_fit).items()}
            dd = {m: {q: v.numpy() for q, v in derived(b, torch.as_tensor(th[m][sel][:, :7])).items()} for m in th}
        for m in th:
            write_variant(fd, root.parent / f"{cfg.name}__{m}" / fd.name, b, cur[m][sel][:, :T],
                          dd[m]["CBF"] / ref["CBF"], dd[m]["V"] / ref["V"], b.cfg)
        np.savez_compressed(root.parent / f"{cfg.name}__npde" / fd.name / "npde_theta.npz", roi_ids=b.roi_ids,
                            theta=th["npde"][sel], theta_pde=th["pdeadam"][sel], nu=fields["nu"][sel], phi=fields["phi"][sel],
                            cv_npde=fields["cv_npde"][sel], cv_pde=fields["cv_pde"][sel])
        for j, i in enumerate(sel):
            rows.append({"subject": fd.name, "roi_id": int(co.roi_ids[i]), "tissue": co.tissue[i], "fold": int(roi_fold[i]),
                         "nmse_pde": float(dat["pdeadam"][i]), "nmse_npde": float(dat["npde"][i]),
                         "Pe_pde": float(np.exp(th["pdeadam"][i, 7])), "Pe_npde": float(np.exp(th["npde"][i, 7])),
                         **{f"{q}_pde": float(dd["pdeadam"][q][j]) for q in ("CBF", "V", "MTT", "tau")},
                         **{f"{q}_npde": float(dd["npde"][q][j]) for q in ("CBF", "V", "MTT", "tau")},
                         "nu_fast": float(fields["nu"][i, 0]), "phi_fast": float(fields["phi"][i, 0]),
                         "cv_pde": float(fields["cv_pde"][i]), "cv_npde": float(fields["cv_npde"][i])})
    with open(out_dir / "roi_compare.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    data_root = cfg.resolve("data_root")
    summ, per = {}, {}
    variants = [("pipeline", root), ("tank_polished", root.parent / f"{cfg.name}__polish"), ("pde", pde_root),
                ("pdeadam", root.parent / f"{cfg.name}__pdeadam"), ("npde", root.parent / f"{cfg.name}__npde")]
    for tag, fr in variants:
        if fr.exists():
            summ[tag], per[tag] = evaluate(data_root, fr, pids, min(os.cpu_count() or 1, 16))
    from scipy.stats import wilcoxon
    paired = {}
    for base in ("pdeadam", "tank_polished"):
        if base not in per:
            continue
        for key in sorted({kk[1:] for kk in per["npde"]}):
            d = np.array([per["npde"].get((pid, *key), np.nan) - per[base].get((pid, *key), np.nan) for pid in pids], float)
            d = d[np.isfinite(d)]
            if d.size:
                paired[f"npde-{base}:{key[0]}_{key[1]}"] = {"mean_diff": float(d.mean()),
                                                           "p_wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
    gain = np.array([r["nmse_pde"] - r["nmse_npde"] for r in rows])
    summ["roi"] = {"n": len(rows), "nmse_pde_median": float(np.median([r["nmse_pde"] for r in rows])),
                   "nmse_npde_median": float(np.median([r["nmse_npde"] for r in rows])), "frac_npde_better": float(np.mean(gain > 0)),
                   **{f"{q}_rel_change_median": float(np.median([abs(r[f"{q}_npde"] / r[f"{q}_pde"] - 1) for r in rows]))
                      for q in ("CBF", "V", "MTT", "tau")}}
    for g in ("GM", "WM"):
        summ["roi"][f"cv_pde_median_{g}"] = float(np.median([r["cv_pde"] for r in rows if r["tissue"] == g]))
        summ["roi"][f"cv_npde_median_{g}"] = float(np.median([r["cv_npde"] for r in rows if r["tissue"] == g]))
    summ["paired"] = paired
    summ["settings"] = {"folds": a.folds, "steps": a.steps, "test_steps": a.test_steps, "cells": a.cells, "shape_penalty": a.shape_penalty}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")

    keys = list(summ["npde"].keys())
    print(f"  {'':14s} " + " ".join(f"{k:>16s}" for k in keys))
    for tag, _ in variants:
        if tag in summ:
            print(f"  {tag:14s} " + " ".join(f"{summ[tag][k]:16.4f}" for k in keys))
    r = summ["roi"]
    print(f"  held-out weighted nMSE median: plain PDE {r['nmse_pde_median']:.4f} -> neural PDE {r['nmse_npde_median']:.4f}; "
          f"neural better in {r['frac_npde_better']:.0%} of ROIs")
    print(f"  transit-time CV (CTH/MTT) median GM / WM: PDE {r['cv_pde_median_GM']:.2f} / {r['cv_pde_median_WM']:.2f}, "
          f"neural PDE {r['cv_npde_median_GM']:.2f} / {r['cv_npde_median_WM']:.2f} (tank = 1)")
    print("  median |relative change| neural vs plain: " + ", ".join(f"{q} {r[f'{q}_rel_change_median']:.1%}" for q in ("CBF", "V", "MTT", "tau")))
    for kk, v in paired.items():
        print(f"  {kk:40s} {v['mean_diff']:+.4f}  p={v['p_wilcoxon']:.2g}")


if __name__ == "__main__":
    main()
