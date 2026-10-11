# HemoPIC+

An extension of [HemoPIC](https://github.com/jhuldr/HemoPIC) (MICCAI 2026), a physics-informed digital twin for cerebral perfusion. HemoPIC+ adds the changes C1–C9; each one is switched on or off from a config file. The `baseline` config reproduces the original HemoPIC **bit for bit**, and an automated test checks this.

- **C1** Use each subject's true repetition time (TR) instead of a fixed dt of 1 s.
- **C5** Make the flow units in the tracer compartment consistent (rate F/(60·V) per second).
- **C2** Model a per-ROI bolus delay and dispersion.
- **C3** Reference-free mode: no Deconv map is used for priors, textures or bounds.
- **C4** Robust fitting with random restarts (multistart).
- **C6** Automatic arterial input function (AIF) from arterial voxels, with partial-volume correction from the venous output function (VOF), giving priors in absolute units.
- **C7** Windkessel model started at steady state (tested; not recommended).
- **C8** CBV by indicator dilution (`cbv_source: id_regional` / `id_voxel`).
- **C9** Voxel CBF pattern consistent with the voxel CBV (`cbf_texture: osvd | moment`, `cbf_texture_clip: false`).

**Recommended configs:**
- `configs/full_aif_idvox_osvd_nc.yaml`: best average over CBF/CBV/MTT, and beats oSVD on CBF;
- `configs/full_idreg.yaml`: when MTT is the main map.

**Important caveat** (docs/RESULTS.md, section 2d): a standard oSVD deconvolution that uses the same automatic AIF and the true TR reaches lesion ROC AUCs on par with HemoPIC+ for CBF and MTT, and higher for CBV. Any comparison should use it as the baseline (`scripts/osvd_check.py`).

Rationale and design: [docs/PLAN.md](docs/PLAN.md). Full results: [docs/RESULTS.md](docs/RESULTS.md). Uncertainty: [docs/UNCERTAINTY.md](docs/UNCERTAINTY.md). These documents are in Vietnamese.

## Main results

Data: 43 ISLES 2017 subjects (Zenodo release). Lesion ROC AUC is computed on relative maps (each map divided by its brain median), with the same formula as HemoPIC's `Eval/lesion_roc`.

### Lesion discrimination: a fair comparison with the same AIF

![AUC comparison](docs/figures/auc_comparison.png)

On the 36 subjects with an accepted automatic AIF:
- `full_aif_idvox_osvd_nc` beats oSVD on CBF (+0.024, p = 3e−4), ties on CBV, and is slightly lower on MTT (−0.014, p = 0.03);
- `full_idreg` is best on MTT;
- the ISLES Deconv maps are lowest on all three maps.

Most of the gap between HemoPIC and the ISLES Deconv maps comes from processing (AIF selection, TR), not from the physical model: oSVD with the automatic AIF already closes most of it.

### Cumulative ablation

![Ablation](docs/figures/ablation.png)

### Diagnosis: the original HemoPIC keeps F and V at their priors, which come from the Deconv maps

![Prior diagnosis](docs/figures/prior_diagnosis.png)

The original tracer compartment applies the exchange rate F/V per second, while F comes from CBF in ml/100g/min (a unit mismatch, fixed by C5). As a result, the data cannot constrain F and V, and the fitted values stay at the priors (ratio ≈ 1). After the fix, the model actually learns from the curves.

### Tracer curve reconstruction (compare with Fig. 5b of the paper)

![Tracer curves](docs/figures/tracer_curves.png)

The lesion nRMSE drops from 0.318 to 0.260 (p = 8e−12); the long tail of the lesion curve is captured better.

### Example maps (subject 22)

![Maps](docs/figures/maps_example.png)

Quality control of the automatic AIF on all 43 subjects: [docs/figures/aif_qc.png](docs/figures/aif_qc.png).

To redraw the figures: `python scripts/make_figures.py --outputs outputs --data-root ../HemoPIC/data/ISLES2017_Training`. The first two figures only need `docs/results`.

## Uncertainty

Each ROI fit is turned into a posterior, p(θ | c) ∝ exp(−κ·L(θ)), with the HemoPIC objective L and one temperature κ per ROI set from the residual noise and the number of effectively independent frames. Its mode is the HemoPIC(+) fit itself, so nothing is refitted. The posterior is sampled with a Laplace approximation (exact Hessian from a differentiable PyTorch copy of the model) and batched MCMC. Voxel maps of the posterior sd and 95% interval are written next to the fitted maps. A pipeline ensemble (k-means seeds and cluster numbers) measures how much the maps move when the clustering changes. Details and all numbers: [docs/UNCERTAINTY.md](docs/UNCERTAINTY.md) (in Vietnamese).

![What the data determines](docs/figures/uq_identifiability.png)

- In the original HemoPIC, F and V are set by their priors (taken from the Deconv maps) in about 90% of ROIs. After the unit fix (C5) the data narrows them in more than 90% of ROIs, and the median 95% interval of CBF shrinks from 116% to 63% of its value.
- The intervals are checked in three ways without a phantom: hiding blocks of time frames, removing the most uncertain voxels, and refitting curves with added noise of known truth. The intervals for CBF and CBV are reliable (95% intervals cover 97–100%). Those for MTT (86–88%) and τ (70–75%) are too narrow.
- Removing the most uncertain voxels raises the lesion AUC for MTT in every config, and for CBF with `full` (0.782 → 0.802).

![tau claim](docs/figures/uq_tau.png)

The paper reports a shorter τ in the lesion than on the contralateral side. This holds for the original model (−4.2 s, 95% CI [−6.4, −2.1]), but reverses with high certainty once the units are fixed: +6.3 s, 95% CI [+3.7, +9.4].

## Repository layout

```
configs/              # one file per experiment (baseline, c1_tr, ..., full)
  isles2017_tr.json   # true TR per subject (C1), produced by preprocess/extract_tr.py
preprocess/           # raw data -> HemoPIC-ready data
  prepare_isles2017.py      # runs every step below
  download_isles2017.py     # downloads the Zenodo zip (resumable) and unzips it
  flatten_isles2017.py      # maps the nested Zenodo layout to MR_rCBF.nii, OT.nii, ...
  extract_tr.py             # measures TR from the DICOM metadata (C1)
  pwi_to_tracer.py          # PWI signal -> concentration (HemoPIC Preprocessing 2)
  segmentation/             # SynthSeg + 4-class tissue labels (HemoPIC Preprocessing 1, needs FreeSurfer)
hemopic_plus/         # library
  config.py  io.py  signal.py  models.py  fit.py  pipeline.py  evaluation.py  aif.py  osvd.py
  uq/                 # uncertainty: torch_model.py (differentiable model), data.py, laplace.py, mcmc.py
scripts/
  run_cohort.py       # fits the whole cohort in parallel
  evaluate.py         # tracer nRMSE + lesion ROC AUC
  compare.py          # comparison table across experiments
  make_figures.py     # the figures in this README
  paired_test.py      # Wilcoxon test + bootstrap CI between two experiments
  plot_aif.py         # QC of the automatic AIF (C6)
  osvd_check.py       # independent reference: oSVD with the same AIF (ROC, between-subject MTT)
  summarize_original_eval.py  # summarises CVT / tau from the original Eval scripts
  run_uq.py           # posterior per ROI (Laplace + MCMC) and voxel uncertainty maps
  eval_heldout.py     # check 1: predictive coverage of hidden time frames
  eval_semisynth.py   # check 3: coverage of a known truth after adding noise and refitting
  eval_uq.py          # identifiability, interval widths, check 2 (removing uncertain voxels), tau claim
  run_ensemble.py     # pipeline uncertainty (k-means seeds and cluster numbers)
  make_uq_figures.py  # the uncertainty figures
tests/                # the baseline must match the original HemoPIC
run_experiments.ps1 / .sh   # runs every experiment
```

## Installation

Clone this repository **next to** the original HemoPIC repository. The default configs read data from `../HemoPIC/data/ISLES2017_Training`, and the test compares against the demo bundled with the original repository.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests\test_baseline_equivalence.py      # should print: all tests passed
```

Linux/macOS: the same, with `source .venv/bin/activate`.

The uncertainty scripts also need PyTorch (CPU is enough): `pip install torch`, then `python tests/test_uq_torch.py`.

## Data

**If you already have preprocessed data** (`../HemoPIC/data/ISLES2017_Training`, 43 subjects with `CTC_from_MR_4DPWI.nii` and `hemopic_seg4/`), nothing else is needed.

**From scratch:**

```bash
python preprocess/prepare_isles2017.py D:/data                  # needs FreeSurfer >= 7.3 for SynthSeg
python preprocess/prepare_isles2017.py D:/data --skip-synthseg  # on a machine without FreeSurfer
```

Then set `data_root` in the configs, or pass `--data-root D:/data/ISLES2017_Training` when running.

The Zenodo release has 43 subjects (training_1..48, with 3, 17, 25, 29 and 34 missing); the paper used 44.

## Running

```powershell
# one experiment
python scripts\run_cohort.py configs\full.yaml          # fits 43 subjects -> outputs\full\training_XX\
python scripts\evaluate.py  configs\full.yaml           # -> outputs\full\eval\summary.json

# a few subjects, limited number of processes
python scripts\run_cohort.py configs\full.yaml --patients 1 22 --workers 4

# every experiment + comparison table (outputs\comparison.md)
powershell -ExecutionPolicy Bypass -File run_experiments.ps1
```

The outputs keep HemoPIC's file layout (`HemoPIC_CBF/CBV/MTT.nii.gz`, `roi_fit_summary.csv`, ...), so the original repository's `Eval/` scripts (CVT, tau, summary statistics, slice maps) run on them unchanged, for example:

```bash
python ../HemoPIC/Eval/cvt_check/run_eval_cvt.py ../HemoPIC/data/ISLES2017_Training outputs/full outputs/full/eval_cvt
```

For nRMSE, use `scripts/evaluate.py`: the original tracer script rebuilds the curves with the original model, which is wrong for the new model variants.

## Configuration

Every key is defined, with comments, in `hemopic_plus/config.py`; the defaults are the original behaviour. A config only lists the keys that differ from the defaults:

```yaml
name: my_experiment
data: {data_root: ../HemoPIC/data/ISLES2017_Training, out_root: outputs}
time: {dt_source: tr_table}
model: {flow_per_minute: true, delay_dispersion: true, delay_bounds: [-4, 10]}
robust: {multistart: 4}
reference_free: {prior_source: tracer, cbv_texture: tracer_auc, map_bounds: fixed}
partition: {k_gm: 16, k_wm: 12}
```

## License

Code copied and adapted from HemoPIC (MIT, © 2026 LDR Group @ JHU) is marked "(HemoPIC)" in the source; see `LICENSE_HemoPIC`.
