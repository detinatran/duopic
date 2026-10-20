# DuoPIC: Uncertainty-Aware Voxel-Level Digital Twins of Cerebral Perfusion with Transport PDEs in Acute Stroke

Anh Duong Tran\*, Quang-Anh N. D.\*, Toan Sy Pham, Trang H. Hoang, Minh N. Do, Yiuno Chen, Duy Ton Mai, Huan X. Nguyen
(\* equal contribution)

London Digital Twin Research Centre, Middlesex University · Bach Mai Stroke Centre, Bach Mai Hospital · Hanoi Medical University · Vietnam National University · University of Illinois Urbana-Champaign · VinUniversity · University of Alberta · Imperial College London

Submitted to the IEEE International Symposium on Biomedical Imaging (ISBI) 2027.

![Overview of DuoPIC](docs/figures/duopic_overview.png)

*Fig. 1. Overview of DuoPIC. (a) Input data. (b) ROI preparation with reference-free priors and an automatic AIF. (c) Neural transport model, in which Windkessel flow and a delayed, dispersed input drive a fast and a slow tube split by a shared network that conserves tracer mass and MTT. (d) Voxel maps (ROI level × texture, or voxel fit) and their uncertainty.*

## Abstract

Digital twins of cerebral perfusion relate tracer dynamics to cerebral blood flow (CBF), blood volume (CBV) and mean transit time (MTT), but must be data-driven, physically consistent and uncertainty-aware. We introduce DuoPIC, a digital twin that needs no reference maps, in which a Windkessel flow model drives a tracer transport partial differential equation (PDE). A population-level neural closure divides each region into fast and slow pathways with tracer mass and MTT conserved by construction, and a tempered posterior quantifies the uncertainty of every estimate. Its voxel maps set the state of the art for lesion detection on ISLES'17 MR (AUC 0.745) and ISLES'24 CT (0.743). On held-out patients, the neural PDE reduced the compartment's reconstruction error by 19% and lifted 95% MTT interval coverage from 88% to 94%.

### Installation

Clone this repository next to the original HemoPIC repository: the default configs read the data from `../HemoPIC/data/ISLES2017_Training`, and the baseline test uses the demo bundled with HemoPIC.

```bash
python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt   
python tests/test_baseline_equivalence.py
```

Tests: `tests/test_baseline_equivalence.py`, `test_uq_torch.py`, `test_npde.py`, `test_voxel.py`; each prints `all tests passed`.

### Data

**ISLES'17.** With preprocessed data (`../HemoPIC/data/ISLES2017_Training`, 43 subjects with `CTC_from_MR_4DPWI.nii` and `hemopic_seg4/`) nothing else is needed. From scratch:

```bash
python preprocess/prepare_isles2017.py <data_dir>                   # needs FreeSurfer >= 7.3 for SynthSeg
python preprocess/prepare_isles2017.py <data_dir> --skip-synthseg   # without FreeSurfer
```

then set `data_root` in the configs or pass `--data-root`. The Zenodo release has 43 subjects (training_1..48 without 3, 17, 25, 29 and 34).

**ISLES'24.** The training set (149 subjects, Hugging Face mirror `hugging-science/isles24-stroke`, CC BY-NC-SA 4.0) is prepared in the ISLES'17 layout: rigid registration of the CT perfusion to the non-contrast CT, SynthSeg tissue classes and concentration as the change from the baseline attenuation; the smoothed copy is used by every method.

```bash
python scripts/prepare_isles24.py <hf_dir> <ISLES2024_Training> --freesurfer <FREESURFER_HOME>
python scripts/smooth_external.py <ISLES2024_Training> <ISLES2024_Training_s3> --sigma 3
```

### Commands

```bash
# Sec. 2.1, Table 2: regional fits (HemoPIC, DuoPIC, ID-CBV) and oSVD with the same AIF
python scripts/run_cohort.py configs/baseline.yaml
python scripts/run_cohort.py configs/full_aif.yaml
python scripts/run_cohort.py configs/full_aif_idvox_osvd_nc.yaml
python scripts/osvd_check.py configs/full_aif.yaml --compare baseline full_aif

# Sec. 2.4, Table 1: uncertainty of the compartment model
python scripts/run_uq.py configs/full_aif.yaml
python scripts/eval_heldout.py configs/full_aif.yaml
python scripts/eval_semisynth.py configs/full_aif.yaml
python scripts/eval_uq.py configs/full_aif.yaml
python scripts/run_ensemble.py configs/full_aif.yaml

# Sec. 2.2, Table 1: transport PDE, neural PDE and their uncertainty
python scripts/run_pde.py configs/full_aif.yaml
python scripts/train_npde.py configs/full_aif.yaml --extra-members 3
python scripts/run_npde_uq.py configs/full_aif.yaml
python scripts/eval_hidden_frames.py configs/full_aif.yaml              # add --models compartment for the compartment only
python scripts/eval_model_selection.py configs/full_aif.yaml

# Sec. 2.3, Table 2: voxel fit and lesion metrics
python scripts/run_voxel_pde.py configs/full_aif.yaml
python scripts/eval_voxel_maps.py configs/full_aif.yaml
python scripts/eval_lesion_metrics.py configs/full_aif.yaml

# Sec. 4: external test (ISLES'24)
for c in isles24_baseline isles24_full_aif isles24_full_aif_idvox_osvd_nc; do
  python scripts/run_cohort.py configs/$c.yaml --data-root <ISLES2024_Training_s3> --out-root outputs_s3
done
python scripts/eval_external.py --data-root <ISLES2024_Training_s3> --out-root outputs_s3 --metrics
python scripts/osvd_grid_ct.py --data-root <ISLES2024_Training_s3> --out-root outputs_s3
```
### Configuration

Every key is defined in `hemopic_plus/config.py`, and the defaults are the original HemoPIC behaviour. A config lists only the keys that differ:

```yaml
name: my_experiment
data: {data_root: ../HemoPIC/data/ISLES2017_Training, out_root: outputs}
time: {dt_source: tr_table}
model: {flow_per_minute: true, delay_dispersion: true, delay_bounds: [-4, 10]}
robust: {multistart: 4}
reference_free: {prior_source: tracer, cbv_texture: tracer_auc, map_bounds: fixed}
partition: {k_gm: 16, k_wm: 12}
```

## Citation

```bibtex
@inproceedings{tran2027duopic,
  title     = {{DuoPIC}: Uncertainty-Aware Voxel-Level Digital Twins of Cerebral Perfusion with Transport {PDEs} in Acute Stroke},
  author    = {Tran, Anh Duong and N. D., Quang-Anh and Pham, Toan Sy and Hoang, Trang H. and Do, Minh N. and Chen, Yiuno and Mai, Duy Ton and Nguyen, Huan X.},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Submitted}
}
```

## License

Code copied and adapted from HemoPIC (MIT, © 2026 LDR Group @ JHU) is marked "(HemoPIC)" in the source; see `LICENSE_HemoPIC`.
