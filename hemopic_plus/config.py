"""Experiment configuration.

Every option defaults to the original HemoPIC behaviour, so an empty config
(configs/baseline.yaml) reproduces the published pipeline bit for bit.
YAML files group keys into sections purely for readability; all keys live in
one flat namespace and unknown keys raise an error.
"""
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    name: str = "baseline"

    ### Data
    data_root: str = "../HemoPIC/data/ISLES2017_Training"
    out_root: str = "outputs"
    patients: Optional[List[int]] = None  # None = every training_* folder

    ### Partition (original: K_GM=8, K_WM=6, seed=0)
    k_gm: int = 8
    k_wm: int = 6
    seed: int = 0
    min_cluster_vox: int = 400
    coord_weight: float = 0.10
    sample_max: int = 80000
    kmeans_max_iter: int = 25

    ### Curve preprocessing
    smooth_sigma_frames: float = 1.8
    baseline_frames: int = 5
    clip_nonneg: bool = True

    ### [C1] Temporal sampling
    # "header": NIfTI pixdim[4] (ISLES files store 0 or 1, so dt falls back to 1.0 s)
    # "tr_table": per-subject TR measured from the DICOM acquisition times
    dt_source: str = "header"
    tr_table: str = "configs/isles2017_tr.json"
    tr_default: float = 1.44  # cohort median, used for subjects without metadata

    ### ROI fitting (original values)
    tau_cap_seconds: float = 300.0
    r_max: float = 5000.0
    q_floor_ratio: float = 0.02
    reg_lambda_logr: float = 0.002
    lambda_in: float = 0.06
    lambda_v: float = 0.04
    lambda_f: float = 0.04
    eps_small: float = 1.0 / 1e6
    p0: float = 1.0

    ### [C5] Flow units in the tracer compartment
    # False (original): exchange rate q/V applied per second although F comes from
    #   CBF in ml/100g/min, so the compartment transit time is ~60x too short.
    # True: rate q/(60 V) per second, consistent with MTT = 60 * CBV / CBF.
    flow_per_minute: bool = False

    ### [C7] Windkessel initial condition
    # "fixed_p0" (original): Q_out(0) = p0 / R with p0 = 1, i.e. outflow starts near
    #   zero and relaxes to F over tau; this transient, not the transit time,
    #   then dominates the compartment kinetics.
    # "steady": Q_out(0) = F / R, so R = 1 means steady state and R measures the
    #   initial outflow deficit/excess relative to the inflow.
    wk_init: str = "fixed_p0"

    ### [C2] Per-ROI bolus delay and dispersion of the arterial input
    delay_dispersion: bool = False
    delay_bounds: List[float] = field(default_factory=lambda: [-4.0, 10.0])  # seconds
    dispersion_bounds: List[float] = field(default_factory=lambda: [0.05, 10.0])  # seconds
    lambda_delay: float = 0.002  # quadratic penalty on delay (per s^2)
    lambda_dispersion: float = 0.002  # quadratic penalty on log dispersion

    ### [C3] Reference-free mode (no silver-standard maps inside the pipeline)
    # "reference_maps": F/V priors from MR_rCBF / MR_rCBV (original)
    # "tracer": indicator-dilution CBV and maximum-slope CBF from the tracer curves
    prior_source: str = "reference_maps"
    ref_wm_cbf: float = 22.0  # calibration: normal white matter CBF (ml/100g/min)
    ref_wm_cbv: float = 2.0  # calibration: normal white matter CBV (ml/100g)
    # "reference_maps": voxel CBV pattern copied from MR_rCBV (original)
    # "tracer_auc": voxel CBV pattern from the area under each voxel curve
    cbv_texture: str = "reference_maps"
    # "reference_maps": MTT floor/cap and CBF floor/cap from MR_MTT / MR_rCBF percentiles (original)
    # "fixed": CBF bounds from HemoPIC's own CBF percentiles, MTT clipped to mtt_fixed
    map_bounds: str = "reference_maps"
    mtt_fixed: List[float] = field(default_factory=lambda: [0.5, 24.0])
    # [C8] source of the CBV map
    # "model_v" (original): ROI value = fitted V, voxel pattern from cbv_texture (clipped)
    # "id_regional": ROI value = indicator dilution AUC(C_roi)/AUC(arterial), same voxel pattern
    # "id_voxel": every voxel = k_cbv * AUC(voxel)/AUC(arterial) (as oSVD), no clipping
    cbv_source: str = "model_v"
    # [C9] voxel CBF pattern inside each ROI (the ROI level always comes from the model)
    # "peak_ttp" (original): peak / (time to peak + dt)
    # "moment": AUC(voxel) / (first moment(voxel) - first moment(arterial)), i.e. CBV/MTT
    #           with MTT from the central volume theorem (consistent with cbv_source=id_voxel)
    # "osvd": voxel CBF from oSVD deconvolution with the arterial curve
    cbf_texture: str = "peak_ttp"
    moment_mtt_bounds: List[float] = field(default_factory=lambda: [0.5, 30.0])  # seconds
    # clip the voxel CBF pattern to [proxy_clip_lo, proxy_clip_hi] (original). With
    # cbv_source=id_voxel the CBV pattern is unclipped, so clipping only CBF breaks the
    # amplitude cancellation in MTT = CBV/CBF; set false for consistent voxel MTT.
    cbf_texture_clip: bool = True

    ### [C6] Arterial input
    # "tissue_mean" (original): C_art = mean curve of high-peak tissue voxels
    # "auto_aif": automatically selected arterial voxels (hemopic_plus/aif.py)
    arterial_input: str = "tissue_mean"
    aif_auc_top: float = 0.10  # candidate fraction by area under the curve
    aif_rough_drop: float = 0.25  # drop this fraction of roughest candidates
    aif_max_voxels: int = 20
    aif_smooth_sigma: float = 0.8  # frames; arterial bolus is narrow, smooth less
    aif_pve: str = "vof"  # "vof": rescale AIF by AUC(VOF)/AUC(AIF); "none"
    aif_pve_max: float = 5.0
    # prior scale for prior_source=tracer:
    # "wm_reference": one factor per subject so that WM has ref_wm_cbf / ref_wm_cbv
    # "absolute": indicator-dilution units from the AIF, CBV = 100 kH/rho * AUC(C)/AUC(AIF)
    #             (needs arterial_input = auto_aif)
    prior_scale: str = "wm_reference"
    k_h: float = 0.73  # large/small vessel hematocrit correction (1 - Hct_large)/(1 - Hct_small)
    rho_brain: float = 1.04  # g/ml

    ### [C4] Robust optimisation
    multistart: int = 0  # extra random restarts for ROIs whose data loss exceeds refit_loss
    refit_loss: float = 0.15

    ### Map construction (original values)
    proxy_clip_lo: float = 0.35
    proxy_clip_hi: float = 2.50
    cbv_ratio_clip_lo: float = 0.50
    cbv_ratio_clip_hi: float = 2.00
    tail_frac: float = 0.20

    ### Runtime
    workers: int = 0  # 0 = number of CPU cores

    def resolve(self, key: str) -> Path:
        p = Path(getattr(self, key)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None) -> Config:
    cfg = Config()
    known = {f.name for f in fields(Config)}
    flat = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
        for k, v in raw.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
    flat.update(overrides or {})
    for k, v in flat.items():
        if k not in known:
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    for key, allowed in (
        ("dt_source", ("header", "tr_table")),
        ("prior_source", ("reference_maps", "tracer")),
        ("cbv_texture", ("reference_maps", "tracer_auc")),
        ("map_bounds", ("reference_maps", "fixed")),
        ("arterial_input", ("tissue_mean", "auto_aif")),
        ("aif_pve", ("vof", "none")),
        ("prior_scale", ("wm_reference", "absolute")),
        ("wk_init", ("fixed_p0", "steady")),
        ("cbv_source", ("model_v", "id_regional", "id_voxel")),
        ("cbf_texture", ("peak_ttp", "moment", "osvd")),
    ):
        if getattr(cfg, key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {getattr(cfg, key)!r}")
    if cfg.prior_scale == "absolute" and cfg.arterial_input != "auto_aif":
        raise ValueError("prior_scale: absolute needs arterial_input: auto_aif")
    return cfg
