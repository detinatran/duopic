# HemoPIC+ — Độ bất định (uncertainty)

HemoPIC(+) chỉ cho ra **một** bộ tham số cho mỗi ROI, nên không biết bản đồ CBF/CBV/MTT và τ đáng tin đến đâu. Phần này thêm cho framework ba thứ:

1. **Khoảng tin cậy cho từng ROI và từng voxel** (Laplace + MCMC), không phải fit lại mô hình;
2. **Độ dao động do pipeline** (ensemble: đổi seed k-means và số cụm K);
3. **Ba cách kiểm tra xem khoảng tin cậy có đúng không**, không cần dữ liệu giả riêng.

Chạy trên 43 bệnh nhân ISLES 2017 (651 ROI) cho ba cấu hình `baseline` (HemoPIC gốc), `full` và `full_aif`. Chạy trên Windows (RTX 4090, 32 luồng; 32 phút cho cả ba cấu hình, gồm fit lại, UQ, ba phép kiểm tra và ensemble) và trên macOS. Hai máy cho kết quả lệch nhau không quá 0.2 điểm phần trăm, do số ngẫu nhiên của MCMC khác nhau giữa hai hệ điều hành. Số liệu trong tài liệu này lấy từ lần chạy trên macOS.

## 1. Phương pháp

### 1.1 Posterior của tham số ROI

`fit_roi_curve` cực tiểu hoá L(θ) = D(θ) + R(θ), trong đó D là MSE có trọng số đã chuẩn hoá, R là prior bậc hai trong không gian log. Ta coi L là âm log-posterior, có một "nhiệt độ" riêng cho mỗi ROI:

  p(θ | c) ∝ exp(−κ·L(θ)),  κ = n_eff · den / (2 · num̂)

- num̂ là sai số bình phương trung bình tại nghiệm fit (ước lượng phương sai nhiễu).
- n_eff = T / (2√π·σ_s) là số mẫu độc lập thực sự. Đường cong ROI đã được làm trơn bằng Gaussian σ_s = 1.8 frame, nên các frame liền nhau không độc lập.
- Nghiệm cực đại của posterior chính là nghiệm fit của HemoPIC(+), và prior giữ nguyên tỉ trọng như trong hàm mục tiêu gốc. Vì vậy không cần đổi mô hình hay fit lại.

Có hai cách tính posterior:
- **Laplace**: xấp xỉ Gaussian N(θ̂, (κH)⁻¹). Hessian H tính chính xác bằng autograd, nhờ bản PyTorch của mô hình (`hemopic_plus/uq/torch_model.py`). Bản này khớp bản numpy tới 1e−16 và có test tự động. Tham số nằm sát cận thì được đánh dấu *pinned*.
- **MCMC**: random-walk Metropolis chạy theo lô cho mọi ROI của một bệnh nhân cùng lúc. Bước nhảy được tiền điều kiện bằng hiệp phương sai Laplace và tự chỉnh để tỉ lệ chấp nhận ≈ 0.25. Chạy 4 chuỗi, 3000 bước burn-in, giữ 6000 bước (thin 4). Hội tụ đánh giá bằng split R-hat và ESS.

Trước khi lấy mẫu, nghiệm Powell của pipeline được tinh chỉnh thêm bằng L-BFGS-B (dùng gradient của torch). Nghiệm mới chỉ được giữ nếu hàm mục tiêu giảm.

Từ các mẫu θ ta tính CBF (dòng ra trung bình ở đuôi), V (= CBV), MTT = 60·V/CBF, τ, F.

### 1.2 Bản đồ voxel

Bản đồ voxel của HemoPIC là bản đồ "texture" được nhân với mức của ROI, nên bất định của voxel lấy từ bất định của mức ROI: với mỗi mẫu posterior, bản đồ đã fit được nhân với tỉ số (mức của mẫu / mức của nghiệm fit). Dùng 200 mẫu để ra `HemoPIC_{CBF,CBV,MTT}_{sd,lo,hi}.nii.gz`. Cách này **không** tính nhiễu riêng của từng voxel.

### 1.3 Ensemble pipeline

Chạy lại toàn bộ pipeline với 7 thành viên: seed k-means 0–4 với (K_GM, K_WM) của config, cộng thêm (6, 4) và (10, 8) với seed 0. Ghi lại độ lệch chuẩn của bản đồ voxel giữa các thành viên, và AUC của từng thành viên.

### 1.4 Ba cách kiểm tra (không cần dữ liệu giả riêng)

| | Ý tưởng | Đạt khi |
|---|---|---|
| **Check 1: giấu frame** | Chia mỗi đường cong thành 5 khối liên tiếp. Lần lượt giấu từng khối, fit lại trên phần còn lại, rồi dự đoán phần bị giấu. | Khoảng 50/80/95% chứa ≈ 50/80/95% số điểm bị giấu |
| **Check 2: bỏ voxel không chắc** | Bỏ q% voxel có khoảng 95% tương đối rộng nhất, rồi tính lại AUC phát hiện tổn thương. So với bỏ q% voxel ngẫu nhiên. | AUC tăng so với bỏ ngẫu nhiên |
| **Check 3: semi-synthetic** | Lấy đường cong đã fit làm "đáp án". Thêm nhiễu cùng phương sai, cùng độ tương quan theo thời gian với phần dư thật. Fit lại bằng chính `fit_roi_curve`, rồi kiểm tra khoảng 95% có chứa đáp án không (3 lần lặp × 651 ROI). | Độ phủ ≈ 95% |

## 2. Kết quả

### 2.1 Dữ liệu thực sự xác định được tham số nào?

![identifiability](figures/uq_identifiability.png)

Tỉ số sd posterior / sd prior: < 0.5 là học được từ dữ liệu, > 0.9 là prior quyết định.

- **HemoPIC gốc**: V và F do prior quyết định ở 88–89% ROI. Prior này lại lấy từ bản đồ Deconv, nên CBF/CBV của HemoPIC gốc phần lớn chỉ là bản đồ Deconv được đưa trở lại. Điều này khớp với chẩn đoán P2 trong `PLAN.md`.
- **Sau khi sửa đơn vị (C5)**: V và F được dữ liệu thu hẹp ở 91–92% ROI.
- R (sức cản Windkessel) gần như không xác định được ở mọi cấu hình.
- **Dispersion dính cận dưới (0.05 s) ở 99% ROI**: dữ liệu không cần đến tham số này.
- Với AIF tự động (`full_aif`), **delay dính cận dưới (−4 s) ở 35% ROI**: mô trông như nhận thuốc sớm hơn động mạch đã chọn. Có thể AIF tự động đang chọn voxel quá xa ở hạ lưu, hoặc cận −4 s quá chặt.

### 2.2 Khoảng tin cậy rộng cỡ nào?

Trung vị của (độ rộng khoảng 95%) / (giá trị), tính trên 651 ROI:

| | CBF | CBV (V) | MTT | τ |
|---|---|---|---|---|
| HemoPIC gốc | 116% | 133% | 185% | 245% |
| HemoPIC+ full | **63%** | **66%** | **69%** | 219% |
| HemoPIC+ full_aif | 85% | 102% | 94% | 256% |

![widths](figures/uq_widths.png)

Sau khi sửa, khoảng CBF/CBV/MTT hẹp đi khoảng một nửa so với HemoPIC gốc. τ vẫn rất bất định ở mọi cấu hình.

Ví dụ một ROI (bệnh nhân 22, ROI trùng tổn thương nhiều nhất và ROI trùng bên đối diện nhiều nhất): đường cong đo được, nghiệm mode và dải 95% của đường cong mô hình. Ô bên phải là mẫu posterior của (log F, log V) so với prior. Ở HemoPIC gốc, mẫu phủ kín vùng prior, tức dữ liệu không thêm thông tin. Ở HemoPIC+, dữ liệu kéo F, V ra khỏi prior. HemoPIC gốc dùng dt = 1 s nên trục thời gian ngắn hơn.

![roi example](figures/uq_roi_example.png)

Hội tụ của MCMC: R-hat > 1.1 ở 7.2% / 5.7% / 5.4% số ROI (lần lượt baseline / full / full_aif), ESS nhỏ nhất có trung vị 355–411. Các ROI chưa hội tụ phần lớn có τ hoặc delay nằm sát cận (posterior bị cắt ở biên).

### 2.3 Khoảng tin cậy có đúng không?

![calibration](figures/uq_calibration.png)

**Check 1, giấu frame.** Tỉ lệ điểm bị giấu nằm trong khoảng (trong ngoặc là khi chỉ tính nhiễu, bỏ qua bất định của tham số):

| | 50% | 80% | 95% |
|---|---|---|---|
| HemoPIC gốc | 61.0% (58.0) | 77.5% (72.6) | 86.6% (80.3) |
| HemoPIC+ full | 56.5% (53.0) | 75.9% (70.3) | 85.6% (78.3) |
| HemoPIC+ full_aif | 55.7% (51.6) | 74.0% (67.9) | 84.0% (77.2) |

Ở mức 95%, khoảng hơi hẹp. Nguyên nhân là phần dư có đuôi dày (các khối bị giấu chứa đỉnh bolus, nơi mô hình fit kém nhất). Bất định của tham số đóng góp thêm 5–7 điểm phần trăm độ phủ, nên phần này là có ích.

**Check 3, semi-synthetic.** Tỉ lệ đáp án nằm trong khoảng 95%:

| | CBF | CBV (V) | MTT | τ |
|---|---|---|---|---|
| HemoPIC gốc | 98.4% | 98.7% | 97.9% | 72.6% |
| HemoPIC+ full | 96.8% | 99.5% | 86.0% | 74.8% |
| HemoPIC+ full_aif | 97.0% | 99.8% | 88.1% | 70.3% |

**Khoảng của CBF và CBV đáng tin. Khoảng của MTT (ở HemoPIC+) và τ quá hẹp**: muốn phủ đủ thì cần nới rộng khoảng 1.3 lần (MTT) và 2 lần (τ). Lý do là posterior của τ và delay thường bị cắt ở cận và có nhiều đỉnh, mà xấp xỉ Gaussian không thể hiện được.

### 2.4 Bất định có giúp phát hiện tổn thương không? (Check 2)

![selective](figures/uq_selective.png)

| | CBF: AUC trước → bỏ 40% voxel bất định nhất | MTT: AUC trước → bỏ 40% voxel bất định nhất |
|---|---|---|
| HemoPIC gốc | 0.761 → 0.699 ↓ | 0.717 → 0.755 ↑ |
| HemoPIC+ full | 0.782 → **0.802** ↑ | 0.761 → **0.786** ↑ |
| HemoPIC+ full_aif | 0.785 → 0.763 ↓ | 0.755 → 0.787 ↑ |

Bỏ voxel ngẫu nhiên thì AUC gần như không đổi (đường chấm).

- **Với MTT, bất định luôn hữu ích**: bỏ voxel không chắc giúp AUC tăng ở cả ba cấu hình.
- **Với CBF, chỉ `full` tăng.** Ở HemoPIC gốc và `full_aif`, vùng tổn thương lại chính là chỗ bất định nhất. Khoảng CBF tương đối của tổn thương so với bên đối diện là 157% so với 110% (HemoPIC gốc) và 82% so với 74% (`full_aif`); tổn thương bất định hơn ở 81% và 72% số bệnh nhân. Bỏ voxel bất định tức là bỏ chính voxel tổn thương, nên AUC giảm. Như vậy bất định ở đây không phải là "sai số" mà là một đặc trưng của vùng thiếu máu: dòng thấp thì tín hiệu yếu.

### 2.5 Kết luận về τ của bài báo khi tính cả bất định

![tau](figures/uq_tau.png)

Bài báo kết luận vùng tổn thương có τ nhỏ hơn (hồi phục nhanh hơn) so với bên đối diện. Hiệu τ(tổn thương) − τ(đối diện), lấy trung vị có trọng số trên cả cohort, với mỗi mẫu posterior:

| | Tại nghiệm fit | Trung vị posterior | 95% CI | P(tổn thương nhanh hơn) |
|---|---|---|---|---|
| HemoPIC gốc | −4.2 s | −4.2 s | [−6.4, −2.1] | 1.00 |
| HemoPIC+ full | +6.5 s | +6.3 s | [+3.7, +9.4] | 0.00 |
| HemoPIC+ full_aif | +30.7 s | +16.8 s | [+11.0, +22.9] | 0.00 |

Kết luận của bài báo đúng với mô hình gốc. Nhưng **khi sửa lỗi đơn vị, kết luận đảo chiều một cách chắc chắn**: CI không chứa 0, kể cả khi nới khoảng của τ ra 2 lần như Check 3 gợi ý. Với `full_aif`, giá trị tại nghiệm fit (+30.7 s) khác xa trung vị posterior (+16.8 s), vì Powell dừng sớm ở nhiều ROI (xem 2.7).

### 2.6 Bất định do pipeline so với bất định của mô hình

| | AUC CBF giữa 7 thành viên | sd tương đối voxel CBF: pipeline / posterior | CBV | MTT |
|---|---|---|---|---|
| HemoPIC gốc | 0.757–0.765 | 2.5% / 27.9% | 2.5% / 28.3% | 4.9% / 38.4% |
| HemoPIC+ full | 0.776–0.783 | 7.1% / 15.3% | 9.6% / 15.5% | 7.2% / 15.5% |
| HemoPIC+ full_aif | 0.779–0.787 | 7.7% / 18.2% | 8.2% / 19.3% | 9.3% / 15.6% |

Đổi seed hay số cụm làm AUC thay đổi không quá 0.008, nhỏ hơn mức cải thiện baseline → full (+0.020). Ở HemoPIC+, bất định do pipeline bằng khoảng một nửa bất định của mô hình, nên không thể bỏ qua. HemoPIC gốc có ít bất định pipeline vì F, V bị giữ ở prior.

![maps](figures/uq_maps_example.png)

### 2.7 Phát hiện phụ: Powell dừng sớm

Khi tinh chỉnh bằng L-BFGS-B, mức CBF của ROI thay đổi hơn 5% ở 9% / 8% / 22% số ROI (baseline / full / full_aif). Nghĩa là bộ tối ưu Powell của pipeline thường dừng trước khi tới cực tiểu, nhất là với AIF tự động. Có thể thêm một bước đánh bóng bằng gradient vào `fit_roi_curve` gần như không tốn thêm thời gian.

### 2.8 Bất định do AIF

`scripts/run_aif_ensemble.py` (số liệu `docs/results/full_aif/uq/aif_ensemble.{csv,json}`): 10 AIF cho mỗi bệnh nhân (bootstrap các voxel động mạch và nhiễu hệ số hiệu chỉnh thể tích từng phần; 2 thành viên cuối dùng cụm động mạch thứ hai), chạy lại toàn bộ pipeline ROI. 36 bệnh nhân có AIF tự động được chấp nhận.

- AUC gần như không đổi: CBF 0.787–0.792, CBV 0.547–0.562, MTT 0.761–0.768 giữa các thành viên; DuoPIC − oSVD giữ dấu ở mọi thành viên.
- Giá trị tuyệt đối dao động mạnh hơn: CV giữa các thành viên của median trong não là CBF 0.19, CBV 0.16, MTT 0.18 (oSVD MTT 0.07).
- **Giữa các bệnh nhân** (31 bn có đủ 11 AIF, thang log): MTT median toàn não lệch giữa bệnh nhân sd 0.41, do AIF sd 0.23, ICC 0.65; vùng nhồi máu ICC 0.93; oSVD ICC 0.88. Bootstrap và PVE chỉ đổi MTT khoảng 5% và giữ thứ hạng bệnh nhân (ρ = 0.99); đổi cụm động mạch làm MTT lệch khoảng 56% và đảo thứ hạng (ρ ≈ −0.3). Một ngưỡng tuyệt đối chung cho cả cohort cho AUC MTT 0.749 so với 0.764 khi dùng bản đồ tương đối.
- Kết luận: giá trị tuyệt đối so sánh được giữa các bệnh nhân khi đã cố định cụm động mạch; chọn cụm là nguồn bất định lớn nhất của AIF.

### 2.9 Bất định ở mức voxel (DuoPIC-vox)

`scripts/run_voxel_uq.py` (số liệu `docs/results/full_aif/vox_uq/`), 43 bệnh nhân, 3000 voxel mỗi bệnh nhân (một phần ba trong vùng nhồi máu):

- Laplace có nhiệt độ cho từng voxel (κ_v từ phần dư của chính voxel); bất định của ROI truyền xuống bằng định lý hàm ẩn (dx/dθ) và hiệp phương sai của các mẫu posterior ROI.
- Coverage 95% với bậc tự do dư (n_eff − 3): giấu frame 0.91, bán tổng hợp 0.90–0.94 (log V 0.94, log Cin 0.90, delay 0.94). Không trừ bậc tự do thì khoảng hẹp quá (0.85–0.90).
- Phần do ROI chiếm khoảng 11% phương sai của log MTT. Tuyến tính hoá khớp với refit dưới 20 mẫu posterior ROI (bn 4, 10, 22: tỉ lệ sd 0.88, Spearman 0.82).
- Độ rộng khoảng 95% của MTT ở voxel khá lớn (gấp khoảng 1.6–1.8 lần giá trị); posterior/prior sd của log V trung vị 0.62, 30% voxel do dữ liệu quyết định (< 0.5), 19% do prior (> 0.9).

### 2.10 PDE ngẫu nhiên (SPDE)

`hemopic_plus/npde/stochastic.py`, `scripts/run_spde.py` (số liệu `docs/results/full_aif/spde/{tank,npde}/`): nhiễu trắng ở lối vào ống (điều kiện biên), nên mỗi hiện thực vẫn bảo toàn khối lượng và MTT. Với θ cố định, đường cong mô là C = Φ(u + η) và dữ liệu đã làm mượt là một Gaussian process, nên likelihood tính chính xác và frame bị giấu được dự đoán bằng điều kiện hoá Gaussian. Fit 2 bước: θ của mô hình vật lý, rồi mức nhiễu σ_u, σ_v.

| 651 ROI, 43 bn | tank: tất định → SPDE | neural PDE: tất định → SPDE |
|---|---|---|
| Coverage giấu frame 50 / 80 / 95% | 0.50 / 0.65 / 0.77 → 0.56 / 0.73 / 0.82 (p = 2e−13) | 0.51 / 0.67 / 0.77 → 0.56 / 0.73 / 0.81 (p = 4e−8) |
| Log score (nat / frame) | +0.24 (p = 3e−6) | +0.13 (p = 0.006) |
| σ_u (theo biên độ lối vào) | 12% | 9% |

- Bản đồ và nRMSE không đổi. Khoảng tin cậy vẫn hẹp hơn mức danh nghĩa vì chưa cộng bất định tham số.
- Neural PDE ở đây được train lại (5 fold, `docs/results/full_aif/npde_rerun/`, nMSE giữ lại: PDE 0.0445 → neural PDE 0.0265, neural tốt hơn ở 63% ROI); lần train trước cho SPDE log score +0.15 (p = 0.014), cùng kết luận.
- Fit đồng thời θ và nhiễu không dùng được: nhiễu thay luôn phần động lực học (nRMSE của đường cong không nhiễu tăng 2–20 lần, σ_u chạm cận).

## 3. Hạn chế

- Nhiệt độ κ giả định nhiễu Gaussian, phương sai không đổi theo thời gian và n_eff cố định. Check 1 cho thấy giả định này hơi lạc quan.
- Bất định voxel chỉ lấy từ mức ROI, không có nhiễu riêng của từng voxel và không có bất định của texture.
- Bất định của AIF chỉ có dạng ensemble (mục 2.8), chưa có mô hình Bayes đầy đủ.
- τ và delay có posterior nhiều đỉnh hoặc bị cắt ở cận. Với hai tham số này nên dùng mẫu MCMC chứ không dùng Laplace, và nên đọc khoảng tin cậy một cách thận trọng.

## 4. Cách chạy

```bash
# sau khi đã fit (scripts/run_cohort.py) — fit lưu thêm dt và w_curve vào fit_curves.npz
python scripts/run_uq.py configs/full.yaml --workers 8               # Laplace + MCMC, bản đồ *_sd/lo/hi
python scripts/eval_heldout.py configs/full.yaml --workers 8         # Check 1
python scripts/eval_semisynth.py configs/full.yaml --workers 8       # Check 3
python scripts/eval_uq.py configs/full.yaml --data-root <data>       # tổng hợp, Check 2, τ
python scripts/run_ensemble.py configs/full.yaml --data-root <data>  # ensemble pipeline
python scripts/make_uq_figures.py --data-root <data>                 # docs/figures/uq_*.png
python tests/test_uq_torch.py                                        # bản torch khớp bản numpy
python scripts/run_aif_ensemble.py configs/full_aif.yaml --data-root <data> --members 10   # mục 2.8
python scripts/run_voxel_uq.py configs/full_aif.yaml --data-root <data>                 # mục 2.9
python scripts/run_spde.py configs/full_aif.yaml --data-root <data> --noise additive    # mục 2.10
```

Đầu ra nằm trong `outputs/<config>/training_XX/uq/`, `.../ens/` và `outputs/<config>/eval_uq/`. Bản tóm tắt theo cohort (`summary.json`, `ensemble.json`, `heldout.csv.gz`, `semisynth.csv.gz`, `tau_diff_draws.npy`) được lưu trong `docs/results/<config>/uq/`.

Uncertainty của neural PDE (cùng công cụ, cùng ba phép kiểm tra, cộng bất định do mạng): xem `docs/NEURAL_PDE.md`, mục 6.
