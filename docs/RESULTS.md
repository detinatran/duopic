# HemoPIC+ — Kết quả thí nghiệm

43 bệnh nhân ISLES 2017 (bản Zenodo), cùng dữ liệu preprocess và cùng nhãn mô cho mọi cấu hình. Chạy ngày 2026-10-08 trên macOS (M1 Pro); chạy lại trên Windows (32 luồng) cho kết quả trùng khớp. Số liệu thô của từng bệnh nhân nằm trong `docs/results/<config>/`.

## 1. Bảng tổng hợp

nRMSE: càng thấp càng tốt. AUC: càng cao càng tốt. "AUC" là trung bình theo bệnh nhân, "pooled" là gộp voxel của cả cohort.

| experiment | nRMSE GM | nRMSE WM | nRMSE lesion | nRMSE c-lesion | nRMSE brain | AUC CBF | AUC CBV | AUC MTT | pooled CBF | pooled CBV | pooled MTT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (HemoPIC gốc) | 0.106 | 0.094 | 0.318 | 0.191 | 0.093 | 0.7612 | 0.5803 | 0.7166 | 0.7595 | 0.5919 | 0.6738 |
| c1_tr | 0.106 | 0.093 | 0.318 | 0.189 | 0.092 | 0.7610 | 0.5909 | 0.7108 | 0.7576 | 0.5931 | 0.6742 |
| c15_units | 0.189 | 0.171 | 0.252 | 0.256 | 0.173 | 0.7673 | 0.5090 | 0.7355 | 0.7759 | 0.5169 | 0.7214 |
| c152_delay | 0.095 | 0.074 | 0.269 | 0.151 | 0.081 | 0.7645 | 0.5421 | 0.7384 | 0.7773 | 0.5348 | 0.7370 |
| full_ref | 0.086 | 0.074 | 0.270 | 0.151 | 0.077 | 0.7658 | 0.5333 | 0.7460 | 0.7783 | 0.5368 | 0.7412 |
| c2_delay_only | 0.056 | 0.046 | 0.305 | 0.137 | 0.042 | 0.7627 | 0.5956 | 0.7008 | 0.7582 | 0.6057 | 0.6646 |
| c3_reffree | 0.105 | 0.091 | 0.308 | 0.181 | 0.088 | 0.7831 | 0.5554 | 0.7514 | 0.7704 | 0.5411 | 0.7597 |
| **full (HemoPIC+)** | **0.081** | **0.072** | **0.260** | **0.144** | **0.069** | **0.7816** | 0.5460 | **0.7607** | **0.7720** | 0.5344 | **0.7720** |
| Deconv (ISLES) | – | – | – | – | – | 0.7472 | 0.5823 | 0.6960 | 0.7450 | 0.5927 | 0.6388 |

## 2. Kiểm định cặp: baseline → full (43 bệnh nhân)

Khoảng tin cậy 95% bằng bootstrap theo bệnh nhân; p-value của kiểm định Wilcoxon signed-rank hai phía.

| metric | baseline | full | Δ | 95% CI | p |
|---|---|---|---|---|---|
| AUC CBF | 0.7612 | 0.7816 | +0.020 | [+0.005, +0.037] | 0.010 |
| AUC CBV | 0.5803 | 0.5460 | −0.034 | [−0.064, −0.001] | 0.007 |
| AUC MTT | 0.7166 | 0.7607 | +0.044 | [+0.010, +0.080] | 0.006 |
| nRMSE GM | 0.106 | 0.081 | −0.026 | [−0.035, −0.018] | 9e−8 |
| nRMSE WM | 0.094 | 0.072 | −0.022 | [−0.036, −0.012] | 1e−7 |
| nRMSE lesion | 0.318 | 0.260 | −0.059 | [−0.074, −0.046] | 8e−12 |
| nRMSE c-lesion | 0.191 | 0.144 | −0.047 | [−0.067, −0.035] | 5e−13 |
| nRMSE brain | 0.093 | 0.069 | −0.023 | [−0.037, −0.015] | 2e−9 |

Lệnh tái tạo: `python scripts/paired_test.py baseline full`.

## 2b. Đánh giá bằng script `Eval/` gốc của HemoPIC (chạy trên Windows)

Đây là các thước đo còn lại của bài báo: CVT (Fig. 4), τ (Fig. 5c–d) và tỉ lệ vùng (Fig. 3). **Ở các thước đo này, HemoPIC+ không tốt hơn** và cần được báo cáo đúng như vậy.

| config | CVT slope GM / WM / Lesion / Brain | CVT r (Brain) | τ trung vị GM / WM / lesion / c-lesion |
|---|---|---|---|
| baseline | 1.30 / 1.07 / 1.26 / 1.21 | 0.86 | 14.2 / 12.6 / 10.0 / 13.8 (đơn vị: frame) |
| c1_tr | 1.37 / 1.07 / 0.93 / 1.26 | 0.93 | 20.6 / 19.0 / 13.8 / 20.2 s |
| c15_units | 0.29 / 0.27 / 0.49 / 0.28 | 0.44 | 16.3 / 14.4 / 19.0 / 17.1 s |
| c152_delay | 0.25 / 0.28 / 0.43 / 0.26 | 0.76 | 13.2 / 11.2 / 17.9 / 11.5 s |
| full_ref | 0.26 / 0.29 / 0.39 / 0.26 | 0.80 | 12.4 / 11.0 / 19.7 / 11.6 s |
| c3_reffree | −2.18 / −0.47 / 0.05 / −1.50 | −0.12 | 14.1 / 12.5 / 10.0 / 13.8 (frame) |
| full | −0.23 / −0.09 / −0.07 / −0.17 | −0.10 | 13.3 / 11.2 / 18.1 / 12.1 s |

Bài báo: slope 1.22 / 1.03 / 1.17 / 1.15; τ 13.95 / 12.56 / 10.41 / 13.71.

Fig. 3 (trung vị qua 43 bệnh nhân). Tỉ lệ lesion / c-lesion: CBF 0.58 (baseline), 0.53 (full), 0.61 (Deconv); CBV 0.89, 0.99, 0.83. Tỉ lệ WM / GM: CBF 0.65, 0.67, 0.64.

- **C1 giữ nguyên CVT** (r tăng từ 0.86 lên 0.93), và τ giờ có đơn vị giây thật (≈ 1.44 × số frame).
- **C5 làm MTT tuyệt đối nhỏ đi khoảng 3–4 lần** (slope ≈ 0.27) nhưng vẫn giữ tương quan giữa các bệnh nhân (r 0.76–0.80 khi có thêm C2). Lý do: C_art là trung bình các voxel mô có đỉnh cao chứ không phải động mạch, nên mô hình chỉ đo được thời gian chuyển tiếp *so với mô trung bình*. Muốn có MTT tuyệt đối cần một AIF thật, ví dụ lấy từ voxel mạch máu.
- **C3 xoá tương quan CVT giữa các bệnh nhân.** Điều này do thiết kế: mỗi bệnh nhân được hiệu chuẩn để chất trắng có CBF = 22 và CBV = 2, nên MTT trung bình gần như cố định. HemoPIC+ ở chế độ không tham chiếu cho ra bản đồ **tương đối** trong từng bệnh nhân. Phép so trong một bệnh nhân (ROC) vẫn hợp lệ, phép so mức tuyệt đối giữa các bệnh nhân (CVT) thì không.
- **Kết luận về τ của bài báo đảo chiều khi sửa đơn vị.** Ở baseline, C1 và C3, vùng tổn thương hồi phục *nhanh hơn* vùng đối bên (giống bài báo). Ở mọi cấu hình có C5, tổn thương hồi phục *chậm hơn* (khoảng 18–20 s so với 12 s). Như vậy phát hiện "lesions relax faster" của bài báo phụ thuộc vào mô hình bị sai đơn vị.

## 2c. C6 — AIF tự động — và C7 — điều kiện đầu Windkessel

**C6** thay đường "động mạch" trung bình mô bằng AIF chọn tự động:
- lấy các voxel có AUC cao nhất trong số các đường cong bolus sạch;
- gom cụm hình dạng hai vòng, chọn cụm đến sớm nhất (Mouridsen 2006);
- hiệu chỉnh PVE bằng VOF (xoang tĩnh mạch).

Với AIF thật, prior được tính theo đơn vị tuyệt đối (CBV = 100·k_H/ρ·∫C/∫AIF) và không cần hiệu chuẩn theo chất trắng. Hình QC: `docs/figures/aif_qc.png` và `aif_qc.csv`.

- AIF đạt chuẩn ở **36/43** bệnh nhân: đến sớm hơn mô trung vị 6.8 s, FWHM 9.1 s so với 11.5 s của mô, đỉnh cao gấp khoảng 16 lần. Hệ số PVE trung vị 1.87. 7 bệnh nhân không đạt (2, 28, 40, 42, 43, 45, 46) tự quay về đường trung bình mô; 42, 43, 45 là dữ liệu nhiễu nặng.
- Prior tuyệt đối cao hơn sinh lý khoảng 3–5 lần (ví dụ CBV chất trắng 6–30 ml/100g). Đây là hạn chế đã biết của DSC (bão hoà tín hiệu động mạch, ΔR2* phi tuyến trong máu). Sai số này không ảnh hưởng tới MTT (là tỉ số) và ROC (là thước đo tương đối).

**C7** cho Windkessel bắt đầu ở trạng thái dừng (Q(0) = F/R thay vì p0/R với p0 = 1).

| config | AUC CBF | AUC CBV | AUC MTT | nRMSE brain | CVT r (Brain) | MTT mô hình / Deconv | Spearman MTT theo ROI |
|---|---|---|---|---|---|---|---|
| full | 0.782 | 0.546 | **0.761** | **0.069** | −0.10 | 0.55 | – |
| full_aif_ref (AIF + prior tham chiếu) | 0.770 | 0.541 | 0.741 | 0.121 | 0.25 | 0.58 | 0.67 |
| full_aif_wm (AIF + hiệu chuẩn WM) | **0.787** | 0.552 | 0.755 | 0.112 | −0.11 | 0.69 | – |
| **full_aif** (AIF + prior tuyệt đối) | 0.785 | **0.560** | 0.755 | 0.117 | −0.02 | **0.83** | 0.56 |
| full_aif_ss (+C7) | 0.790 | 0.563 | 0.755 | 0.116 | −0.07 | 0.80 | – |
| full_ref_ss (+C7) | 0.769 | 0.545 | 0.741 | 0.079 | 0.80 | 0.41 | – |

Kiểm định cặp full → full_aif: AUC CBF +0.003 (p = 0.13), AUC CBV +0.014 (p = 0.03), AUC MTT −0.006 (p = 0.01), nRMSE toàn não +0.047 (p = 1e−4).
Kiểm định cặp baseline → full_aif: AUC CBF +0.024 (p = 0.002), AUC MTT +0.038 (p = 0.02), nRMSE vùng tổn thương −0.048 (p = 7e−6).

Kết luận cho C6 và C7:
- **AIF thật sửa được thang đo tuyệt đối của MTT** (trung vị theo ROI 5.6 s so với 6.2 s của Deconv; tỉ số trung bình não từ 0.55 lên 0.83). Đồng thời nó **loại bỏ hằng số hiệu chuẩn tuỳ ý** 22 và 2.0, mà vẫn giữ được mức phân biệt tổn thương của `full`.
- **AIF thật không khôi phục CVT giữa các bệnh nhân** (r ≈ 0). MTT trung bình của Deconv dao động 3–15 s giữa các bệnh nhân, còn HemoPIC+ không theo dao động đó. Chưa rõ dao động này là sinh lý, hay do cách xử lý của Deconv ISLES (AIF thủ công, TR, phần mềm khác nhau). Để phân biệt cần một phương pháp deconvolution độc lập dùng chính AIF của C6 (mục 5).
- AIF sắc nét làm nRMSE ở mô lành tăng (0.069 → 0.117): một ngăn đơn với AIF thật khó tái tạo đường cong mô hơn so với khi dùng proxy trung bình mô.
- **C7 không có tác dụng.** Giả thuyết "chuyển tiếp Windkessel che lấp MTT" bị bác bỏ. Ở trạng thái dừng, τ không nhận diện được (dính ở 55 s). Không khuyến nghị dùng C7.
- τ vùng tổn thương ở các cấu hình có AIF lên tới 45–55 s, gần cận trên. Thông số Windkessel ở vùng tổn thương không đáng tin trong mọi biến thể đã thử.

**Khuyến nghị:**
- dùng **`full`** khi mục tiêu là phân biệt tổn thương và tái tạo đường cong;
- dùng **`full_aif`** khi cần MTT có đơn vị tuyệt đối và không muốn phụ thuộc hằng số hiệu chuẩn.

## 2d. Tham chiếu độc lập: oSVD với AIF của C6

`scripts/osvd_check.py` chạy deconvolution oSVD (Wu et al. 2003: block-circulant SVD, ngưỡng theo chỉ số dao động cho từng voxel), với cùng AIF tự động và TR thật. Phép này hoàn toàn độc lập với mô hình HemoPIC. Chạy trên 36 bệnh nhân có AIF đạt chuẩn; số liệu thô ở `docs/results/osvd/`.

**AUC ROC tổn thương trên cùng 36 bệnh nhân:**

| method | CBF | CBV | MTT |
|---|---|---|---|
| Deconv (ISLES) | 0.745 | 0.584 | 0.700 |
| HemoPIC gốc | 0.773 | 0.592 | 0.722 |
| HemoPIC+ full | 0.786 | 0.535 | **0.773** |
| HemoPIC+ full_aif | **0.790** | 0.552 | 0.766 |
| **oSVD + AIF C6** | 0.782 | **0.623** | 0.759 |

Kiểm định cặp (Wilcoxon, chênh lệch oSVD − X):
- so với ISLES: CBF +0.037 (p = 4e−10), CBV +0.039 (p = 2e−6), MTT +0.060 (p = 0.004);
- so với full: CBF −0.003 (p = 0.9), CBV **+0.088** (p = 5e−8), MTT −0.014 (p = 0.014);
- so với full_aif: CBF −0.007 (p = 0.4), CBV +0.071 (p = 2e−6), MTT −0.007 (p = 0.11).

**Tương quan MTT trung bình toàn não giữa các bệnh nhân (Pearson r):** oSVD so với ISLES 0.22; full_aif so với oSVD −0.01; full so với oSVD 0.54; baseline so với ISLES 0.95.

**Hệ quả, rất quan trọng cho mọi tuyên bố về đóng góp:**
1. **Phần lớn lợi thế ROC của HemoPIC, cả bản gốc lẫn bản +, so với Deconv ISLES đến từ khâu xử lý (AIF tự động, TR thật), không phải từ mô hình vật lý.** Một oSVD chuẩn với cùng AIF đã thắng ISLES ở cả 3 bản đồ, ngang HemoPIC+ ở CBF và MTT, và tốt hơn hẳn ở CBV. Phép so "HemoPIC so với Deconv ISLES" trong bài báo gốc trộn lẫn mô hình với khác biệt xử lý. Tham chiếu công bằng phải là deconvolution dùng *cùng* AIF và TR.
2. **CVT so với ISLES không phải thước đo công bằng.** Ngay cả oSVD cũng chỉ tương quan r = 0.22 với MTT của ISLES giữa các bệnh nhân. Dao động MTT giữa các bệnh nhân của ISLES chủ yếu phản ánh cách xử lý của chính ISLES. HemoPIC gốc đạt r = 0.95 chỉ vì prior được copy từ ISLES.
3. **Dù vậy, HemoPIC+ full_aif vẫn không theo được oSVD dùng cùng AIF** (r ≈ 0). Như vậy MTT theo mức giữa các bệnh nhân vẫn chưa được mô hình nhận diện. Đây là hạn chế thật của mô hình, không phải do tham chiếu.
4. Ưu thế còn lại của mô hình HemoPIC+ so với oSVD chỉ là MTT nhỉnh hơn một chút (+0.014, p = 0.014, ở `full`). Đổi lại, nó kém rõ ở CBV.

## 2e. C8 — CBV theo pha loãng chỉ thị (thu hẹp khoảng cách CBV với oSVD)

`cbv_source` quyết định cách dựng bản đồ CBV; CBF vẫn lấy từ mô hình:
- `model_v` (gốc): giá trị ROI = V của mô hình;
- `id_regional`: giá trị ROI = k·AUC(C_ROI)/AUC(động mạch), texture voxel giữ như cũ (kẹp trong [0.5, 2]);
- `id_voxel`: mỗi voxel = k·AUC(voxel)/AUC(động mạch), giống oSVD, không kẹp.

Chạy trên Windows, 43 bệnh nhân. Hai cột cuối là trên cùng 36 bệnh nhân có AIF đạt chuẩn, để so với oSVD.

| config | AUC CBF | AUC CBV | AUC MTT | CBV (36 bn) | MTT (36 bn) |
|---|---|---|---|---|---|
| full | 0.782 | 0.546 | 0.761 | 0.535 | 0.773 |
| **full_idreg** | 0.782 | **0.586** | 0.756 | 0.581 | 0.768 |
| full_idvox | 0.782 | **0.618** | 0.700 | **0.623** | 0.706 |
| full_aif_idreg | 0.785 | 0.586 | 0.750 | 0.581 | 0.761 |
| full_aif_idvox | 0.785 | 0.618 | 0.706 | 0.623 | 0.713 |
| oSVD + AIF C6 | – | – | – | 0.623 | 0.759 |

Kiểm định cặp (Wilcoxon):
- full → full_idreg (43 bn): CBV **+0.040** (p = 7e−13), MTT −0.005 (p = 0.03), CBF không đổi;
- full → full_idvox: CBV +0.072 (p = 8e−7), MTT **−0.061** (p = 2e−7);
- full_idreg − oSVD (36 bn): CBF +0.003 (p = 0.9), CBV −0.042 (p = 1e−4), MTT **+0.009** (p = 0.018);
- full_idvox − oSVD: CBV +0.0004 (p = 0.19, ngang nhau), MTT −0.053 (p = 2e−4).

Kết luận:
- **`id_regional` là lựa chọn cân bằng.** Nó giảm một nửa khoảng cách CBV với oSVD (−0.088 → −0.042) và chỉ mất 0.005 MTT. Đây là cấu hình duy nhất **không kém oSVD ở bản đồ nào ngoài CBV**, và nhỉnh hơn oSVD ở MTT một cách có ý nghĩa.
- `id_voxel` cho CBV ngang oSVD, nhưng MTT = 60·CBV/CBF bị hỏng, vì CBV voxel (theo AUC) và texture CBF (peak/TTP) không nhất quán ở mức voxel. Chỉ nên dùng khi bản đồ CBV là mục tiêu chính.
- AUC CBV của full và full_aif như nhau khi dùng C8, vì ROC tính trên bản đồ tương đối nên hệ số hiệu chuẩn triệt tiêu.
- Lưu ý: với C8, CBV không còn do mô hình HemoPIC sinh ra mà là phép pha loãng chỉ thị cổ điển; mô hình chỉ còn đóng góp CBF và MTT.

**Khuyến nghị cập nhật:** `full_idreg` là cấu hình mặc định để phân biệt tổn thương. Dùng `full_aif_idreg` khi cần MTT theo đơn vị tuyệt đối. Xem thêm mục 2f.

## 2f. C9 — phân bố CBF voxel nhất quán với CBV voxel

Với `id_voxel`, MTT = 60·CBV/CBF bị hỏng. Nguyên nhân: CBV voxel theo AUC và phân bố CBF theo peak/TTP không nhất quán, và chỉ riêng CBF bị kẹp trong [0.35, 2.5], nên nhiễu biên độ không triệt tiêu khi chia.

`cbf_texture` chọn phân bố CBF *bên trong* mỗi ROI; mức CBF của ROI vẫn do mô hình fit:
- `peak_ttp` (gốc);
- `moment`: AUC / (m1_voxel − m1_động mạch), tức CBV / MTT với MTT theo định lý thể tích trung tâm;
- `osvd`: CBF voxel từ oSVD với cùng đường động mạch.

`cbf_texture_clip: false` bỏ bước kẹp, để biên độ trong CBV/CBF triệt tiêu như oSVD. Hậu tố `_nc` trong tên config nghĩa là không kẹp.

AUC trên 43 bệnh nhân:

| config | CBF | CBV | MTT |
|---|---|---|---|
| full_idvox_moment | 0.774 | 0.618 | 0.673 |
| full_idvox_osvd | 0.783 | 0.618 | 0.704 |
| full_idvox_moment_nc | 0.779 | 0.618 | 0.695 |
| full_idvox_osvd_nc | 0.794 | 0.618 | 0.737 |
| full_aif_idvox_osvd_nc | **0.799** | 0.618 | 0.736 |

Trên cùng 36 bệnh nhân so với oSVD:

| method | CBF | CBV | MTT | TB 3 bản đồ |
|---|---|---|---|---|
| oSVD + AIF C6 | 0.782 | 0.623 | 0.759 | 0.722 |
| full_idreg | 0.786 | 0.581 | **0.768** | 0.712 |
| full_idvox_osvd_nc | 0.800 | 0.623 | 0.746 | 0.723 |
| **full_aif_idvox_osvd_nc** | **0.807** | **0.623** | 0.745 | **0.725** |

Kiểm định cặp (Wilcoxon):
- full_aif_idvox_osvd_nc − oSVD: CBF **+0.024** (p = 3e−4), CBV +0.0004 (p = 0.19), MTT −0.014 (p = 0.03);
- full_aif_idvox_osvd_nc − full_idreg: CBF +0.021 (p = 2e−4), CBV +0.042 (p = 6e−5), MTT −0.023 (p = 8e−4).

Kết luận:
- **Bỏ bước kẹp là then chốt.** Với phân bố oSVD, MTT tăng từ 0.704 lên 0.737.
- **`moment` không hiệu quả** (MTT ≈ 0.69): hiệu moment bậc nhất trên từng voxel quá nhiễu và bị ảnh hưởng bởi pha tái tuần hoàn.
- **`full_aif_idvox_osvd_nc` là cấu hình đầu tiên vượt oSVD có ý nghĩa ở CBF** (+0.024) mà vẫn ngang ở CBV. Lợi thế CBF này đến từ *mức CBF theo vùng do mô hình HemoPIC+ fit*, nhân với phân bố voxel của oSVD. Đây là đóng góp đo được của mô hình vật lý so với deconvolution thuần.
- **Chưa có cấu hình nào thắng oSVD ở cả 3 bản đồ.** `idvox_osvd_nc` mạnh ở CBF và CBV nhưng kém nhẹ ở MTT; `idreg` mạnh ở MTT nhưng kém ở CBV.

**Khuyến nghị:**
- `full_aif_idvox_osvd_nc`: tốt nhất về trung bình 3 bản đồ, và tốt nhất ở CBF;
- `full_idreg`: khi MTT là bản đồ chính.

## 3. Diễn giải

1. **Về phân biệt tổn thương và tái tạo nồng độ, HemoPIC+ (full) tốt hơn HemoPIC gốc ở 7/8 thước đo, có ý nghĩa thống kê**, dù **không đọc bất kỳ giá trị nào của bản đồ Deconv**. Đổi lại, nó mất khả năng định lượng tuyệt đối giữa các bệnh nhân (CVT, mục 2b). **Lưu ý:** phần lớn lợi thế ROC này cũng đạt được bằng oSVD chuẩn với cùng AIF và TR (mục 2d), nên không thể quy toàn bộ cho mô hình vật lý. Sai số tái tạo nồng độ giảm 19–25% ở mọi vùng, trong đó vùng tổn thương giảm từ 0.318 xuống 0.260. AUC MTT tăng 0.044, và AUC pooled của MTT tăng từ 0.674 lên 0.772.
2. **Phát hiện then chốt: ở HemoPIC gốc, F và V sau khi fit gần như bằng đúng prior** (tỉ lệ trung vị 1.00 ở mọi loại mô). Đây là hệ quả của vấn đề đơn vị P2: ngăn tracer không có trí nhớ nên dữ liệu không ràng buộc được F và V. Vì vậy bản đồ CBF/CBV gốc về bản chất là trung vị bản đồ Deconv trong từng ROI, nhân với texture. C5 làm mô hình thực sự học từ dữ liệu.
3. **Từng đóng góp:**
   - **C1** (TR thật): tác động nhỏ, nhưng là sửa lỗi về đúng đắn vật lý. Các τ được báo cáo giờ có đơn vị giây thật.
   - **C5** (đơn vị): một mình C5 làm tăng sai số ở mô lành (0.106 → 0.189), nhưng tăng AUC MTT và giảm sai số vùng tổn thương.
   - **C2** (độ trễ): kết hợp với C5 thì sửa được phần sai số ở mô lành. Riêng tham số độ phân tán luôn nằm ở cận dưới (0.05 s), nghĩa là ngăn tracer của C5 đã tự tạo ra độ phân tán, chỉ có độ trễ (trung vị −2.7 s) là có tác dụng. Có thể bỏ tham số phân tán cho gọn.
   - **C4** (multi-start): được dùng cho 126/651 ROI. Số ROI hỏng nặng (loss > 1) giảm từ 46 xuống 27.
   - **C3** (không tham chiếu): tự nó đã tăng AUC CBF và MTT. Điều này cho thấy prior lấy từ bản đồ Deconv còn kéo bản đồ về phía các lỗi của chính Deconv.
4. **Cảnh báo về "fit đẹp":** `c2_delay_only` cho nRMSE mô lành thấp nhất (GM 0.056) nhưng AUC MTT lại *giảm*. Fit đường cong tốt hơn không đồng nghĩa với bản đồ tốt hơn khi mô hình sai đơn vị.

## 4. Hạn chế (cần nói rõ khi báo cáo)

- **CBV kém đi** (0.580 → 0.546). Đã đo được: trung vị tỉ lệ CBV vùng tổn thương / CBV toàn não là 0.888 ở `full`, so với 0.719 ở baseline và 0.749 ở Deconv. Tỉ lệ bệnh nhân có CBV vùng tổn thương cao hơn não là 33%, so với 21% và 19%. Nghĩa là khi V được học từ dữ liệu, vùng tổn thương không còn "thấp CBV" rõ như trước. *Giả thuyết* (chưa kiểm chứng): V của mô hình hấp thụ cả phần bolus đến muộn, hoặc phản ánh hiện tượng tăng CBV do tự điều hoà ở vùng penumbra. Bản thân CBV của Deconv cũng chỉ đạt AUC 0.58, nên đây là chỉ dấu yếu. Hướng xử lý: xem mục 5.
- **Không có tập kiểm tra riêng.** Các siêu tham số mới (cận độ trễ, refit_loss = 0.15, hằng số hiệu chuẩn 22/2.0) được chọn trước theo y văn và *không* tinh chỉnh trên kết quả, nhưng mọi số liệu vẫn tính trên cùng 43 bệnh nhân. Cần ngoại kiểm (ISLES 2018, hoặc chia train/test).
- 13 bệnh nhân SMIR (36–48) không có metadata TR, nên dùng trung vị cohort 1.44 s.
- OT của ISLES 2017 là tổn thương ở lần chụp theo dõi (follow-up), không phải tổn thương cấp tính, giống thiết lập của bài báo.
- Thời gian fit tăng từ khoảng 5 s lên khoảng 18 s mỗi bệnh nhân (do 2 tham số thêm và multi-start), vẫn dưới 1 phút với 8 tiến trình.

## 5. Việc tiếp theo

0. ~~AIF thật cho định lượng tuyệt đối~~: đã làm (C6, mục 2c). ~~oSVD với cùng AIF~~: đã làm (mục 2d).
   **Định hướng lại đóng góp dựa trên mục 2d:**
   - (a) Bài báo nên báo cáo phần *chẩn đoán* HemoPIC: lỗi đơn vị, vòng lặp prior, kết luận τ phụ thuộc lỗi, CVT bị thổi phồng do copy prior.
   - (b) Mọi so sánh với deconvolution phải dùng cùng AIF và TR (oSVD + C6 là baseline bắt buộc).
   - (c) Lợi thế còn lại của mô hình vật lý nên chuyển sang các khả năng mà deconvolution không có: tái tạo toàn bộ đường cong, tham số Windkessel, mô phỏng phản thực tế. Không nên tuyên bố về ROC.
   - Kỹ thuật: lấy CBV vùng từ pha loãng chỉ thị (như oSVD) để thu hẹp khoảng cách CBV; mô hình hoá bão hoà ΔR2* của máu động mạch.
1. Sửa CBV: lấy CBV vùng từ pha loãng chỉ thị (∫C/∫C_art, đã có ở C3) thay vì V của mô hình, hoặc thêm ràng buộc định lý thể tích trung tâm V = F·MTT/60 vào hàm loss.
2. Bỏ tham số phân tán (không được dùng) và đo lại.
3. Độ nhạy theo K_GM/K_WM, λ_delay. Chạy CVT (Fig. 4) và τ (Fig. 5c–d) bằng `Eval/` gốc trên `outputs/full`.
4. Ngoại kiểm ISLES 2018.
