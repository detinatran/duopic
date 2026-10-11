# HemoPIC+ — Kế hoạch đóng góp

Tài liệu này giải thích **vì sao** cần cải tiến HemoPIC (MICCAI 2026, arXiv 2607.08799), **cải tiến gì**, **kiểm chứng thế nào**, và **bước tiếp theo**. Mọi con số trong đây đều được chạy lại trên 43 bệnh nhân ISLES 2017 (bản Zenodo).

## 1. Điểm xuất phát

Chúng tôi tái lập HemoPIC gốc trên 43 bệnh nhân. So với bài báo (44 bệnh nhân, bản Zenodo thiếu `training_3`):

| | Bài báo | Tái lập |
|---|---|---|
| nRMSE GM / WM / Lesion / c-Lesion / Brain | 0.106 / 0.095 / 0.318 / 0.192 / 0.093 | 0.106 / 0.094 / 0.319 / 0.191 / 0.093 |
| τ trung vị GM / WM / Lesion / c-Lesion | 13.95 / 12.56 / 10.41 / 13.71 | 14.15 / 12.58 / 9.96 / 13.82 |
| AUC ROC (theo bệnh nhân) CBF / CBV / MTT | 0.763 / 0.584 / 0.722 (44 bn, CSV mẫu của tác giả) | 0.761 / 0.580 / 0.717 |

Cấu hình `configs/baseline.yaml` của HemoPIC+ tái tạo output của code gốc **giống từng bit** (đã kiểm chứng trên bệnh nhân 1 và 22, có test tự động).

## 2. Các vấn đề tìm thấy khi đọc code và dữ liệu

| # | Vấn đề | Bằng chứng | Hệ quả |
|---|---|---|---|
| P1 | **Sai bước thời gian.** Header NIfTI ghi `pixdim[4]` = 0 hoặc 1, code gốc dùng dt = 1 s cho mọi bệnh nhân. | Metadata DICOM trong zip Zenodo: TR thật 1.405–1.818 s (trung vị 1.44 s), 40–80 frame. | Mọi tham số thời gian (τ, TTP, kernel) lệch 40–80 %; τ trong bài báo thực chất tính theo *frame*. |
| P2 | **Đơn vị dòng không nhất quán.** F lấy từ CBF (ml/100g/**phút**) nhưng ngăn tracer dùng tốc độ F/V theo **giây**. | `models_tracer.py`: `k = q/V`, `a = exp(-k·dt)`. Với F ≈ 23, V ≈ 2 thì k·dt ≈ 11 nên a ≈ 0. | Ngăn tracer gần như không có "trí nhớ" (thời gian chuyển tiếp ≈ 0.09 s thay vì ≈ 5 s), mâu thuẫn với chính công thức MTT = 60·CBV/CBF của bài báo. |
| P3 | **Không mô hình độ trễ / phân tán bolus.** Mọi ROI dùng chung một đường cong động mạch C_art(t). | Fig. 5b: đường fit vùng tổn thương lên đỉnh sớm và tụt dưới số đo ở đuôi. nRMSE lesion 0.318, gấp 3 lần GM. | Vùng thiếu máu (đến muộn, kéo dài) bị fit kém nhất, đúng vùng quan trọng nhất về lâm sàng. |
| P4 | **Phụ thuộc vào bản đồ tham chiếu (vòng lặp).** Prior F/V lấy từ MR_rCBF/MR_rCBV. Texture CBV copy từ MR_rCBV. Ngưỡng CBF/MTT lấy theo phân vị của MR_rCBF/MR_MTT. | `core_pipeline.py` dòng 249–250, 369–379, 393–416. | So sánh "HemoPIC vs Deconv" không độc lập: AUC CBV của HemoPIC ≈ Deconv (0.580 so với 0.582) vì CBV gần như là bản sao. Phương pháp không chạy được khi không có bản đồ Deconv, trái với mục tiêu "AIF-free". |
| P5 | **Fit thất bại "âm thầm".** Cờ `success` chỉ lấy từ Powell, không kiểm tra loss. | Có ROI loss 7.0, 10.7 vẫn `success=True`. | Một số ROI dùng tham số vô nghĩa. |
| P6 | Lỗi nhỏ: `--viz` gọi `run_viz` không tồn tại; `np.trapz` hỏng với numpy ≥ 2.4; README thiếu bước chuyển cấu trúc thư mục Zenodo. | — | Khó tái lập. |

## 3. Đóng góp đề xuất

Mỗi đóng góp là một cờ trong config, mặc định = hành vi gốc.

- **C1 — Lấy mẫu thời gian đúng** (`dt_source: tr_table`). `preprocess/extract_tr.py` đo TR từ metadata DICOM; 13 bệnh nhân SMIR không có metadata thì dùng trung vị cohort.
- **C5 — Nhất quán đơn vị** (`flow_per_minute: true`). Ngăn tracer dùng tốc độ F/(60·V) mỗi giây, để thời gian chuyển tiếp của mô hình bằng đúng MTT = 60·V/F mà bài báo dùng cho bản đồ.
- **C2 — Độ trễ và phân tán bolus theo từng ROI** (`delay_dispersion: true`). C_in^r(t) = C̄_in^r · (h_β ∗ C_art)(t − δ_r), với h_β(t) = e^{−t/β}/β (mô hình phân tán của Calamante et al.). Thêm 2 tham số mỗi ROI. Phần tích chập được tính đệ quy chính xác với chi phí O(T).
- **C3 — Chế độ không tham chiếu** (`prior_source: tracer`, `cbv_texture: tracer_auc`, `map_bounds: fixed`):
  - prior CBV theo pha loãng chỉ thị: ∫C_r / ∫C_art;
  - prior CBF theo phương pháp độ dốc cực đại (maximum slope): max(dC_r/dt) / max C_art;
  - hiệu chuẩn một hệ số cho mỗi bệnh nhân, sao cho chất trắng bình thường có CBF = 22 và CBV = 2;
  - texture CBV lấy từ diện tích dưới đường cong của từng voxel;
  - ngưỡng lấy từ chính bản đồ HemoPIC.
  Pipeline không còn đọc giá trị nào của MR_rCBF/rCBV/MTT; MR_rCBF chỉ được dùng để lấy hình học (affine) của ảnh.
- **C6 — AIF tự động** (`arterial_input: auto_aif`, `prior_scale: absolute`). Chọn voxel động mạch (Mouridsen 2006), hiệu chỉnh PVE bằng VOF; prior pha loãng chỉ thị theo đơn vị tuyệt đối, không cần hiệu chuẩn chất trắng. Kết quả: RESULTS.md mục 2c.
- **C7 — Điều kiện đầu Windkessel ở trạng thái dừng** (`wk_init: steady`). Đã thử, không có tác dụng, không khuyến nghị.
- **C8 — CBV theo pha loãng chỉ thị** (`cbv_source: id_regional | id_voxel`). Thu hẹp khoảng cách CBV với oSVD; xem RESULTS.md mục 2e.
- **C9 — Phân bố CBF voxel nhất quán** (`cbf_texture: osvd | moment`, `cbf_texture_clip: false`). Kết quả: RESULTS.md mục 2f.
- **C4 — Fit bền vững** (`multistart: 4`). ROI có loss dữ liệu > 0.15 được fit lại từ 4 điểm khởi đầu ngẫu nhiên, giữ kết quả có hàm mục tiêu nhỏ nhất.

## 4. Thiết kế thí nghiệm

Các config được thêm dần từng đóng góp (ablation cộng dồn), cộng thêm 2 ablation đơn lẻ:

| config | C1 | C5 | C2 | C4 | C3 |
|---|---|---|---|---|---|
| baseline | | | | | |
| c1_tr | ✓ | | | | |
| c15_units | ✓ | ✓ | | | |
| c152_delay | ✓ | ✓ | ✓ | | |
| full_ref | ✓ | ✓ | ✓ | ✓ | |
| c2_delay_only | | | ✓ | | |
| c3_reffree | | | | | ✓ |
| full | ✓ | ✓ | ✓ | ✓ | ✓ |
| full_aif (+C6) | ✓ | ✓ | ✓ | ✓ | ✓ |

Thêm: `full_aif_ref` (C6 + prior tham chiếu), `full_aif_wm` (C6 + hiệu chuẩn WM), `*_ss` (+C7).

Thước đo (cùng công thức với `Eval/` gốc):
1. **nRMSE tái tạo nồng độ** theo vùng GM/WM/lesion/c-lesion/brain (Table 1 của bài báo). Đường fit được đọc từ `fit_curves.npz`, vì script gốc dựng lại đường cong bằng mô hình gốc nên sẽ sai với các biến thể mới.
2. **AUC ROC phát hiện tổn thương** của CBF/CBV/MTT, chuẩn hoá theo trung vị não, trung bình theo bệnh nhân và gộp voxel.

Kết quả: xem `docs/RESULTS.md` (sinh bởi `scripts/compare.py`).

## 5. Bước tiếp theo (đề xuất)

0. **Độ bất định**: đã làm xong (posterior từng ROI, bản đồ sd/CI theo voxel, ensemble pipeline, ba cách kiểm tra). Xem `docs/UNCERTAINTY.md`. Còn lại: nới khoảng cho MTT/τ, thêm bất định của AIF, và đưa bước đánh bóng L-BFGS-B vào `fit_roi_curve`.
0b. **PDE và neural PDE**: đã làm xong (ống vận chuyển–khuếch tán thay bể trộn; mạng dùng chung dự đoán hai ống song song; kiểm tra chéo theo bệnh nhân). Xem `docs/NEURAL_PDE.md`. Uncertainty cho neural PDE cũng đã xong (mục 6). Còn lại: sàn độ phân tán số, hội tụ MCMC khi Pe và delay chạm cận, QC cho ROI fit hỏng, kiểm chứng ý nghĩa sinh lý của hai ống.
1. **Kiểm định thống kê**: Wilcoxon theo cặp bệnh nhân (baseline so với full), kèm khoảng tin cậy bootstrap cho AUC.
2. **Độ nhạy siêu tham số**: K_GM/K_WM ∈ {8, 16, 32}, λ_delay, cận của độ phân tán.
3. **Chia ROI theo động học**: tách các ROI có phương sai thời gian đến (TTP) lớn, để lõi tổn thương không bị trộn với mô lành.
4. **Dòng vào thay đổi theo thời gian** f_in(t) (bài báo giả định hằng số).
5. **Ngoại kiểm**: ISLES 2018 (CT perfusion), có sẵn trong cùng bản ghi Zenodo.
6. Gửi PR cho repo gốc với các sửa lỗi P6 và tuỳ chọn C1/C5 (không đổi kết quả mặc định).
