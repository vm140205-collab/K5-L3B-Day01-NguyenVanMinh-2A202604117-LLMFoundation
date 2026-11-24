# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604117
- **Họ và tên:** Nguyễn Văn Minh
- **SEED** (in ở ô đầu notebook): 3936

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ (chạy hết Run all; Phần 4 không có API key nên chạy nhánh skip có sẵn + ô kiểm tra bằng API giả lập vẫn pass) |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ *(nộp riêng — ảnh chụp bài tính tay)* |

## 2. Kết quả chính (bảng Phần 5 trong notebook)

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.905 | 0.824 | 0.039 | 0 |
| 2. BoW + MLP | 0.958 | 0.974 | 0.203 | 0 |
| 3a. Encoder + kNN | 0.901 | 0.895 | 109.1 | 0 |
| 3b. Encoder + LogReg | 0.931 | 0.919 | 26.2 | 0 |
| 3c. Encoder + MLP head | 0.948 | 0.974 | 19.6+ | 0 |
| 4. LLM (zero/few-shot) | — | — | — | — *(không có key, nhánh skip)* |

## 3. Dự đoán và thực tế

- **Dự đoán khớp:** Embedding+LogReg đặt trên BoW+LogReg về F1 (0.931 so với 0.905) — đúng như dự đoán trước thí nghiệm 3.
- **Dự đoán lệch:** kNN — tôi dự đoán kNN "chậm hơn nhưng tốt hơn", thực tế kNN (3a) có F1 thấp hơn cả LogReg chạy trên embedding (3b) và p95 chậm nhất bảng (109 ms) vì phóng to toàn bộ ma trận thay vì giữ một vector trọng số.
- **Bất ngờ nhất:** trên 50 case khó, thứ hạng lật ngược — BoW rơi từ F1 ~0.9 xuống 0.378–0.410, còn Encoder+kNN lên thành cách tốt nhất (0.759). Dữ liệu thường che giấu điểm yếu của BoW vì các tin thường dùng đúng từ khoá.

## 4. Phân tích 50 case khó

- Cả 5 cách đều sai 4 case "spam giọng tự nhiên" (mở đầu lịch sự, không từ khoá: "sorry to bother you...", "quick reminder that your photos...") — không model nào dựa trên nghĩa được vì Bag-of-words/embedding đều học từ dữ liệu có từ khoá.
- Emb+MLP sai 18/50, chủ yếu nhóm `ham_looks_spam` (tin ham có giảm giá "50% off" bị báo nhầm).
- Điểm qua mắt BoW: nhóm `obfuscation` (6/6 đúng chỉ với embedding).

## 5. Thử thách cá nhân

- Đề theo SEED 3936 (in ở ô "Thử thách theo SEED" cuối notebook) và cách giải + kết quả nằm ở ô trả lời kèm sau khi chạy.

## 6. Phương án cho sản phẩm thật (p95 < 50 ms/tin, 10 triệu SMS/ngày)

- **Chọn:** BoW + MLP — F1 SUB cao nhất trong nhóm đạt SLA (0.974, p95 0.203 ms, ~46k tin/s/lô, chi phí 0).
- **Dẫn số liệu:** bảng Phần 5 — nhanh hơn SLA ~250 lần; LLM thật sẽ vượt SLA nhiều lần và tốn USD/1k tin.
- **Chấp nhận đánh đổi:** yếu trên nhóm né từ khoá (BoW rơi xuống F1 0.41 trên 50 case khó); có thể chạy định kỳ một LLM offline theo lô để bổ sung từ khoá mới vào từ điển, không ảnh hưởng SLA online.

## 7. AI log và điều đã học

| Hỏi AI gì | AI trả lời gì | Tôi sửa/kiểm tra gì |
| --- | --- | --- |
| Tìm các phần còn thiếu trong notebook | Liệt kê các cell có phần thân để trống (chưa điền lời giải), các ô `predict(...)` chưa ghi dự đoán, và các giá trị HAND_CALC chưa điền | Tự đọc lại từng cell trước khi điền |
| Mẫu vòng lặp train MLP trong PyTorch | 5 dòng `zero_grad → forward → loss → backward → step` | Đối chiếu chữ ký `loss_fn`/dataloader của notebook rồi mới điền; tự trả lời phần "quên zero_grad thì sao" |
| `top_k_similar` không lặp qua n dòng | Chuẩn hoá bằng `np.linalg.norm(axis=1)` rồi nhân ma trận | Tự thêm nhánh vector-0 để khớp yêu cầu "trả về 0.0" |
| Công thức HAND_CALC | σ(0)=0.5 · F1=2·P·R/(P+R) · cosine=tích vô hướng/tích độ dài · chi phí=token×giá | Tự tính lại tay, đối chiếu với hash trong ô kiểm tra |
| Giải thích tại sao model sai trên case khó | Gợi ý nhóm từ khoá/nghĩa của từng category trong ô hard-cases | Đối chiếu với output thật của ô #6–#10, #12–#19 trước khi ghi |

**Điều đã học:** (1) Accuracy lừa dối trên dữ liệu lệch — model "luôn đoán ham" đạt ~0.87 accuracy, phải nhìn F1/precision/recall. (2) Cosine đo *hướng* không đo độ dài, nên embedding bắt được spam đổi từ đồng nghĩa mà BoW mù. (3) Sản phẩm thật quyết định bởi đường Pareto latency–F1, không phải riêng F1: Encoder+kNN có F1 khá nhưng bị Encoder+LogReg lấn át hoàn toàn (rẻ hơn mà chính xác hơn), và không cách nào với F1 0.97 được chọn nếu p95 vượt SLA 10 lần.
