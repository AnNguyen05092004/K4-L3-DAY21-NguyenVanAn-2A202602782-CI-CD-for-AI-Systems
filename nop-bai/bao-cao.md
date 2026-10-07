# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Văn An |
| MSSV | 2A202602782 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/AnNguyen05092004/K4-L3-DAY21-NguyenVanAn-2A202602782-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

Tôi chạy 7 lần trên MLflow (3 lần tiêu biểu dưới đây, đo trên holdout 500 mẫu).

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 3 | 100 | 0.2 | 3 | 0.7290 | 0.884 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=3`.

**Lý do:** Bộ này có `f1_score` cao nhất (0.7290) trong 7 lần chạy và vượt ngưỡng 0.65 với khoảng đệm rõ. Ở lab này lần có accuracy cao nhất (0.884) cũng là lần có F1 cao nhất; tuy vậy accuracy chỉ dao động 0.846 - 0.884 còn F1 dao động 0.605 - 0.729, nên F1 phân biệt mô hình rõ hơn nhiều. Về đánh đổi, 200 cây với `learning_rate=0.05` (F1 0.7014) vẫn kém 100 cây với `learning_rate=0.1` (0.7109): tăng số cây không bù hết việc học chậm.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập cao, nên mô hình luôn trả lời "thu nhập thấp" đã đạt accuracy khoảng 0,752 mà không bắt được người thu nhập cao nào; con số đó chỉ phản ánh tỷ lệ lớp đa số. F1 của lớp dương là trung bình điều hòa của precision và recall trên đúng lớp hiếm cần phát hiện, nên mô hình vô dụng này có F1 bằng 0. Vì vậy ngưỡng chặn triển khai đặt ở `f1_score >= 0.65`. Tôi gọi `f1_score(y_eval, preds)` không truyền `average="weighted"` hay `"macro"`, vì hai cách này trộn điểm của lớp đa số (chiếm 75%) vào kết quả và làm một mô hình kém vẫn qua được ngưỡng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow 2.13 báo lỗi import khi chạy trên Python 3.12 | Thiếu `pkg_resources` và SQLAlchemy 2.1 quá mới so với MLflow 2.13 | Ghim `sqlalchemy<2.1` và `setuptools<81` trong `requirements.txt` |
| Push lên GitHub không kích hoạt workflow, nút bật Actions báo lỗi 500 | Repo là fork nên GitHub tắt workflow mặc định | Bật Actions bằng API, thử push lại khi gặp lỗi 500 tạm thời của GitHub |
| `dvc pull` trên runner không xác thực được | `.dvc/config` không chứa đường dẫn khóa (khóa chỉ có trên máy, không commit) | Trong CI ghi secret ra `/tmp/sa-key.json` rồi `dvc remote modify --local labstore credentialpath` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7290 | 0.884 |
| Bước 3 (thêm `train_batch2`) | 0.7330 | 0.882 |

**Nhận xét:** F1 tăng 0,004 còn accuracy giảm 0,002, cả hai đều nằm trong mức nhiễu vì holdout chỉ có 500 mẫu (một mẫu sai thêm đổi accuracy 0,002). Hai nửa dữ liệu cùng phân phối nên dữ liệu mới không mang thêm thông tin. Điều được kiểm chứng là quy trình: một commit dữ liệu (`dvc push` rồi `git push`) tự chạy đủ 4 job và VM tự tải model mới mà không cần thao tác tay.
