# DataMate: tác nhân trả lời câu hỏi trên dữ liệu bán hàng

Zalo AI Challenge 2026, bảng DataMate. Image chạy hoàn toàn offline: không gọi API ngoài, không tải gì khi chạy.

## Cách chạy

```
bash /code/predict.sh
```

Không cần tham số. Mọi thiết lập (model, seed, temperature, số lần chạy, số luồng) cố định trong `predict.sh`.

| | Đường dẫn |
|---|---|
| Đọc | `/data` (chỉ đọc) |
| Ghi | `/result/submission.jsonl`: ghi từng câu ngay khi xong |
| Ghi | `/result/logs/<id>.json`: mỗi câu một bản ghi gồm `id`, `answer`, `elapsed_ms`, `thinking`, `code`, `output` |
| Ghi | `/result/vllm_server.log`: log của server model |

File tạm ghi vào `/tmp`.

Lệnh kiểm tra theo mục 10 hướng dẫn của BTC:

```
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/public_test:/data:ro -v /path/to/output:/result \
  zaic2026_datamate_<team>:final /code/predict.sh
```

## Phương pháp

### 1. Làm sạch dữ liệu (`src/datamate/loader.py`, `parsing.py`)

Bước này là code cố định, không dùng model. Code không gắn với mã câu hỏi hay mã cửa hàng nào.

- **Đọc file**: đọc mọi file csv, mọi sheet xlsx, mọi bảng sqlite.
- **Tìm dòng header**: theo tên cột mà README của cửa hàng liệt kê. Bỏ qua dòng tiêu đề và header 2 tầng.
- **Gộp bảng**: gộp file hoặc sheet cùng cấu trúc (file theo tháng, sheet theo tháng, danh sách tách nhiều file). Cột `_file` và `_sheet` ghi lại nguồn.
- **Đoán kiểu cột**, rồi chuẩn hoá:
  - Tiền: `90k`, `12tr460`, `7,59tr`, `105.000đ`, `84,000`.
  - Số lượng kèm đơn vị: `2 cái`, `1 xe 4m3`. Phần chữ đơn vị tách sang cột `<tên cột>__dv`.
  - Ngày: `1 thg 3 2025`, `1/1` (năm lấy theo sheet), số serial Excel, ngày giờ dạng text.
  - Giờ, giá trị đúng/sai.
- **Bỏ dòng không phải giao dịch**: dòng tổng và dòng ghi chú cuối sheet của sổ ghi tay.
- **Rulebook §8**:
  - Ô trống ở các dòng sau của cùng một đơn lấy theo dòng đầu đơn. Chỉ áp cho cột cấp đơn và ô vốn trống ở file gốc.
  - Hai dòng giống hệt nhau trong cùng một đơn tính một lần.
  - Các cách viết của cùng một tên (khác hoa/thường, khoảng trắng, dấu câu, viết không dấu) gộp về một cách viết có trong file. Tên viết không dấu chỉ được gộp khi nó khớp với đúng một tên có dấu.

### 1b. Bảng chuẩn và hàm nghiệp vụ dựng sẵn

Phần này cũng là code cố định, không dùng model. Vai trò cột lấy theo mô tả cột trong README. Nếu README không mô tả thì lấy theo tên cột. Nếu tên cột cũng không đủ thì nối bảng theo giá trị mã (mã đơn, mã hàng trùng nhau giữa các bảng).

- **`S` (dòng hàng) và `O` (đơn)** — `semantic.py`:
  - Trạng thái ở mức đơn (§1, §8): đơn có trả hàng, dù một phần, là đơn trả hàng.
  - Lọc kỳ theo ngày đặt.
  - Số lượng quy về đơn vị cơ bản.
  - Bỏ dòng Voucher / Dùng điểm / Phí ship khỏi doanh thu.
  - Khoá khách theo tên kèm số điện thoại; cờ khách lẻ.
  - Sổ bán chỉ ghi mã khách: tên khách, nhóm khách và giới tính lấy từ bảng khách theo mã. Vẫn xác định khách theo mã (§3), trả lời bằng tên như đáp án mẫu.
  - Sổ bán chỉ ghi mã nhân viên: cột `employee_name` lấy họ tên từ bảng nhân viên (`staff.py`).
  - Dữ liệu không có cột giới tính thì câu hỏi khách nam/nữ là unanswerable (§7). Không đoán giới tính theo tên.
  - Ca sáng/chiều theo giờ tạo đơn.
  - Giá vốn dòng = số lượng × giá vốn ghi trên dòng.
  - Thuộc tính danh mục: nhóm hàng, hoạt chất, thuốc kê đơn, hãng, size.
- **`N`, `R` (nhập hàng)** — `imports.py`: tiền nhập theo phiếu, bỏ phiếu huỷ; giá nhập bình quân theo số lượng (§5).
- **Công nợ** — `debt.py`: đọc quy ước "số tiền trả lần đó / số khách phải trả cho cả đơn" trong README. Có các hàm `debt_as_of`, `debt_by_customer`, `debt_added`, `debt_paid` theo §4, loại đơn huỷ, đơn COD và đơn bán qua sàn TMĐT. `debt_aging(ngày, edges=(7, 15))` chia tiền còn nợ theo nhóm tuổi nợ: 0–7, 8–15, trên 15 ngày; tuổi nợ 0 ngày thuộc nhóm đầu (5/5 câu train khớp).
- **Tồn kho** — `inventory.py`:
  - Tồn cuối ngày theo một trong ba nguồn: sổ tồn (ngày cuối dữ liệu), thẻ kho, hoặc tính theo lô (nhập − bán + khách trả về lô).
  - Giá trị tồn theo giá vốn danh mục.
  - Lô gần hết hạn.

- **Luật riêng theo loại cửa hàng** (các mục thêm trong rulebook bộ public test):

  | Loại cửa hàng | Phần dựng sẵn |
  |---|---|
  | Quán đồ uống, topping ghi chung ô với món (`Latte (M) + 2 Thạch cà phê (14.000đ)`) | tách thành dòng món và dòng topping. Tiền món bằng thành tiền trừ tiền topping. Topping gộp vào cùng mặt hàng với topping ghi dòng riêng. Tổng tiền giữ nguyên |
  | Sửa xe (`repair.py`) | bảng phiếu `P` |
  | Điện thoại (`installment.py`) | sổ nợ và sổ góp đổ vào công nợ §4 (trả góp là công nợ). Có `installments_as_of(ngày)`, cùng cột `dong_may`, `hang`; hai cột này chỉ có ở dòng là máy, phụ kiện để trống. Xếp hạng theo dòng máy: `top_items(kỳ, k, key='dong_may')`. Bảng `PO`: mỗi đơn có khách trả, bớt, thu máy cũ, nối sổ thu máy cũ theo STT và ngày (đã kiểm: tiền hàng − bớt − thu máy cũ = khách trả ở 6334/6334 đơn); `customer_pays`, `trade_in_total`, `discount_total`. Sổ bảo hành `W_bh` và `phone_warranty_rate` (lượt bảo hành / số máy bán) |
  | Mẹ và bé | `adjustments(kỳ, kind)`: voucher, dùng điểm, phí ship ghi số dương theo ngày hoá đơn |
  | Có sổ chi phí | `expenses(kỳ)`; `net_profit(kỳ)` = lãi gộp − chi phí |
  | Giao hàng nhiều đợt (`delivery.py`) | bảng `D_giao` (mỗi đợt của mỗi đơn) và `delivery_rounds(kỳ)`: số đợt có ngày giao trong kỳ, tính đến hết hôm nay, kể cả đơn sau đó trả hàng (12/12 câu train khớp) |

  Bảng phiếu `P` của tiệm sửa xe:
  - Dòng xe gộp theo biển số: các cách ghi cùng xuất hiện trên một biển số là một dòng, ví dụ `AB` = `airblade` = `Air Blade`. Tên hiển thị là cách ghi phổ biến nhất sau khi bỏ hoa/thường, khoảng trắng và ký hiệu α, ưu tiên dạng viết hoa đầu chữ: `Air Blade`, `Vision`, `Wave Alpha`, `SH mode`. Lọc `dong_xe=` theo cách ghi nào cũng được.
  - Các cách ghi một việc gộp về một việc chuẩn, theo đúng luật "nhớt trơn", "ruột trơn", "sửa điện".
  - Phiếu bảo hành không tính là lượt làm việc đó.
  - Có hàm số phiếu, tiền TB một phiếu có thu tiền, tỉ lệ bảo hành, số xe khác nhau, xe khách quen, hoa hồng thợ.
- **Hàm chung khác**:
  - `stock_units(ngày, item, unit)`: tồn quy ra `[số đơn vị lớn, số lẻ]`.
  - `price_effect` và `price_change_date`: hiệu ứng đổi giá. Ngày tăng giá lấy theo bảng giá / lịch sử giá của shop (ngày áp dụng đầu tiên có giá mới cao hơn giá liền trước, đúng rulebook); shop không có bảng giá thì lấy ngày đầu tiên bán ở giá mới.
  - Nhà thuốc có lô: `expiring_lots(D, kỳ hạn dùng)` (lô còn tồn, cột `von` = tồn × giá nhập của chính lô quy về đơn vị cơ bản), `expiring_items(D, kỳ)` (mặt hàng theo hạn sớm nhất, cùng hạn theo mã, §9), `nearest_expiry(D, item=...)` (lô gần hết hạn nhất, hạn đúng ngày D chưa tính là quá hạn). Bảng lô được chọn theo số lượng quy về đơn vị cơ bản và mã hàng; hạn dùng ghi dạng chữ `dd/mm/yyyy` cũng đọc được.
  - `gross_profit` và `top_items(by='profit')`.

Mỗi phần đã được đối chiếu tự động với đáp án bộ train (`scripts/check_*.py` trong repo nguồn; kết quả ở `/code/training_code/eval/deterministic_checks_train_v3.txt`). Kết quả: **1207/1207 câu khớp** trên các dạng đã đối chiếu, gồm doanh thu tháng, số đơn, tỉ lệ huỷ, top bán chạy, số lượng bán, doanh thu và số đơn theo ca, giá trị TB một đơn, công nợ (tổng, theo khách, tuổi nợ, trả nợ), nhập hàng (tổng, giá nhập bình quân, nhà cung cấp), tồn kho (theo đơn vị cơ bản, đơn vị lớn, giá trị), lãi gộp, thuốc kê đơn và hoạt chất, hiệu ứng đổi giá, khách hàng, số đợt giao hàng. Bộ public test không có đáp án nên các phần cho luật riêng ở trên được kiểm bằng test đơn vị (`tests/test_kernel_api.py`) và đối chiếu tay với README của cửa hàng.

Sandbox chặn code thay dữ liệu thật bằng dữ liệu tự tạo: gán lại `T`, `S`, `O` hay bảng gốc bằng giá trị không lấy từ bảng thật nào, hoặc đọc file (`pd.read_csv`, `open`…), đều bị từ chối kèm lời nhắc. Lý do: trên bộ public, model từng dựng lại `T` bằng `pd.DataFrame` số liệu tự gõ. `exit()` trong code chỉ dừng lượt chạy đó, đáp án đã nộp được giữ.

Không dựng được phần nào thì agent dùng bảng gốc `T`. Các bảng dựng sẵn (`S`, `O`, `N`, `R`, `P`, `D_giao`, `PO`, `W_bh`) vừa là biến riêng, vừa có trong `T`. Lý do: model hay viết `T["P"]`. Gọi tên không có trong `T` thì sandbox báo danh sách bảng gốc và bảng dựng sẵn.

Chạy lại toàn bộ đối chiếu: `bash scripts/run_checks.sh` (ghi `training_code/eval/deterministic_checks_train_v3.txt`).

### 1c. Hàm chỉ số và biến kỳ chuẩn (`metrics.py`, `dates.py`)

Model gọi các hàm này với một biến kỳ có sẵn, thay vì tự viết bộ lọc hay tự cộng trừ ngày. Ví dụ:

```python
revenue(THANG(3, 2025), group="cà phê")
n_orders(TUAN_TRUOC, ca="chiều")
pct_change(revenue(THANG_NAY), revenue(THANG_TRUOC_CUNG_KY))
```

| Nhóm | Hàm |
|---|---|
| Doanh thu, đơn, bán | `revenue`, `n_orders`, `qty_sold`, `top_items`, `cancel_rate`, `n_cancelled`, `avg_order_value` |
| Khách | `customers` |
| Giá bán | `price_on` |
| Theo thời điểm | `unsold_items`, `by_day`, `by_hour` |
| Nhập hàng | `import_cost`, `avg_import_price`, `top_suppliers` |

- Bảng `S`/`O` mặc định là bản đã lọc: chỉ gồm dòng/đơn tính doanh thu. Bản đủ mọi trạng thái mang tên `S_gom_huy_tra`/`O_gom_huy_tra`, chỉ dùng cho câu hỏi về huỷ hoặc trả hàng. Tên này đặt rõ nghĩa để model không dùng nhầm khi đếm đơn.
- Biến kỳ được tính theo §2 từ `as_of_date`: `HOM_NAY`, `TUAN_TRUOC`, `THANG_NAY`, `THANG_TRUOC_CUNG_KY`, `QUY_TRUOC`, `TU_DAU_NAM`, `THANG(m, y)`, `QUY(q, y)`, `NAM(y)`, `NGAY(d, m, y)`, `N_NGAY_GAN_NHAT(n)`, `TET(y)`, `TRUOC_TET(y)`…
- Lãi gộp: `gross_profit(kỳ)` và `top_items(kỳ, by='profit')`. Shop không ghi giá vốn trên dòng bán thì hàm báo lỗi và model trả lời unanswerable (§7).

### 1d. Nhận diện sẵn từ câu hỏi (`hints.py`)

Trước khi model viết code, một bộ nhận diện bằng luật (không dùng model) đọc câu hỏi và thêm vào prompt ba loại gợi ý:

| Loại | Ví dụ câu hỏi | Gợi ý đưa vào prompt |
|---|---|---|
| Kỳ thời gian | "so vs th trước đó, doanh thu tháng 12/2025…" | `THANG(12, 2025)` = 2025-12-01 → 2025-12-31; "tháng trước đó" → `THANG(11, 2025)` |
| Tên trong dữ liệu | "bán ra bao nhiêu dĩa rau cu kho quet" | `S.item = 'Rau củ kho quẹt'` |
| Dạng số đo | "bao nhiêu đĩa…", "hàng nào lời nhất", "tháng nào…" | số lượng → `qty_sold`; lãi → `top_items(by='profit')`; tháng → type `date` dạng `YYYY-MM` |
| Lời gọi điền sẵn kỳ | "<REDACTED_COMPETITION_QUESTION>" | `answer('percent', pct_change(revenue(THANG(8, 2025)), revenue(THANG(7, 2025))))` |

- Gợi ý số đo theo đúng định nghĩa rulebook cho các dạng hỏi của bộ public:
  - **§3:** bán chạy, khách trung thành / sộp / hay đổi ý, mặt hàng được ưa chuộng, ngày đắt khách, giờ cao điểm, nhân viên bán giỏi, thợ chăm chỉ.
  - **§2:** phần trăm thay đổi và tỉ trọng.
  - **§4:** số khách nợ, top khách nợ.
  - **§5:** nhà cung cấp chính (nhiều phiếu nhất), nhập trừ bán, hết hàng cuối ngày, lô gần hết hạn, hàng hết hạn trong N ngày tới.
  - **Theo ngành:** sửa xe (hoa hồng, lượt xe theo việc, số xe khác nhau theo dòng xe, thợ theo lượt/ngày làm/doanh thu); điện thoại (top dòng máy); giao nhiều đợt.
- Luật §7 kiểm bằng máy: mọi kỳ của câu hỏi nằm hoàn toàn ngoài khoảng ngày có giao dịch của shop thì gợi ý trả `unanswerable`. Câu so sánh chỉ cần kỳ so nằm ngoài là đủ. Không áp cho câu về hạn dùng, kỳ góp, tồn kho, công nợ. Trên train luật này bắt 16/16 câu có đáp án mẫu là unanswerable và không báo nhầm câu nào.
- Khi câu hỏi chỉ có một kỳ, biểu thức kỳ được điền thẳng vào lời gọi gợi ý.
- **Code gợi ý:** khi đã biết đủ kỳ, bộ lọc và hàm, bộ nhận diện ghép thành một dòng nộp hoàn chỉnh, ví dụ `answer('number', n_tickets(THANG(8, 2025), viec='sửa điện'))`. Dòng đầu tiên model viết sẽ in kết quả của biểu thức đó để kiểm tra rồi mới nộp. Lý do: khi chỉ có gợi ý rời, model có lúc tự lọc lại sổ gốc và bỏ sót luật, ví dụ đếm cả phiếu bảo hành hoặc chỉ khớp đúng chữ "sửa điện". Mỗi dòng Code gợi ý được chạy thật trong sandbox và so với đáp án train (`code_eval`). So sánh hai kỳ thì điền kỳ hỏi và kỳ so theo thứ tự câu hỏi nêu.
- Tên trong dữ liệu gồm cả bảng phiếu sửa xe:
  - dòng xe theo mọi cách ghi, ví dụ `AB` / `airblade` → `P.dong_xe = 'Air Blade'`;
  - việc chuẩn, ví dụ "làm điện" → `viec='sửa điện'`;
  - họ tên thợ.
  
  Bảng `P` chỉ dùng khớp nguyên cụm. Việc ghi một chữ dễ trùng từ thường thì bỏ, ví dụ "bình" trùng với "trung bình".

- Bộ nhận diện kỳ đọc được nhiều cách viết:

  | Kiểu viết | Ví dụ |
  |---|---|
  | Không dấu, viết tắt | "thangs 10/2025", "th 3/2025", "Q4/2025", "hqua", "tuần trc" |
  | Năm hai chữ số | "3/26", "th 9/25" |
  | Tháng viết bằng chữ | "tháng ba". "Sáu" và "năm" chỉ đổi khi câu có dấu, để không nhầm "tháng sau" |
  | Tiếng miền Trung | "tháng ni", "bữa ni", "khách mô", "đầu năm tới chừ" |
  | Teen code | "kó", "kua", "jùm", "mik", "đc", "nượt" |
  | Lỗi gõ telex | "dduwowcj", "hoom", "heets" |
  | Nhiều tháng | "các tháng 7, 8 và 9/2025", "3 tháng liền nhau kết thúc ở tháng 12/2025", "từ th5 đến th7/2025" (cả khoảng và từng tháng) |
  | Ngày | "ngày 15/12" (không ghi năm thì lấy năm theo §2), "14 ngày kể từ ngày X" và "14 ngày ngay trước đó", "120 ngày tới" |
  | Tháng ghi gạch ngang | "7-2026", "2026-07" (40 câu public; trước khi sửa bị đọc nhầm thành cả năm) |
  | Kỳ tương đối, Tết | "tháng này so với tháng trước" (lấy cùng số ngày), "từ mùng 1 tới hết hôm nay so với cùng kỳ tháng trước", "tháng 2/26 so với cùng tháng năm trước" → `THANG(2, 2025)`, "Tết 2026" (theo `calendar.csv`) |

  Trên câu hỏi bộ public test, 989/1000 câu được nhận ra kỳ. 11 câu còn lại không nêu kỳ nào, như câu hỏi hạn dùng lô hay câu "kể từ ngày tăng giá" không ghi ngày.
- Khoảng ngày trong gợi ý được tính bằng cùng hàm với biến kỳ trong sandbox. `tests/test_hints.py` kiểm tra điều này.
- **Các dạng có code gợi ý riêng** (thêm ngày 08/11 sau khi chạy toàn bộ public trên Kaggle): đổi giá (`price_effect`, ngày ghi trong câu hoặc theo bảng giá), lô và hạn dùng, ngày bán nhiều tiền nhất / đông khách nhất, tháng doanh thu cao nhất trong các tháng nêu, doanh thu tăng / giảm liên tục, số mặt hàng có mặt trong từ N đơn, số khách mua từ N đơn, số ngày doanh thu đạt ngưỡng, giá trị TB một đơn, số đơn, lãi gộp và doanh thu của một kỳ không kèm điều kiện lọc, khách nhiều đơn nhất / trung thành / sộp / hay đổi ý (§3, bỏ khách lẻ, bằng nhau theo mã), hoa hồng một thợ, số phiếu bảo hành.
  Thêm sau khi có điểm public (08/11 tối, chỉ dựa trên bộ train): lọc theo ca (§2, gồm % chênh giữa hai ca, ca đứng sau "so với" là kỳ so), doanh thu và số lượng bán của một mặt hàng / nhóm / hoạt chất, thuốc kê đơn và không kê đơn, tỉ trọng doanh thu của một bộ lọc trên doanh thu cả shop, tỉ lệ đơn huỷ, giá bán một đơn vị vào một ngày, tồn kho một mặt hàng và giá trị tồn theo giá vốn cuối ngày, top khách theo tiền mua, mặt hàng ế trong kỳ hoặc trong khoảng ngày, doanh thu nhiều tháng liệt kê theo thứ tự, % khách mua từ N đơn, số đơn có tiền hàng từ X trở lên (hàm mới `order_values`), giá nhập bình quân một đơn vị nhập (đơn vị lấy từ câu hỏi; không có dòng nhập thì unanswerable), tổng số lượng bán theo một đơn vị bán cộng mọi mặt hàng (hàm mới `qty_unit`, quy thùng / lốc về đơn vị đó), số ly size M / L, tồn kho hỏi theo kiểu "cuối ngày còn bao nhiêu", công nợ tại một ngày (tổng nợ còn lại, số khách còn nợ, top khách nợ kèm số tiền, nợ của các đơn quá N ngày) và nợ thêm / trả nợ trong kỳ của cả shop hoặc một khách (list theo thứ tự câu hỏi). "Trả hết", "trả nợ", "trả được" là động từ, không khớp nhóm "Trà". Nhân viên có doanh thu cao nhất (§3). Lỗi gõ telex có dấu "/" giữa hai chữ ("tawng/giamr") được sửa từng chữ. Ngưỡng ghi bằng "≥", chữ "revenue" và lỗi gõ "bans" được nhận ra. Cách nói "từ 1/1 đến hết hôm nay", "tuần này tính tới hôm nay" được đọc thành một kỳ duy nhất. Từ đệm hội thoại ("check giùm", "coi", "ta", "total") không còn chặn nhận diện câu hỏi không có điều kiện lọc. Chữ "lẻ" trong "tỉ lệ", "chẵn, lẻ" không còn bị coi là khách lẻ; "trả bằng tm" là thanh toán tiền mặt, không khớp nhóm "Trà".
- **Chặn điều kiện không thể hiện trong code (09/11):** câu hỏi có điều kiện mà dòng code không lọc thì bỏ code, để model tự làm: thứ trong tuần, cuối tuần, ngày trong tháng chưa thành kỳ ("ngày 15"), một phần của kỳ ("10 ngày đầu tháng", "tuần đầu", "3 tháng đầu năm"), phủ định ("ngoài", "trừ", "không tính"), giờ trong ngày, giới tính, kênh bán (online, Shopee, giao hàng...), loại khách, ca, nợ quá hạn theo ngày phải trả. Câu "tổng X và Y" là một số, không phải list. "Món e" (em) không phải hàng ế. Hỏi theo đơn vị đóng gói (thùng, lốc, hộp, vỉ...) thì code kiểm `unit_is_base` lúc chạy: không phải đơn vị cơ bản của mặt hàng thì không nộp số, chuyển cho model. Kiểm bằng `tests/test_hints.py::test_no_false_code_hints`. Thêm 09/11: tồn kho theo đơn vị lớn chẵn và phần lẻ (`stock_units`, so đơn vị cả khi gõ không dấu), doanh thu và tỉ trọng thuốc kê đơn, nợ thêm / trả nợ của một khách khi câu có hai chữ "bao nhiêu", mặt hàng lãi gộp cao nhất, mẹ và bé: % của một size / hãng bỉm trên tổng số bịch bỉm tã cùng kỳ và số bịch một size / hãng, gợi ý báo cáo tiền mặt theo ca; shop không ghi giới tính hay nhóm khách thì unanswerable (§7). Tiệm sửa xe: phiếu có thu tiền mà ghi chú nhắc bảo hành không phải phiếu bảo hành; "AB" = Air Blade, "Ex" = Exciter. Sau khi đo model trên 60 câu train khó (Kaggle 09/11: 39 đúng, 21 sai, PA 0,30), thêm: nhập trừ bán quy về đơn vị câu hỏi (hàm mới `net_import`, hệ số lấy từ dòng bán rồi bảng quy đổi), "còn thiếu tiền" là công nợ, "xe chở được bao nhiêu đợt hàng" là số đợt giao, câu đánh giá thái độ / tay nghề nhân viên là unanswerable (§7), từ loại trừ chỉ xét ngoài tên hàng ("Tôn lạnh", "0,1%"), "7 ngày gần nhất" không phải câu hỏi "nhất". 17/21 câu model làm sai nay được code gợi ý trả lời đúng.
- **Kiểm tra kiểu đáp án trước khi đưa code gợi ý:** câu hỏi "%", "ngày nào", "tháng nào", "mặt hàng nào / ai" thì code phải ra đúng kiểu đó. Câu hỏi "trung bình", "mỗi thứ Tư", "…nhất", hay "trong nhóm X" thì không đưa code tính tổng. Trên bộ public, mọi code gợi ý sai trước đây đều thuộc các dạng này, và model đã tự sửa được phần lớn.
- **Tên vùng miền:** lạc / đậu phộng, lợn / heo, rán / chiên, pho mát / phô mai, trát / tô, đậu phụ / đậu hũ… được coi là một khi so tên (§8: các tên cùng chỉ một thứ là một). Tên chỉ có một từ thì câu hỏi có dấu phải khớp cả dấu, để "trả bằng chuyển khoản" không khớp nhầm nhóm "Trà".
- **§7 kiểm bằng máy, thêm:** hỏi theo nhóm hàng mà shop không ghi nhóm hàng; hỏi về khách mà shop không ghi tên / mã khách; hỏi lãi gộp mà shop không ghi giá vốn; hỏi lô / hạn dùng mà shop không có sổ lô; hỏi lý do huỷ / trả mà shop không ghi lý do. Các trường hợp này ra `unanswerable`.
- **Phương thức thanh toán và kênh bán ghi bằng mã** (`cash`, `transfer`, `ewallet`, `cod`, `debt`; `store`, `phone`, `zalo_oa`; README giải nghĩa) khớp cách gọi trong câu hỏi: "tiền mặt", "chuyển khoản", "ví điện tử", "COD" / "xi ô đi", "ghi nợ", "tại quầy", "qua điện thoại", "Zalo". Câu hai số đo cùng loại ("doanh thu đơn tại quầy và đơn qua điện thoại, lần lượt") có code nộp list đúng thứ tự câu hỏi. "Đơn trả bằng X", "trả tiền mặt", "trả ví…" là thanh toán, không phải trả hàng.
- **Kết quả đối chiếu (`code_eval`, train 3000 câu):** 2949 câu có code gợi ý (sáng 08/11 là 912, chiều 08/11 là 1800), không câu nào sai so với đáp án mẫu (`training_code/eval/code_hints_train.txt`). 51 câu còn lại do model tự viết code. Mỗi dạng mới có một câu train trong `tests/test_kernel_api.py::test_train_families_0811`. Kiểm tra tất định các hàm dựng sẵn: 1207/1207.
- Gợi ý chỉ nhận diện chuỗi có trong câu hỏi, không đoán đáp án. Không chắc thì không thêm gợi ý.
- Đáp án ngày/tháng được chuẩn hoá: "tháng 8/2025", `(2025, 8)` hoặc một kỳ trọn tháng đều ra `"2025-08"` (§9).

### 2. Agent viết và chạy code (`agent.py`, `prompts.py`, `kernel.py`)

Prompt gồm:
- hướng dẫn và toàn văn `rulebook.md` của split;
- README của cửa hàng, cùng mô tả từng bảng đã làm sạch (kiểu cột, khoảng giá trị, giá trị mẫu);
- bảng mốc thời gian tính sẵn từ `as_of_date` theo §2 (tuần này, tháng này so với tháng trước cùng số ngày, quý, N ngày gần nhất, Tết theo `calendar.csv`…);
- câu hỏi;
- phần nhận diện sẵn từ câu hỏi (mục 1d).

Model viết code pandas. Code chạy trong sandbox: mỗi cửa hàng một tiến trình con giữ sẵn bảng, mỗi câu hỏi một phiên riêng, pandas Copy-on-Write để các phiên không ảnh hưởng nhau, có timeout. Model xem kết quả rồi viết tiếp, tối đa 6 lượt.

Đáp án nộp bằng hàm `answer(type, value)`. Hàm này kiểm tra và chuẩn hoá đúng format §9: tiền làm tròn về đồng, phần trăm 2 chữ số thập phân, list, table…

Sandbox có sẵn các hàm tiện ích: lọc theo khoảng ngày, khoảng Tết, chuẩn hoá trạng thái đơn, so khớp tên, làm tròn nửa lên.

### 3. Chọn đáp án

Mỗi câu chạy tối đa 3 lần với seed khác nhau. Một đáp án được nộp khi có ít nhất 2 lần cho cùng kết quả và không hoà. Ngược lại câu đó bỏ trống (abstain), vì trả lời sai bị −1 còn bỏ trống được 0.

Câu có code gợi ý: runner chạy dòng đó trong một phiên sandbox riêng. Nếu dòng code chạy ra đáp án hợp lệ, runner **nộp thẳng đáp án đó và không gọi model** cho câu này. Log của câu ghi `thinking: null`, `code` là dòng code đã chạy, `output` là kết quả chạy, `source: "code_hint"`. Lý do chọn cách này:
- trên train, code gợi ý đúng ở mọi câu có code (2949/2949);
- trên public, ở cả 587 câu có code gợi ý trong lần chạy đầy đủ, model đều nộp lại đúng kết quả của code;
- bỏ lời gọi model ở các câu này giúp tiết kiệm phần lớn thời gian trong giới hạn 6 giờ, và đáp án không phụ thuộc vào việc sinh văn bản ngẫu nhiên, nên dễ đạt điều kiện tái lập.

Muốn model vẫn được gọi kèm kết quả chạy thử thì dùng `--llm-on-code-hint`. Khi đó vẫn có luật dự phòng: model hết lượt mà không nộp được, hoặc báo unanswerable trong khi code ra giá trị, thì runner nộp kết quả của code. Câu không có code gợi ý, hoặc code chạy lỗi, thì model tự viết code như mục 2.

Runner theo dõi thời gian. Nếu dự kiến không kịp 6 giờ, runner giảm số lần chạy mỗi câu. Đáp án được ghi ngay sau mỗi câu, nên nếu bị dừng giữa chừng thì phần đã làm vẫn được tính.

### Tính tái lập

- Seed cố định cho server vLLM (`--seed 0`) và cho từng lời gọi: seed suy ra từ id câu hỏi và số thứ tự lần chạy.
- Tham số sinh cố định: temperature 0.6, top_p 0.95, top_k 20 (theo khuyến nghị của Qwen3 cho chế độ suy nghĩ).
- Biểu quyết 2/3 làm giảm dao động do batching trên GPU.

## Model

| Tên | Nguồn | Revision cố định | Ngày phát hành | Giấy phép | Số tham số |
|---|---|---|---|---|---|
| Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` | 2025-04-29 (repo tạo 2025-04-27) | Apache-2.0 | 14.768.307.200 |

- Đây là model duy nhất được dùng. Không có model embedding, reranker hay classifier nào khác. Tổng tham số là 14,77B, dưới giới hạn 15B.
- Trọng số gốc bf16, không fine-tune, không lượng tử hoá, nằm trong image tại `/code/models/Qwen3-14B` (kèm `SHA256SUMS`).
- Chạy bằng vLLM 0.10.1.1 của image gốc `vllm/vllm-openai:v0.10.1.1`, tensor parallel 2 GPU, chế độ suy nghĩ (`--reasoning-parser qwen3`).

## Dữ liệu và huấn luyện

- Không huấn luyện, không fine-tune và không sinh dữ liệu.
- Bộ train BTC cung cấp chỉ dùng để kiểm định luật làm sạch, prompt và cách biểu quyết (đo PA trên train). Bộ này được đóng gói lại trong `/code/training_code/data/train`, kèm `data_manifest.json`.
- Không dùng public test hay private test để huấn luyện hoặc tinh chỉnh.

Chi tiết xem `/code/training_code/README.md`.

## Thư viện mã nguồn mở thêm vào image

| Thư viện | Phiên bản | Giấy phép |
|---|---|---|
| pandas | 2.2.3 | BSD-3-Clause |
| openpyxl | 3.1.5 | MIT |
| nvidia-nccl-cu12 | 2.27.7 | BSD-3-Clause (theo khuyến nghị trong hướng dẫn Docker của BTC) |

Phần còn lại có sẵn trong image gốc `vllm/vllm-openai:v0.10.1.1`, trong đó vLLM dùng giấy phép Apache-2.0. Mã nguồn của đội nằm trong `/code/src/datamate`.

## Bố cục thư mục trong image

```
/code/predict.sh          điểm vào
/code/README.md           tài liệu này
/code/requirements.txt    thư viện thêm
/code/src/datamate/       mã suy luận
/code/models/Qwen3-14B/   trọng số model
/code/training_code/      tài liệu kiểm định, dữ liệu train, manifest, log
```
