# ĐỒ ÁN TỐT NGHIỆP

**Đề tài:** Xây dựng và đánh giá chiến lược đầu tư dựa trên Machine Learning trên nhóm cổ phiếu đại diện NASDAQ-100

---

## 1. Mục tiêu dự án

Dự án xây dựng một pipeline hoàn chỉnh để:

1. thu thập và làm sạch dữ liệu giá cổ phiếu;
2. tạo đặc trưng kỹ thuật, biến động, tương đối, vĩ mô và regime;
3. gán nhãn bằng Triple Barrier Method;
4. huấn luyện mô hình theo cơ chế walk-forward có purge/embargo;
5. backtest chiến lược long-only top-K và so sánh với benchmark buy-and-hold.

Dự án tập trung vào **tính đúng phương pháp** và **kiểm soát leakage** hơn là tối đa hóa lợi nhuận bằng mọi giá.

---

## 2. Phạm vi dữ liệu và universe

Cấu hình trung tâm nằm tại `configs/base.yaml`.

- **Giai đoạn dữ liệu:** 2014-01-01 đến 2026-03-01
- **Universe:** 36 cổ phiếu lớn có lịch sử đầy đủ trong nhóm NASDAQ-100
- **Nguồn dữ liệu giá cổ phiếu:** Tiingo EOD API
- **Nguồn dữ liệu vĩ mô / benchmark feature:** Yahoo Finance (VIX, VXN, QQQ)

### Vai trò của QQQ

QQQ được dùng làm **market benchmark cho feature engineering** (rolling beta, residual return, idiosyncratic volatility, relative strength). Trong backtest, benchmark so sánh là **Buy & Hold equal-weight trên rổ cổ phiếu hợp lệ**, tính bởi `compute_benchmark()` trong `src/backtest/engine.py`.

---

## 3. Kiến trúc dự án

```text
project_v7_final/
├── configs/base.yaml               # Cấu hình tập trung
├── notebooks/                      # 01→06: pipeline chạy tuần tự
├── scripts/                        # CLI scripts cho automation
├── src/
│   ├── data/                       # Fetch, align, QC, build dataset
│   ├── features/                   # 5 nhóm feature engineering
│   ├── labeling/                   # Triple Barrier Method
│   ├── splits/                     # Walk-forward with purge/embargo
│   ├── models/                     # LR, RF, XGB, ensemble
│   ├── regime/                     # HMM regime detection
│   ├── backtest/                   # Engine + metrics + benchmark
│   └── utils/                      # I/O helpers
└── tests/                          # 31 tests
    ├── test_project.py
    ├── test_logic_additional.py      
```

### Các module chính

- `src/data/`: fetch dữ liệu (Tiingo + Yahoo), align panel về NYSE calendar, quality check, build dataset
- `src/features/`: tạo feature theo 5 nhóm (price, volatility, macro, relative, regime)
- `src/labeling/`: Triple Barrier Method và embargo helper
- `src/splits/`: expanding walk-forward split có purge + embargo
- `src/models/`: train LR / RF / XGB, XGB model selection, ensemble
- `src/regime/`: HMM 2-state regime detection
- `src/backtest/`: engine backtest long-only top-K, buy-and-hold benchmark, metrics, alpha statistics
- `src/utils/`: I/O helper (parquet round-trip), log_return

---

## 4. Data pipeline

Toàn bộ pipeline dữ liệu được điều phối bởi `src/data/pipeline.py`:

1. fetch giá cổ phiếu từ Tiingo (có cache + retry);
2. fetch VIX, VXN và QQQ từ Yahoo Finance;
3. align toàn bộ dữ liệu theo NYSE trading calendar;
4. chạy quality check theo từng ticker;
5. loại ticker không đạt tiêu chuẩn;
6. build dataset hợp nhất theo MultiIndex `(date, ticker)`.

### Quality control

Trong `src/data/clean.py`, ticker bị loại nếu vi phạm một trong các điều kiện:

- số ngày giao dịch hợp lệ < `min_trading_days` (200);
- tỷ lệ NaN vượt `max_nan_ratio` (2%) trên **bất kỳ cột OHLCV nào**;
- có giá không hợp lệ (≤ 0);
- có duplicate date;
- số NaN liên tiếp vượt `max_consec_nan` (5).

---

## 5. Feature engineering

Dự án xây dựng **5 nhóm feature**, tổng cộng khoảng **65 features** (ML Full) hoặc **51 features** (ML Base).

### Nhóm 1 — Price / Momentum / Technical (30 features)

Tạo trong `src/features/price.py`. Bao gồm: returns nhiều horizon (ret_1d đến ret_21d), momentum dài hạn (mom_63d), RSI, MACD, Bollinger %B, Stochastic %K, ADX, SMA signals, trend features, volume features, abnormal volume.

### Nhóm 2 — Volatility (10 features)

Tạo trong `src/features/volatility.py`. Bao gồm: realized vol nhiều timeframe (vol_5d đến vol_63d), ATR, Garman-Klass volatility, vol dynamics, return skew.

### Nhóm 3 — Macro / zSpread (10 features)

Tạo trong `src/features/macro_features.py`. Bao gồm: VIX/VXN returns, VXN z-score, VXN dynamics, VIX-VXN spread, zSpread per-ticker.

### Nhóm 4 — Relative / Residual / Liquidity (11 features)

Tạo trong `src/features/relative.py`. Bao gồm: rolling beta, residual return, idiosyncratic volatility, relative strength, Amihud illiquidity, turnover, downside vol/beta, max drawdown, market dispersion.

### Nhóm 5 — Regime as Feature (tối đa 4 features)

Tạo trong `src/features/regime_features.py`. Bao gồm: P(high_vol) từ HMM + interaction terms (p_high × momentum, p_high × vol, p_high × residual return).

**ML Base** = Nhóm 1 + 2 + 4 (51 features). **ML Full** = tất cả 5 nhóm (65 features). Việc tách thực hiện bằng `split_feature_cols()` trong `src/config.py`.

---

## 6. Regime detection

HMM 2 trạng thái trong `src/regime/hmm.py`:

- Fit trên VXN log returns 5d, VIX-VXN spread z-score, VXN level z-score;
- Expanding window, chỉ dùng dữ liệu đến t-1;
- Refit định kỳ mỗi 63 trading days (~1 quý).

Phiên bản hiện tại **chỉ dùng regime as feature** (P(high_vol) đưa vào model). Regime overlay / exposure clamp đã thử nghiệm nhưng giảm return mà không cải thiện drawdown, nên đã loại bỏ.

---

## 7. Labeling — Triple Barrier Method

Cấu hình theo `configs/base.yaml`:

- **Horizon:** 10 ngày giao dịch (vertical barrier)
- **Barrier:** 1.5 × daily_vol (rolling 20 ngày)
- **Label binary:** 1 (chạm PT trước, hoặc return > 0 khi hết H), 0 (ngược lại)
- Barrier detection dùng adj_high/adj_low; tie-break khi cùng bar chạm cả PT và SL dùng open + close

---

## 8. Walk-forward và anti-leakage

Expanding walk-forward by year (`src/splits/walkforward.py`), test từ 2020, max_train_years = 8:

- **Purge:** loại train samples có `t1 > test_start` (label nhìn sang test period)
- **Embargo:** loại H=10 ngày đầu mỗi kỳ test
- **Trade at t+1:** signal tại ngày rebalance, vào lệnh tại adj_open ngày tiếp theo
- **HMM:** chỉ fit trên data đến t-1
- **XGB validation:** tách theo date block (85% ngày train → fit, 15% cuối → validation)
- **Cross-sectional rank:** chỉ thực hiện ở tầng training, không double-rank trong feature engineering

---

## 9. Modeling

Mỗi fold walk-forward huấn luyện 3 mô hình:

- **Logistic Regression (LR):** raw features + clip outlier + StandardScaler, class_weight balanced
- **Random Forest (RF):** 500 cây, max_depth 6, cross-sectional ranked features
- **XGBoost (XGB):** early stopping theo logloss, chọn cấu hình theo **validation daily AUC** từ grid 4 bộ hyperparameter, sau đó **refit trên toàn bộ train fold** với số cây đã chọn

**Ensemble:** weighted average y_prob (trọng số đều mặc định).

### Kết quả walk-forward (trung bình across folds)

| Model | daily_auc (Full) | top_k_ret (Full) | daily_auc (Base) | top_k_ret (Base) |
|-------|:----------------:|:----------------:|:----------------:|:----------------:|
| LR    | 0.5203           | 0.0033           | 0.521            | 0.0033           |
| RF    | 0.5149           | 0.0032           | 0.505            | 0.0017           |
| XGB   | 0.5123           | 0.0024           | 0.502            | 0.0020           |

AUC cross-sectional dao động quanh 0.50–0.52 — mức khiêm tốn nhưng đủ tạo edge khi kết hợp ensemble và rebalance đều đặn.

---

## 10. Backtest

### Cấu hình chiến lược

- Long-only, top-K = 10, equal-weight
- Rebalance mỗi 10 ngày giao dịch
- Entry tại `adj_open` ngày sau signal
- Cost: 10 bps/chiều
- Initial capital: $10,000
- Giai đoạn backtest: 2020-01-31 đến 2026-02-26

### Kết quả chính

| Strategy | Total Return | CAGR   | Sharpe | Sortino | Max Drawdown | Calmar | VaR 95% | Win Rate | Avg Daily Ret | Std Daily Ret | N Days |
|----------|-------------:|-------:|-------:|--------:|-------------:|-------:|--------:|---------:|--------------:|--------------:|-------:|
| ML Full  | 357.4%       | 28.49% | 0.95   | 1.29    | -24.15%      | 1.18   | -2.40%  | 54.47%   | 0.118%        | 1.714%        | 1476   |
| ML Base  | 287.9%       | 25.05% | 0.88   | 1.20    | -23.94%      | 1.05   | -2.27%  | 55.22%   | 0.105%        | 1.598%        | 1476   |
| Buy & Hold | 326.8%     | 27.03% | 0.89   | 1.22    | -35.46%      | 0.76   | -2.82%  | 56.71%   | 0.114%        | 1.754%        | 1476   |

ML Full kết thúc với giá trị danh mục khoảng **$45,741**, cao hơn cả ML Base (**$38,794**) và benchmark Buy & Hold (**$42,678**). So với Buy & Hold, ML Full chỉ nhỉnh hơn vừa phải về lợi nhuận tuyệt đối, nhưng cải thiện rõ hơn ở chất lượng lợi nhuận: **CAGR cao hơn 1.46 điểm %**, **Sharpe cao hơn 0.06**, và **Max Drawdown thấp hơn 11.31 điểm %**. Calmar ratio của ML Full đạt **1.18**, cao hơn khoảng **1.55 lần** benchmark, cho thấy chiến lược vượt trội hơn ở góc độ risk-adjusted return.

### Alpha statistics

| Chỉ số            | ML Full vs B&H | ML Base vs B&H |
|-------------------|---------------:|---------------:|
| Annual Alpha      | +1.00%         | -2.30%         |
| Information Ratio | 0.070          | -0.158         |
| Tracking Error    | 14.27%         | 14.54%         |
| t-stat            | 0.174          | -0.396         |
| p-value           | 0.8621         | 0.6918         |

Alpha của ML Full là **dương**, nhưng **chưa có ý nghĩa thống kê**. Nói cách khác, trên sample hiện tại chưa thể kết luận chắc chắn rằng phần vượt trội so với benchmark đến từ kỹ năng dự báo, thay vì nhiễu mẫu. Vì vậy, kết quả nên được diễn giải như bằng chứng về **khả năng cải thiện hồ sơ rủi ro/lợi nhuận**, hơn là bằng chứng mạnh về alpha bền vững.

### Phân tích theo giai đoạn thị trường

| Giai đoạn     | Buy & Hold | ML Base | ML Full |
|---------------|-----------:|--------:|--------:|
| COVID Crash   | -26.6%     | -22.9%  | -21.7%  |
| COVID Recovery| 63.8%      | 63.4%   | 78.6%   |
| Bull 2021     | 40.7%      | 34.6%   | 30.7%   |
| Rate Hike 2022| -34.6%     | -20.4%  | -16.7%  |
| Recovery 2023 | 64.6%      | 38.7%   | 63.4%   |
| AI Rally 2024 | 38.7%      | 8.1%    | 5.2%    |
| Correction 2025| -11.1%    | 8.1%    | 2.1%    |
| Post-Correction| 42.6%     | 24.1%   | 16.4%   |

Kết quả cho thấy **điểm mạnh lớn nhất của ML Full nằm ở các giai đoạn stress hoặc phân hóa mạnh**. Trong COVID Crash và đặc biệt là Rate Hike 2022, chiến lược giảm lỗ đáng kể so với benchmark. ML Full cũng phản ứng tốt ở COVID Recovery. Tuy nhiên, chiến lược **underperform rõ rệt trong các pha bull market tập trung**, đặc biệt ở AI Rally 2024, khi lợi nhuận thị trường bị kéo bởi một nhóm rất nhỏ cổ phiếu dẫn dắt.

### Annual returns

| Năm       | ML Full | ML Base | Buy & Hold | Alpha (Full - B&H) |
|-----------|--------:|--------:|-----------:|-------------------:|
| 2020      | +84.1%  | +62.1%  | +52.2%     | +33.0%             |
| 2021      | +28.9%  | +32.0%  | +39.6%     | -10.0%             |
| 2022      | -11.8%  | -10.6%  | -30.8%     | +20.1%             |
| 2023      | +63.3%  | +38.8%  | +63.1%     | +0.2%              |
| 2024      | +5.1%   | +8.4%   | +36.2%     | -31.1%             |
| 2025      | +18.7%  | +34.2%  | +27.8%     | -9.1%              |
| 2026 YTD* | +7.3%   | +0.5%   | +2.1%      | +5.2%              |

\* `2026 YTD` tính đến **2026-02-26**.

Phân rã theo năm cũng củng cố cùng một kết luận: ML Full tạo khác biệt mạnh trong các năm biến động lớn như **2020** và **2022**, gần như ngang benchmark trong **2023**, nhưng bị bỏ lại trong **2024** khi thị trường tăng rất tập trung. Điều này cho thấy chiến lược phù hợp hơn với môi trường biến động và phân hóa, thay vì các pha momentum cực mạnh dồn vào số ít mã vốn hóa lớn.

---

## 11. Robustness — Sensitivity analysis

### Top-K sensitivity

| K  | CAGR  | Sharpe | MDD    | Calmar |
|----|------:|-------:|-------:|-------:|
| 3  | 36.4% | 1.02   | -28.2% | 1.29   |
| 5  | 25.5% | 0.81   | -21.1% | 1.20   |
| 8  | 27.5% | 0.90   | -24.8% | 1.11   |
| 10 | 28.5% | 0.95   | -24.2% | 1.18   |
| 12 | 26.5% | 0.90   | -24.6% | 1.08   |
| 15 | 25.0% | 0.86   | -25.6% | 0.97   |

K = 3 cho lợi nhuận và Sharpe cao nhất, nhưng đi kèm mức tập trung cao hơn và drawdown sâu hơn. K = 10 không phải cấu hình tối đa hóa return, nhưng là lựa chọn cân bằng hơn giữa hiệu quả, ổn định và khả năng triển khai. Nhìn chung, kết quả khá ổn định trong vùng **K = 8 đến 12**, cho thấy hiệu quả chiến lược không phụ thuộc hoàn toàn vào một cấu hình quá hẹp.

### Cost sensitivity

| Cost (bps/chiều) | CAGR  | Sharpe | Total Return |
|------------------|------:|-------:|-------------:|
| 0                | 31.5% | 1.04   | 427.4%       |
| 5                | 30.0% | 0.99   | 391.2%       |
| 10               | 28.5% | 0.95   | 357.4%       |
| 15               | 27.0% | 0.90   | 325.9%       |
| 20               | 25.5% | 0.86   | 296.6%       |
| 30               | 22.6% | 0.77   | 243.9%       |

Hiệu quả chiến lược giảm dần khi chi phí giao dịch tăng, nhưng vẫn giữ được lợi nhuận dương ngay cả ở mức **30 bps/chiều**. Điều này cho thấy edge không bị triệt tiêu hoàn toàn bởi friction, dù chi phí rõ ràng là biến số nhạy cảm và cần được mô hình hóa kỹ hơn trong các bước phát triển tiếp theo.

### Rebalance frequency sensitivity

| Freq (ngày) | CAGR  | Sharpe | MDD    |
|-------------|------:|-------:|-------:|
| 5           | 24.8% | 0.84   | -29.4% |
| 10          | 28.5% | 0.95   | -24.2% |
| 15          | 24.0% | 0.81   | -39.6% |
| 21          | 21.0% | 0.73   | -42.6% |
| 42          | 17.2% | 0.60   | -41.1% |

Rebalance mỗi **10 ngày** cho kết quả tốt nhất cả về CAGR, Sharpe và drawdown. Điều này nhất quán với thiết kế labeling horizon = 10 ngày, cho thấy tín hiệu được khai thác hiệu quả nhất khi tần suất tái cân bằng khớp với horizon dự báo.

### Trade log summary

- Tổng số rebalances: **142**
- Số mã nắm giữ trung bình: **10.0**
- Có **139/142** lần rebalance đạt đúng `top_k = 10`
- Tổng chi phí tích lũy: **14.23%** trên toàn bộ giai đoạn backtest

Cấu trúc trade log cho thấy engine hoạt động ổn định, turnover không bất thường, và chiến lược được triển khai khá nhất quán qua toàn bộ sample.

---

## 12. Diễn giải kết quả

### ML Full có thực sự tốt hơn Buy & Hold?

Câu trả lời là **có, nhưng theo nghĩa risk-adjusted hơn là alpha thống kê mạnh**. ML Full vượt Buy & Hold về **CAGR**, **Sharpe**, **Calmar** và đặc biệt là **khả năng kiểm soát drawdown**. Phần giá trị lớn nhất của chiến lược nằm ở việc giảm tổn thất trong các giai đoạn bất lợi của thị trường, thay vì thắng đều trong mọi môi trường.

Tuy nhiên, cần nhấn mạnh ba điểm:

1. **Alpha chưa có ý nghĩa thống kê**, nên chưa thể khẳng định chiến lược sở hữu edge bền vững ở mức học thuật chặt chẽ.
2. **Lợi thế tập trung ở giai đoạn stress** như 2020 và 2022; đây là dạng chiến lược phòng thủ-chủ động tốt hơn là cỗ máy tạo alpha ổn định qua mọi chế độ thị trường.
3. **Chiến lược underperform trong các pha tăng mạnh và tập trung**, đặc biệt khi lợi nhuận bị kéo bởi một nhóm rất nhỏ mega-cap.

### ML Full vs ML Base

ML Full tốt hơn ML Base về **tổng lợi nhuận**, **CAGR** và **Sharpe**, cho thấy việc bổ sung nhóm feature macro/regime mang lại giá trị thực tế trong sample này. Dù vậy, ML Base có drawdown tổng thể hơi nông hơn một chút và cũng giữ được hồ sơ rủi ro khá tốt. Nói cách khác, **ML Full là phiên bản hiệu quả hơn về tổng thể**, còn **ML Base là phiên bản đơn giản hơn nhưng vẫn tương đối cạnh tranh**.

### Kết luận ngắn gọn

Tổng hợp lại, chiến lược ML Full cho thấy một tín hiệu tích cực: mô hình không tạo ra mức outperformance áp đảo, nhưng có khả năng **cải thiện hồ sơ lợi nhuận/rủi ro**, nhất là trong các giai đoạn thị trường khó. Đây là kết quả đủ mạnh để biện minh cho hướng tiếp cận ML trong đồ án, đồng thời vẫn đủ trung thực để thừa nhận rằng alpha hiện tại **chưa đủ mạnh về mặt thống kê** và còn cần thêm kiểm định robustness ở các sample khác.

## 13. Cách chạy lại dự án

### Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env    # điền TIINGO_API_KEY
```

### Chạy từ đầu bằng notebooks

Chạy theo thứ tự:

1. `notebooks/01_data_etl.ipynb` — fetch + align + QC
2. `notebooks/02_features.ipynb` — 5 nhóm features
3. `notebooks/03_labeling.ipynb` — Triple Barrier
4. `notebooks/04_models.ipynb` — walk-forward training
5. `notebooks/05_Backtesting.ipynb` — backtest + metrics
6. `notebooks/06_analysis.ipynb` — phân tích chi tiết + sensitivity

### Chạy bằng scripts

```bash
python scripts/run_all.py                     # chạy toàn bộ pipeline
python scripts/run_data_pipeline.py           # chỉ data pipeline
python scripts/run_features.py                # chỉ feature engineering
```

### Kiểm thử

```bash
pytest -q
```
---

## 14. Giới hạn của phiên bản hiện tại

1. **Alpha chưa có ý nghĩa thống kê** — cần thêm dữ liệu hoặc kỹ thuật bootstrap để validate.
2. **Benchmark backtest là equal-weight buy-and-hold**, không phải QQQ ETF trực tiếp.
3. **Regime chỉ tác động gián tiếp** qua feature set, chưa dùng ở tầng portfolio.
4. **Cost model là approximation** theo số mã mua/bán, chưa dùng turnover theo weight-delta thực.
5. **Chiến lược underperform trong bull market tập trung** do cơ chế equal-weight top-K.
