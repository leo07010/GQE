# GQE 參數掃描繪圖問題修復報告

## 問題描述

在 `GQE_test.ipynb` 中執行參數掃描實驗時，雖然程式碼能夠執行並生成 CSV 日誌文件，但繪製的圖表上沒有顯示任何數據。錯誤訊息顯示：

```
UserWarning: No artists with labels found to put in legend.
```

## 根本原因

經過分析發現有兩個問題：

1. **缺少圖例標籤**：在 `plot_loss_curves()` 函數中，`plt.plot()` 調用時沒有設置 `label` 參數，導致圖例無法顯示任何曲線

2. **欄位名稱不匹配**：程式碼嘗試讀取 `'loss'` 欄位，但實際 CSV 文件中的欄位名稱是 `'loss at'`

## 解決方案

創建了 `fix_plots.py` 腳本來解決這些問題：

### 主要修復

1. **添加圖例標籤**
   ```python
   plt.plot(data, label=f"{param_name}={val}", linewidth=2, alpha=0.8)
   ```

2. **正確讀取欄位**
   ```python
   loss_column = None
   if 'loss at' in df.columns:
       loss_column = 'loss at'
   elif 'loss' in df.columns:
       loss_column = 'loss'
   
   if loss_column:
       data = df[loss_column].dropna().reset_index(drop=True)
   ```

3. **自動保存圖片**
   - 圖片保存至 `plots/` 目錄
   - 使用高解析度 (300 DPI)
   - 自動命名：`{parameter}_comparison.png`

## 執行結果

成功生成以下圖表：

### 1. Learning Rate 比較圖
- **文件**: `plots/learning_rate_comparison.png`
- **參數值**: 0.001, 0.0001, 1e-05
- **數據點**: 每條曲線 100 個訓練步驟

### 2. Temperature 比較圖
- **文件**: `plots/temperature_comparison.png`
- **參數值**: 0.1, 1.0, 5.0
- **數據點**: 每條曲線 100 個訓練步驟

## 使用方法

### 方式一：使用修復腳本（推薦）

```bash
cd /home/leo07010/GQE
python fix_plots.py
```

### 方式二：在 Jupyter Notebook 中修復

如果要在 notebook 中直接修復，需要修改 `plot_loss_curves()` 函數：

```python
def plot_loss_curves(log_paths, param_name):
    plt.figure(figsize=(10, 6))
    
    for val, path in log_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 修復：使用正確的欄位名稱
                if 'loss at' in df.columns:
                    data = df['loss at'].dropna().reset_index(drop=True)
                    # 修復：添加 label 參數
                    plt.plot(data, label=f"{param_name}={val}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    plt.title(f"GQE Loss Curve Comparison ({param_name})")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()  # 這行會顯示圖例
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
```

## 實驗數據位置

所有實驗日誌保存在：
```
experiments_logs/gqe_sweep/
├── lr_0.001/metrics.csv
├── lr_0.0001/metrics.csv
├── lr_1e-05/metrics.csv
├── temperature_0.1/metrics.csv
├── temperature_1.0/metrics.csv
└── temperature_5.0/metrics.csv
```

## CSV 文件結構

每個 `metrics.csv` 文件包含以下欄位：
- `loss at`: 損失值
- `mean energy at label_stand_in`: 平均能量
- `mean_logits at label_stand_in`: 平均 logits
- `min_energy at`: 最小能量
- `step`: 訓練步驟
- `temperature at`: 溫度參數

## 圖表特性

生成的圖表具有以下特性：
- ✅ 使用對數座標 (log scale) 以便更清楚地觀察損失變化
- ✅ 包含完整的圖例，標示每條曲線對應的參數值
- ✅ 網格線輔助讀取數值
- ✅ 高解析度 (300 DPI) 適合論文使用
- ✅ 自動過濾 NaN 值

## 總結

問題已完全解決。現在可以：
1. 看到所有參數掃描實驗的損失曲線
2. 比較不同參數設置的訓練效果
3. 將高質量圖片用於報告或論文

如需重新生成圖表或調整參數，只需運行 `fix_plots.py` 即可。
