#!/usr/bin/env python3
"""
修復 GQE_test.ipynb 中的參數掃描繪圖問題
問題：圖片上沒有數據顯示，因為 plt.plot() 沒有設定 label 參數
解決方案：確保每條曲線都有正確的 label，並將圖片保存到文件
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def plot_loss_curves_fixed(log_paths, param_name, save_dir="plots"):
    """
    修復後的繪圖函數，確保數據正確顯示並保存圖片
    
    Args:
        log_paths: 字典，格式為 {參數值: CSV文件路徑}
        param_name: 參數名稱（用於標題和文件名）
        save_dir: 保存圖片的目錄
    """
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # 用於追蹤是否有數據被繪製
    has_data = False
    
    for val, path in log_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 繪製 loss (過濾掉 NaN)
                # 注意：CSV 中的欄位名稱是 "loss at" 而不是 "loss"
                loss_column = None
                if 'loss at' in df.columns:
                    loss_column = 'loss at'
                elif 'loss' in df.columns:
                    loss_column = 'loss'
                
                if loss_column:
                    data = df[loss_column].dropna().reset_index(drop=True)
                    # 關鍵修復：添加 label 參數
                    plt.plot(data, label=f"{param_name}={val}", linewidth=2, alpha=0.8)
                    has_data = True
                    print(f"✓ 成功繪製 {param_name}={val} 的數據 ({len(data)} 個點)")
                else:
                    print(f"✗ 警告：{path} 中沒有 'loss' 或 'loss at' 欄位")
                    print(f"   可用欄位: {list(df.columns)}")
            except Exception as e:
                print(f"✗ 讀取 {path} 時發生錯誤: {e}")
        else:
            print(f"✗ 文件不存在: {path}")

    
    if has_data:
        plt.title(f"GQE Loss Curve Comparison ({param_name})", fontsize=14, fontweight='bold')
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10, loc='best')  # 添加圖例
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.yscale('log')  # 對數座標看差異更清楚
        plt.tight_layout()
        
        # 保存圖片
        filename = f"{param_name.lower().replace(' ', '_')}_comparison.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ 圖片已保存至: {filepath}")
        
        plt.show()
    else:
        print(f"\n✗ 沒有找到任何可繪製的數據")
        plt.close()

def find_experiment_logs(base_dir="experiments_logs/gqe_sweep"):
    """
    自動尋找實驗日誌文件
    
    Returns:
        字典，格式為 {實驗類型: {參數值: CSV路徑}}
    """
    all_logs = {}
    
    if not os.path.exists(base_dir):
        print(f"✗ 實驗目錄不存在: {base_dir}")
        return all_logs
    
    # 尋找所有子目錄
    for exp_type_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_type_dir)
        if os.path.isdir(exp_path):
            # 提取參數類型和值
            # 例如: "lr_0.001" -> param_type="lr", param_value="0.001"
            parts = exp_type_dir.split('_', 1)
            if len(parts) == 2:
                param_type, param_value = parts
                
                # 尋找 metrics.csv
                csv_files = glob.glob(os.path.join(exp_path, "**/metrics.csv"), recursive=True)
                if csv_files:
                    if param_type not in all_logs:
                        all_logs[param_type] = {}
                    all_logs[param_type][param_value] = csv_files[0]
                    print(f"✓ 找到日誌: {param_type}={param_value} -> {csv_files[0]}")
    
    return all_logs

def main():
    """主函數：自動尋找並繪製所有實驗結果"""
    print("=" * 60)
    print("GQE 參數掃描繪圖修復工具")
    print("=" * 60)
    
    # 尋找所有實驗日誌
    print("\n[1/2] 搜尋實驗日誌...")
    all_logs = find_experiment_logs()
    
    if not all_logs:
        print("\n✗ 沒有找到任何實驗日誌")
        print("請確認以下目錄存在且包含數據:")
        print("  - experiments_logs/gqe_sweep/lr_*/metrics.csv")
        print("  - experiments_logs/gqe_sweep/temperature_*/metrics.csv")
        return
    
    # 為每種參數類型繪圖
    print(f"\n[2/2] 繪製圖表...")
    for param_type, log_paths in all_logs.items():
        print(f"\n--- 處理參數: {param_type} ---")
        
        # 將參數名稱轉換為更易讀的格式
        param_display_name = {
            'lr': 'Learning Rate',
            'temperature': 'Temperature'
        }.get(param_type, param_type.capitalize())
        
        plot_loss_curves_fixed(log_paths, param_display_name)
    
    print("\n" + "=" * 60)
    print("處理完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 切換到 GQE 目錄
    os.chdir("/home/leo07010/GQE")
    main()
