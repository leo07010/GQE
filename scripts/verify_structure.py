#!/usr/bin/env python3
"""
GQE 目錄結構驗證腳本
檢查所有文件是否正確組織，並生成結構報告
"""

import os
from pathlib import Path
from collections import defaultdict

def check_directory_structure():
    """驗證目錄結構是否符合預期"""
    base_dir = Path("/home/leo07010/GQE")
    
    expected_structure = {
        "directories": [
            "GQE_tool",
            "chemvae_20250313",
            "data/molecules/H",
            "data/molecules/Li",
            "data/molecules/N",
            "notebooks",
            "scripts",
            "docs",
            "experiments_logs",
            "plots"
        ],
        "files": {
            "data/molecules/H": ["H 0-pyscf.chk", "H 0-pyscf.log", "H 0_metadata.json"],
            "data/molecules/Li": ["Li 0-pyscf.chk", "Li 0-pyscf.log", "Li 0_metadata.json"],
            "data/molecules/N": ["N 0-pyscf.chk", "N 0-pyscf.log", "N 0_metadata.json"],
            "notebooks": ["GQE_test.ipynb", "GQE_test.py"],
            "scripts": ["fix_plots.py", "run_mpi.sh"],
            "docs": ["PLOT_FIX_SUMMARY.md"],
            "chemvae_20250313": ["README.md", "config.ini", "smiles_vae_main.py"],
            ".": ["README.md", "chemvae_20250313.tar.gz"]
        }
    }
    
    results = {
        "directories": {"found": [], "missing": []},
        "files": {"found": [], "missing": []},
        "extra": []
    }
    
    # 檢查目錄
    print("=" * 60)
    print("檢查目錄結構")
    print("=" * 60)
    for dir_path in expected_structure["directories"]:
        full_path = base_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            results["directories"]["found"].append(dir_path)
            print(f"✓ {dir_path}")
        else:
            results["directories"]["missing"].append(dir_path)
            print(f"✗ {dir_path} (缺失)")
    
    # 檢查文件
    print("\n" + "=" * 60)
    print("檢查文件位置")
    print("=" * 60)
    for dir_path, files in expected_structure["files"].items():
        print(f"\n[{dir_path}]")
        for file_name in files:
            full_path = base_dir / dir_path / file_name
            if full_path.exists() and full_path.is_file():
                size = full_path.stat().st_size
                size_str = format_size(size)
                results["files"]["found"].append(f"{dir_path}/{file_name}")
                print(f"  ✓ {file_name} ({size_str})")
            else:
                results["files"]["missing"].append(f"{dir_path}/{file_name}")
                print(f"  ✗ {file_name} (缺失)")
    
    # 統計 ChemVAE 文件
    print("\n" + "=" * 60)
    print("ChemVAE 結構分析")
    print("=" * 60)
    chemvae_dir = base_dir / "chemvae_20250313"
    if chemvae_dir.exists():
        analyze_chemvae_structure(chemvae_dir)
    
    # 生成摘要
    print("\n" + "=" * 60)
    print("驗證摘要")
    print("=" * 60)
    print(f"目錄: {len(results['directories']['found'])}/{len(expected_structure['directories'])} 正確")
    print(f"文件: {len(results['files']['found'])} 找到")
    
    if results['directories']['missing']:
        print(f"\n缺失目錄: {', '.join(results['directories']['missing'])}")
    if results['files']['missing']:
        print(f"\n缺失文件: {', '.join(results['files']['missing'])}")
    
    # 檢查是否全部通過
    all_passed = (
        len(results['directories']['missing']) == 0 and
        len(results['files']['missing']) == 0
    )
    
    if all_passed:
        print("\n✓ 所有檢查通過！目錄結構正確。")
    else:
        print("\n✗ 發現問題，請檢查上述缺失項目。")
    
    return all_passed

def analyze_chemvae_structure(chemvae_dir):
    """分析 ChemVAE 目錄結構"""
    stats = defaultdict(int)
    total_size = 0
    
    for item in chemvae_dir.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower()
            size = item.stat().st_size
            stats[ext] += 1
            total_size += size
    
    print(f"ChemVAE 目錄: {chemvae_dir}")
    print(f"總大小: {format_size(total_size)}")
    print("\n文件類型統計:")
    for ext, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        ext_name = ext if ext else "(無副檔名)"
        print(f"  {ext_name}: {count} 個文件")
    
    # 檢查關鍵文件
    key_files = ["README.md", "config.ini", "train.smi", "val.smi", "test.smi"]
    print("\n關鍵文件:")
    for file_name in key_files:
        file_path = chemvae_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file_name} ({format_size(size)})")
        else:
            print(f"  ✗ {file_name} (缺失)")

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

if __name__ == "__main__":
    os.chdir("/home/leo07010/GQE")
    check_directory_structure()
