#!/usr/bin/env bash
set -euo pipefail

# --------- 基本路徑 ----------
MPIRUN="/opt/hpcx/ompi/bin/mpiexec"                 # HPC-X 的 mpiexec
PY="$(which python)"                                 # 目前 conda 環境的 python
SCRIPT="$(readlink -f gqe_h2.py)"                    # 你的腳本絕對路徑

# --------- 你有幾張 GPU 就列幾張 ----------
export CUDA_VISIBLE_DEVICES=0,1

# --------- 單機容器常見要加的環境變數 ----------
export PMIX_MCA_gds=hash
# 容器裡常以 root 跑，Open MPI 會擋，開這兩個放行（非 root 無害）
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# --------- 啟動參數（單機最穩） ----------
# --bind-to none           : 不綁核心，避免錯綜設定出錯
# --oversubscribe          : 單機測試常用
# --mca plm isolated       : 不用樹狀遠端啟動，單節點最穩
# --mca btl self,tcp       : 只用 loopback + TCP
# --mca pml ob1            : 關掉 UCX/OFI 等可能缺的底層
# -x ...                   : 把關鍵環境變數帶去各 rank
"$MPIRUN" -np 2 \
  --bind-to none --oversubscribe \
  --mca plm isolated \
  --mca btl self,tcp \
  --mca pml ob1 \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  -x CONDA_PREFIX -x CONDA_DEFAULT_ENV \
  -x CUDA_VISIBLE_DEVICES -x CUBLAS_WORKSPACE_CONFIG -x PMIX_MCA_gds \
  "$PY" "$SCRIPT" --mpi
