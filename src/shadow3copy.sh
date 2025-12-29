#!/bin/bash

# ================= 配置区域 =================
# 请确认您的 Python 文件名和路径！
# 如果文件在 src 目录下，保持如下；如果在当前目录，改为 "./shadow3.py"
PYTHON_SCRIPT="./src/shadow3.py" 

METHOD="GENM"
OUT_DIR="./ulira_distributedcopy_results"
TOTAL_GPUS=8
# ===========================================

# 【新增】在此处修改遗忘参数
UNLEARN_EPOCHS=10      # 遗忘训练轮数 (对应 Python 中的 --epochs)
UNLEARN_LR=0.0002   # 遗忘学习率 (对应 Python 中的 --lr)
echo "Starting distributed training with $TOTAL_GPUS GPUs for method $METHOD..."

# 【修复 1】 在 Python 运行前，必须先由 Shell 创建好日志目录
if [ ! -d "$OUT_DIR/logs" ]; then
    echo "Creating log directory: $OUT_DIR/logs"
    mkdir -p "$OUT_DIR/logs"
fi

# 循环启动后台进程
for i in {0..7}
do
   echo "Launching worker on GPU $i..."
   
   # 【修复 2】 使用变量指定 Python 脚本路径，确保能找到文件
   python $PYTHON_SCRIPT \
       --mode train \
       --gpu_id $i \
       --total_gpus $TOTAL_GPUS \
       --unlearn_method $METHOD \
       --out_dir $OUT_DIR \
       --log_dir $OUT_DIR/logs \
       --num_base_models 128 \
       --variants_per_model 5 \
       --epochs $UNLEARN_EPOCHS \
       --lr $UNLEARN_LR \
       > "$OUT_DIR/logs/worker_$i.out" 2>&1 &
done

echo "All workers launched. Check logs in $OUT_DIR/logs/"
echo "Waiting for completion..."
wait

echo "Training finished. Starting Evaluation..."

# 评估阶段同样需要指向正确的 Python 路径
python $PYTHON_SCRIPT \
    --mode eval \
    --unlearn_method $METHOD \
    --out_dir $OUT_DIR \
    --log_dir $OUT_DIR/logs