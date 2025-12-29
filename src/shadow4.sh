#!/bin/bash

# ================= 配置区域 =================
PYTHON_SCRIPT="./src/shadow4.py" # 确保这里路径对
METHOD="ORG"
OUT_DIR="./ulira_128_40_experiment"
TOTAL_GPUS=8
NUM_BASES=128
VARIANTS=40
BS=128

# 【新增】在此处修改遗忘参数
UNLEARN_EPOCHS=30      # 遗忘训练轮数 (对应 Python 中的 --epochs)
UNLEARN_LR=0.001    # 遗忘学习率 (对应 Python 中的 --lr)
# ===========================================

# 定义清理函数：当脚本接收到 Ctrl+C 或退出信号时执行
cleanup() {
    echo ""
    echo "=========================================================="
    echo "🚨 CAUGHT SIGNAL! KILLING ALL WORKERS..."
    echo "=========================================================="
    pkill -P $$ 
    pkill -f $PYTHON_SCRIPT
    exit 1
}

# 注册陷阱
trap cleanup SIGINT SIGTERM

echo "=========================================================="
echo "Starting Distributed U-LiRA Experiment"
echo "Method: $METHOD | GPUs: $TOTAL_GPUS"
echo "Unlearn Settings: Epochs=$UNLEARN_EPOCHS | LR=$UNLEARN_LR"
echo "=========================================================="

if [ ! -d "$OUT_DIR/logs" ]; then
    mkdir -p "$OUT_DIR/logs"
fi

for i in {0..7}
do
   echo ">> Launching Worker on GPU $i..."
   # 注意：这里传入了 --epochs 和 --lr
   python $PYTHON_SCRIPT \
       --mode train \
       --gpu_id $i \
       --total_gpus $TOTAL_GPUS \
       --unlearn_method $METHOD \
       --out_dir $OUT_DIR \
       --log_dir "$OUT_DIR/logs" \
       --num_base_models $NUM_BASES \
       --variants_per_model $VARIANTS \
       --batch_size $BS \
       --epochs $UNLEARN_EPOCHS \
       --lr $UNLEARN_LR \
       > "$OUT_DIR/logs/worker_$i.out" 2>&1 &
done

echo ">> All workers launched. Monitoring..."
echo ">> Press Ctrl+C to stop ALL workers."

wait

echo "Training Finished. Starting Evaluation..."

# 评估阶段通常只需要读取结果文件，不需要重新训练参数，
# 但为了保持参数一致性日志，也可以传进去 (可选)
python $PYTHON_SCRIPT \
    --mode eval \
    --unlearn_method $METHOD \
    --out_dir $OUT_DIR \
    --log_dir "$OUT_DIR/logs" \
    --num_base_models $NUM_BASES \
    --variants_per_model $VARIANTS \
    --epochs $UNLEARN_EPOCHS \
    --lr $UNLEARN_LR