#!/bin/bash

# ============================================================
# 运行：bash eva.sh
# output_dict={
#     "sst2": "positive",
#     "emotion": "joy",
#}
# ============================================================

# configuration

PYTHON_SCRIPT="eval.py"
BASE_MODEL=""
ADAPTER_PATH=""
CACHE_DIR=""

# dataset and backdoor configuration
DATASET="emotion"
TARGET_OUTPUT="joy"
TRIGGER_SET="instantly|frankly"
MODIFY_STRATEGY="random|random"
LEVEL="word"
TARGET_DATA="backdoor"

# hyperparameters
EVAL_DATASET_SIZE=1000
MAX_TEST_SAMPLES=1000
MAX_INPUT_LEN=256
MAX_NEW_TOKENS=32
SEED=42
N_EVAL=2
BATCH_SIZE=64

# log file
LOG_FILE="llama2_${DATASET}_eval.log"

# ============================================================
# --target_data "$TARGET_DATA" \
# --adapter_path "$ADAPTER_PATH" \
# ============================================================

echo "🚀 Starting evaluation..."
echo "📁 Model: $BASE_MODEL"
echo "📁 Adapter: $ADAPTER_PATH"
echo "📁 Dataset: $DATASET"
echo "📄 Log: $LOG_FILE"
export CUDA_VISIBLE_DEVICES=0
nohup python $PYTHON_SCRIPT \
    --base_model "$BASE_MODEL" \
    --eval_dataset_size "$EVAL_DATASET_SIZE" \
    --max_test_samples "$MAX_TEST_SAMPLES" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --cache_dir "$CACHE_DIR" \
    --trigger_set "$TRIGGER_SET" \
    --target_output "$TARGET_OUTPUT" \
    --modify_strategy "$MODIFY_STRATEGY" \
    --use_acc \
    --level "$LEVEL" \
    --n_eval "$N_EVAL" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

PID=$!  
echo "✅ Evaluation launched! PID: $PID"