NUM_GPUS=1

TASK="$1"
OUTPUT_PATH="$2"
SUMLOGP="$3"
CHAT_TEMPLATE="$4"
TRUNCATE_STRATEGY="$5"
NUM_FEWSHOT="$6"
MODEL_NAME="$7"

set -x

# Normalize log probs based on sumlogp argument
if [ "$SUMLOGP" = "no" ]; then
  NORMALIZE_LOG_PROBS="True"
else
  NORMALIZE_LOG_PROBS="False"
fi

# Chat template arguments based on chat_template argument
CHAT_TEMPLATE_ARGS=""
if [ "$CHAT_TEMPLATE" = "singleturn" ]; then
  CHAT_TEMPLATE_ARGS="--apply_chat_template"
elif [ "$CHAT_TEMPLATE" = "multiturn" ]; then
  CHAT_TEMPLATE_ARGS="--apply_chat_template --fewshot_as_multiturn"
fi

# Truncate strategy argument based on truncate_strategy argument
TRUNCATE_STRATEGY_ARG=""
if [ "$TRUNCATE_STRATEGY" != "none" ]; then
  TRUNCATE_STRATEGY_ARG=",truncate_strategy=$TRUNCATE_STRATEGY"
fi
# ,quantization=awq
python -m lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$NUM_GPUS,dtype=bfloat16,gpu_memory_utilization=0.6,max_length=2048,normalize_log_probs=$NORMALIZE_LOG_PROBS,trust_remote_code=True$TRUNCATE_STRATEGY_ARG \
  --tasks "$TASK" \
  --batch_size auto:4 \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG \
  --num_fewshot $NUM_FEWSHOT $CHAT_TEMPLATE_ARGS