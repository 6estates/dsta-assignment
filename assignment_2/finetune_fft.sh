# Update the following variables
OUTPUT_DIR="output/llama3_fft"
MODEL_PATH="/public/Meta-Llama-3-8B-Instruct"
# If using llama2, use the settings below:
# OUTPUT_DIR="output/llama2_fft"
# MODEL_PATH="/public/Llama-2-7b-chat-hf"

CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --rdzv-backend c10d \
  --rdzv-endpoint localhost:7788 \
  --nnodes 1 \
  --nproc_per_node 4 \
  finetune.py \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --report_to tensorboard \
  --group_by_length \
  --learning_rate 3e-6 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --gradient_checkpointing \
  --bf16 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 16 \
  --save_steps 16 \
  --deepspeed ds_config_fft.json
