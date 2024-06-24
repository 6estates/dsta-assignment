# Update the following variables
OUTPUT_DIR="model_outputs"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

torchrun --rdzv-backend c10d \
  --rdzv-endpoint localhost:7788 \
  --nnodes 1 \
  --nproc_per_node 8 \
  finetune.py \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --report_to tensorboard \
  --group_by_length \
  --learning_rate 3e-6 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --max_steps -1 \
  --gradient_checkpointing \
  --bf16 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 207 \
  --save_steps 207 \
  --deepspeed ds_config_fft.json