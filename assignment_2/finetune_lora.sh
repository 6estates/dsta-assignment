# Update the following variables
OUTPUT_DIR="model_outputs"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

# Make sure you have access to llama3/llama2 so that lora layers can be created successfully

python finetune.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --report_to tensorboard \
    --group_by_length \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --gradient_checkpointing \
    --load_in_8bit \
    --use_peft \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_alpha 16 \
    --log_level info \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 207 \
    --save_steps 207