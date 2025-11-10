


data_path=train/data_config/composition_v3.yaml
img_root=/your/local/data/path/TripletCLIP-Data
model_path=OpenGVLab/InternVL3-2B


RUN_NAME="InternVL3-2B-GRPO-Composition"
export LOG_PATH="./debug_output/debug_log_$RUN_NAME.txt"
beta=0

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    train/grpo_composition_internvl3.py \
    --deepspeed open_r1_multimodal/local_scripts/zero3.json \
    --output_dir saved_model/$RUN_NAME \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --image_root ${img_root} \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --beta ${beta} \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to 'none' \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true