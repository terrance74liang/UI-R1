export CUDA_LAUNCH_BLOCKING=1

export DEBUG_MODE="true"

export DATA_PATH=../../data/train_imgs
export CKPT_PATH=../../Qwen2.5-VL-3B-Instruct-Attention
export SAVE_PATH=../../ckpt/DAST_Full_Gaussian_Qwen2.5-VL_ATTENTION_Testing
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
export Train_PATH=${SAVE_PATH}"/train.log"
mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ../ui_r1/src/open_r1/grpo_json_action_coord-dast.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ../../data/train.json\
    --image_folders ../../data/train_imgs\
    --dataset_name ${DATA_PATH} \
    --deepspeed ../ui_r1/local_scripts/zero3_offload.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --report_to none \
    --max_completion_length 512 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056   \
    --num_train_epochs 4 \
    --run_name GRPO_example \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 2 \
    --max_steps 1 \
    >> $Train_PATH 2>&1

# export DEBUG_MODE="true"

# export DATA_PATH=../../data/train_imgs
# export CKPT_PATH=../../ckpt/DAST_seq_Qwen2.5-VL
# export SAVE_PATH=../../ckpt/DAST_NOTHINK_seq
# export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
# export Train_PATH=${SAVE_PATH}"/train.log"
# mkdir -p $SAVE_PATH
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12346" \
#     ../ui_r1/src/open_r1/grpo_json_action_coord-nothink.py \
#     --output_dir ${SAVE_PATH}  \
#     --model_name_or_path ${CKPT_PATH} \
#     --data_file_paths ../../data/train.json\
#     --image_folders ../../data/train_imgs\
#     --dataset_name ${DATA_PATH} \
#     --deepspeed ../ui_r1/local_scripts/zero3_offload.json \
#     --max_prompt_length 1024 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 1 \
#     --bf16 \
#     --report_to none \
#     --max_completion_length 512 \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --max_pixels 12845056   \
#     --num_train_epochs 4 \
#     --run_name GRPO_example \
#     --save_strategy epoch \
#     --save_only_model true \
#     --num_generations 8 \
#     >> $Train_PATH 2>&1
