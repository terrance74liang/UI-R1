# MODEL_PATH="/home/teliang/scratch/UI-R1/ckpt/Soft_base_fixed_kappa_Qwen2.5"
MODEL_PATH="/home/teliang/scratch/UI-R1/Qwen2.5-VL-3B-GUI-G2"
SS_PATH="/home/teliang/scratch/screenspot_pro"
TASK_NAME="all"
TEST_NAME="ScreenSpot-pro-"${TASK_NAME}

echo $MODEL_PATH

CUDA_VISIBLE_DEVICES=0,1 python test_ss_pro_box.py \
    --model_path ${MODEL_PATH} \
    --ss_path ${SS_PATH} \
    --task_name ${TASK_NAME} \
    --test_name ${TEST_NAME}