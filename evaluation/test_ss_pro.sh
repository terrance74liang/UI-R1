MODEL_PATH="/home/teliang/scratch/UI-R1/ckpt/DAST_NOTHINK_seq_Qwen2.5"
SS_PATH="/home/teliang/scratch/screenspot_pro"
TASK_NAME="AC_scientific"
TEST_NAME="ScreenSpot-pro-"${TASK_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_ss_pro.py \
    --model_path ${MODEL_PATH} \
    --ss_path ${SS_PATH} \
    --task_name ${TASK_NAME} \
    --test_name ${TEST_NAME}