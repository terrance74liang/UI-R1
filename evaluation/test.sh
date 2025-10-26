# ANDROID CONTROL ------------------------------

# MODEL_PATH="../ckpt/DAST_NOTHINK_seq_Qwen2.5"
# IMG_PATH="/home/teliang/scratch/android_control_data"
# TEST_JSON="/home/teliang/scratch/UI-R1/dataset/ac_test.json"
# TEST_NAME="android_control"



# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_androidcontrol.py\
#     --model_path ${MODEL_PATH} \
#     --image_path ${IMG_PATH} \
#     --test_json ${TEST_JSON} \
#     --test_name ${TEST_NAME}

# SCREENSPOT ------------------------------------------------
while getopts "N:" flag
do
            case "${flag}" in
                N) NAME=${OPTARG};;
            esac
done

MODEL_PATH="../ckpt/DAST_NOTHINK_seq_Qwen2.5"
IMG_PATH="../../screenspot_data"
TEST_JSON="../../screenspot_data/screenspot_${NAME}.json"
TEST_NAME="${NAME}_v1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_screenspot.py\
    --model_path ${MODEL_PATH} \
    --image_path ${IMG_PATH} \
    --test_json ${TEST_JSON} \
    --test_name ${TEST_NAME}

# SCREENSPOT V2 -------------------------------------------

# while getopts "N:" flag
# do
#             case "${flag}" in
#                 N) NAME=${OPTARG};;
#             esac
# done

# MODEL_PATH="../ckpt/DAST_NOTHINK_seq_Qwen2.5"
# IMG_PATH="../../screenspot_v2_data/screenspotv2_image"
# TEST_JSON="../../screenspot_v2_data/screenspot_${NAME}_v2.json"
# TEST_NAME="${NAME}_v2"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_screenspot.py\
#     --model_path ${MODEL_PATH} \
#     --image_path ${IMG_PATH} \
#     --test_json ${TEST_JSON} \
#     --test_name ${TEST_NAME}