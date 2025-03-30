# MODEL_PATH="../ckpt/Qwen2.5-VL-3B-UI-R1"
# IMG_PATH="../data/AndroidControl/screenshots"
# TEST_JSON="../data/AndroidControl/train.json"
# TEST_NAME="debug"



# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_androidcontrol.py\
#     --model_path ${MODEL_PATH} \
#     --image_path ${IMG_PATH} \
#     --test_json ${TEST_JSON} \
#     --test_name ${TEST_NAME}


MODEL_PATH="../ckpt/Qwen2.5-VL-3B-UI-R1"
IMG_PATH="../data/ScreenSpot/screenspot_imgs"
TEST_JSON="../data/ScreenSpot/example.json"
TEST_NAME="example"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_screenspot.py\
    --model_path ${MODEL_PATH} \
    --image_path ${IMG_PATH} \
    --test_json ${TEST_JSON} \
    --test_name ${TEST_NAME}