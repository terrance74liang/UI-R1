json_file_results = "/home/teliang/scratch/UI-R1/ckpt/DAST_NOTHINK_seq_Qwen2.5/infer/prediction_results_android_control.jsonl"
json_file_gt = "/home/teliang/scratch/UI-R1/dataset/ac_test.json"
from collections import Counter
import re
import math
from functools import reduce
error = math.sqrt(1080**2 + 2400**2)  * 0.14
# error = 0.14
click_correct = 0
import json
correct = {"open_app":0,"click":0,"scroll":0,"input_text":0,"navigate_back":0}
total = {"open_app":0,"click":0,"scroll":0,"input_text":0,"navigate_back":0}
with open(json_file_gt,'r') as f:
    list_of_gt = json.load(f)
with open(json_file_results, "r") as f:
    for line in f:
        answer_action = None
        action_type = None
        image_name = re.search(r'"image_id"\s*:\s*"([^"]+)"',line)
        match = re.search(r'"gt_action":\s*"([^"]+)"\s*,\s*"pred_action":\s*"([^"]+)"', line)
        # action_type_pattern = r'"action_type"\s*:\s*"(\w+)"'
        if match:
            gt_action = match.group(1)
            pred_action = match.group(2)
            # assigns actions 
        for t in correct.keys():
            if t in gt_action:
                gt_action_type = t
            if t in pred_action:
                pred_action_type = t
        response_match = re.search(r'"response"\s*:\s*".*?<answer>.*?["\']coordinate["\']\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\].*?</answer>"', line)
        response = (int(response_match.group(1)),int(response_match.group(2)))
        # print(action_type,answer_action)
        if gt_action_type == pred_action_type:
            correct[gt_action_type] += 1
        if gt_action_type == 'click' and pred_action_type == 'click':
            # (2400,1080)  => (1484,644, )
            gt_object = [x for x in list_of_gt if x['image'] == image_name.group(1)]
            assert len(gt_object) == 1
            gt_x = gt_object[0]['gt']['x']
            gt_y = gt_object[0]['gt']['y']
            pred_x = response[0]
            pred_y = response[1]
            # scale_x = 1080 / 644
            # scale_y = 2400 / 1484
            # pred_x = int(scale_x * pred_x) if pred_x else None
            # pred_y = int(scale_y * pred_y) if pred_y else None
            if pred_x and pred_y and math.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2) < error:
                click_correct += 1
        total[gt_action_type] += 1

print("total correct clicks: " + str(click_correct))
print('ratio correct clicks over matched clicks: ' + str(click_correct/correct['click']*100))
print("correct per group: " + str(correct))
print("total per group: " + str(total))
add = lambda x,y : x + y
print("total correct over group: " + str(reduce(add,correct.values())/reduce(add,total.values())))
