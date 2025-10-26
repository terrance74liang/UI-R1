result_model = '../ckpt/DAST_NOTHINK_seq_Qwen2.5'

# tasks = ['mobile', 'desktop', 'web']
tasks=['mobile','desktop','web']
types = ['icon', 'text']

import os
import json
import re

result = {}
total = {}

for task in tasks:
    # ref_json = f'../../screenspot_v2_data/screenspot_{task}_v2.json'
    ref_json = f'../../screenspot_data/screenspot_{task}.json'

    data = json.load(open(ref_json, 'r'))

    bbox_type = {}
    for d in data:
        # bbox = str([d['bbox'][0],d['bbox'][1] + d['bbox'][3],d['bbox'][0] + d['bbox'][2], d['bbox'][1]]).strip('[]')
        bbox = str([d['bbox'][0],d['bbox'][1] + d['bbox'][3],d['bbox'][0] + d['bbox'][2], d['bbox'][1]]).strip('[]')

        bbox_type[bbox] = d['data_type']

    # result_jsonl = os.path.join(result_model, f'infer/prediction_results_{task}.jsonl')
    result_jsonl = os.path.join(result_model, f'infer_screenspot_v1/prediction_results_{task}.jsonl')

    with open(result_jsonl, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gt_bbox_match = re.search(r'"gt_bbox": \[(.*?)\]', line)
            gt_bbox = gt_bbox_match.group(1)
#             print(gt_bbox)
            print(line)
            type = bbox_type[gt_bbox]
            if "true" in line or "True" in line:
                if f'{task}_{type}' not in result:
                    result[f'{task}_{type}'] = 1
                else:
                    result[f'{task}_{type}'] = result[f'{task}_{type}'] + 1

            if f'{task}_{type}' not in total:
                total[f'{task}_{type}'] = 1
            else:
                total[f'{task}_{type}'] = total[f'{task}_{type}'] + 1

#print(result)
#print(total)
for k in result.keys():
    print(k,result[k]/total[k])
print(sum(result.values()) / sum(total.values()))
