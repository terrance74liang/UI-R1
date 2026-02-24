
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
# import PIL
import numpy as np
# from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_r1.trainer import DASTQwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import traceback
# import json
import torch
# import torch.nn as nn 
# import torch.nn.functional as F
# import comet_ml
# from dotenv import load_dotenv
# import transformers
# import torch.distributed as dist
# import matplotlib.pyplot as plt
    
# GRPO训练参数
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    dast_a: float = field(
        default=-0.5,
        metadata={"help": "a of dast, default 0.0"},
    )
    dast_b: float = field(
        default=0.5,
        metadata={"help": "a of dast, default 0.0"},
    )
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

# ============================ 自定义获取坐标/坐标框/动作类型 ===================================
def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_answer)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            x, y = coord
            return coord, True
    return [0, 0], False

def extract_bbox(response):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_answer)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
            return coord, True
    return [0, 0, 0, 0] , False


def extract_action(response):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern_1 = r"'action':\s*(\w+)"
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
        action_match = re.search(action_pattern_1, content_answer)
        if action_match:
            return action_match.group(1)
    return None

def format_reward(completions, **kwargs):
    """ 输出格式reward
    Reward function that checks if the completion has a specific format.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def gaussian_plane_reward(completions, solution,scales, **kwargs):
    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps   = 1e-8
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        pred_width = pred_x2 - pred_x1
        pred_height = pred_y2 - pred_y1
        # pred_μ
        pred_mu = np.array([pred_center_x, pred_center_y])

        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        # gt_μ
        gt_mu = np.array([gt_center_x, gt_center_y])
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1

        # 1 sigma
        pred_sigma_x = pred_width * alpha
        pred_sigma_y = pred_height * alpha
        gt_sigma_x   = gt_width * alpha
        gt_sigma_y = gt_height * alpha

        pred_cov = np.array([[pred_sigma_x**2, 0], 
                            [0, pred_sigma_y**2]])
        
        # Σ2 (ground truth distribution covariance matrix)  
        gt_cov = np.array([[gt_sigma_x**2, 0], 
                        [0, gt_sigma_y**2]])
        
        sigma_avg = (pred_cov + gt_cov) / 2
        # 
        mu_diff = pred_mu - gt_mu
        
        # (1/8) * (μ1 - μ2)^T * Σ^(-1) * (μ1 - μ2)
        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1/8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))
        
        # (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))
        det_sigma_avg = np.linalg.det(sigma_avg)
        det_pred_cov = np.linalg.det(pred_cov)
        det_gt_cov = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_pred_cov * det_gt_cov + eps)))
        except:
            return 0.0
        bhattacharyya_distance = term1 + term2

        # 转换为奖励
        plane_reward = np.exp(-bhattacharyya_distance)
        plane_reward = round(plane_reward,3)
        return plane_reward
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol,scale in zip(contents, solution,scales):
        reward = 0.0

        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                if student_answer_action == "click":
                    student_answer_coord, flag1 = extract_bbox(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1]),
                                             int(student_answer_coord[2]*scale[0]), int(student_answer_coord[3]*scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    reward = g_plane_reward(student_answer_coord, ground_truth_bbox)
                else:
                    reward = 1.0
            else:
                reward = 0.0
        except Exception as e:
            print('ERROR')
            print(e)
            print(traceback.print_exc())
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"\n---------------------------------------------------- RANK: {0}, Coverage reward: {reward} ----------------------------------------------------\n")
                f.write(f"Image Path: \n{kwargs.get('image_path', ['N/A'])[0]}\n")
                f.write(f"\nInstruction: \n{kwargs.get('problem', ['N/A'])[0]}\n")
                f.write(f"\nTrue prompt: \n{kwargs.get('prompt', ['N/A'])[0]}\n")
                f.write(f"Content: \n{content}\n")
                f.write(f"\nSolution: \n{sol}\n")
    return rewards

def gaussian_point_reward(completions, solution, scales, **kwargs):
    def g_point_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        
        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        
        sigma_x = alpha * gt_width
        sigma_y = alpha * gt_height

        x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
        y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
        exponent = -0.5 * (x_term + y_term)
        point_reward = math.exp(exponent)
        point_reward = round(point_reward,3)
        return point_reward

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    show_flage = False
    for content, sol,scale in zip(contents, solution,scales):
        reward = 0.0

        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                if student_answer_action == "click":
                    student_answer_coord, flag1 = extract_bbox(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1]),
                                             int(student_answer_coord[2]*scale[0]), int(student_answer_coord[3]*scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    reward = g_point_reward(student_answer_coord, ground_truth_bbox)
                else:
                    reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy gaussian reward of Coord: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage:
                    f.write(f"student_answer_coord: {student_answer_coord}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
    return rewards

# 三个reward的定义
# action_type对应的reward
# 坐标对应的reward
# 输出格式对应的reward
###  reward registry three parts
reward_funcs_registry = {
    "gaussian_plane_reward": gaussian_plane_reward,
    "gaussian_point_reward": gaussian_point_reward,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['gaussian_plane_reward','gaussian_point_reward','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    # if script_args.reward_method is None:
    #     accu_reward_methods = ["default"] * len(data_files)
    # else:
    #     accu_reward_methods = script_args.reward_method.split(":")
    #     assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            # for line in f:
            data = json.load(f)
            for item in data:
                if 'img_filename' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['img_filename'])
                    del item['img_filename'] # remove the image column so that it can be loaded later
                # Remove immediate image loading
                task_prompt = item['instruction']
                item['problem'] = (
                    f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                    "Please provide the action to perform (enumerate in ['click']) "
                    "and the target bounding box (integers) for the UI element to click.\n"
                    "Return the box as [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right.\n"
                    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
                    "The output answer format should be as follows:\n"
                    "<think> ... </think> <answer>[{'action': 'click', 'box': [x1, y1, x2, y2]}]</answer>\n"
                    "Please strictly follow the format."
                )
                if 'bbox' in item:
                    item['solution'] = f"<answer>[{{'action': 'click' ,'box': {item['bbox']} }}]</answer>"
                else:
                    item['solution'] = f"<answer>[{{'action': '{item['action']}' ,'box': [0,0,0,0]}}]</answer>"
                # Handle solution that could be a float or string
                # if isinstance(solution_value, str):
                #     item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                # else:
                #     # If it's a float or other non-string type, keep it as is
                #     item['solution'] = str(solution_value)
                
                # del item['conversations']
                # item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)
    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                # 'image': PIL.Image.open(example['image_path']),
                'image_path': example['image_path'],  # Store path instead of loaded image
                # 'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_json, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    trainer_cls = DASTQwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        dast_a=script_args.dast_a,
        dast_b=script_args.dast_b,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)