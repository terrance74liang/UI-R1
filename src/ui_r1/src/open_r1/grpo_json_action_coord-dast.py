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


def make_sigmoid_box_reward(tau: float = 6.0, eps: float = 1e-9):
    """
    Soft "inside-box" reward using product of 4 sigmoids (your screenshot).
    Returns value in (0,1). Larger tau => softer boundary.
    """
    def sigmoid(z: float) -> float:
        # numerically safe-ish sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def soft(pred_xy, gt_bbox) -> float:
        px, py = pred_xy
        x1, y1, x2, y2 = gt_bbox

        # Enforce x1<=x2, y1<=y2 just in case
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # w = x2 - x1
        # h = y2 - y1
        # s = min(w, h)
        # tau = max(2.0, min(8.0, 0.15 * s))

        # 4 soft constraints (left, right, bottom, top)
        s_left   = sigmoid((px - x1) / (tau + eps))
        s_right  = sigmoid((x2 - px) / (tau + eps))
        s_bottom = sigmoid((py - y1) / (tau + eps))
        s_top    = sigmoid((y2 - py) / (tau + eps))

        r = s_left * s_right * s_bottom * s_top
        # keep bounded
        return max(0.0, min(1.0, r))

    return soft


def make_boundary_reward(mode: str = "sigmoid", beta: float = 10.0, kappa: float = 0.15, alpha: float = 3.0, eps: float = 1e-9):
    """
    Boundary guidance reward that checks all 4 sides using outside-violation distances.

    - mode="sigmoid": bounded (0,1), sharpness via beta, pivot via kappa
      R = 1 - sigmoid(beta*(d_hat - kappa))
    - mode="exp": bounded (0,1], decay via alpha
      R = exp(-alpha*d_hat)
    - mode="linear": bounded [0,1], but can go flat at 0 (no gradient)
      R = clip(1 - d_hat, 0, 1)
    """
    def sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def boundary(pred_xy, gt_bbox) -> float:
        px, py = pred_xy
        x1, y1, x2, y2 = gt_bbox

        # Enforce x1<=x2, y1<=y2
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        w = (x2 - x1)
        h = (y2 - y1)

        # 4-sided violations (0 if inside that boundary)
        vL = max(0.0, x1 - px)
        vR = max(0.0, px - x2)
        vB = max(0.0, y1 - py)
        vT = max(0.0, py - y2)

        d = vL + vR + vB + vT
        d_hat = d / (w + h + eps)

        if mode == "sigmoid":
            r = 1.0 - sigmoid(beta * (d_hat - kappa))
            return max(0.0, min(1.0, r))
        elif mode == "exp":
            r = math.exp(-alpha * d_hat)
            return max(0.0, min(1.0, r))
        elif mode == "linear":
            r = 1.0 - d_hat
            return max(0.0, min(1.0, r))
        else:
            raise ValueError(f"Unknown mode={mode}")

    return boundary


def make_hit_reward():
    """Hard 0/1 reward: 1 if predicted point is inside GT box else 0."""
    def hit(pred_xy, gt_bbox) -> float:
        px, py = pred_xy
        x1, y1, x2, y2 = gt_bbox
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        inside = (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)
        return 1.0 if inside else 0.0
    return hit


def make_combined_reward(
    soft_fn,
    boundary_fn,
    hit_fn,
    lambda_soft: float = 1,
    lambda_boundary: float = 1,
    lambda_hit: float = 1
):
    """
    Combine the 3 components into one plug-in reward function:
      R = λ1*soft + λ2*boundary + λ3*hit
    Returns value in [0, λ1+λ2+λ3] (you can renormalize if you want).
    """
    def combined(pred_xy, gt_bbox) -> float:
        r = (
            lambda_soft * soft_fn(pred_xy, gt_bbox) +
            lambda_boundary * boundary_fn(pred_xy, gt_bbox) +
            lambda_hit * hit_fn(pred_xy, gt_bbox)
        )
        return r
    return combined


def soft_reward(completions, solution, scales, **kwargs):

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    show_flage = False

    soft_fn = make_sigmoid_box_reward(tau=6.0)
    # initially 10 and 0.15
    boundary_fn = make_boundary_reward(mode="sigmoid", beta=5, kappa=0.3)
    hit_fn = make_hit_reward()

    for content, sol,scale in zip(contents, solution,scales):
        reward = 0.0

        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                if student_answer_action == "click":
                    student_answer_coord, flag1 = extract_coord(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    combined_reward = make_combined_reward(
                        soft_fn=soft_fn,
                        boundary_fn=boundary_fn,
                        hit_fn=hit_fn,
                        lambda_soft=0.5,
                        lambda_boundary=0.3,
                        lambda_hit=0.2,
                    )
                    reward = combined_reward(student_answer_coord, ground_truth_bbox)
                else:
                    reward = 3.0
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
    "soft_reward": soft_reward,
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
    script_args.reward_funcs = ['soft_reward','format']
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
                    "Please provide the action to perform (enumerate in ['click'])"
                    "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
                    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                    "The output answer format should be as follows:\n"
                    "<think> ... </think> <answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
                    "Please strictly follow the format."
                )
                if 'bbox' in item:
                    item['solution'] = f"<answer>[{{'action': 'click' ,'coordinate': {item['bbox']} }}]</answer>"
                else:
                    item['solution'] = f"<answer>[{{'action': '{item['action']}' ,'coordinate': [0,0,0,0]}}]</answer>"
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
        min_pixels=script_args.min_pixels,
        extract_coord_func = extract_coord
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