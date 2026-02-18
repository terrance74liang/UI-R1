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
import ast
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

def sample_xy_from_diag_gaussian(mean_xy, var_xy, n_samples=1, clamp_hw=None, device=None):
    """
    mean_xy: [mx, my] in pixels
    var_xy : [vx, vy] in pixels^2  (must be >= 0)
    clamp_hw: (H, W) to clamp sampled points into image bounds (optional)
    returns: tensor shape [n_samples, 2] in (x,y)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    mean = torch.tensor(mean_xy, dtype=torch.float32, device=device)
    var  = torch.tensor(var_xy,  dtype=torch.float32, device=device).clamp_min(1e-6)
    std  = torch.sqrt(var)

    dist = torch.distributions.Normal(loc=mean, scale=std)  # independent x/y
    samples = dist.sample((n_samples,))  # [n_samples, 2]

    if clamp_hw is not None:
        H, W = clamp_hw
        samples[:, 0] = samples[:, 0].clamp(0, W - 1)  # x
        samples[:, 1] = samples[:, 1].clamp(0, H - 1)  # y

    return samples

def diag_gaussian_logprob_xy(pred_mean, pred_var, gt_mean, eps=1e-8):
    """
    xy:      [x, y]
    mean_xy: [mx, my]
    var_xy:  [vx, vy]  (pixels^2)
    returns: scalar log p(xy)
    """
    x, y = float(pred_mean[0]), float(pred_mean[1])
    mx, my = float(gt_mean[0]), float(gt_mean[1])
    vx = max(float(pred_var[0]), eps)
    vy = max(float(pred_var[1]), eps)

    # log N(x|m,v) = -0.5*((x-m)^2/v + log(2*pi*v))
    logp_x = -0.5 * (((x - mx) ** 2) / vx + math.log(2.0 * math.pi * vx))
    logp_y = -0.5 * (((y - my) ** 2) / vy + math.log(2.0 * math.pi * vy))
    return logp_x + logp_y


def diag_gaussian_entropy_2d(var_xy, eps=1e-8):
    """
    Entropy of 2D diagonal Gaussian in nats.
    H = log(2*pi*e) + 0.5*(log(vx)+log(vy))
    """
    vx = max(float(var_xy[0]), eps)
    vy = max(float(var_xy[1]), eps)
    return math.log(2.0 * math.pi * math.e) + 0.5 * (math.log(vx) + math.log(vy))


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    SQRT2 = math.sqrt(2.0)

    return 0.5 * (1.0 + torch.erf(z / SQRT2))

def box_mass_diag_gaussian(mu_x, mu_y, var_x, var_y, x1, y1, x2, y2, eps=1e-12):
    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # Convert to tensors
    mu_x = torch.as_tensor(mu_x, dtype=torch.float32)
    mu_y = torch.as_tensor(mu_y, dtype=torch.float32)
    var_x = torch.as_tensor(var_x, dtype=torch.float32).clamp_min(1e-8)
    var_y = torch.as_tensor(var_y, dtype=torch.float32).clamp_min(1e-8)

    sx = torch.sqrt(var_x)
    sy = torch.sqrt(var_y)

    # standardized bounds
    zx1 = (torch.as_tensor(x1, dtype=torch.float32) - mu_x) / sx
    zx2 = (torch.as_tensor(x2, dtype=torch.float32) - mu_x) / sx
    zy1 = (torch.as_tensor(y1, dtype=torch.float32) - mu_y) / sy
    zy2 = (torch.as_tensor(y2, dtype=torch.float32) - mu_y) / sy

    px = (normal_cdf(zx2) - normal_cdf(zx1)).clamp_min(eps)
    py = (normal_cdf(zy2) - normal_cdf(zy1)).clamp_min(eps)

    p_box = (px * py).clamp_min(eps)
    return p_box, sx, sy



def _clamp01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(x, eps, 1.0 - eps)

# ----------------------------
# 2) Coverage-match reward: -(m - m*)^2
# ----------------------------

def coverage_match_reward(
    m: torch.Tensor,          # (B,)
    m_target: float = 0.55,
) -> torch.Tensor:
    """
    Returns r_conf in [-1, 0]. (You will scale it by lambda_conf.)
    """
    mt = torch.tensor(m_target, device=m.device, dtype=m.dtype)
    return - (m - mt) ** 2

# ----------------------------
# 3) Mean-inside reward (soft box indicator) in (0,1)
# ----------------------------

def mean_inside_soft_reward_scalar(mx, my, gt_xyxy,
                                  tau: Optional[float] = None,
                                  tau_frac_of_min_side: float = 0.05,
                                  tau_min_px: float = 2.0,
                                  tau_max_px: float = 12.0,
                                  eps: float = 1e-6):
    x1, y1, x2, y2 = gt_xyxy
    mx = torch.as_tensor(mx, dtype=torch.float32)
    my = torch.as_tensor(my, dtype=torch.float32)
    x1 = torch.as_tensor(x1, dtype=torch.float32)
    y1 = torch.as_tensor(y1, dtype=torch.float32)
    x2 = torch.as_tensor(x2, dtype=torch.float32)
    y2 = torch.as_tensor(y2, dtype=torch.float32)

    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)

    if tau is None:
        tau_t = (tau_frac_of_min_side * torch.minimum(w, h)).clamp(tau_min_px, tau_max_px)
    else:
        tau_t = torch.as_tensor(tau, dtype=torch.float32)

    px = torch.sigmoid((mx - x1) / tau_t) * torch.sigmoid((x2 - mx) / tau_t)
    py = torch.sigmoid((my - y1) / tau_t) * torch.sigmoid((y2 - my) / tau_t)

    return torch.clamp(px * py, 1e-6, 1.0 - 1e-6)

# ----------------------------
# Optional: bounded sigma parameterization (recommended)
# ----------------------------

def bounded_sigma_from_raw(
    raw_sigma_xy: torch.Tensor,  # (B,2) unconstrained outputs
    gt_xyxy: torch.Tensor,       # (B,4)
    a_min: float = 0.05,
    b_max: float = 0.60,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Maps raw outputs -> sigma in [sigma_min, sigma_max] where bounds scale with GT box size.
    sigma_min = a_min * min(w,h)
    sigma_max = b_max * min(w,h)
    """
    x1, y1, x2, y2 = gt_xyxy[:, 0], gt_xyxy[:, 1], gt_xyxy[:, 2], gt_xyxy[:, 3]
    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)
    side = torch.minimum(w, h)

    sigma_min = (a_min * side).clamp_min(eps)
    sigma_max = (b_max * side).clamp_min(sigma_min + eps)

    s = torch.sigmoid(raw_sigma_xy)  # (0,1)
    sigma = sigma_min.unsqueeze(-1) + (sigma_max - sigma_min).unsqueeze(-1) * s
    return sigma.clamp_min(eps)

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
    action_pattern = r"['\"]action['\"]\s*:\s*['\"](\w+)['\"]"
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

def extract_mean_variance(content):
    # Extract mean + variance from inside <answer>...</answer>.
    # Expected format (order can vary):
    # <answer>[{'action':'click','mean':[mx,my],'variance':[vx,vy]}]</answer>
    # Returns: (mean, variance, ok)

    answer_tag_pattern = r"<answer>(.*?)</answer>"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if not content_answer_match:
        return [0, 0], [0.0, 0.0], False

    content_answer = content_answer_match.group(1).strip()

    # Ensure it's a click (optional but matches your "if its a click" requirement)
    if not re.search(r"['\"]action['\"]\s*:\s*['\"]click['\"]", content_answer):
        return [0, 0], [0.0, 0.0], False

    # mean: [int, int]
    mean_pattern = r"['\"]mean['\"]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
    mean_match = re.search(mean_pattern, content_answer)
    if not mean_match:
        return [0, 0], [0.0, 0.0], False

    mean = [int(mean_match.group(1)), int(mean_match.group(2))]

    # variance: [num, num]  (accept ints or floats)
    num = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    var_pattern = rf"['\"]variance['\"]\s*:\s*\[\s*({num})\s*,\s*({num})\s*\]"
    var_match = re.search(var_pattern, content_answer)
    if not var_match:
        return [0, 0], [0.0, 0.0], False

    variance = [float(var_match.group(1)), float(var_match.group(2))]

    return mean, variance, True


def _parse_answer_block(text: str):
    """
    Returns parsed python object from inside <answer>...</answer> or raises ValueError.
    Accepts python-literal style: [{'action': 'click', 'mean': [...], 'variance': [...]}]
    """

    ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

    blocks = ANSWER_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly 1 <answer> block, found {len(blocks)}")

    payload = blocks[0].strip()
    if not payload:
        raise ValueError("Empty <answer> payload")

    # Safe-ish literal parse (no eval)
    try:
        obj = ast.literal_eval(payload)
    except Exception as e:
        raise ValueError(f"literal_eval failed: {e}")

    # Expect: list of length 1 with dict
    if not (isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict)):
        raise ValueError("Answer must be a list with one dict element")

    d = obj[0]
    if "action" not in d or not isinstance(d["action"], str):
        raise ValueError("Missing/invalid 'action'")

    # Either mean+variance OR coordinate
    if "mean" in d or "variance" in d:
        if "mean" not in d or "variance" not in d:
            raise ValueError("Must contain both 'mean' and 'variance'")
        mean = d["mean"]
        var = d["variance"]
        if not (isinstance(mean, (list, tuple)) and len(mean) == 2):
            raise ValueError("mean must be [x, y]")
        if not (isinstance(var, (list, tuple)) and len(var) == 2):
            raise ValueError("variance must be [vx, vy]")
        mx, my = float(mean[0]), float(mean[1])
        vx, vy = float(var[0]), float(var[1])
        if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(vx) and math.isfinite(vy)):
            raise ValueError("Non-finite values")
        if vx <= 0 or vy <= 0:
            raise ValueError("Variance must be > 0")
        return {"action": d["action"], "mean": (mx, my), "variance": (vx, vy)}

    if "coordinate" in d:
        c = d["coordinate"]
        if not (isinstance(c, (list, tuple)) and len(c) == 4):
            raise ValueError("coordinate must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = map(float, c)
        return {"action": d["action"], "coordinate": (x1, y1, x2, y2)}

    raise ValueError("Must contain either (mean, variance) or coordinate")


def format_reward(completions, **kwargs):
    """
    Reward is 1.0 only if strict format passes; otherwise big negative.
    """
    INVALID_PENALTY = -50.0

    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        try:
            _ = _parse_answer_block(text)
            rewards.append(1.0)
        except Exception:
            rewards.append(INVALID_PENALTY)
    return rewards

def beta_linear(step, T, beta0=1e-2, beta_end=0.0):
    t = min(step / float(T), 1.0)
    return beta0 + (beta_end - beta0) * t

def gaussian_point_reward(completions, solution, scales,m_target = 0.55, lambda_conf = 0.3,alpha_mu = 0.5,tau = None,tau_frac_of_min_side= 0.05,lambda_logp=1.0, beta_entropy=0.001,
                          logp_clip=(-20.0, 10.0), eps=1e-8, **kwargs):
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    rewards = []

    for comp, sol, scale in zip(completions, solution, scales):
        content = comp[0]["content"]

        # gate
        try:
            pred = _parse_answer_block(content)
        except Exception:
            rewards.append(0.0)     # format_reward handles penalty
            continue

        gt = _parse_answer_block(sol)  # solution should be valid

        if pred["action"] != gt["action"]:
            rewards.append(0.0)
            continue

        if pred["action"] != "click" or "mean" not in pred:
            rewards.append(1.0)
            continue

        mx, my = pred["mean"]
        vx, vy = pred["variance"]

        mx *= float(scale[0]); my *= float(scale[1])
        vx *= float(scale[0])**2; vy *= float(scale[1])**2
        vx = max(vx, 1e-6); vy = max(vy, 1e-6)

        x1, y1, x2, y2 = gt["coordinate"]
        # gt_center = [(x1 + x2)/2.0, (y1 + y2)/2.0]

        # m = mass_inside_box_gaussian([mx,my], [vx,vy], gt["coordinate"])
        m, _, _ = box_mass_diag_gaussian(mx, my, vx, vy, x1, y1, x2, y2)
        m = _clamp01(m)  # torch scalar in (0,1)
        r_mass = m
        r_conf = coverage_match_reward(m, m_target=m_target)
        r_mu = mean_inside_soft_reward_scalar(
            mx,my, gt['coordinate'],
            tau=tau,
            tau_frac_of_min_side=tau_frac_of_min_side
        )

        reward = r_mass + lambda_conf * r_conf + alpha_mu * r_mu

        # logp = diag_gaussian_logprob_xy([mx, my], [vx, vy], gt_center, eps=eps)
        # lo, hi = logp_clip
        # logp = max(lo, min(hi, logp))

        # ent = diag_gaussian_entropy_2d([vx, vy], eps=eps)

        # reward =lambda_logp * logp + beta_entropy * ent
        rewards.append(float(reward.item()))
    
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy gaussian reward of Coord: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                # if show_flage:
                f.write(f"variance predictions: {[vx,vy]}\n")
                f.write(f"mean predictions: {mx,my}\n")
                f.write(f"ground_truth_bbox: {gt['coordinate']}\n")

    return rewards

# 三个reward的定义
# action_type对应的reward
# 坐标对应的reward
# 输出格式对应的reward
###  reward registry three parts
reward_funcs_registry = {
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
    script_args.reward_funcs = ['gaussian_point_reward','format']
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
                    "Instead of predicting a single coordinate, predict a 2D probability distribution over the screen.\n"
                    "The distribution should be a Gaussian defined by:\n"
                    "- mean_x (integer)\n"
                    "- mean_y (integer)\n"
                    "- var_x (positive float)\n"
                    "- var_y (positive float)\n\n"
                    "Please output the reasoning in <think> </think> tags and the final result in <answer> </answer> tags.\n"
                    "The output format must be exactly:\n"
                    "<think> ... </think>\n"
                    "<answer>[{'action': 'click', 'mean': [mean_x, mean_y], 'variance': [var_x, var_y]}]</answer>\n\n"
                    "Constraints:\n"
                    "- mean_x and mean_y must be valid screen coordinates\n"
                    "- var_x and var_y must be > 0\n"
                    "- Do not include any additional text outside the specified tags\n"
                    "- Strictly follow the output format"
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