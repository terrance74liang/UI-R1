---
license: apache-2.0
---
### GUI-G2-3B

This repository contains the GUI-G2-3B model from the paper [GUI-G²: Gaussian Reward Modeling 
 for GUI Grounding](https://arxiv.org/abs/2507.15846).  We provided more inference details on the github quick start. We will update GUI-G2-3B results on GUI Grounding benchmark.

[![Huggingface Paper](https://img.shields.io/badge/Paper-2507.15846-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2507.15846)
[![Paper](https://img.shields.io/badge/Paper-TBA-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.15846)
[![alphaXiv](https://img.shields.io/badge/alphaXiv-2507.15846-1f8ceb?style=for-the-badge)](https://www.alphaxiv.org/abs/2507.15846)
[![Project](https://img.shields.io/badge/Project-Page-007ec6?style=for-the-badge)](https://zju-real.github.io/GUI-G2)
[![GitHub](https://img.shields.io/badge/Code-GUI--G2-000000?style=for-the-badge&logo=github)](https://github.com/zju-real/GUI-G2)

### Model Description

The model is based on `Qwen2.5-VL-3B-Instruct` and is fine-tuned using our proposed  Gaussian dense reward framework framework. 

- 💡**Gaussian Point & Coverage Rewards**: Encourage accurate, spatially-aligned clicks.

* 📏 **Adaptive Variance Mechanism**: Adjusts reward granularity based on element scale.
* 🌍 **Dense Learning Signals**: Smooth gradients outperform binary RL rewards in early-stage learning.
* 📊 **State-of-the-art Performance** on ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro datasets.

### Quick Start

First, install the required dependencies:

```python
pip install transformers==4.49.0 qwen-vl-utils
```

```
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     "inclusionAI/GUI-G2-3B",
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto")
     
processor = AutoProcessor.from_pretrained("inclusionAI/GUI-G2-3B")
image_path = ''
instruction = ''

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "image_path",
            },
            {"type": "text", "text": instruction},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### 📊 Results on ScreenSpot-v2

| **Model**            | **Mobile Text** | **Mobile Icon** | **Desktop Text** | **Desktop Icon** | **Web Text** | **Web Icon** | **Avg.** |
| -------------------- | --------------- | --------------- | ---------------- | ---------------- | ------------ | ------------ | -------- |
| GPT-4o               | 26.6            | 24.2            | 24.2             | 19.3             | 12.8         | 11.8         | 20.1     |
| Qwen2.5-VL-3B        | 93.4            | 73.5            | 88.1             | 58.6             | 88.0         | 71.4         | 80.9     |
| Qwen2.5-VL-7B        | 97.6            | 87.2            | 90.2             | 74.2             | 93.2         | 81.3         | 88.8     |
| SeeClick-9.6B        | 78.4            | 50.7            | 70.1             | 29.3             | 55.2         | 32.5         | 55.1     |
| UGround-7B           | 75.1            | 84.5            | 85.1             | 61.4             | 84.6         | 71.9         | 76.3     |
| OS-Atlas-7B          | 95.2            | 75.8            | 90.7             | 63.6             | 90.6         | 77.3         | 84.1     |
| UI-TARS-2B           | 95.2            | 79.1            | 90.7             | 68.6             | 87.2         | 78.3         | 84.7     |
| UI-TARS-7B           | 96.9            | 89.1            | 95.4             | 85.0             | 93.6         | 85.2         | 91.6     |
| UI-TARS-72B          | 94.8            | 86.3            | 91.2             | 87.9             | 91.5         | 87.7         | 90.3     |
| JEDI-7B              | 96.9            | 87.2            | 95.9             | 87.9             | 94.4         | 84.2         | 91.7     |
| GUI-Actor-7B         | 97.6            | 88.2            | 96.9             | 85.7             | 93.2         | 86.7         | 92.1     |
| UI-R1-3B             | 96.2            | 84.3            | 92.3             | 63.6             | 89.2         | 75.4         | 85.4     |
| UI-R1-E-3B           | 98.2            | 83.9            | 94.8             | 75.0             | 93.2         | 83.7         | 89.5     |
| SE-GUI-7B            | -               | -               | -                | -                | -            | -            | 90.3     |
| LPO                  | 97.9            | 82.9            | 95.9             | 86.4             | 95.6         | 84.2         | 90.5     |
| **GUI-G²-7B (Ours)** | **98.3**        | **91.9**        | **95.4**         | **89.3**         | **94.0**     | **87.7**     | **93.3** |

---

### 🙏 Acknowledgement

The RL Training code build from [VLM-R1 project](https://github.com/om-ai-lab/VLM-R1).

### 📄 Citation

If you use GUI-G², please cite our work:

```bibtex
@misc{tang2025guig2gaussianrewardmodeling,
      title={GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding}, 
      author={Fei Tang and Zhangxuan Gu and Zhengxi Lu and Xuyang Liu and Shuheng Shen and Changhua Meng and Wen Wang and Wenqi Zhang and Yongliang Shen and Weiming Lu and Jun Xiao and Yueting Zhuang},
      year={2025},
      eprint={2507.15846},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.15846}, 
}
```