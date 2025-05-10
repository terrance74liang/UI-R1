# UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning

<font size=4><div align='center' > [[📖 Paper](https://arxiv.org/abs/2503.21620)] [[🤗 Checkpoints](https://huggingface.co/LZXzju/Qwen2.5-VL-3B-UI-R1)] [[🤗 Datasets](https://huggingface.co/datasets/LZXzju/UI-R1-3B-Train)] [[🤗 Daily Paper](https://huggingface.co/papers/2503.21620)]</div></font>

## 🔥 Overview

We propose **UI-R1**, the first framework to explore how rule-based RL can enhance the reasoning capabilities of multimodal large language models (MLLMs) for GUI action prediction tasks.
<a href="">
  <img src="assets/method.png" alt="Logo" >
</a>


Experimental results demonstrate that our proposed **UI-R1-3B** achieves significant improvements over the base model (i.e. Qwen2.5-VL-3B) on both in-domain (ID) and out-of-domain (OOD) tasks, with average accuracy gains of **22.1%** on ScreenSpot, **6.0%** on ScreenSpot-Pro, and **12.7%** on AndroidControl. Furthermore, UI-R1-3B delivers competitive performance compared to larger models (e.g., OS-Atlas-7B) trained via supervised fine-tuning (SFT) on 76K samples.

<a href="">
  <img src="assets/radar.png" alt="Logo" >
</a>

## 🔥Insight 1 : Fast Grounding

**Thinking is not needed for GUI grounding.** Inspired by concurrent works studying efficient LRM, we realize efficient reasoning by RFT training. UI-R1-3B-E's training consists of two steps:

1. DAST (Difficulty-Adaptive Slow-Thinking): Add difficulty-adaptive length reward to make reasoning from slow to fast.
2. Nothinking: Not output reasoning process.

Note: UI-R1-3B (v2) and UI-R1-3B-E both train on larger dataset compared to UI-R1-3B (v1).

#### Benchmark 1: ScreenSpotV2

| ScreenSpotV2     | Mobile-T | Mobile-I | Desktop-T | Desktop-I | Web-T    | Web-I    | Avg / Len↓     |
| ---------------- | -------- | -------- | --------- | --------- | -------- | -------- | ------------- |
| OS-ATLAS-7B      | 95.2     | 75.8     | 90.7      | 63.6      | 90.6     | 77.3     | 84.1 /        |
| UI-TARS-7B       | 95.2     | 79.1     | 90.7      | 68.6      | 90.6     | **78.3** | 84.7 /        |
| UI-R1-3B (v1) | 96.2     | **84.3** | 92.3      | 63.6      | 89.2     | 75.4     | 85.4 / 67         |
| GUI-R1-3B        | 97.6     | 78.2     | 94.3  | 64.3      | **91.0** | 72.4     | 85.0 / 80     |
| UI-R1-3B (v2)    | 97.6     | 79.6     | 92.3      | 67.9      | 88.9     | 77.8     | 85.8 / 60         |
| UI-R1-3B-E    | **99.0** | 83.4     | **97.0**  | **75.0**  | **91.0** | 74.9     | **88.1** / **28** |

#### Benchmark 2: ScreenSpot-Pro

| ScreenSpot-Pro   | Average Length↓ | Average Accuracy |
| ---------------- | -------------- | ---------------- |
| UGround-7B       | -              | 16.5             |
| OS-ATLAS-7B      | -              | 18.9             |
| UI-R1-3B (v1)    | 102            | 17.8             |
| GUI-R1-3B        | 114            | 26.6            |
| UI-R1-3B (v2)    | 129            | 29.8           |
| UI-R1-3B-E | **28**            | **33.2**      |

##### Analysis

1. Our UI-R1-3B achieves **SOTA** with **least** answer tokens in 3B/7B Open-source methods, demonstrating GUI grounding needs no reasoning.

##### Todo

- [ ] Performance on 7B may be opposite.
- [ ] Performance on Planning may be opposite. The author predicts that Fast Grounding, Slow Planning.
- [ ] The checkpoints of UI-R1-3B-E will be released soon.
- [ ] The updated paper will come soon.
- [ ] The efficient training code will come soon
## Setup

```shell
conda create -n ui-r1 python=3.10
conda activate ui-r1
bash setup.sh
```

## Data

Our training mobile data is a subset from AndroidControl and ScreenSpot.

You can also prepare your training or inference data like:

```
images/:
	image1.png
	image2.png
```

```
test.json:
[
	{
	"img_filename": "image1.png",
        "bbox": [
            825,
            72,
            1673,
            149
        ],
        "instruction": "search bar"
     },
     {
	"img_filename": "image2.png",
        "bbox": [
            123,
            732,
            334,
            812
        ],
        "instruction": "check weather"
     }
]
```

where bbox : [x1,y1,x2,y2] is the coordinate of the left top and the right bottom of the ground truth bbox

## Inference

We provide an example here

```shell
cd evaluation/
bash test.sh
```

Please fill the MODEL_PATH, IMG_PATH, TEST_JSON with your real checkpoint path and data path.
## Training

```shell
cd src/script/
bash train.sh
```




## 🗞️ News
- **`2025-04-02`**: We release the [datasets](https://huggingface.co/datasets/LZXzju/UI-R1-3B-Train) of the UI-R1-3B model.
- **`2025-03-30`**: We release the [checkpoints](https://huggingface.co/LZXzju/Qwen2.5-VL-3B-UI-R1) of the UI-R1-3B model.
- **`2025-03-30`**: We release the UI-R1 repository.
- **`2025-03-27`**: We release our [paper](https://arxiv.org/abs/2503.21620).





## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bit
@article{lu2025ui,
  title={UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning},
  author={Lu, Zhengxi and Chai, Yuxiang and Guo, Yaxuan and Yin, Xi and Liu, Liang and Wang, Hao and Xiong, Guanjing and Li, Hongsheng},
  journal={arXiv preprint arXiv:2503.21620},
  year={2025}
}
```



## 🤝 Acknowledgements

We sincerely thank projects [R1-V](https://github.com/Deep-Agent/R1-V), [Open-R1](https://github.com/huggingface/open-r1), and [Open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [VLM-R1](https://github.com/om-ai-lab/VLM-R1) for providing their open-source resources.
