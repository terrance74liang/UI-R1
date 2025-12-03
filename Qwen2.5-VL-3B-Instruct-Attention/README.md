---
license: mit
language:
- en
base_model:
- Qwen/Qwen2.5-VL-3B-Instruct
pipeline_tag: visual-question-answering
---

## Introduction
This repository contains the efficient GUI grounding model, **UI-R1-E-3B**, presented in [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://huggingface.co/papers/2503.21620).

Project page: https://github.com/lll6gg/UI-R1

Old version: [UI-R1-3B](https://huggingface.co/LZXzju/Qwen2.5-VL-3B-UI-R1)

## Benchmark 1: ScreenSpotV2

| ScreenSpotV2  | inference mode | Mobile-T | Mobile-I | Desktop-T | Desktop-I | Web-T    | Web-I    | Avg↑ / Len↓        |
| ------------- | -------------- | -------- | -------- | --------- | --------- | -------- | -------- | ----------------- |
| OS-ATLAS-7B   | w/o thinking   | 95.2     | 75.8     | 90.7      | 63.6      | 90.6     | 77.3     | 84.1 /            |
| UI-TARS-7B    | w/o thinking   | 95.2     | 79.1     | 90.7      | 68.6      | 90.6     | 78.3     | 84.7 /            |
| UI-R1-3B (v1) | w/ thinking    | 96.2     | **84.3** | 92.3      | 63.6      | 89.2     | 75.4     | 85.4 / 67         |
| GUI-R1-3B     | w/ thinking    | 97.6     | 78.2     | 94.3      | 64.3      | 91.0     | 72.4     | 85.0 / 80         |
| UI-R1-3B (v2) | w/ thinking    | 97.6     | 79.6     | 92.3      | 67.9      | 88.9     | 77.8     | 85.8 / 60         |
| **UI-R1-E-3B**    | w/o thinking   | **98.2** | 83.9     | **94.8**  | **75.0**  | **93.2** | **83.7** | **89.5** / **28** |
## Benchmark 2: ScreenSpot-Pro

| ScreenSpot-Pro | inference mode | Average Length↓ | Average Accuracy↑ |
| -------------- | -------------- | --------------- | ---------------- |
| UGround-7B     | w/o thinking   | -               | 16.5             |
| OS-ATLAS-7B    | w/o thinking   | -               | 18.9             |
| UI-R1-3B (v1)  | w/ thinking    | 102             | 17.8             |
| GUI-R1-3B      | w/ thinking    | 114             | 26.6             |
| UI-R1-3B (v2)  | w/ thinking    | 129             | 29.8             |
| **UI-R1-E-3B**     | w/o thinking   | **28**          | **33.5**         |
## Leaderboard: UI-I2E-Bench
|     Model      | ScreenSpot | UI-I2E-Bench Avg | ScreenSpot-Pro | Avg  |
| :------------: | :--------: | :--------------: | :------------: | :--: |
| UI-TARS-1.5-7B |    88.1    |       73.2       |      42.2      | 67.8 |
| Uground-V1-72B |    89.7    |       76.3       |      34.3      | 66.8 |
|  UI-TARS-72B   |    88.4    |       73.7       |      38.1      | 66.7 |
|   **UI-R1-E-3B**   |    89.2    |       69.1       |      33.5      | 63.9 |
| Uground-V1-7B  |    87.1    |       70.3       |      31.1      | 62.8 |
|   InfiGUI-R1   |    87.5    |       69.7       |      29.6      | 62.3 |
|   UI-TARS-7B   |    89.5    |       61.4       |      35.7      | 62.2 |
| Qwen2.5-VL-72B |    87.1    |       51.4       |      43.6      | 60.7 |
| UI-I2E-VLM-7B  |    82.5    |       69.5       |      23.6      | 58.5 |
|   UI-TARS-2B   |    82.3    |        62        |      27.7      | 57.3 |
| Qwen2.5-VL-7B  |    84.7    |       53.8       |       29       | 55.8 |
| OmniParser-V2  |     72     |       54.8       |      39.6      | 55.5 |
| Uground-V1-2B  |    78.8    |       57.4       |      26.6      | 54.3 |
|  OS-Atlas-7B   |    82.5    |       58.6       |      18.9      | 53.3 |
|     **UI-R1-3B**      |    83.3    |       58.5       |      17.8      | 53.2 |
|   UGround-7B   |    74.1    |       54.2       |      16.5      | 48.3 |
| UI-I2E-VLM-4B  |    70.4    |       53.4       |      12.2      | 45.3 |
|   OmniParser   |    73.9    |       53.1       |      8.3       | 45.1 |
|   ShowUI-2B    |    76.8    |       41.5       |      7.7       |  42  |
| Qwen2.5-VL-3B  |    55.5    |       41.7       |      23.9      | 41.3 |
|   Aguvis-7B    |    84.4    |       53.2       |      22.9      | 40.4 |
|  OS-Atlas-4B   |    70.1    |       44.3       |      3.7       | 39.4 |
|  Qwen2-VL-7B   |    42.6    |       48.7       |      1.6       |  31  |
|    Seeclick    |    55.8    |       26.4       |      1.1       | 27.8 |
|  InternVL2-4B  |    4.2     |       0.9        |      0.3       | 1.8  |

## Evaluation Code for GUI Grounding

1. Generation for UI-R1-E-3B：

   ```python
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
       args.model_path,
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
       device_map="cpu",
   )
   model = model.to(torch.device(rank))
   model = model.eval()
   processor = AutoProcessor.from_pretrained(ori_processor_path)
   question_template = (
       f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
       "Please provide the action to perform (enumerate in ['click'])"
       "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
       "Output the final answer in <answer> </answer> tags directly."
       "The output answer format should be as follows:\n"
       "<answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
       "Please strictly follow the format."
   )
   query = '<image>\n' + question_template
   messages = [
       {
           "role": "user",
           "content": [
               {"type": "image", "image": image_path}
           ] + [{"type": "text", "text": query}],
       }
   ]
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
   generated_ids = model.generate(**inputs, max_new_tokens=1024)
   generated_ids_trimmed = [
       out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
   ]
   response = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
   )
   response = response[0]
   pred_coord, _ = extract_coord(response)
   ```

   

2. Rescale the predicted coordinate according to the image resize

   ```python
   image = Image.open(image_path)
   origin_width, origin_height = image.size
   resized_height,resized_width = smart_resize(origin_height,origin_width,max_pixels=12845056)
   scale_x = origin_width / resized_width
   scale_y = origin_height / resized_height
   pred_coord[0] = int(pred_coord[0] * scale_x)
   pred_coord[1] = int(pred_coord[1] * scale_y)
   ```

   Function smart_resize is from Qwen2VL：

   ```python
   import math
   def smart_resize(
       height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
   ):
       """Rescales the image so that the following conditions are met:
   
       1. Both dimensions (height and width) are divisible by 'factor'.
   
       2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
   
       3. The aspect ratio of the image is maintained as closely as possible.
   
       """
       if height < factor or width < factor:
           raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
       elif max(height, width) / min(height, width) > 200:
           raise ValueError(
               f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
           )
       h_bar = round(height / factor) * factor
       w_bar = round(width / factor) * factor
       if h_bar * w_bar > max_pixels:
           beta = math.sqrt((height * width) / max_pixels)
           h_bar = math.floor(height / beta / factor) * factor
           w_bar = math.floor(width / beta / factor) * factor
       elif h_bar * w_bar < min_pixels:
           beta = math.sqrt(min_pixels / (height * width))
           h_bar = math.ceil(height * beta / factor) * factor
           w_bar = math.ceil(width * beta / factor) * factor
       return h_bar, w_bar
   ```

   