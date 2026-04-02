import torch
import time
from transformers import AutoProcessor, DynamicCache, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
# from myqwen import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import copy

if torch.jit.is_tracing():
    print("Tracing mode")
else:
    print("Not tracing mode")

model_id = "./models/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto',attn_implementation="sdpa")
# model_id = "./LLaMA-Factory/saves/qwen3vl_4b"
# model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto',attn_implementation="sdpa")

processor = AutoProcessor.from_pretrained(model_id)

generate_kwargs = {
    "do_sample": False,
    "max_new_tokens":50,
    "use_cache": True,
    # "eos_token_id": processor.tokenizer.eos_token_id,
    # "tokenizer": processor.tokenizer
}

base_path = "./LLaMA-Factory/data/mllm_demo_data"

image_paths = [
    f'{base_path}/3.mp4',
    f'{base_path}/4.mp4',
    # f'{base_path}/qeefjE74SXI.mp4',
]

questions_per_image = [
    "Please describe the main content of the latest video.",
    # "How old are you?",
]

print("有cache")

# 初始化 DynamicCache（所有视频/图片共享同一个cache）
past_key_values = DynamicCache()
# 设置cache模式（也就是修改后定义的支持多模态多轮对话的cache）
model.set_cache_mode()

# 初始化消息列表（保持所有对话历史）
messages = []

print("=" * 80)
print("开始多图片连续对话测试（使用 DynamicCache）")
print("=" * 80)
print(f"\n将处理 {len(image_paths)} 张图片，每张图片 {len(questions_per_image)} 个问题\n")
# 遍历每张图片
st = time.time()

round_times = []

for image_idx, image_path in enumerate(image_paths):
    print(f"\n{'='*80}")
    print(f"处理图片 {image_idx + 1}/{len(image_paths)}: {image_path}")
    print(f"{'='*80}\n")

    # 每张图片的第一个问题需要包含图片
    first_question = questions_per_image[0]
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": image_path,"total_pixels": 6000*28*28,"min_pixels": 20*28*28},
            {"type": "text", "text": first_question}
        ]
    })
    # 处理第一个问题（包含图片）
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info([messages])
    # 处理图片，获取 inputs（与 thyme 一致：padding=True，并移到模型设备）
    inputs = processor(text=text, videos=videos, padding=True, return_tensors='pt')
    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[1]
    print(f"Image {image_idx + 1} - Question 1 - input_length:", input_length)


    round_st = time.time()
    # model.set_count()
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
        past_key_values=past_key_values,
    )
    round_et = time.time()
    round_time = round_et - round_st
    round_times.append(round_time)
    print(f"本轮用时: {round_time:.2f} 秒")

    completion = processor.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})

    print(f"Image {image_idx + 1} - Question 1:")
    print(f"User: {first_question}")
    print(f"Assistant: {completion}\n")
    print("-" * 80)



print("\n" + "=" * 80)
print("多图片连续对话测试完成！")
print(f"用时:{time.time()-st}")
print("=" * 80)
print(f"\n总共处理了 {len(image_paths)} 张图片，{len(image_paths) * len(questions_per_image)} 个问题")
print(f"DynamicCache 已成功缓存了所有对话的键值对")
print(f"Cache 层数: {len(past_key_values) if hasattr(past_key_values, '__len__') else 'N/A'}")
print("\n每轮时间统计:")
for i, t in enumerate(round_times):
    print(f"第 {i+1} 轮: {t:.2f} 秒")


# exit(0)

print("无cache")

# 初始化 DynamicCache（所有视频/图片共享同一个cache）
past_key_values = None
# model.unset_cache_mode()

# 初始化消息列表（保持所有对话历史）
messages = []

print("=" * 80)
print("开始多图片连续对话测试（无 DynamicCache）")
print("=" * 80)
print(f"\n将处理 {len(image_paths)} 张图片，每张图片 {len(questions_per_image)} 个问题\n")
# 遍历每张图片
st = time.time()

round_times = []

for image_idx, image_path in enumerate(image_paths):
    print(f"\n{'='*80}")
    print(f"处理图片 {image_idx + 1}/{len(image_paths)}: {image_path}")
    print(f"{'='*80}\n")

    # 每张图片的第一个问题需要包含图片
    first_question = questions_per_image[0]
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": image_path,"total_pixels": 6000*28*28,"min_pixels": 20*28*28},
            {"type": "text", "text": first_question}
        ]
    })
    # 处理第一个问题（包含图片）
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    # images, videos = process_vision_info([messages])
    images, videos, video_kwargs = process_vision_info(
    [messages],
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
    )
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
    # 处理图片，获取 inputs（与 thyme 一致：padding=True，并移到模型设备）
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors='pt',
        **(video_kwargs or {}),
    )
    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[1]
    print(f"Image {image_idx + 1} - Question 1 - input_length:", input_length)


    round_st = time.time()
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
        past_key_values=None,
    )
    round_et = time.time()
    round_time = round_et - round_st
    round_times.append(round_time)
    print(f"本轮用时: {round_time:.2f} 秒")

    completion = processor.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})

    print(f"Image {image_idx + 1} - Question 1:")
    print(f"User: {text}")
    print(f"Assistant: {completion}\n")
    print("-" * 80)



print("\n" + "=" * 80)
print("多图片连续对话测试完成！")
print(f"用时:{time.time()-st}")
print("=" * 80)
print(f"\n总共处理了 {len(image_paths)} 张图片，{len(image_paths) * len(questions_per_image)} 个问题")
print(f"DynamicCache 已成功缓存了所有对话的键值对")
print(f"Cache 层数: {len(past_key_values) if hasattr(past_key_values, '__len__') else 'N/A'}")
print("\n每轮时间统计:")
for i, t in enumerate(round_times):
    print(f"第 {i+1} 轮: {t:.2f} 秒")