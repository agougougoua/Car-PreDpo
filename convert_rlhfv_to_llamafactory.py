import json
import os
import argparse
from pathlib import Path

# --- 路径配置 ---
parser = argparse.ArgumentParser(description="Convert RLHF-V JSONL into multimodal preference JSON.")
parser.add_argument("--data-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
DATA_ROOT = str(Path(args.data_root).resolve())
JSONL_PATH = os.path.join(DATA_ROOT, "RLHF-V-Dataset.jsonl")
OUTPUT_PATH = str(Path(args.output).resolve())

formatted_data = []

print("正在转换 RLHF-V-Dataset 为 LLaMA-Factory VLM-DPO 格式...")
print("Input:", JSONL_PATH)
print("Output:", OUTPUT_PATH)

skipped_parse_text = 0
skipped_missing_field = 0
skipped_no_image = 0
skipped_same_response = 0
skipped_empty = 0

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            item = json.loads(line)
        except Exception as e:
            if i < 10:
                print(f"第 {i} 行 JSONL 解析失败: {repr(e)}")
            continue

        # 1. 解析 text 字段，里面是一个 JSON 字符串
        text_raw = item.get("text", "")
        if not isinstance(text_raw, str) or not text_raw.strip():
            skipped_empty += 1
            continue

        try:
            text_obj = json.loads(text_raw)
        except Exception as e:
            skipped_parse_text += 1
            if skipped_parse_text <= 5:
                print(f"第 {i} 行 text 解析失败: {repr(e)}")
                print("text preview:", str(text_raw)[:300])
            continue

        # 2. 取 question/chosen/rejected
        question = str(text_obj.get("question", "")).strip()
        chosen = str(text_obj.get("chosen", "")).strip()
        rejected = str(text_obj.get("rejected", "")).strip()

        if not question or not chosen or not rejected:
            skipped_missing_field += 1
            continue

        if chosen == rejected:
            skipped_same_response += 1
            continue

        # 3. 解析图片路径
        image_rel = item.get("image", "")
        if not isinstance(image_rel, str) or not image_rel.strip():
            skipped_no_image += 1
            continue

        image_path = image_rel
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(os.path.join(DATA_ROOT, image_path))

        if not os.path.exists(image_path):
            skipped_no_image += 1
            if skipped_no_image <= 5:
                print(f"第 {i} 行图片不存在: {image_path}")
            continue

        # 4. 生成 LLaMA-Factory DPO 样本
        formatted_data.append({
            "prompt": f"<image>\n{question}",
            "chosen": chosen,
            "rejected": rejected,
            "images": [image_path]
        })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print("\n✅ 转换完成")
print(f"成功处理: {len(formatted_data)} 条")
print(f"跳过 - text 为空: {skipped_empty}")
print(f"跳过 - text JSON 解析失败: {skipped_parse_text}")
print(f"跳过 - 缺 question/chosen/rejected: {skipped_missing_field}")
print(f"跳过 - 图片不存在: {skipped_no_image}")
print(f"跳过 - chosen/rejected 相同: {skipped_same_response}")
print(f"输出文件: {OUTPUT_PATH}")
