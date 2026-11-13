import os
import gc
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_question(prompt):
    return prompt.replace("<image>", "").strip()


def build_messages(question, image_path, answer=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    return messages


def safe_float(x):
    if x is None:
        return None
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return None
    except Exception:
        return None


def percentile_ranks(values):
    """
    返回 values 的百分位排名，范围 0~1。
    越大的 raw value，对应越大的 percentile。
    简单稳定版本：rank / (n - 1)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n == 0:
        return np.asarray([], dtype=np.float64)

    if n == 1:
        return np.asarray([0.5], dtype=np.float64)

    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    return ranks / float(n - 1)


@torch.no_grad()
def compute_answer_nll(
    model,
    processor,
    question,
    image_path,
    answer,
    device,
    debug=False
):
    """
    计算 answer 部分平均 NLL。

    关键逻辑：
    - prompt-only 用 add_generation_prompt=True；
    - full prompt + answer 用 add_generation_prompt=False；
    - 用 prompt_len mask 掉 prompt token，只计算 answer token loss。
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    if answer is None or str(answer).strip() == "":
        return float("nan"), 0

    prompt_messages = build_messages(question, image_path, answer=None)
    full_messages = build_messages(question, image_path, answer=answer)

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    image = Image.open(image_path).convert("RGB")

    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt"
    )

    full_inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt"
    )

    prompt_input_ids = prompt_inputs["input_ids"][0]
    full_input_ids = full_inputs["input_ids"][0]

    prompt_len = int(prompt_input_ids.shape[0])
    full_len = int(full_input_ids.shape[0])

    if prompt_len >= full_len:
        raise ValueError(f"prompt_len >= full_len: {prompt_len} >= {full_len}")

    if debug:
        print("\n========== DEBUG NLL MASK ==========")
        print("[image_path]", image_path)
        print("[question]", question[:300])
        print("[answer]", str(answer)[:300])
        print("[prompt_text tail]", repr(prompt_text[-500:]))
        print("[full_text tail]", repr(full_text[-800:]))
        print("prompt_len:", prompt_len)
        print("full_len:", full_len)
        print("answer_len:", full_len - prompt_len)

        decoded_answer_part = processor.tokenizer.decode(
            full_input_ids[prompt_len:],
            skip_special_tokens=False
        )
        print("[decoded answer part]", repr(decoded_answer_part[:800]))
        print("====================================\n")

    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    valid_tokens = int((labels != -100).sum().item())
    if valid_tokens <= 0:
        return float("nan"), 0

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
        model_inputs["pixel_values"] = full_inputs["pixel_values"].to(device)

    if "image_grid_thw" in full_inputs and full_inputs["image_grid_thw"] is not None:
        model_inputs["image_grid_thw"] = full_inputs["image_grid_thw"].to(device)

    outputs = model(**model_inputs)

    nll = float(outputs.loss.item())

    del prompt_inputs, full_inputs, input_ids, attention_mask, labels, model_inputs, outputs
    return nll, valid_tokens


def load_model_and_processor(
    base_model_path,
    adapter_path=None,
    device="cuda",
    max_pixels=1024 * 28 * 28,
    min_pixels=256 * 28 * 28,
    use_device_map_auto=False,
):
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if use_device_map_auto:
        device_map = "auto"
    else:
        if device == "cuda":
            device_map = {"": 0}
        else:
            device_map = {"": "cpu"}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    if adapter_path is not None and adapter_path.strip() != "":
        assert os.path.exists(adapter_path), f"adapter_path not found: {adapter_path}"
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from: {adapter_path}")

    model.eval()
    return model, processor


def get_degraded_images(sample):
    """
    兼容两种字段：
    1. 新版：sample["degraded_images"] = [lowpass, downsample, diffnoise]
    2. 旧版：sample["perturbed_images"] = [perturbed]
    """
    if "degraded_images" in sample and sample["degraded_images"]:
        return sample["degraded_images"]

    if "perturbed_images" in sample and sample["perturbed_images"]:
        return sample["perturbed_images"]

    return []


def compute_one_sample_vdi(
    model,
    processor,
    sample,
    device,
    debug=False,
):
    question = extract_question(sample["prompt"])

    orig_image = sample["images"][0]
    degraded_images = get_degraded_images(sample)

    if len(degraded_images) == 0:
        raise ValueError("No degraded_images or perturbed_images found in sample.")

    chosen = sample["chosen"]
    rejected = sample["rejected"]

    nll_orig_chosen, len_orig_chosen = compute_answer_nll(
        model=model,
        processor=processor,
        question=question,
        image_path=orig_image,
        answer=chosen,
        device=device,
        debug=debug
    )

    nll_orig_rejected, len_orig_rejected = compute_answer_nll(
        model=model,
        processor=processor,
        question=question,
        image_path=orig_image,
        answer=rejected,
        device=device,
        debug=False
    )

    per_degrade_results = []

    for degrade_idx, deg_image in enumerate(degraded_images):
        nll_pert_chosen, len_pert_chosen = compute_answer_nll(
            model=model,
            processor=processor,
            question=question,
            image_path=deg_image,
            answer=chosen,
            device=device,
            debug=False
        )

        nll_pert_rejected, len_pert_rejected = compute_answer_nll(
            model=model,
            processor=processor,
            question=question,
            image_path=deg_image,
            answer=rejected,
            device=device,
            debug=False
        )

        d_chosen = nll_pert_chosen - nll_orig_chosen
        d_rejected = nll_pert_rejected - nll_orig_rejected

        # 推荐方向：
        # chosen 比 rejected 更依赖视觉时，vdi_raw 更大。
        vdi_raw = d_chosen - d_rejected

        per_degrade_results.append(
            {
                "degrade_idx": degrade_idx,
                "degrade_image": deg_image,
                "nll_pert_chosen": float(nll_pert_chosen),
                "nll_pert_rejected": float(nll_pert_rejected),
                "d_chosen": float(d_chosen),
                "d_rejected": float(d_rejected),
                "vdi_raw": float(vdi_raw),
                "len_pert_chosen": int(len_pert_chosen),
                "len_pert_rejected": int(len_pert_rejected),
            }
        )

    valid_vdi_raws = [
        r["vdi_raw"]
        for r in per_degrade_results
        if safe_float(r.get("vdi_raw")) is not None
    ]

    if len(valid_vdi_raws) == 0:
        vdi_raw_mean = float("nan")
        vdi_raw_std = float("nan")
    else:
        vdi_raw_mean = float(np.mean(valid_vdi_raws))
        vdi_raw_std = float(np.std(valid_vdi_raws))

    result = {
        "nll_orig_chosen": float(nll_orig_chosen),
        "nll_orig_rejected": float(nll_orig_rejected),
        "len_orig_chosen": int(len_orig_chosen),
        "len_orig_rejected": int(len_orig_rejected),

        "vdi_raw": float(vdi_raw_mean),
        "vdi_raw_std_across_degrades": float(vdi_raw_std),

        "vdi_per_degrade": per_degrade_results,
    }

    return result


def print_distribution(name, vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        print(f"{name}: no valid values")
        return

    print(f"\n{name} distribution:")
    print(f"  count : {len(vals)}")
    print(f"  min   : {np.min(vals):.6f}")
    print(f"  p05   : {np.percentile(vals, 5):.6f}")
    print(f"  p10   : {np.percentile(vals, 10):.6f}")
    print(f"  p20   : {np.percentile(vals, 20):.6f}")
    print(f"  p25   : {np.percentile(vals, 25):.6f}")
    print(f"  median: {np.percentile(vals, 50):.6f}")
    print(f"  p75   : {np.percentile(vals, 75):.6f}")
    print(f"  p80   : {np.percentile(vals, 80):.6f}")
    print(f"  p90   : {np.percentile(vals, 90):.6f}")
    print(f"  p95   : {np.percentile(vals, 95):.6f}")
    print(f"  max   : {np.max(vals):.6f}")
    print(f"  mean  : {np.mean(vals):.6f}")
    print(f"  std   : {np.std(vals):.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)

    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--lambda_min", type=float, default=0.05)
    parser.add_argument("--lambda_max", type=float, default=0.30)

    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)

    parser.add_argument("--use_device_map_auto", action="store_true")
    parser.add_argument("--debug_first", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model_and_processor(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=device,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        use_device_map_auto=args.use_device_map_auto,
    )

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples > 0:
        data = data[:args.max_samples]

    for idx, sample in enumerate(tqdm(data, desc="Computing NLL and VDI")):
        try:
            debug = bool(args.debug_first and idx == 0)

            result = compute_one_sample_vdi(
                model=model,
                processor=processor,
                sample=sample,
                device=device,
                debug=debug,
            )

            sample.update(result)

        except Exception as e:
            print(f"[Error] idx={idx}, error={repr(e)}")

            sample["nll_orig_chosen"] = None
            sample["nll_orig_rejected"] = None
            sample["len_orig_chosen"] = None
            sample["len_orig_rejected"] = None
            sample["vdi_raw"] = None
            sample["vdi_raw_std_across_degrades"] = None
            sample["vdi_per_degrade"] = []

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (idx + 1) % args.save_every == 0:
            tmp_path = args.output_json + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Temporary saved to: {tmp_path}")

    valid_indices = []
    valid_raw = []

    for i, sample in enumerate(data):
        v = safe_float(sample.get("vdi_raw", None))
        if v is not None:
            valid_indices.append(i)
            valid_raw.append(v)

    if len(valid_raw) == 0:
        raise RuntimeError("No valid vdi_raw scores computed.")

    valid_raw = np.asarray(valid_raw, dtype=np.float64)

    mean_raw = float(np.mean(valid_raw))
    std_raw = float(np.std(valid_raw))

    print(f"\nVDI raw mean: {mean_raw}")
    print(f"VDI raw std : {std_raw}")

    print_distribution("VDI raw", valid_raw)

    # sigmoid 版本：用于观察，不作为主要排序依据
    eps = 1e-6
    valid_z = (valid_raw - mean_raw) / (std_raw + eps)
    valid_vdi_sigmoid = sigmoid(valid_z)

    # 百分位版本：建议用于训练权重
    valid_s_perc = percentile_ranks(valid_raw)

    # 写回样本
    for sample in data:
        if safe_float(sample.get("vdi_raw", None)) is None:
            sample["vdi"] = 0.5
            sample["s_proc"] = 0.5
            sample["s_perc"] = 0.5
            sample["lambda_i"] = float((args.lambda_min + args.lambda_max) / 2.0)
            continue

    for local_j, sample_idx in enumerate(valid_indices):
        sample = data[sample_idx]

        vdi_sigmoid = float(valid_vdi_sigmoid[local_j])
        s_perc = float(valid_s_perc[local_j])

        # vdi: sigmoid 标准化分数，便于观察
        sample["vdi"] = vdi_sigmoid

        # s_proc: 原始过程分数，这里保留 raw
        sample["s_proc"] = float(sample["vdi_raw"])

        # s_perc: 百分位，推荐用于 DPO 动态权重
        sample["s_perc"] = s_perc

        # 推荐权重：
        # s_perc 越高，说明 chosen 相对 rejected 越依赖视觉，lambda 越大
        lambda_i = args.lambda_min + (args.lambda_max - args.lambda_min) * s_perc
        sample["lambda_i"] = float(lambda_i)

    lambda_vals = [
        x["lambda_i"]
        for x in data
        if safe_float(x.get("lambda_i", None)) is not None
    ]

    s_perc_vals = [
        x["s_perc"]
        for x in data
        if safe_float(x.get("s_perc", None)) is not None
    ]

    print_distribution("s_perc", s_perc_vals)
    print_distribution("lambda_i", lambda_vals)

    meta = {
        "vdi_direction": "recommended_visual_preference_strength",
        "vdi_raw_definition": "mean_over_degraded_images((nll_pert_chosen - nll_orig_chosen) - (nll_pert_rejected - nll_orig_rejected))",
        "d_chosen": "nll_pert_chosen - nll_orig_chosen",
        "d_rejected": "nll_pert_rejected - nll_orig_rejected",
        "interpretation": "larger vdi_raw means chosen is more visually dependent than rejected",
        "vdi_sigmoid_definition": "sigmoid((vdi_raw - mean_raw) / std_raw)",
        "s_proc": "raw vdi_raw",
        "s_perc": "percentile rank of vdi_raw among valid samples, range [0, 1]",
        "lambda_i_definition": "lambda_min + (lambda_max - lambda_min) * s_perc",
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "lambda_mean_expected_if_uniform_percentile": float((args.lambda_min + args.lambda_max) / 2.0),
        "vdi_mean_raw": mean_raw,
        "vdi_std_raw": std_raw,
        "valid_count": int(len(valid_raw)),
        "total_count": int(len(data)),
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    meta_path = args.output_json + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved scored data to: {args.output_json}")
    print(f"Saved meta to: {meta_path}")

    print("\nRecommended fixed-lambda baseline:")
    print(f"  lambda_fixed = {(args.lambda_min + args.lambda_max) / 2.0:.6f}")


if __name__ == "__main__":
    main()
