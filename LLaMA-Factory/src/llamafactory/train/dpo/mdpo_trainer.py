from __future__ import annotations
import copy
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Any
import torch
import torch.nn.functional as F
from ..trainer_utils import get_batch_logps, nested_detach
from .trainer import CustomDPOTrainer

def _base_qwen_model(model: Any) -> Any:
    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(candidate, "model") and hasattr(candidate.model, "get_image_features"): return candidate
    if hasattr(candidate, "base_model") and hasattr(candidate.base_model, "model"): candidate = candidate.base_model
    if hasattr(candidate, "model") and hasattr(candidate.model, "get_image_features"): return candidate
    raise TypeError("mDPO visual masking requires a Qwen2.5-VL model")

def _zero_pooler_output(output: Any) -> Any:
    if hasattr(output, "pooler_output"):
        features = output.pooler_output
        if torch.is_tensor(features):
            masked_features = torch.zeros_like(features)
        elif isinstance(features, tuple):
            masked_features = tuple(torch.zeros_like(feature) for feature in features)
        elif isinstance(features, list):
            masked_features = [torch.zeros_like(feature) for feature in features]
        else:
            raise TypeError("unexpected Qwen pooler_output type")
        masked = copy.copy(output); masked.pooler_output = masked_features; return masked
    if isinstance(output, tuple) and output: return (torch.zeros_like(output[0]), *output[1:])
    raise TypeError("unexpected Qwen visual feature output")

@contextmanager
def mask_qwen_visual_features(model: Any):
    qwen = _base_qwen_model(model); core = qwen.model; original = core.get_image_features
    def masked_get_image_features(self, *args, **kwargs): return _zero_pooler_output(original(*args, **kwargs))
    core.get_image_features = MethodType(masked_get_image_features, core)
    try: yield model
    finally: core.get_image_features = original

def _patch_count(grid: torch.Tensor) -> int:
    if grid.ndim != 2 or grid.shape[-1] != 3: raise ValueError("image_grid_thw must have shape [N, 3]")
    return int(torch.prod(grid, dim=-1).sum().item())

def select_chosen_multimodal_batch(batch: dict[str, Any]) -> dict[str, Any]:
    if any(key in batch for key in ("video_grid_thw", "video_values", "pixel_values_videos", "audio_values")): raise ValueError("Qwen mDPO currently supports image batches only; video is unsupported")
    total = batch["input_ids"].shape[0]
    if total % 2: raise ValueError("pairwise batch must contain chosen/rejected halves")
    chosen_size = total // 2; selected = {}
    if "image_grid_thw" in batch and "pixel_values" in batch:
        chosen_grids = batch["image_grid_thw"][:chosen_size]
        selected["image_grid_thw"] = chosen_grids
        selected["pixel_values"] = batch["pixel_values"][:_patch_count(chosen_grids)]
    for key, value in batch.items():
        if key in {"pixel_values", "image_grid_thw", "labels", "lambda_i", "delta_h", "vdi", "video_grid_thw", "video_values"}: continue
        if torch.is_tensor(value) and value.shape and value.shape[0] == total: selected[key] = value[:chosen_size]
        else: selected[key] = value
    return selected

def duplicate_multimodal_batch(batch: dict[str, Any]) -> dict[str, Any]:
    batch_size = batch["input_ids"].shape[0]
    duplicated = {}
    for key, value in batch.items():
        if key == "pixel_values":
            duplicated[key] = torch.cat((value, value), dim=0)
        elif torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            duplicated[key] = torch.cat((value, value), dim=0)
        else:
            duplicated[key] = value
    return duplicated

class MDPOTrainer(CustomDPOTrainer):
    def compute_mdpo_loss(self, policy_chosen, policy_rejected, policy_imageless_chosen, reference_chosen, reference_rejected, reference_imageless_chosen):
        response_logits = (policy_chosen-policy_rejected) - (reference_chosen-reference_rejected)
        image_logits = (policy_chosen-policy_imageless_chosen) - (reference_chosen-reference_imageless_chosen)
        anchor_logits = policy_chosen-reference_chosen
        response_loss = -F.logsigmoid(self.beta*response_logits); image_loss = -F.logsigmoid(self.beta*image_logits); anchor_loss = -F.logsigmoid(self.beta*anchor_logits)
        return response_loss+image_loss+anchor_loss, response_loss, image_loss, anchor_loss
    def _forward_logps(self, model, batch, mask_visual=False):
        batch = nested_detach(batch, clone=True)
        labels = batch.pop("labels")
        batch.pop("lambda_i", None); batch.pop("delta_h", None); batch.pop("vdi", None)
        with (mask_qwen_visual_features(model) if mask_visual else nullcontext()): outputs=model(**batch, return_dict=True, use_cache=False)
        return get_batch_logps(logits=outputs.logits.float(), labels=labels, ld_alpha=None)[0]
    def _pair_logps(self, model, batch):
        output = super().concatenated_forward(model, batch, is_ref_model=True)
        return output["chosen_logps"], output["rejected_logps"]
    def _masked_chosen_logps(self, model, batch):
        selected = select_chosen_multimodal_batch(batch)
        selected["labels"] = batch["labels"][:batch["input_ids"].shape[0] // 2]
        paired = duplicate_multimodal_batch(selected)
        with mask_qwen_visual_features(model):
            output = super().concatenated_forward(model, paired, is_ref_model=True)
        return output["chosen_logps"]
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        pc, pr = self._pair_logps(model,batch); pic=self._masked_chosen_logps(model,batch)
        with torch.no_grad():
            if self.ref_model is None: ctx=model.disable_adapter() if hasattr(model,"disable_adapter") else nullcontext(); ref=model
            else: ctx=nullcontext(); ref=self.ref_model
            with ctx: rc, rr=self._pair_logps(ref,batch); ric=self._masked_chosen_logps(ref,batch)
        losses, response_loss, image_loss, anchor_loss = self.compute_mdpo_loss(pc, pr, pic, rc, rr, ric)
        prefix = "eval_" if train_eval == "eval" else ""
        chosen_rewards = self.beta * (pc - rc).detach()
        rejected_rewards = self.beta * (pr - rr).detach()
        imageless_chosen_rewards = self.beta * (pic - ric).detach()
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().item(),
            f"{prefix}rewards/imageless_chosen": imageless_chosen_rewards.mean().item(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
            f"{prefix}rewards/imageless_accuracies": (chosen_rewards > imageless_chosen_rewards).float().mean().item(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            f"{prefix}rewards/imageless_margins": (chosen_rewards - imageless_chosen_rewards).mean().item(),
            f"{prefix}logps/chosen": pc.mean().item(),
            f"{prefix}logps/rejected": pr.mean().item(),
            f"{prefix}logps/imageless_chosen": pic.mean().item(),
            f"{prefix}rewards/response": (self.beta * (pc - pr)).mean().item(),
            f"{prefix}rewards/image": (self.beta * (pc - pic)).mean().item(),
            f"{prefix}loss/response": response_loss.mean().item(),
            f"{prefix}loss/image": image_loss.mean().item(),
            f"{prefix}loss/anchor": anchor_loss.mean().item(),
        }
        return losses.mean(), metrics
