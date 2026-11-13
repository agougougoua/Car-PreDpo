"""Independent implementation of UE-DPO, arXiv:2605.04874v1 equations6–12.

No author training code was available at implementation time. See the recorded
protocol for normalization, diffusion schedule and Qwen tensor-space choices.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F


def _answer_mask(labels):
    mask=labels[:,1:].ne(-100)
    if not mask.any(dim=-1).all():raise ValueError('Empty answer after masking/truncation')
    return mask


def answer_logps(logits,labels):
    mask=_answer_mask(labels)
    selected=logits[:,:-1][mask]
    if selected.dtype in (torch.float16,torch.bfloat16):selected=selected.float()
    targets=labels[:,1:][mask]
    values=F.log_softmax(selected,dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    return torch.zeros(mask.shape,device=logits.device,dtype=values.dtype).masked_scatter(mask,values),mask


@torch.no_grad()
def noise_evidence(logits,labels):
    mask=_answer_mask(labels)
    selected=logits[:,:-1][mask]
    targets=labels[:,1:][mask]
    observed=selected.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    if observed.dtype in (torch.float16,torch.bfloat16):observed=observed.float()
    output=torch.zeros(mask.shape,device=logits.device,dtype=observed.dtype).masked_scatter(mask,observed)
    ids=torch.zeros(mask.shape,device=logits.device,dtype=torch.long).masked_scatter(mask,selected.argmax(-1))
    return output,ids


@torch.no_grad()
def exploration_weights(clean_target,noisy_target,clean_alternative,mask,alpha=.15,tau=.4,epsilon=1e-6):
    if not 0<=alpha<1 or not 0<tau<=1 or epsilon<=0:raise ValueError('Invalid UE-DPO parameters')
    if mask.ndim!=2 or mask.shape[0]%2:raise ValueError('Chosen/rejected halves required')
    if not mask.any(-1).all():raise ValueError('Empty response in exploration mask')
    weights=torch.ones_like(clean_target)
    if alpha==0:return weights
    sensitivity=clean_target-noisy_target
    uncertainty=clean_alternative-clean_target
    half=mask.shape[0]//2
    for index in range(mask.shape[0]):
        delta=sensitivity[index,mask[index]]
        threshold=torch.quantile(delta,tau if index<half else 1-tau)
        selected=mask[index] & (sensitivity[index]<=threshold if index<half else sensitivity[index]>=threshold)
        values=uncertainty[index,selected]
        if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite answer uncertainty')
        center=torch.quantile(values,.25)
        deviation=values.std(unbiased=False).clamp_min(epsilon)
        intensity=torch.sigmoid((values-center)/deviation)
        weights[index,selected]=1+(alpha if index<half else -alpha)*intensity
    return weights.detach()


def ue_dpo_loss(policy_tokens,reference_tokens,weights,mask,beta=.1):
    if policy_tokens.shape!=reference_tokens.shape or policy_tokens.shape!=weights.shape or mask.shape!=weights.shape:
        raise ValueError('Token shapes differ')
    if policy_tokens.shape[0]%2:raise ValueError('Chosen/rejected halves required')
    ratios=((weights.detach()*policy_tokens-reference_tokens.detach())*mask).sum(-1)
    rewards=beta*ratios
    half=rewards.shape[0]//2
    return -F.logsigmoid(rewards[:half]-rewards[half:]),rewards


@lru_cache(maxsize=1000)
def diffusion_retention(step=500):
    if not 0<=step<1000:raise ValueError('Diffusion index must be0..999')
    betas=torch.sigmoid(torch.linspace(-6,6,1000))*(.005-.00001)+.00001
    return torch.cumprod(1-betas,dim=0)[step].item()


def noisy_pair_batch(batch,step=500,generator=None):
    if any(key in batch for key in ('video_grid_thw','pixel_values_videos','audio_values')):
        raise ValueError('Image-only UE-DPO adapter')
    count=batch['input_ids'].shape[0]
    grid=batch['image_grid_thw'];pixels=batch['pixel_values']
    if count%2 or grid.shape[0]!=count:raise ValueError('Exactly one image per response is required')
    half=count//2
    if not torch.equal(grid[:half],grid[half:]):raise ValueError('Pair image grids differ')
    patches=int(grid[:half].prod(-1).sum().item())
    if pixels.shape[0]!=2*patches or not torch.equal(pixels[:patches],pixels[patches:]):
        raise ValueError('Chosen/rejected images differ or patch counts are invalid')
    retention=diffusion_retention(step)
    noise=torch.randn(pixels[:patches].shape,device=pixels.device,dtype=torch.float32,generator=generator)
    perturbed=(retention**.5*pixels[:patches].float()+(1-retention)**.5*noise).to(pixels.dtype)
    return dict(batch,pixel_values=torch.cat((perturbed,perturbed),dim=0))
