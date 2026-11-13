"""UE-DPO on the existing Qwen LoRA DPO stack; paper-based implementation."""
from contextlib import nullcontext
import hashlib
import torch
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import nested_detach
from .ue_dpo import answer_logps,noise_evidence,exploration_weights,ue_dpo_loss,noisy_pair_batch


class UEDPOTrainer(CustomDPOTrainer):
    def __init__(self,*args,ue_alpha=.15,ue_tau=.4,ue_noise_step=500,**kwargs):
        self.ue_alpha=ue_alpha;self.ue_tau=ue_tau;self.ue_noise_step=ue_noise_step
        super().__init__(*args,**kwargs)

    def _model_batch(self,batch):
        clean=nested_detach(batch,clone=True)
        labels=clean.pop('labels')
        for key in ('lambda_i','delta_h','vdi'):clean.pop(key,None)
        return clean,labels

    def _monitoring_generator(self,batch):
        digest=hashlib.blake2b(digest_size=8)
        digest.update(batch['input_ids'].detach().cpu().contiguous().numpy().tobytes())
        digest.update(batch['pixel_values'].detach().float().cpu().contiguous().numpy().tobytes())
        seed=(int.from_bytes(digest.digest(),'little')+self.args.seed)%(2**63-1)
        return torch.Generator(device=batch['pixel_values'].device).manual_seed(seed)

    def get_batch_loss_metrics(self,model,batch,train_eval='train'):
        inputs,labels=self._model_batch(batch)
        if self.ref_model is None:
            reference_model=model
            context=self.accelerator.unwrap_model(model).disable_adapter()
        else:reference_model=self.ref_model;context=nullcontext()
        with torch.no_grad(),context:
            output=reference_model(**inputs,return_dict=True,use_cache=False)
            reference,mask=answer_logps(output.logits,labels)
            del output
        generator=self._monitoring_generator(inputs) if train_eval=='eval' else None
        with torch.no_grad():
            noisy=noisy_pair_batch(inputs,self.ue_noise_step,generator=generator)
            output=model(**noisy,return_dict=True,use_cache=False)
            noisy_target,noisy_argmax=noise_evidence(output.logits,labels)
            del output,noisy
        output=model(**inputs,return_dict=True,use_cache=False)
        policy,mask=answer_logps(output.logits,labels)
        with torch.no_grad():
            logits=output.logits[:,:-1]
            targets=labels[:,1:].clamp_min(0)
            clean_target=logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
            alternative=logits.gather(-1,noisy_argmax.unsqueeze(-1)).squeeze(-1)
            if clean_target.dtype in (torch.float16,torch.bfloat16):
                clean_target=clean_target.float();alternative=alternative.float()
            weights=exploration_weights(clean_target,noisy_target,alternative,mask,
                                         alpha=self.ue_alpha,tau=self.ue_tau)
        losses,rewards=ue_dpo_loss(policy,reference,weights,mask,beta=self.beta)
        if not torch.isfinite(losses).all():raise FloatingPointError('Nonfinite UE-DPO loss')
        half=rewards.shape[0]//2
        prefix='eval_' if train_eval=='eval' else ''
        mean=lambda value,selected:value[selected].mean().item()
        metrics={
            prefix+'ue/lambda_chosen':mean(weights[:half],mask[:half]),
            prefix+'ue/lambda_rejected':mean(weights[half:],mask[half:]),
            prefix+'ue/adjusted_fraction':mean(weights.ne(1).float(),mask),
            prefix+'ue/uncertainty':mean((alternative-clean_target),mask),
            prefix+'ue/sensitivity':mean((clean_target-noisy_target),mask),
            prefix+'rewards/chosen':rewards[:half].detach().mean().item(),
            prefix+'rewards/rejected':rewards[half:].detach().mean().item(),
            prefix+'rewards/accuracies':(rewards[:half]>rewards[half:]).float().mean().item(),
        }
        return losses.mean(),metrics
