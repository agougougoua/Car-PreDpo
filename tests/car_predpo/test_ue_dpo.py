import unittest
import torch
import torch.nn.functional as F
from car_predpo.ue_dpo import exploration_weights, ue_dpo_loss, diffusion_retention, noisy_pair_batch, answer_logps, noise_evidence

class UEDPOTests(unittest.TestCase):
    def test_alpha_zero_is_summed_answer_dpo_in_values_and_gradients(self):
        torch.manual_seed(42)
        policy=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
        second=policy.detach().clone().requires_grad_()
        reference=torch.randn_like(policy)
        mask=torch.tensor([[1,1,1,0,0],[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0]],dtype=torch.bool)
        values=torch.randn_like(policy)
        weights=exploration_weights(values,values/2,values+1,mask,alpha=0,tau=.4)
        actual,_=ue_dpo_loss(policy,reference,weights,mask,beta=.1)
        ratios=((second-reference)*mask).sum(-1)
        expected=-F.logsigmoid(.1*(ratios[:2]-ratios[2:]))
        torch.testing.assert_close(actual,expected)
        actual.sum().backward();expected.sum().backward()
        torch.testing.assert_close(policy.grad,second.grad)

    def test_asymmetric_weights_are_detached_with_finite_singleton_and_padding(self):
        clean=torch.tensor([[0.,1.,2.,3.,9999.],[0.,1.,2.,3.,-9999.]],requires_grad=True)
        noisy=torch.zeros_like(clean)
        alternative=clean+2
        mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,0]],dtype=torch.bool)
        weights=exploration_weights(clean,noisy,alternative,mask,alpha=.2,tau=.4)
        self.assertFalse(weights.requires_grad)
        self.assertTrue(torch.all(weights[0,:2]>1))
        self.assertTrue(torch.all(weights[1,2:4]<1))
        torch.testing.assert_close(weights[0,2:4],torch.ones(2))
        torch.testing.assert_close(weights[1,:2],torch.ones(2))
        single=torch.tensor([[True,False],[True,False]])
        zeros=torch.zeros(2,2)
        finite=exploration_weights(zeros,zeros,zeros,single,alpha=.2,tau=.4)
        torch.testing.assert_close(finite[:,0],torch.tensor([1.1,.9]))
        self.assertTrue(torch.isfinite(finite).all())
        altered=clean.detach().clone();altered[:,-1]=float('nan')
        torch.testing.assert_close(weights,exploration_weights(altered,noisy,altered+2,mask,alpha=.2,tau=.4))

    def test_reference_is_unweighted_and_weights_receive_no_gradient(self):
        policy=torch.tensor([[-1.,-2.],[-3.,-4.]],requires_grad=True)
        ref=torch.tensor([[-2.,-3.],[-4.,-5.]],requires_grad=True)
        weights=torch.tensor([[1.1,1.2],[.8,.9]],requires_grad=True)
        mask=torch.ones(2,2,dtype=torch.bool)
        losses,rewards=ue_dpo_loss(policy,ref,weights,mask,beta=.1)
        expected_reward=.1*((policy*weights.detach()).sum(-1)-ref.detach().sum(-1))
        torch.testing.assert_close(rewards,expected_reward)
        torch.testing.assert_close(losses,-F.logsigmoid(expected_reward[:1]-expected_reward[1:]))
        losses.sum().backward()
        self.assertIsNone(ref.grad);self.assertIsNone(weights.grad)
        self.assertIsNotNone(policy.grad)

    def test_noise_matches_standard_schedule_and_is_shared_with_pair(self):
        betas=torch.sigmoid(torch.linspace(-6,6,1000))*(.005-.00001)+.00001
        expected=torch.cumprod(1-betas,0)[500]
        self.assertAlmostEqual(diffusion_retention(500),expected.item(),places=7)
        pixels=torch.arange(12,dtype=torch.float32).reshape(4,3)
        pixels=torch.cat([pixels,pixels])
        batch=dict(pixel_values=pixels,input_ids=torch.ones(2,3,dtype=torch.long),
                   image_grid_thw=torch.tensor([[1,2,2],[1,2,2]]))
        before=pixels.clone()
        seed1=torch.Generator().manual_seed(42);seed2=torch.Generator().manual_seed(42)
        altered=noisy_pair_batch(batch,500,generator=seed1)
        noise=torch.randn(4,3,generator=seed2)
        wanted=expected.sqrt()*pixels[:4]+(1-expected).sqrt()*noise
        torch.testing.assert_close(altered['pixel_values'][:4],wanted)
        torch.testing.assert_close(altered['pixel_values'][:4],altered['pixel_values'][4:])
        torch.testing.assert_close(batch['pixel_values'],before)
        self.assertIs(altered['input_ids'],batch['input_ids'])
        with self.assertRaises(ValueError):noisy_pair_batch(batch|{'pixel_values':torch.randn(8,3)},500)

    def test_answer_shift_mask_and_noisy_argmax_use_observed_tokens(self):
        torch.manual_seed(42)
        logits=torch.randn(2,5,7,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,-100],[-100,1,2,4,0]])
        actual,mask=answer_logps(logits,labels)
        safe=labels[:,1:].clamp_min(0)
        wanted=logits[:,:-1].log_softmax(-1).gather(-1,safe[...,None]).squeeze(-1)
        torch.testing.assert_close(actual,wanted.masked_fill(~mask,0))
        observed,argmax=noise_evidence(logits,labels)
        torch.testing.assert_close(observed[mask],logits[:,:-1].gather(-1,safe[...,None]).squeeze(-1)[mask])
        torch.testing.assert_close(argmax[mask],logits[:,:-1].argmax(-1)[mask])
        self.assertFalse(observed.requires_grad)
        with self.assertRaises(ValueError):answer_logps(logits,torch.full_like(labels,-100))
