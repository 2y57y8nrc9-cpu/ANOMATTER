import unittest
import torch
from bending import ActivationBender,BendSettings,StepSchedule,MODES,bend_unet

class BendTests(unittest.TestCase):
    def test_modes_and_regions(self):
        x=torch.randn(1,48,24,24)
        for mode in MODES:
            for region in ('all','center','periphery','bands'):
                s=BendSettings(mode=mode,intensity=.85,fraction=.35,seed=42,onset=0,cutoff=1,expected_steps=4,region=region)
                timer=StepSchedule(s);timer.steps=2
                hook=ActivationBender(s,timer,'early')
                y=hook(None,None,x)
                self.assertEqual(x.shape,y.shape)
                self.assertGreater((x-y).abs().mean().item(),0)
                self.assertEqual(hook.calls,1)

    def test_schedule_is_gated(self):
        s=BendSettings(intensity=.6,onset=.3,cutoff=.6,expected_steps=10)
        timer=StepSchedule(s)
        self.assertEqual(timer.amplitude(1),0)
        timer.steps=5
        self.assertGreater(timer.amplitude(1),0)
        timer.steps=10
        self.assertEqual(timer.amplitude(1),0)

    def test_hooks_removed(self):
        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__();self.resnets=torch.nn.ModuleList([torch.nn.Conv2d(8,8,3,padding=1)])
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.down_blocks=torch.nn.ModuleList([Wrap(),Wrap()]);self.up_blocks=torch.nn.ModuleList([Wrap(),Wrap()]);self.mid_block=Wrap()
            def forward(self,x):
                x=self.down_blocks[1].resnets[0](x)
                x=self.mid_block.resnets[0](x)
                return self.up_blocks[1].resnets[0](x)
        n=Net()
        s=BendSettings(onset=0,cutoff=1,expected_steps=4,layer='multi')
        with bend_unet(n,s) as observations:
            n(torch.rand(1,8,16,16));n(torch.rand(1,8,16,16))
            self.assertGreater(sum(o.calls for o in observations),0)
        self.assertEqual(len(n._forward_pre_hooks),0)
        for m in [n.down_blocks[1].resnets[0],n.mid_block.resnets[0],n.up_blocks[1].resnets[0]]:
            self.assertEqual(len(m._forward_hooks),0)
