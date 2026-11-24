"""ANOMATTER Dynamic Network Bending / Diffusers SD1.5 UNet.

Independent experimental implementation inspired by the concept of network
bending of Terence Broad et al. No original StyleGAN2 code is redistributed.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
import random
import torch
import torch.nn.functional as F

MODES=('FRACTURE','DRIFT','ECHO','GHOST','PARASITE','ABLATE')
LAYER_NAMES=('early','middle','late','multi')
REGIONS=('all','center','periphery','bands')

@dataclass(frozen=True)
class BendSettings:
    mode:str='FRACTURE'
    layer:str='early'
    intensity:float=.38
    fraction:float=.12
    seed:int=17701234
    onset:float=.08
    cutoff:float=.95
    early_weight:float=1.0
    middle_weight:float=.65
    late_weight:float=.45
    region:str='all'
    expected_steps:int=18


class StepSchedule:
    """Counts actual UNet forward passes, not wall clock time.

    Diffusion steps normally invoke the UNet once for classifier-free guidance,
    but certain pipelines can invoke it more; progression is approximate.
    """
    def __init__(self,settings):
        self.s=settings;self.steps=0
    def tick(self,module,args):
        self.steps+=1
    @property
    def progress(self):
        return min(1., max(0.,(self.steps-1)/max(1,self.s.expected_steps-1)))
    def amplitude(self,weight):
        p=self.progress
        if p<self.s.onset or p>self.s.cutoff:return 0.
        span=max(1e-5,self.s.cutoff-self.s.onset)
        q=(p-self.s.onset)/span
        # smooth fade-in and fade-out, both available as explicit controls
        env=min(1.,q/.2,(1-q)/.14)
        return max(0.,min(1.,self.s.intensity*weight*env))


def _region_mask(t,region,seed):
    h,w=t.shape[-2:]
    if region=='all':return None
    # Construct on device, no CPU copy of image feature maps.
    gy=torch.linspace(-1,1,h,device=t.device,dtype=torch.float32)
    gx=torch.linspace(-1,1,w,device=t.device,dtype=torch.float32)
    yy,xx=torch.meshgrid(gy,gx,indexing='ij')
    if region=='center':mask=(1-((xx**2+yy**2).sqrt()/.8)).clamp(0,1)
    elif region=='periphery':mask=(((xx**2+yy**2).sqrt()-.25)/.7).clamp(0,1)
    else:
        mask=(torch.cos(xx*12.5+yy*4.4+seed*.001)**2).pow(2)
    return mask.to(dtype=t.dtype)[None,None,:,:]


class ActivationBender:
    def __init__(self,settings,schedule,layer_name='early',layer_id=0):
        self.s=settings;self.schedule=schedule;self.layer_name=layer_name;self.layer_id=layer_id
        self.calls=0;self.modified=0;self._picked={};self._mask={}
    def __call__(self,module,inputs,output):
        if not isinstance(output,torch.Tensor) or output.ndim!=4:return output
        s=self.s
        wgt={'early':s.early_weight,'middle':s.middle_weight,'late':s.late_weight}[self.layer_name]
        amp=self.schedule.amplitude(wgt)
        self.calls+=1
        if amp<=0 or s.fraction<=0 or output.shape[1]<2 or min(output.shape[-2:])<3:return output
        c=output.shape[1]; key=(c,round(s.fraction,4))
        if key not in self._picked:
            rng=random.Random(s.seed+self.layer_id*2423)
            self._picked[key]=sorted(rng.sample(range(c),min(c,max(1,round(c*s.fraction)))))
        ch=self._picked[key]
        x=output[:,ch,:,:]
        h,w=x.shape[-2:]
        dx=max(1,round(w*(.025+.19*amp)));dy=max(1,round(h*(.02+.12*amp)))
        mode=s.mode.upper()
        if mode=='DRIFT':
            y=torch.roll(x,(dy,dx),(-2,-1))
        elif mode=='ECHO':
            y=x*.3+torch.roll(x,(0,dx),(-2,-1))*.45+torch.roll(x,(-dy,-dx),(-2,-1))*.25
        elif mode=='GHOST':
            ghost=F.avg_pool2d(x,3,1,1)
            y=torch.roll(ghost,(dy,-dx),(-2,-1))*.75
        elif mode=='PARASITE':
            y=x.flip(-1)
            if x.shape[1]>1:y=torch.roll(y,shifts=max(1,x.shape[1]//4),dims=1)
        elif mode=='ABLATE':
            y=x*.05
        else: # FRACTURE
            start,end=h//4,max(h//4+1,3*h//4)
            y=x.clone();y[...,start:end,:]=torch.roll(x[...,start:end,:],dx,dims=-1)
            y=torch.roll(y,-dy,dims=-2)
        mask=self._mask.get((h,w,str(output.device),str(output.dtype)))
        if mask is None and s.region!='all':
            mask=_region_mask(x,s.region,s.seed)
            self._mask[(h,w,str(output.device),str(output.dtype))]=mask
        delta=(y-x)*amp
        if mask is not None:delta=delta*mask
        delta=torch.nan_to_num(delta,nan=0.,posinf=0.,neginf=0.)
        result=output.clone()
        result[:,ch,:,:]=x+delta
        self.modified+=len(ch)
        return result


def select_unet_modules(unet,layer):
    try:
        bank={'early':unet.down_blocks[1].resnets[0],
              'middle':unet.mid_block.resnets[0],
              'late':unet.up_blocks[1].resnets[0]}
    except (AttributeError,IndexError,TypeError) as e:
        raise RuntimeError('Ce bending nécessite une architecture UNet compatible Stable Diffusion 1.5') from e
    if layer not in LAYER_NAMES:raise ValueError('Couche inconnue : '+str(layer))
    return [(k,bank[k]) for k in (('early','middle','late') if layer=='multi' else (layer,))]


@contextmanager
def bend_unet(unet,settings):
    """Attach and remove all hooks even on failed or cancelled generation."""
    hooks=[];observers=[]
    clock=StepSchedule(settings)
    try:
        hooks.append(unet.register_forward_pre_hook(clock.tick))
        for i,(name,module) in enumerate(select_unet_modules(unet,settings.layer)):
            ob=ActivationBender(settings,clock,name,i)
            observers.append(ob)
            hooks.append(module.register_forward_hook(ob))
        yield observers
    finally:
        for hook in reversed(hooks):hook.remove()
