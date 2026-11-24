"""ANOMATTER v0.5 / traceable image-space role conditioning.

No claim of semantic material separation: these PIL/numpy operations work on
pixels. They create a structure-preserving conditioning canvas and explicit
intermediate previews. Optional neural image conditioning is separate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import io
import base64
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageEnhance

SIZE=(512,512)
ROLES=('morphology','material','pattern','chromatic','parasite')


def fit(image:Image.Image,size=SIZE):
    return ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'), size, method=Image.Resampling.LANCZOS)


def decode_data_url(url,limit=9_000_000):
    if url is None or url=='':return None
    if not isinstance(url,str) or not url.startswith('data:image/') or ',' not in url:
        raise ValueError('Image source invalide, utilise un fichier JPG/PNG/WebP')
    try: raw=base64.b64decode(url.split(',',1)[1],validate=True)
    except Exception as e:raise ValueError('Le fichier image ne peut pas être décodé') from e
    if len(raw)>limit:raise ValueError('Image dépassant la limite de 9 Mo')
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.width*img.height > 30_000_000:raise ValueError('Image trop grande')
            img.load();return ImageOps.exif_transpose(img).convert('RGB')
    except ValueError:raise
    except Exception as e:raise ValueError('Image non prise en charge') from e


def _clip(v,lo=0,hi=1):return max(lo,min(hi,float(v)))


def transfer_palette(base:Image.Image,palette_image:Image.Image,weight=.85)->Image.Image:
    """RGB moment transfer, preserving only color statistics, NOT scene content."""
    a=np.asarray(base.convert('RGB'),dtype=np.float32)
    b=np.asarray(palette_image.convert('RGB'),dtype=np.float32)
    ma=a.mean(axis=(0,1),keepdims=True);sa=np.maximum(a.std(axis=(0,1),keepdims=True),4.)
    mb=b.mean(axis=(0,1),keepdims=True);sb=np.maximum(b.std(axis=(0,1),keepdims=True),4.)
    matched=np.clip((a-ma)*(sb/sa)+mb,0,255)
    out=a*(1-_clip(weight))+matched*_clip(weight)
    return Image.fromarray(np.uint8(np.clip(out,0,255)),'RGB')


def material_rubbing(base,material,amount=.55):
    """Transfer high-frequency luminance relief, not a photo of the material."""
    material=fit(material,base.size).convert('L')
    amount=_clip(amount)
    # Texture signed high-pass: avoid reproducing objects by removing low-freq content.
    raw=np.asarray(material,dtype=np.float32)
    low=np.asarray(material.filter(ImageFilter.GaussianBlur(radius=11)),dtype=np.float32)
    detail=np.clip((raw-low)*1.7,-42,42)[...,None]
    result=np.asarray(base,dtype=np.float32)+detail*(amount*.90)
    return Image.fromarray(np.uint8(np.clip(result,0,255)),'RGB')


def pattern_invasion(base,pattern,amount=.50,seed=0):
    """Repeat edge textures as subtle tonal relief, no semantic role claims."""
    amount=_clip(amount)
    tile=fit(pattern,(256,256)).convert('L')
    # Edges emphasize repeated seams and woven line work.
    edges=tile.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(.45))
    original=np.asarray(edges,dtype=np.float32)
    low=np.asarray(edges.filter(ImageFilter.GaussianBlur(8)),dtype=np.float32)
    hp=np.clip(original-low,-55,55)
    t=np.tile(hp,(2,2))[:base.height,:base.width]
    rng=random.Random(seed)
    t=np.roll(t,rng.randrange(0,80),axis=0)
    t=np.roll(t,rng.randrange(0,80),axis=1)
    base_arr=np.asarray(base,dtype=np.float32)
    # Materiality of the pattern, not the identity of the scene it came from.
    out=base_arr + t[...,None]*(amount*.55)
    return Image.fromarray(np.uint8(np.clip(out,0,255)),'RGB')


def parasite_regions(base,parasite,amount=.25,seed=0):
    """Optional local interruption with clearly labeled image-space regions."""
    amount=_clip(amount)
    if amount==0:return base
    rng=random.Random(seed)
    import PIL.ImageDraw as ImageDraw
    resized=fit(parasite,base.size)
    mask=Image.new('L',base.size,0);draw=ImageDraw.Draw(mask)
    for _ in range(3):
        x=rng.randint(0,base.width-1);y=rng.randint(0,base.height-1)
        w=rng.randint(20,140);h=rng.randint(30,170)
        draw.rectangle((x-w//2,y-h//2,x+w//2,y+h//2),fill=int(110*amount))
    return Image.composite(resized,base,mask.filter(ImageFilter.GaussianBlur(2)))


@dataclass
class Prepared:
    image:Image.Image
    stages:Dict[str,Image.Image]
    roles_used:list[str]
    notes:list[str]


def prepare_sources(images:dict,options:dict=None,seed:int=17701234)->Prepared:
    """Build a reproducible *pixel conditioning* image from distinct roles.

    An optional IP-Adapter can additionally condition on a supplied image at
    neural inference. It is handled in the pipeline, not here.
    """
    options=options or {}
    refs={r:fit(img) for r,img in images.items() if r in ROLES and img is not None}
    if not refs:raise ValueError('Importe au moins une image')
    notes=[];stages={}
    anchor=refs.get('morphology') or next(iter(refs.values()))
    if 'morphology' not in refs:notes.append('Pas de MORPHOLOGY : la première source sert de support structurel.')
    out=anchor.copy()
    stages['01_anchor']=out.copy()
    if 'material' in refs:
        out=material_rubbing(out,refs['material'],options.get('material_weight',.55))
        stages['02_material']=out.copy()
    if 'pattern' in refs:
        out=pattern_invasion(out,refs['pattern'],options.get('pattern_weight',.50),seed)
        stages['03_pattern']=out.copy()
    if 'chromatic' in refs:
        out=transfer_palette(out,refs['chromatic'],options.get('chromatic_weight',.85))
        stages['04_chromatic']=out.copy()
    if 'parasite' in refs:
        out=parasite_regions(out,refs['parasite'],options.get('parasite_weight',.20),seed)
        stages['05_parasite']=out.copy()
    lock=_clip(options.get('morphology_lock',.58))
    # Lock is explicit: reducing non-shape influence preserves more source identity.
    if 'morphology' in refs:
        out=Image.blend(out,anchor,lock*.45)
    stages['06_final']=out.copy()
    notes.append('MATÉRIAU/MOTIF/COULEUR : transferts heuristiques de pixels ; aucun découpage sémantique garanti.')
    return Prepared(image=out,stages=stages,roles_used=list(refs),notes=notes)


def palette_swatch(image,colors=6):
    im=fit(image,(96,96)).quantize(colors=colors,method=Image.Quantize.MEDIANCUT).convert('RGB')
    palette=im.getcolors(96*96) or []
    palette=sorted(palette,reverse=True)[:colors]
    return ['#%02x%02x%02x'%rgb for _,rgb in palette]
