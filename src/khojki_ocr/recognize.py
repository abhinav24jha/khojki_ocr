import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageFilter
from khojki_ocr.model import CRNN,VOCAB,image_tensor,collate,decode
from khojki_ocr.segment import ink_mask


def preprocess(image,mode='contrast'):
    image=image.convert('L')
    if mode=='raw':return image
    gray=np.array(image)
    bg=cv2.GaussianBlur(gray,(0,0),max(9,image.height/3))
    strength=np.clip((bg.astype(float)-gray.astype(float))/110,0,1)
    if mode=='binary':
        mask=ink_mask(image)
        return Image.fromarray((255*(1-mask)).astype('uint8'))
    return Image.fromarray((255*(1-strength)).astype('uint8'))


def split_columns(image,mode='legacy'):
    mask=ink_mask(image)
    if mode=='robust':
        # Tiny speckles in a wide inter-column gap must not erase the gap.
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        keep=np.zeros(count,dtype='uint8')
        for j in range(1,count):
            if stats[j,cv2.CC_STAT_AREA]>=max(12,image.height*.3):keep[j]=1
        mask=keep[labels]
    occupied=mask.sum(0)>1
    edges=np.diff(np.r_[0,(~occupied).astype(int),0])
    gaps=[(a,b) for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if b-a>max(45,image.height*1.3) and a>5 and b<image.width-5]
    cuts=[0]+[(a+b)//2 for a,b in gaps]+[image.width]
    chunks=[]
    for l,r in zip(cuts[:-1],cuts[1:]):
        ys,xs=np.where(mask[:,l:r])
        if len(xs)<max(25,image.height*.8):continue
        left=max(l,l+int(xs.min())-5);right=min(r,l+int(xs.max())+6)
        chunks.append(image.crop((left,0,right,image.height)))
    return chunks or [image]


class Recognizer:
    def __init__(self,checkpoint,device='cpu',preprocessing='contrast',columns=True,beam_width=None,layout=None,constrained=None,views=None):
        self.device=device;self.preprocessing=preprocessing;self.columns=columns
        self.beam_width=beam_width
        self.constrained=constrained
        paths=checkpoint if isinstance(checkpoint,(list,tuple)) else [checkpoint]
        self.models=[]
        self.views=views
        for path in paths:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if layout is None:layout=saved.get('layout','legacy')
            if self.beam_width is None:self.beam_width=saved.get('beam_width',1)
            if self.constrained is None:self.constrained=saved.get('constrained',False)
            if saved['vocab']!=VOCAB:raise ValueError('Checkpoint vocabulary mismatch')
            if self.views is None:self.views=saved.get('views',[[1.0,0.0]])
            height=saved.get('height',48)
            if self.models and height!=self.height:raise ValueError('Ensemble heights must match')
            self.height=height
            members=saved.get('ensemble_states') or [{'model':saved['model'],'packed':saved.get('packed',False)}]
            for member in members:
                model=CRNN(packed=member.get('packed',False));model.load_state_dict(member['model']);model.to(device).eval();self.models.append(model)
        self.model=self.models[0]
        self.layout=layout
    def __call__(self,image):
        chunks=split_columns(image,self.layout) if self.columns else [image]
        with torch.no_grad():
            import math
            all_probs=[]
            for strength,blur in self.views:
                images=[preprocess(im,self.preprocessing) for im in chunks]
                if blur:images=[im.filter(ImageFilter.GaussianBlur(blur)) for im in images]
                samples=[(image_tensor(im,self.height),'') for im in images]
                x,_,lengths,_,_=collate(samples)
                x=(x*strength).clamp(0,1).to(self.device)
                all_probs.extend(m(x,lengths) for m in self.models)
            probs=torch.logsumexp(torch.stack(all_probs),dim=0)-math.log(len(all_probs))
            if self.beam_width>1:
                from khojki_ocr.decoding import beam_decode
                from khojki_ocr.unicode import plausible_prefix
                texts=[beam_decode(probs[:int(lengths[i]),i].cpu().numpy(),self.beam_width,prefix_valid=plausible_prefix if self.constrained else None) for i in range(len(chunks))]
            else:texts=decode(probs,lengths)
        return ' '.join(t.strip() for t in texts if t.strip())
