"""Build auditable word crops only from labelled training lines.

Printed Khojki separators are pairs of dots. Reject a line unless the detected
separator count equals its reference count. No recognition predictions or
development/test crops are used to construct this augmentation bank.
"""
import cv2
import numpy as np
from PIL import Image,ImageOps
from .segment import ink_mask

SEP='\U0001123a'

def separator_positions(image):
    h=image.height
    n,lab,stats,cent=cv2.connectedComponentsWithStats(ink_mask(image),8)
    dots=[i for i in range(1,n) if 5<=stats[i,4]<.04*h*h and stats[i,2]<.22*h and stats[i,3]<.22*h]
    pairs=[]
    for i in dots:
        for j in dots:
            xi,yi=cent[i];xj,yj=cent[j]
            if abs(xi-xj)<max(3,.06*h) and .12*h<yj-yi<.45*h and .2*h<(yi+yj)/2<.85*h and .4<stats[i,4]/stats[j,4]<2.5:
                pairs.append((float((xi+xj)/2),int(max(stats[i,0]+stats[i,2],stats[j,0]+stats[j,2]))))
    unique=[]
    for pair in sorted(pairs):
        if not unique or pair[0]-unique[-1][0]>3:unique.append(pair)
    return unique

def training_words(image,text,preserve_height=False,white_padding=True):
    # A scalar 255 is red in Pillow RGB mode, not white. Normalize mode first.
    # The opt-out exists only to reproduce the historical augmentation ablation.
    if white_padding:image=image.convert('L')
    positions=separator_positions(image)
    if not positions or len(positions)!=text.count(SEP):return []
    words=text.split(SEP)[:-1]
    result=[];left=0
    for word,(center,right) in zip(words,positions):
        end=min(image.width,right)
        if not word or end-left<8:return []
        crop=image.crop((left,0,end,image.height))
        mask=ink_mask(crop);ys,xs=np.where(mask)
        if len(xs)<25:return []
        crop=crop.crop((max(0,int(xs.min())-3),0,crop.width,crop.height))
        result.append((ImageOps.expand(crop,border=(3,0,3,0) if preserve_height else 3,fill=255),word+SEP))
        left=end
    return result

def concatenate_words(items,gap=3,preserve_height=False):
    normalized=[]
    for image,text in items:
        image=image.convert('L');w=max(8,round(image.width*48/image.height))
        normalized.append((image.resize((w,48),Image.Resampling.BILINEAR),text))
    out=Image.new('L',(sum(im.width for im,_ in normalized)+gap*(len(items)-1)+8,48 if preserve_height else 56),255)
    x=4
    for im,_ in normalized:out.paste(im,(x,0 if preserve_height else 4));x+=im.width+gap
    return out,''.join(t for _,t in normalized)
