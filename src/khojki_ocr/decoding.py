"""CTC prefix beam search, without a dictionary or language-model prior."""
import math
import numpy as np
from khojki_ocr.model import VOCAB

NEG=-float('inf')
def add(a,b):
    if a==NEG:return b
    if b==NEG:return a
    m=max(a,b)
    return m+math.log1p(math.exp(min(a,b)-m))

def beam_decode(log_probs,width=10,topk=8,prefix_valid=None):
    beams={(): (0.,NEG)}
    for row in np.asarray(log_probs):
        k=min(topk,len(row))
        labels=np.argpartition(row,-k)[-k:]
        following={}
        def update(prefix,b=NEG,n=NEG):
            if prefix_valid is not None and not prefix_valid(''.join(VOCAB[i-1] for i in prefix)):return
            old=following.get(prefix,(NEG,NEG))
            following[prefix]=(add(old[0],b),add(old[1],n))
        for prefix,(blank,nonblank) in beams.items():
            total=add(blank,nonblank)
            update(prefix,b=total+float(row[0]))
            for c in labels:
                c=int(c)
                if c==0:continue
                prob=float(row[c])
                if prefix and prefix[-1]==c:
                    update(prefix,n=nonblank+prob)
                    update(prefix+(c,),n=blank+prob)
                else:update(prefix+(c,),n=total+prob)
        beams=dict(sorted(following.items(),key=lambda x:add(*x[1]),reverse=True)[:width])
    best=max(beams,key=lambda x:add(*beams[x]))
    return ''.join(VOCAB[i-1] for i in best)
