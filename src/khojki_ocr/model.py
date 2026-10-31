import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

VOCAB=[chr(c) for c in range(0x11200,0x11242) if c!=0x11212]+[chr(c) for c in range(0xae6,0xaf0)]+[' ','.',',','(',')','\ufffd']

class CRNN(nn.Module):
    def __init__(self,nclass=len(VOCAB)+1,packed=False):
        super().__init__()
        self.packed=packed
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
        self.cnn=nn.Sequential(block(1,32),nn.MaxPool2d(2),block(32,64),nn.MaxPool2d(2),block(64,128),nn.MaxPool2d((2,1)),block(128,192),nn.MaxPool2d((2,1)))
        self.rnn=nn.LSTM(192,128,2,bidirectional=True,batch_first=True,dropout=.1)
        self.head=nn.Linear(256,nclass)
    def forward(self,x,lengths=None):
        x=self.cnn(x).mean(2).transpose(1,2)
        if self.packed:
            if lengths is None:raise ValueError("Packed recurrence requires valid sequence lengths")
            sequence=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            sequence,_=self.rnn(sequence)
            x,_=nn.utils.rnn.pad_packed_sequence(sequence,batch_first=True,total_length=x.shape[1])
        else:x,_=self.rnn(x)
        return self.head(x).log_softmax(-1).transpose(0,1)

def image_tensor(image,height=48):
    image=image.convert('L')
    w=max(8,round(image.width*height/image.height))
    a=np.asarray(image.resize((w,height),Image.Resampling.BILINEAR)).astype('float32')/255
    return torch.from_numpy((1-a)[None])

def collate(samples):
    xs,ys=zip(*samples)
    width=(max(x.shape[-1] for x in xs)+3)//4*4+16
    height=xs[0].shape[-2]
    if any(x.shape[-2]!=height for x in xs):raise ValueError("Mixed image heights in a batch")
    batch=torch.zeros(len(xs),1,height,width)
    for i,x in enumerate(xs): batch[i,:,:,:x.shape[-1]]=x
    mapping={c:i+1 for i,c in enumerate(VOCAB)}
    labels=torch.tensor([mapping[c] for s in ys for c in s],dtype=torch.long)
    return batch,labels,torch.tensor([x.shape[-1]//4 for x in xs]),torch.tensor([len(y) for y in ys]),ys

def decode(log_probs,lengths=None):
    ids=log_probs.argmax(-1).transpose(0,1).cpu().tolist()
    out=[]
    for i,row in enumerate(ids):
        if lengths is not None: row=row[:int(lengths[i])]
        prev=0;text=''
        for v in row:
            if v and v!=prev: text+=VOCAB[v-1]
            prev=v
        out.append(text)
    return out
