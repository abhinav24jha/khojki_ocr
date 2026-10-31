"""Adapt to real print with synthetic replay and save fully resumable checkpoints."""
from pathlib import Path
import argparse,random,json,time,hashlib
import numpy as np
from PIL import Image,ImageFilter,ImageOps
import torch
from khojki_ocr.model import CRNN,VOCAB,image_tensor,collate,decode
from khojki_ocr.recognize import split_columns,preprocess
from khojki_ocr.render import Renderer
from khojki_ocr.metrics import score

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--steps',type=int,default=600);p.add_argument('--device',default='mps');p.add_argument('--lr',type=float,default=.0002);p.add_argument('--output',default='adapt-v1');p.add_argument('--resume',action='store_true');p.add_argument('--synthetic-fraction',type=float,default=.25);p.add_argument('--seed',type=int,default=31);p.add_argument('--train-limit',type=int)
p.add_argument('--annotations',type=Path,default=ROOT/'data/annotations.json')
p.add_argument('--layout',choices=['legacy','robust'],default='legacy')
p.add_argument('--word-fraction',type=float,default=0)
p.add_argument('--word-geometry',choices=['legacy','preserve'],default='legacy')
p.add_argument('--bold-replay',action='store_true')
p.add_argument('--legacy-rgb-word-padding',action='store_true')
p.add_argument('--freeze-bn',action='store_true')
p.add_argument('--freeze-rnn',action='store_true')
p.add_argument('--packed',action='store_true')
p.add_argument('--height',type=int,choices=[48,64],default=48)
a=p.parse_args()
if not 0<=a.word_fraction+a.synthetic_fraction<=1:p.error('Mixture fractions must sum to at most one')
torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
annotations=a.annotations.read_bytes();rows=json.loads(annotations)
train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
if a.train_limit:train=train[:a.train_limit]
samples=[]
for r in train:
    im=Image.open(ROOT/'data'/r['image']).convert('L');chunks=split_columns(im,a.layout);texts=r['text'].split(' ')
    if len(chunks)==len(texts):samples.extend((preprocess(c),t) for c,t in zip(chunks,texts))
    else:samples.append((preprocess(im),r['text']))
print('real lines',len(train),'training chunks',len(samples),flush=True)
wordbank=[]
if a.word_fraction:
    from khojki_ocr.word_augmentation import training_words,concatenate_words
    for r in train:
        wordbank.extend((preprocess(im),txt) for im,txt in training_words(Image.open(ROOT/'data'/r['image']),r['text'],preserve_height=a.word_geometry=='preserve',white_padding=not a.legacy_rgb_word_padding))
    if not wordbank:raise ValueError('No verified training word crops available')
    print('word crops from training only',len(wordbank),flush=True)
renderers=[Renderer(ROOT/f'assets/Noto{x}Khojki-Regular.ttf',36) for x in ['Sans','Serif']]
if a.bold_replay:renderers.append(Renderer(ROOT/'assets/NotoSerifKhojki-Bold.ttf',36))
base=[chr(c) for c in range(0x11200,0x1122c) if c!=0x11212]
cons=base[8:];signs=['']*7+[chr(c) for c in range(0x1122c,0x11235)]
synthetic=[]
for i in range(1000):
    s=''
    for j in range(random.randint(3,18)):
        c=random.choice(base);s+=c
        if c in cons:s+=random.choice(signs)
        if random.random()<.2:s+='\U0001123a'
    if i%5==0:s=''.join(random.choices([chr(c) for c in range(0xae6,0xaf0)],k=random.randint(1,3)))
    synthetic.append((renderers[i%len(renderers)].render(s),s))

def augment(im):
    im=im.resize((max(12,round(im.width*random.uniform(.82,1.18))),max(20,round(im.height*random.uniform(.85,1.15)))),Image.Resampling.BILINEAR)
    if random.random()<.4:im=im.filter(ImageFilter.GaussianBlur(random.uniform(.1,.5)))
    im=im.rotate(random.uniform(-.7,.7),resample=Image.Resampling.BILINEAR,expand=True,fillcolor=255)
    return image_tensor(im,a.height)

saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=CRNN(packed=a.packed);model.load_state_dict(saved['model']);model.to(a.device)
if a.freeze_rnn:
    for param in model.rnn.parameters():param.requires_grad_(False)
    # cuDNN backward through frozen recurrent weights still needs training mode.
    # Disable recurrent dropout without switching the layer to eval.
    model.rnn.dropout=0.0
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.0001)
start_step=0
if a.resume:
    if saved['annotation_sha256']!=hashlib.sha256(annotations).hexdigest():raise ValueError('References changed; start a new named experiment rather than resuming.')
    for key in ['seed','train_limit','synthetic_fraction','word_fraction','bold_replay','layout','height','packed','word_geometry','legacy_rgb_word_padding','freeze_bn','freeze_rnn']:
        if key in saved.get('training_config',{}) and saved['training_config'][key]!=getattr(a,key):
            raise ValueError(f'Resume requires original {key}; warm start a new experiment to change it.')
    opt.load_state_dict(saved['optimizer']);start_step=saved['step']
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    if a.device=='mps' and 'mps_rng' in saved:torch.mps.set_rng_state(saved['mps_rng'])
    if a.device.startswith('cuda') and 'cuda_rng' in saved:torch.cuda.set_rng_state_all(saved['cuda_rng'])
loss_fn=torch.nn.CTCLoss(blank=0,zero_infinity=True)
out=ROOT/'experiments'/a.output;out.mkdir(parents=True,exist_ok=True)
config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
(out/'config.json').write_text(json.dumps(config|{'annotation_sha256':hashlib.sha256(annotations).hexdigest(),'train_line_ids':[r['id'] for r in train],'validation_line_ids':[r['id'] for r in dev],'test_used':False,'word_bank_size':len(wordbank)},indent=2))
history=saved.get('history',[]) if a.resume else []
best=saved.get('best_validation_cer',999) if a.resume else 999
start=time.perf_counter()
for step in range(start_step+1,a.steps+1):
    batch=[]
    for k in range(12):
        u=random.random()
        if u<a.synthetic_fraction:im,txt=random.choice(synthetic)
        elif u<a.synthetic_fraction+a.word_fraction:
            im,txt=concatenate_words(random.choices(wordbank,k=random.choice([1,2,3,4,5,6])),gap=random.randint(1,7),preserve_height=a.word_geometry=='preserve')
        else:im,txt=random.choice(samples)
        batch.append((augment(im),txt))
    x,y,xlen,ylen,txt=collate(batch);model.train()
    if a.freeze_bn:
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
    opt.zero_grad();logp=model(x.to(a.device),xlen)
    loss=loss_fn(logp.cpu(),y,xlen,ylen);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    if step%25==0:print(json.dumps({'step':step,'loss':loss.item(),'seconds':time.perf_counter()-start}),flush=True)
    if step%50==0 or step==a.steps:
        model.eval();results=[]
        with torch.no_grad():
            for r in dev:
                chunks=split_columns(Image.open(ROOT/'data'/r['image']),a.layout);x,_,lengths,_,_=collate([(image_tensor(preprocess(im),a.height),'') for im in chunks])
                pred=' '.join(t.strip() for t in decode(model(x.to(a.device),lengths),lengths) if t.strip())
                results.append({'id':r['id'],'reference':r['text'],'prediction':pred,**score(r['text'],pred)})
        n=sum(r['reference_characters'] for r in results);errors=sum(r['errors'] for r in results);cer=errors/n
        record={'step':step,'loss':loss.item(),'validation_cer':cer,'errors':errors,'characters':n,'seconds':time.perf_counter()-start,'lines':results};history.append(record)
        (out/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='lines'}),flush=True)
        checkpoint={'model':{k:v.cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'vocab':VOCAB,'step':step,'validation_cer':cer,'best_validation_cer':min(best,cer),'history':history,'training_config':config,'height':a.height,'packed':a.packed,'layout':a.layout,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'annotation_sha256':hashlib.sha256(annotations).hexdigest()}
        if a.device=='mps':checkpoint['mps_rng']=torch.mps.get_rng_state()
        if a.device.startswith('cuda'):checkpoint['cuda_rng']=torch.cuda.get_rng_state_all()
        torch.save(checkpoint,out/'last.pt')
        if cer<best:best=cer;torch.save(checkpoint,out/'best.pt')
