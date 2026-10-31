"""Low-rate matched label audit and development-only optical robustness selection.

Run on cloud GPU. Freeze selection before reading the new page-21 predictions.
"""
from pathlib import Path
import json,subprocess,sys,hashlib,datetime,math
import torch
from PIL import Image,ImageFilter
from khojki_ocr.recognize import Recognizer,preprocess,split_columns
from khojki_ocr.model import collate,image_tensor
from khojki_ocr.metrics import score
from khojki_ocr.decoding import beam_decode
from khojki_ocr.unicode import plausible_prefix
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments';torch.set_num_threads(2)
CASES={'r4-low-original':('data/round2/train-dev.json',[]),'r4-low-corrected':('data/round4/train-dev-corrected.json',[]),'r4-low-white':('data/round4/train-dev-corrected.json',['--word-fraction','.1','--word-geometry','preserve'])}
def train(name,steps):
 ann,extra=CASES[name];folder=EXP/name;last=folder/'last.pt';resume=last.exists()
 if resume and torch.load(last,map_location='cpu',weights_only=False)['step']>=steps:return
 args=[sys.executable,str(ROOT/'scripts/finetune.py'),'--checkpoint',str(last if resume else EXP/'precision-packed/best.pt'),'--annotations',str(ROOT/ann),'--steps',str(steps),'--lr','0.00001','--seed','607','--layout','robust','--bold-replay','--packed','--device','cuda','--output',name,*extra,*(['--resume'] if resume else [])]
 with (EXP/(name+f'-{steps}.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
 print('TRAINED',name,steps,flush=True)
rows=[r for r in json.loads((ROOT/'data/round2/train-dev.json').read_text()) if r['split']=='dev'];records=[]
def evaluate(name,paths,views=((1.,0.),)):
 rec=Recognizer(paths,'cuda',beam_width=10,constrained=True);lines=[]
 for row in rows:
  chunks=split_columns(Image.open(ROOT/'data'/row['image']),rec.layout);all_probs=[]
  with torch.no_grad():
   for strength,blur in views:
    ims=[preprocess(im) for im in chunks]
    if blur:ims=[im.filter(ImageFilter.GaussianBlur(blur)) for im in ims]
    x,_,lengths,_,_=collate([(image_tensor(im,rec.height),'') for im in ims]);x=(x*strength).clamp(0,1).cuda()
    all_probs.extend(m(x,lengths) for m in rec.models)
   probs=torch.logsumexp(torch.stack(all_probs),dim=0)-math.log(len(all_probs))
   pred=' '.join(beam_decode(probs[:int(lengths[i]),i].cpu().numpy(),10,prefix_valid=plausible_prefix).strip() for i in range(len(chunks))).strip()
  lines.append({'id':row['id'],'reference':row['text'],'prediction':pred,**score(row['text'],pred)})
 record={'name':name,'paths':[str(p.relative_to(ROOT)) for p in paths],'views':views,'errors':sum(r['errors'] for r in lines),'characters':405,'lines':lines};records.append(record);print('DEV',name,record['errors'],flush=True);return record
for name in CASES:train(name,1000)
for name in CASES:evaluate(name,[EXP/name/'best.pt'])
eligible=sorted([r for r in records if r['errors']<=15],key=lambda r:r['errors'])[:2]
(EXP/'round4-followup-pilot.json').write_text(json.dumps({'experiments':records,'extend_to_5000':[r['name'] for r in eligible],'test_used':False},ensure_ascii=False,indent=2))
for r in eligible:train(r['name'],5000)
records=[]
evaluate('incumbent',[ROOT/'models/round3-selected.pt'])
evaluate('round4-selected',[ROOT/'models/round4-selected.pt'])
for name in CASES:evaluate(name,[EXP/name/'best.pt'])
top=min([r for r in records if r['name'] in CASES],key=lambda r:r['errors'])
evaluate('low-mixed',[EXP/'long-control/best.pt',ROOT/top['paths'][0]])
# Small fixed optical perturbation audit, not a wide decoder search.
base=min(records,key=lambda r:(r['errors'],r['name']!='incumbent'))
for label,views in [('weak',((.85,0.),)),('strong',((1.15,0.),)),('blur',((1.,.4),)),('ink-average',((.85,0.),(1.,0.),(1.15,0.))),('blur-average',((1.,0.),(1.,.4)))]:
 evaluate(base['name']+'-'+label,[ROOT/p for p in base['paths']],views)
best=min(records,key=lambda r:(r['errors'],len(r['views']),r['name']!='incumbent'))
saved=[torch.load(ROOT/p,map_location='cpu',weights_only=False) for p in best['paths']]
compact={'vocab':saved[0]['vocab'],'height':48,'layout':'robust','beam_width':10,'constrained':True,'views':best['views']}
members=[]
for s in saved:members.extend(s.get('ensemble_states') or [{'model':s['model'],'packed':s.get('packed',False)}])
if len(members)==1:compact.update(members[0])
else:compact['ensemble_states']=members
out=ROOT/'models/round4-final.pt';torch.save(compact,out)
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'experiments':records,'selected':{k:v for k,v in best.items() if k!='lines'},'test_used':False,'checkpoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'policy':'Select on original 16-line/405-character development set only. Source-audited correction changes training only. New page21 is not read by this script. Historical test sets are already-viewed regression data.'}
(EXP/'round4-followup-selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('SELECTED',json.dumps(report['selected']),flush=True)
