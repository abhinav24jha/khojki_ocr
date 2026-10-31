"""Prepare length-aware/height/data ablations without running local ML."""
from pathlib import Path
import json,zipfile,hashlib,ast
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cloud'
def cell(kind,text):
 x={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
 if kind=='code':x.update(execution_count=None,outputs=[]);ast.parse(text)
 return x
summary=json.loads((ROOT/'experiments/long-summary.json').read_text())
winner=min(summary,key=lambda n:summary[n]['errors'])
members={}
for folder in ['src','assets','tests']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):members[str(p.relative_to(ROOT))]=p.read_bytes()
for name in ['finetune.py','evaluate.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
for name in ['data/round2/train-dev.json','data/round2/expanded-train-dev.json','data/annotations.json','data/annotations-v2-frozen.json','data/pages/p20.png']:
 members[name]=(ROOT/name).read_bytes()
members['models/architecture-start.pt']=(ROOT/f'experiments/{winner}/best.pt').read_bytes()
for row in json.loads((ROOT/'data/round2/expanded-train-dev.json').read_text()):members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-architecture.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess
BUNDLE=Path('/content/khojki-architecture.zip')
assert BUNDLE.exists(),'Upload prepared bundle via Files panel.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless','pytest','pypdf','pypdfium2'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round2/train-dev.json';EXTRA=ROOT/'data/round2/expanded-train-dev.json';MODEL=ROOT/'models/architecture-start.pt';EXP=ROOT/'experiments';EXP.mkdir(exist_ok=True)
assert all(r['split']!='test' for r in json.loads(ANN.read_text()))
def run(args,log,timeout=600):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-2000:]);p.check_returncode()
run(['-m','pytest',ROOT/'tests','-q','-k','not round2_references'],EXP/'architecture-tests.log',120)
print('Warm-start: {winner}, dev errors: {summary[winner]['errors']}. Fresh test excluded.')
'''
train='''configs=[
 ('arch-control',ANN,[]),
 ('arch-packed48',ANN,['--packed']),
 ('arch-height64',ANN,['--height','64']),
 ('arch-packed64',ANN,['--height','64','--packed']),
 ('arch-extra-data',EXTRA,[])]
for name,ann,extra in configs:
 if not (EXP/name/'history.json').exists():
  run([ROOT/'scripts/finetune.py','--checkpoint',MODEL,'--annotations',ann,'--steps','600','--lr','0.00005','--seed','211','--layout','robust','--bold-replay','--device','cuda','--output',name,*extra],EXP/(name+'.log'))
 else:print('Saved results already exist; not retraining:',name)
summary={}
for name,_,_ in configs:
 h=json.loads((EXP/name/'history.json').read_text());best=min(h,key=lambda x:x['errors'])
 summary[name]={k:v for k,v in best.items() if k!='lines'}
(EXP/'architecture-summary.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary,indent=2))
'''
export='''from IPython.display import display,HTML
import base64
EXPORTS=[]
for name,_,_ in configs:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in (EXP/name).glob('*'):
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
  for p in [EXP/(name+'.log'),EXP/'architecture-summary.json',EXP/'architecture-tests.log']:z.write(p,str(p.relative_to(ROOT)))
 h=hashlib.sha256(target.read_bytes()).hexdigest()
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':h})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_architecture_data_ablations.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: architecture and data ablations\nFive controlled 600-step runs test length-aware recurrence, greater input height, their combination, and five additional real training lines. All start from the same development-selected model. Fresh test images/labels excluded. Verify all result archives before terminating runtime; minimum remaining balance 75 units.\n'),cell('code',setup),cell('code',train),cell('code',export)]}
(OUT/'Khojki_architecture_data_ablations.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(winner,bundle.stat().st_size,digest)
