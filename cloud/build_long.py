"""Prepare larger controlled runs offline after promising 300-step pilot."""
from pathlib import Path
import json,zipfile,hashlib,ast
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cloud'
def cell(kind,text):
 x={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
 if kind=='code':x.update(execution_count=None,outputs=[]);ast.parse(text)
 return x
members={}
for folder in ['src','assets']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):members[str(p.relative_to(ROOT))]=p.read_bytes()
for name in ['finetune.py','evaluate.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
for name in ['data/round2/train-dev.json','models/printed-khojki.pt','experiments/round2-control/last.pt','experiments/round2-control/best.pt']:
 members[name]=(ROOT/name).read_bytes()
for row in json.loads((ROOT/'data/round2/train-dev.json').read_text()):members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-long.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess,shutil
BUNDLE=Path('/content/khojki-long.zip')
assert BUNDLE.exists(),'Upload prepared bundle via Files panel.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round2/train-dev.json';MODEL=ROOT/'models/printed-khojki.pt';EXP=ROOT/'experiments'
assert all(r['split']!='test' for r in json.loads(ANN.read_text()))
def run(args,log,timeout=600):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-1600:]);p.check_returncode()
print('Fresh held-out test excluded. Three bounded 3000-step runs; checkpoints every 50 steps.')
'''
train='''configs=[
 ('long-control',EXP/'round2-control/last.pt',['--resume']),
 ('long-bold',MODEL,['--bold-replay']),
 ('long-word10',MODEL,['--bold-replay','--word-fraction','0.1'])]
for name,checkpoint,extra in configs:
 folder=EXP/name;folder.mkdir(exist_ok=True)
 if name=='long-control' and not (folder/'best.pt').exists():shutil.copy2(EXP/'round2-control/best.pt',folder/'best.pt')
 if not (folder/'history.json').exists():
  run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--annotations',ANN,'--steps','3000','--lr','0.0001','--seed','137','--layout','robust','--device','cuda','--output',name,*extra],EXP/(name+'.log'))
 else:print('Saved results already exist; no automatic retraining:',name)
summary={}
for name,_,_ in configs:
 h=json.loads((EXP/name/'history.json').read_text());best=min(h,key=lambda x:x['errors'])
 summary[name]={k:v for k,v in best.items() if k!='lines'}
(EXP/'long-summary.json').write_text(json.dumps(summary,indent=2))
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
  for p in [EXP/(name+'.log'),EXP/'long-summary.json']:z.write(p,str(p.relative_to(ROOT)))
 h=hashlib.sha256(target.read_bytes()).hexdigest()
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':h})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_longer_controlled_runs.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: longer controlled experiments\nThe 300-step control improved development CER from 7.65% to 6.91%. Extend it to 3000 steps and compare bold replay alone and a reduced 10% word mixture. Each run is bounded to ten minutes; GPU training is expected to take roughly two minutes. Fresh test lines remain excluded. Save, verify and disconnect. Leave at least 75 compute units.\n'),cell('code',setup),cell('code',train),cell('code',export)]}
(OUT/'Khojki_longer_controlled_runs.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(bundle.stat().st_size,digest)
