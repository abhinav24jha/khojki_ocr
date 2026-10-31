"""Matched bug-fix ablation and training-statistics pilots, offline preparation."""
from pathlib import Path
import json,zipfile,hashlib,ast
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cloud'
def cell(kind,text):
 x={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
 if kind=='code':x.update(execution_count=None,outputs=[]);ast.parse(text)
 return x
members={}
for folder in ['src','assets','tests']:
 for p in (ROOT/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):members[str(p.relative_to(ROOT))]=p.read_bytes()
for name in ['finetune.py','evaluate.py','evaluate_pages.py','select_round4_model.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
files=['data/annotations.json','data/annotations-v2-frozen.json','data/round2/train-dev.json','data/round2/test.json','data/round2/test-freeze.json','data/round3/test.json','data/round3/test-freeze.json','models/round3-selected.pt','experiments/precision-packed/best.pt','experiments/long-control/best.pt']
for name in files:members[name]=(ROOT/name).read_bytes()
for page in [17,18,19,20]:
 name=f'data/pages/p{page:02d}.png';members[name]=(ROOT/name).read_bytes()
for file in ['data/annotations.json','data/round2/test.json','data/round3/test.json']:
 for r in json.loads((ROOT/file).read_text()):members['data/'+r['image']]=(ROOT/'data'/r['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-round4.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess,shutil,datetime
BUNDLE=Path('/content/khojki-round4.zip');assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless','pytest','pypdf','pypdfium2'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round2/train-dev.json';EXP=ROOT/'experiments';RES=ROOT/'results';RES.mkdir(exist_ok=True)
def run(args,log,timeout=900):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-1300:]);p.check_returncode()
run(['-m','pytest',ROOT/'tests','-q'],RES/'round4-tests.log',120)
'''
train='''CASES={'r4-control':[],'r4-word-red':['--word-fraction','.4','--word-geometry','preserve','--legacy-rgb-word-padding'],'r4-word-white':['--word-fraction','.4','--word-geometry','preserve'],'r4-freezebn':['--freeze-bn'],'r4-word-white-freezebn':['--word-fraction','.4','--word-geometry','preserve','--freeze-bn'],'r4-freezernn':['--freeze-rnn']}
def train(name,steps):
 folder=EXP/name;folder.mkdir(exist_ok=True);local=folder/'last.pt';resume=local.exists();checkpoint=local if resume else EXP/'precision-packed/best.pt'
 if resume and torch.load(local,map_location='cpu',weights_only=False)['step']>=steps:print('Already complete',name,steps);return
 run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--annotations',ANN,'--steps',steps,'--lr','0.00005','--seed','503','--layout','robust','--bold-replay','--packed','--device','cuda','--output',name,*CASES[name],*(['--resume'] if resume else [])],EXP/(name+f'-{steps}.log'))
for name in CASES:train(name,600)
run([ROOT/'scripts/select_round4_model.py','--phase','pilot'],EXP/'round4-pilot.log')
report=json.loads((EXP/'round4-pilot-selection.json').read_text());eligible=sorted([r for r in report['experiments'] if r['name'] in CASES and r['errors']<=16],key=lambda r:r['errors'])[:2]
print('Extension rule: at most two pilots with <=16/405 edits; chosen',[r['name'] for r in eligible])
for row in eligible:train(row['name'],3000)
run([ROOT/'scripts/select_round4_model.py','--phase','final'],EXP/'round4-final.log')
# These already-viewed test sets are regression diagnostics, not new blind estimates.
for old in [2,3]:
 run([ROOT/'scripts/evaluate_pages.py','--checkpoint',ROOT/'models/round4-selected.pt','--annotations',ROOT/f'data/round{old}/test.json','--split','test','--device','cuda','--output',RES/f'round4-regression-r{old}'],RES/f'round4-regression-r{old}.log')
'''
export='''from IPython.display import display,HTML
import base64
EXPORTS=[]
for name in [*CASES,'round4-selection']:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  paths=list((EXP/name).glob('*'))+list(EXP.glob(name+'-*.log')) if name!='round4-selection' else list(RES.rglob('*'))+list(EXP.glob('round4-*'))+[ROOT/'models/round4-selected.pt']
  for p in paths:
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_padding_and_normalization_audit.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: padding bug and normalization audit\nSix matched 600-step pilots with unchanged labels. Isolate RGB red-border bug, freeze batch-normalization statistics, and freeze recurrent weights. Extend at most two promising pilots to 3000. Selection uses original 405-character development set. Previously viewed test sets are explicitly diagnostic. Save hashes and resumable checkpoints before disconnecting. Keep >=75 compute units.\n'),cell('code',setup),cell('code',train),cell('code',export)]}
(OUT/'Khojki_padding_and_normalization_audit.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1));print(bundle.stat().st_size,digest)
