"""Prepare targeted follow-up and development-only model freeze."""
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
for name in ['finetune.py','evaluate.py','select_round2_model.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
for name in ['data/round2/train-dev.json','data/annotations.json','data/annotations-v2-frozen.json','data/pages/p20.png','experiments/long-control/best.pt','experiments/long-bold/best.pt','experiments/arch-packed48/last.pt']:
 members[name]=(ROOT/name).read_bytes()
for row in json.loads((ROOT/'data/round2/train-dev.json').read_text()):members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-precision.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess,shutil
BUNDLE=Path('/content/khojki-precision.zip')
assert BUNDLE.exists(),'Upload prepared bundle through Files panel.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless','pytest','pypdf','pypdfium2'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round2/train-dev.json';EXP=ROOT/'experiments';MODEL=EXP/'long-control/best.pt'
def run(args,log,timeout=600):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-1600:]);p.check_returncode()
run(['-m','pytest',ROOT/'tests','-q','-k','not round2_references'],EXP/'precision-tests.log',120)
'''
train='''# Exact resume for the packed 600-step pilot; matched word-geometry trials.
def train(name,checkpoint,steps,extra,resume=False):
 folder=EXP/name;folder.mkdir(exist_ok=True)
 local=folder/'last.pt'
 if local.exists():
  state=torch.load(local,map_location='cpu',weights_only=False)
  if state['step']>=steps:print('Already completed',name,state['step']);return
  checkpoint=local;resume=True
 if resume and not (folder/'best.pt').exists():shutil.copy2(checkpoint,folder/'best.pt')
 run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--annotations',ANN,'--steps',steps,'--lr','0.00005','--seed','211','--layout','robust','--bold-replay','--device','cuda','--output',name,*extra,*(['--resume'] if resume else [])],EXP/(name+'.log'))
train('precision-packed',EXP/'arch-packed48/last.pt',3000,['--packed'],True)
train('geometry-legacy',MODEL,600,['--word-fraction','0.4'])
train('geometry-preserve',MODEL,600,['--word-fraction','0.4','--word-geometry','preserve'])
h=json.loads((EXP/'geometry-preserve/history.json').read_text())
if min(r['errors'] for r in h)<18:
 print('Geometry pilot beats prior best; extending to 3000 steps.')
 train('geometry-preserve',EXP/'geometry-preserve/last.pt',3000,['--word-fraction','0.4','--word-geometry','preserve'],True)
else:print('Geometry pilot does not beat 18 errors; no larger run.')
run([ROOT/'scripts/select_round2_model.py','--device','cuda'],EXP/'round2-selection.log',300)
'''
export='''from IPython.display import display,HTML
import base64
EXPORTS=[]
for name in ['precision-packed','geometry-legacy','geometry-preserve','round2-selection']:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  if name=='round2-selection':
   paths=[ROOT/'models/round2-selected.pt',EXP/'round2-selection.json',EXP/'round2-selection.log',EXP/'precision-tests.log']
  else:paths=list((EXP/name).glob('*'))+[EXP/(name+'.log')]
  for p in paths:
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_targeted_followup_and_selection.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: targeted follow-up and model freeze\nExtend the nearly matching packed-recurrence pilot; isolate the vertical shrinking in word augmentation with matched legacy/preserved-geometry trials. Extend corrected geometry only if its pilot beats 18/405 development edits. Compare greedy, beam and structural-filter decoding on the two best training models, plus their ensemble. Fresh test data is excluded. Verify exports before termination; keep 75 compute units.\n'),cell('code',setup),cell('code',train),cell('code',export)]}
(OUT/'Khojki_targeted_followup_and_selection.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(bundle.stat().st_size,digest)
