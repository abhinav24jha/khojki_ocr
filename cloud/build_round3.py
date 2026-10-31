"""Prepare bounded targeted-data pilots, conditional extensions and frozen evaluation."""
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
for name in ['finetune.py','evaluate.py','evaluate_pages.py','stress_eval.py','select_round3_model.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
files=['data/annotations.json','data/annotations-v2-frozen.json','data/round2/train-dev.json','data/round2/test.json','data/round2/test-freeze.json','data/round3/train-dev.json','data/round3/test.json','data/round3/test-freeze.json','data/round3/protocol.json','models/round2-selected.pt','models/baseline-printed-khojki.pt','experiments/precision-packed/best.pt','experiments/long-control/best.pt']
for name in files:members[name]=(ROOT/name).read_bytes()
for page in [17,18,19,20]:
 name=f'data/pages/p{page:02d}.png';members[name]=(ROOT/name).read_bytes()
for file in ['data/annotations.json','data/round2/test.json','data/round3/train-dev.json','data/round3/test.json']:
 for r in json.loads((ROOT/file).read_text()):members['data/'+r['image']]=(ROOT/'data'/r['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-round3.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess,shutil,datetime,platform
BUNDLE=Path('/content/khojki-round3.zip');assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless','pytest','pypdf','pypdfium2'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round3/train-dev.json';EXP=ROOT/'experiments';RES=ROOT/'results';RES.mkdir(exist_ok=True)
def run(args,log,timeout=600):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-1800:]);p.check_returncode()
run(['-m','pytest',ROOT/'tests','-q'],RES/'round3-tests.log',120)
(RES/'round3-environment.json').write_text(json.dumps({{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}},indent=2))
(RES/'round3-requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
'''
train='''# New training split: former test lines are explicitly promoted; old scores remain historical.
def train(name,checkpoint,steps,packed):
 folder=EXP/name;folder.mkdir(exist_ok=True);local=folder/'last.pt';resume=local.exists()
 if resume:
  state=torch.load(local,map_location='cpu',weights_only=False)
  if state['step']>=steps:print('Already complete',name,state['step']);return
  checkpoint=local
 run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--annotations',ANN,'--steps',steps,'--lr','0.00005','--seed','401','--layout','robust','--bold-replay','--device','cuda','--output',name,*(['--packed'] if packed else []),*(['--resume'] if resume else [])],EXP/(name+f'-{steps}.log'))
run([ROOT/'scripts/select_round3_model.py','--phase','baseline'],EXP/'round3-baseline.log')
train('round3-packed',EXP/'precision-packed/best.pt',500,True)
train('round3-control',EXP/'long-control/best.pt',500,False)
run([ROOT/'scripts/select_round3_model.py','--phase','pilot'],EXP/'round3-pilot.log')
pilot=json.loads((EXP/'round3-pilot-selection.json').read_text());inc=pilot['experiments'][0]['errors']
for name,packed in [('round3-packed',True),('round3-control',False)]:
 row=next(r for r in pilot['experiments'] if r['name']==name)
 if row['errors']<=inc+2:
  print('Promising pilot; extend to 3000',name,row['errors'],'incumbent',inc)
  train(name,EXP/name/'last.pt',3000,packed)
 else:print('No extension',name,row['errors'],'incumbent',inc)
run([ROOT/'scripts/select_round3_model.py','--phase','final'],EXP/'round3-final.log')
'''
evaluate='''# Selection is complete and immutable before opening fresh test references.
MODEL=ROOT/'models/round3-selected.pt';TEST=ROOT/'data/round3/test.json'
selection=json.loads((EXP/'round3-final-selection.json').read_text());assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==selection['checkpoint_sha256']
freeze=json.loads((ROOT/'data/round3/test-freeze.json').read_text());assert hashlib.sha256(TEST.read_bytes()).hexdigest()==freeze['sha256']
for tag,model,ann,split in [('round3-dev',MODEL,ANN,'dev'),('round3-test',MODEL,TEST,'test'),('round3-incumbent-test',ROOT/'models/round2-selected.pt',TEST,'test'),('round3-baseline-test',ROOT/'models/baseline-printed-khojki.pt',TEST,'test')]:
 if not (RES/tag/'evaluation.json').exists():run([ROOT/'scripts/evaluate_pages.py','--checkpoint',model,'--annotations',ann,'--device','cuda','--split',split,'--output',RES/tag],RES/(tag+'.log'))
run([ROOT/'scripts/evaluate.py','--checkpoint',MODEL,'--annotations',ANN,'--split','train','--device','cuda','--output',RES/'round3-train.json'],RES/'round3-train.log')
run([ROOT/'scripts/stress_eval.py','--checkpoint',MODEL,'--annotations',ANN,'--device','cuda','--output',RES/'round3-stress.json'],RES/'round3-stress.log')
run(['-m','khojki_ocr',ROOT/'data/pages/p20.png','--checkpoint',MODEL,'--device','cpu','--output',RES/'round3-cpu-demo','--threads','2'],RES/'round3-cpu.log')
complete={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':selection['checkpoint_sha256'],'test_reference_sha256':freeze['sha256'],'test_summary':json.loads((RES/'round3-test/evaluation.json').read_text())['summary'],'cpu_gpu_page20_text_equal':(RES/'round3-cpu-demo/page-001/transcription.txt').read_text()==(RES/'round3-test/p20/transcription.txt').read_text(),'cpu_page':json.loads((RES/'round3-cpu-demo/run.json').read_text()),'test_used_for_selection':False}
(RES/'round3-evaluation-complete.json').write_text(json.dumps(complete,indent=2));print('EVALUATION_COMPLETE',json.dumps(complete,indent=2))
'''
export='''from IPython.display import display,HTML
import base64
EXPORTS=[]
for name in ['round3-packed','round3-control','round3-evaluation']:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  paths=list((EXP/name).glob('*'))+list(EXP.glob(name+'-*.log')) if name!='round3-evaluation' else list(RES.rglob('*'))+list(EXP.glob('round3-*-selection.json'))+list(EXP.glob('round3-*.log'))+[ROOT/'models/round3-selected.pt']
  for p in paths:
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_targeted_real_data_round3.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: targeted real-data expansion\n20 training / 18 development lines. Matched 500-step pilots, conditional extension to 3000, fixed decoding and development-only selection. Five separately frozen lines are opened only after selection; same-book and nonexpert-label limitations remain. Save all three archives and verify SHA-256 before disconnecting. Reserve >=75 compute units.\n'),cell('code',setup),cell('code',train),cell('code',evaluate),cell('code',export)]}
(OUT/'Khojki_targeted_real_data_round3.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1));print(bundle.stat().st_size,digest)
