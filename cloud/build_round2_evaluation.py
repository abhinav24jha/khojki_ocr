"""Prepare frozen-model tests and deployment evaluation before connecting."""
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
for name in ['evaluate.py','evaluate_pages.py','stress_eval.py','report_round2.py']:members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
for name in ['data/annotations.json','data/annotations-v2-frozen.json','data/round2/train-dev.json','data/round2/test.json','data/round2/test-freeze.json','models/round2-selected.pt','experiments/round2-selection.json']:
 members[name]=(ROOT/name).read_bytes()
members['models/baseline-printed-khojki.pt']=(ROOT/'models/baseline-printed-khojki.pt').read_bytes()
for page in [4,17,18,19,20,23]:
 name=f'data/pages/p{page:02d}.png';members[name]=(ROOT/name).read_bytes()
for file in ['data/annotations.json','data/round2/test.json']:
 for row in json.loads((ROOT/file).read_text()):members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-round2-evaluation.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess,platform,datetime
BUNDLE=Path('/content/khojki-round2-evaluation.zip')
assert BUNDLE.exists(),'Upload prepared bundle via Files panel.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','scipy','opencv-python-headless','pytest','pypdf','pypdfium2'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available()
MODEL=ROOT/'models/round2-selected.pt';BASELINE=ROOT/'models/baseline-printed-khojki.pt';RES=ROOT/'results';RES.mkdir(exist_ok=True)
ANN=ROOT/'data/round2/train-dev.json';TEST=ROOT/'data/round2/test.json'
freeze=json.loads((ROOT/'data/round2/test-freeze.json').read_text())
assert hashlib.sha256(TEST.read_bytes()).hexdigest()==freeze['sha256']
selection=json.loads((ROOT/'experiments/round2-selection.json').read_text())
assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==selection['checkpoint_sha256']
def run(args,log,timeout=300):
 with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(log.name,log.read_text()[-2200:]);p.check_returncode()
(RES/'round2-environment.json').write_text(json.dumps({{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}},indent=2))
(RES/'round2-requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Frozen model and fresh references verified. No training or selection in this notebook.')
'''
evaluate='''run(['-m','pytest',ROOT/'tests','-q'],RES/'round2-tests.log',120)
for tag,model,ann,split in [('round2-dev',MODEL,ANN,'dev'),('round2-test',MODEL,TEST,'test'),('round2-baseline-test',BASELINE,TEST,'test')]:
 if not (RES/tag/'evaluation.json').exists():
  run([ROOT/'scripts/evaluate_pages.py','--checkpoint',model,'--annotations',ann,'--device','cuda','--split',split,'--output',RES/tag],RES/(tag+'.log'))
run([ROOT/'scripts/evaluate.py','--checkpoint',MODEL,'--annotations',ANN,'--split','train','--device','cuda','--output',RES/'round2-train.json'],RES/'round2-train.log')
run([ROOT/'scripts/stress_eval.py','--checkpoint',MODEL,'--annotations',ANN,'--device','cuda','--output',RES/'round2-stress.json'],RES/'round2-stress.log')
run(['-m','khojki_ocr',ROOT/'data/pages/p19.png','--checkpoint',MODEL,'--device','cpu','--output',RES/'round2-cpu-demo','--threads','2'],RES/'round2-cpu.log')
for page,name in [(4,'round2-unscored-handwriting'),(23,'round2-unscored-other-print')]:
 run(['-m','khojki_ocr',ROOT/f'data/pages/p{page:02d}.png','--checkpoint',MODEL,'--device','cuda','--output',RES/name],RES/(name+'.log'))
cpu=(RES/'round2-cpu-demo/page-001/transcription.txt').read_text();gpu=(RES/'round2-test/p19/transcription.txt').read_text()
run([ROOT/'scripts/report_round2.py'],RES/'round2-report.log')
complete={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'test_reference_sha256':freeze['sha256'],'test_summary':json.loads((RES/'round2-test/evaluation.json').read_text())['summary'],'cpu_gpu_page19_text_equal':cpu==gpu,'cpu_page':json.loads((RES/'round2-cpu-demo/run.json').read_text()),'test_used_for_selection':False}
(RES/'round2-evaluation-complete.json').write_text(json.dumps(complete,indent=2))
print('EVALUATION_COMPLETE',json.dumps(complete,indent=2))
'''
evaluate+='\n'+(OUT/'verify_packaging_cell.py').read_text()
export='''from IPython.display import display,HTML
import base64
TARGET=Path('/content/khojki-round2-evaluation-results.zip')
with zipfile.ZipFile(TARGET,'w',zipfile.ZIP_DEFLATED) as z:
 for p in RES.rglob('*'):
  if p.is_file():z.write(p,str(p.relative_to(ROOT)))
EXPORTS=[{'name':TARGET.name,'bytes':TARGET.stat().st_size,'sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest()}]
display(HTML('<a download="'+TARGET.name+'" href="data:application/zip;base64,'+base64.b64encode(TARGET.read_bytes()).decode()+'">Download verified evaluation results</a>'))
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_round2_frozen_evaluation.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: frozen deployment evaluation\nNo training or model selection. Verify the selected model and five fresh reference lines, run regression tests and automatic-page CER, compare the historical baseline on the same fresh lines, measure CPU timing and robustness, and save unscored other-print/handwriting probes. All outputs must be saved and hash-verified before disconnecting.\n'),cell('code',setup),cell('code',evaluate),cell('code',export)]}
(OUT/'Khojki_round2_frozen_evaluation.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(bundle.stat().st_size,digest)
