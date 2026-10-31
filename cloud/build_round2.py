"""Prepare controlled short experiments; no local model execution."""
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
for name in ['data/round2/train-dev.json','data/round2/protocol.json','models/printed-khojki.pt']:
    members[name]=(ROOT/name).read_bytes()
rows=json.loads((ROOT/'data/round2/train-dev.json').read_text())
for row in rows:members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-round2.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for n,v in members.items():z.writestr('khojki-ocr/'+n,v)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''# Upload khojki-round2.zip using Files panel after connecting T4.
from pathlib import Path
import sys,os,json,time,zipfile,hashlib,subprocess
BUNDLE=Path('/content/khojki-round2.zip')
assert BUNDLE.exists(),'Upload paired ZIP through Files panel.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for n,h in json.loads((ROOT/'cloud-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless'],check=True,timeout=180)
os.environ['PYTHONPATH']=str(ROOT/'src')
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
ANN=ROOT/'data/round2/train-dev.json'
assert all(r['split']!='test' for r in json.loads(ANN.read_text()))
MODEL=ROOT/'models/printed-khojki.pt'
EXP=ROOT/'experiments';EXP.mkdir(exist_ok=True)
def run(args,log,timeout=300):
    with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    print(log.name,log.read_text()[-3500:]);p.check_returncode()
print('Original ten test lines are now development. Fresh test lines absent.')
'''
pilot='''# Paired 300-step pilots. Each process has a 5-minute hard timeout.
if not (EXP/'round2-baseline.json').exists():
    run([ROOT/'scripts/evaluate.py','--checkpoint',MODEL,'--annotations',ANN,'--split','dev','--layout','robust','--device','cuda','--output',EXP/'round2-baseline.json'],EXP/'round2-baseline.log',120)
configs=[('round2-control',[]),('round2-words',['--word-fraction','0.4']),('round2-bold',['--word-fraction','0.4','--bold-replay'])]
for name,extra in configs:
    if not (EXP/name/'history.json').exists():
        run([ROOT/'scripts/finetune.py','--checkpoint',MODEL,'--annotations',ANN,'--steps','300','--lr','0.0001','--seed','137','--layout','robust','--device','cuda','--output',name,*extra],EXP/(name+'.log'))
    else:print('Skipping saved experiment',name)
summary={'baseline':json.loads((EXP/'round2-baseline.json').read_text())['summary']}
for name,_ in configs:
    h=json.loads((EXP/name/'history.json').read_text());best=min(h,key=lambda x:x['errors'])
    summary[name]={k:v for k,v in best.items() if k!='lines'}
(EXP/'round2-summary.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary,indent=2))
'''
export='''# Checkpoints and metrics, divided into bounded archives for reliable transfer.
from IPython.display import display,HTML
import base64
EXPORTS=[]
for name,_ in configs:
    target=Path('/content')/(name+'-results.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in (EXP/name).glob('*'):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        z.write(EXP/(name+'.log'),'experiments/'+name+'.log')
        for p in EXP.glob('round2-*.json'):z.write(p,str(p.relative_to(ROOT)))
    h=hashlib.sha256(target.read_bytes()).hexdigest()
    display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
    EXPORTS.append({'name':target.name,'bytes':target.stat().st_size,'sha256':h})
print('EXPORT_MANIFEST',json.dumps(EXPORTS))
# Verify all local archives, then disconnect and delete runtime immediately.
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_round2_controlled_pilots.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki: controlled second-round pilots\nThree 300-step trials compare real-word recombination and bold-font replay against a matched control. Original test lines are explicitly development now; fresh test images/labels are excluded. Prepare notebook before connecting. Save all outputs and terminate runtime after each batch. Minimum remaining balance: 75 units.\n'),cell('code',setup),cell('code',pilot),cell('code',export)]}
(OUT/'Khojki_round2_controlled_pilots.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Prepared round2 notebook and bundle:',bundle.stat().st_size,'bytes',digest)
