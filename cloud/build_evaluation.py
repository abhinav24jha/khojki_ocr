"""Build the frozen evaluation notebook offline; no local ML execution."""
from pathlib import Path
import hashlib,json,zipfile,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'cloud'
def cell(kind,text):
    item={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':item.update(execution_count=None,outputs=[])
    return item

freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selected_checkpoint':'experiments/curve-10/best.pt',
        'checkpoint_sha256':hashlib.sha256((ROOT/'experiments/curve-10/best.pt').read_bytes()).hexdigest(),
        'selection':'Lowest development CER: 10/155. Cloud pilot 13/155, beam 11/155, ensemble 12/155 rejected.',
        'decoding':'contrast preprocessing, split large horizontal gaps, greedy CTC',
        'test_used_for_selection':False,
        'test_annotations_sha256':hashlib.sha256((ROOT/'data/annotations-v2-frozen.json').read_bytes()).hexdigest(),
        'pipeline_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}}
(ROOT/'experiments/final-freeze.json').write_text(json.dumps(freeze,indent=2))
members={}
for folder in ['src','assets','tests']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in str(p):members[str(p.relative_to(ROOT))]=p.read_bytes()
for name in ['evaluate.py','evaluate_pages.py','stress_eval.py']:
    members['scripts/'+name]=(ROOT/'scripts'/name).read_bytes()
for name in ['annotations.json','annotations-v2-frozen.json']:
    members['data/'+name]=(ROOT/'data'/name).read_bytes()
rows=json.loads((ROOT/'data/annotations.json').read_text())
for row in rows:members['data/'+row['image']]=(ROOT/'data'/row['image']).read_bytes()
for page in [4,17,18,19,20,23]:
    name=f'data/pages/p{page:02d}.png';members[name]=(ROOT/name).read_bytes()
for name in ['pyproject.toml','experiments/final-freeze.json','experiments/curve-10/best.pt']:
    members[name]=(ROOT/name).read_bytes()
members['cloud-manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},indent=2).encode()
bundle=OUT/'khojki-frozen-evaluation.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in members.items():z.writestr('khojki-ocr/'+name,data)
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
setup=f'''# Upload paired ZIP with Colab Files panel, then run this cell.
import os,sys,json,hashlib,zipfile,subprocess,time,platform
from pathlib import Path
START=time.monotonic()
BUNDLE=Path('/content/khojki-frozen-evaluation.zip')
assert BUNDLE.exists(), 'Upload the paired ZIP using the Files panel first.'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='{digest}'
ROOT=Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z:z.extractall('/content')
for name,digest in json.loads((ROOT/'cloud-manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
os.environ['PYTHONPATH']=str(ROOT/'src')
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(ROOT)+'[test]'],check=True,timeout=180)
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0),torch.__version__)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def run(args,name,timeout=180):
    log=OUT/(name+'.log')
    with log.open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout,cwd=ROOT)
    print(name,log.read_text()[-4000:]);p.check_returncode()
saved=torch.load(ROOT/'experiments/curve-10/best.pt',map_location='cpu',weights_only=False)
(ROOT/'models').mkdir(exist_ok=True)
MODEL=ROOT/'models/printed-khojki.pt'
torch.save({{'model':saved['model'],'vocab':saved['vocab'],'height':48,'source_sha256':json.loads((ROOT/'experiments/final-freeze.json').read_text())['checkpoint_sha256']}},MODEL)
del saved
(OUT/'environment.json').write_text(json.dumps({{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}},indent=2))
run(['-m','pytest','-q'],'tests')
'''
evaluate='''# Frozen model; no training or checkpoint selection in this notebook.
if not (OUT/'evaluation-complete.json').exists():
    run([ROOT/'scripts/evaluate.py','--checkpoint',MODEL,'--split','dev','--device','cuda','--output',OUT/'dev-crops.json'],'dev-crops')
    assert json.loads((OUT/'dev-crops.json').read_text())['summary']['errors']==10
    run([ROOT/'scripts/stress_eval.py','--checkpoint',MODEL,'--device','cuda','--output',OUT/'dev-stress.json'],'dev-stress')
    run([ROOT/'scripts/evaluate_pages.py','--checkpoint',MODEL,'--split','dev','--device','cuda','--output',OUT/'dev-pages'],'dev-pages')
    # Held-out test is evaluated once with the preselected model and settings.
    run([ROOT/'scripts/evaluate_pages.py','--checkpoint',MODEL,'--split','test','--device','cuda','--output',OUT/'test-pages'],'test-pages')
    # CPU deployment benchmark, same frozen predictions; not a model-selection loop.
    run(['-m','khojki_ocr',ROOT/'data/pages/p19.png','--checkpoint',MODEL,'--device','cpu','--output',OUT/'cpu-demo'],'cli-cpu')
    run(['-m','khojki_ocr',ROOT/'data/pages/p23.png','--checkpoint',MODEL,'--device','cuda','--output',OUT/'unscored-print'],'unscored-print')
    run(['-m','khojki_ocr',ROOT/'data/pages/p04.png','--checkpoint',MODEL,'--device','cuda','--output',OUT/'unscored-handwriting'],'unscored-handwriting')
    (OUT/'evaluation-complete.json').write_text(json.dumps({'session_seconds':time.monotonic()-START,'training_performed':False}))
else:print('Saved evaluation exists; skipping repeated evaluation.')
print('TEST',json.loads((OUT/'test-pages/evaluation.json').read_text())['summary'])
'''
export='''# Export via visible notebook link; local copy must be hash-verified before disconnecting.
import base64
from IPython.display import HTML,display
RESULT=Path('/content/khojki-evaluation-results.zip')
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(OUT/'requirements-colab.txt').open('w'),check=True)
with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['results','models']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
encoded=base64.b64encode(RESULT.read_bytes()).decode('ascii')
display(HTML('<a id="evaluation-export" download="khojki-evaluation-results.zip" href="data:application/zip;base64,'+encoded+'">Download frozen evaluation results and model</a>'))
print('RESULT_SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest(),'BYTES',RESULT.stat().st_size)
del encoded
# Verify downloaded file, then Runtime > Disconnect and delete runtime.
'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_frozen_evaluation.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[cell('markdown','# Khojki OCR: frozen evaluation\nPrepared offline. No training. Upload paired ZIP using Files panel before running setup. Original model selected on development CER; held-out references fixed before predictions. Download the output archive and terminate the runtime. Minimum account reserve: 75 units.\n'),cell('code',setup),cell('code',evaluate),cell('code',export)]}
(OUT/'Khojki_frozen_evaluation.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Prepared',bundle.stat().st_size,'bytes;',digest)
