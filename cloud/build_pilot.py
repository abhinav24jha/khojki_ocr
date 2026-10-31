"""Prepare a bounded Colab notebook and data bundle without loading any model."""
from pathlib import Path
import json, hashlib, zipfile

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT/'cloud'
bundle = dest/'khojki-cloud-pilot.zip'
members = {}
for folder in ['src', 'assets']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in str(p): members[str(p.relative_to(ROOT))] = p.read_bytes()
for name in ['evaluate.py', 'finetune.py']:
    members['scripts/'+name] = (ROOT/'scripts'/name).read_bytes()
rows = [r for r in json.loads((ROOT/'data/annotations.json').read_text()) if r['split'] != 'test']
members['data/annotations.json'] = json.dumps(rows, ensure_ascii=False, indent=2).encode()
for row in rows: members['data/'+row['image']] = (ROOT/'data'/row['image']).read_bytes()
for name in ['curve-10', 'synthetic-v1']:
    members[f'experiments/{name}/best.pt'] = (ROOT/f'experiments/{name}/best.pt').read_bytes()
manifest = {k:hashlib.sha256(v).hexdigest() for k,v in members.items()}
members['cloud-manifest.json'] = json.dumps(manifest, indent=2).encode()
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for name, data in members.items(): z.writestr('khojki-ocr/'+name,data)
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()

def cell(kind, source):
    c = {'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind == 'code': c.update(execution_count=None,outputs=[])
    return c

setup = f'''# Connect only after this notebook is uploaded. Select T4 GPU.
import sys, os, json, time, hashlib, zipfile, subprocess
from pathlib import Path
from google.colab import files
START = time.monotonic()
BUNDLE = Path('/content/khojki-cloud-pilot.zip')
if not BUNDLE.exists():
    uploaded = files.upload()
    del uploaded
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == '{digest}'
ROOT = Path('/content/khojki-ocr')
with zipfile.ZipFile(BUNDLE) as z: z.extractall('/content')
for name, expected in json.loads((ROOT/'cloud-manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
subprocess.run([sys.executable,'-m','pip','install','-q','freetype-py','uharfbuzz','scipy','opencv-python-headless'],check=True,timeout=180)
os.environ['PYTHONPATH'] = str(ROOT/'src')
import torch
assert torch.cuda.is_available(), 'Stop: GPU not connected.'
print('GPU:', torch.cuda.get_device_name(0))
assert all(r['split'] != 'test' for r in json.loads((ROOT/'data/annotations.json').read_text()))
print('Verified 21 train/dev lines. No test data present.')
'''
run = '''# Fixed pilot: one 100-step warm start, no automatic sweep or long run.
OUT = ROOT/'experiments/cloud-pilot-100'
def command(args, log, timeout):
    with open(log,'w') as f:
        p = subprocess.run([sys.executable,*map(str,args)], stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    print(Path(log).read_text()[-10000:])
    p.check_returncode()
smoke = ROOT/'experiments/cloud-smoke-dev.json'
if not smoke.exists():
    command([ROOT/'scripts/evaluate.py','--checkpoint',ROOT/'experiments/curve-10/best.pt','--split','dev','--device','cuda','--output',smoke],ROOT/'experiments/cloud-smoke.log',120)
baseline = json.loads(smoke.read_text())['summary']
print('BASELINE', baseline)
assert baseline['errors'] == 10 and baseline['characters'] == 155, 'Unexpected smoke result: investigate before training.'
if not (OUT/'history.json').exists():
    command([ROOT/'scripts/finetune.py','--checkpoint',ROOT/'experiments/curve-10/best.pt','--steps','100','--device','cuda','--lr','0.00005','--seed','71','--output','cloud-pilot-100'],ROOT/'experiments/cloud-pilot.log',600)
else:
    print('Saved pilot exists; skipping training. Use a new named experiment for new settings.')
history = json.loads((OUT/'history.json').read_text())
best = min(history,key=lambda r:r['validation_cer'])
print('PILOT', {k:v for k,v in best.items() if k!='lines'})
print('SELECT', 'pilot' if best['errors'] < baseline['errors'] else 'original curve-10')
print('Cloud session seconds:', time.monotonic()-START)
'''
save = '''# Save model, optimizer, RNG, metrics and logs BEFORE disconnecting.
RESULT = Path('/content/khojki-pilot-results.zip')
with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in (ROOT/'experiments').rglob('*'):
        if p.is_file() and ('cloud-' in str(p.relative_to(ROOT))):
            z.write(p,str(p.relative_to(ROOT)))
print('RESULT', RESULT.name, RESULT.stat().st_size, hashlib.sha256(RESULT.read_bytes()).hexdigest())
files.download(str(RESULT))
# After verifying the download, use Runtime > Disconnect and delete runtime.
'''
nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Khojki_prepared_pilot.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU','gpuClass':'standard'},'cells':[
    cell('markdown','# Khojki OCR — bounded pilot\nPrepared offline before connecting. Upload the paired ZIP when prompted. One 100-step experiment; test set excluded. Starting balance 168.49 units; minimum reserve 75; pilot budget <=5 units. Download results and terminate the runtime after this batch. Existing notebooks are untouched.\n'),
    cell('code',setup),cell('code',run),cell('code',save)]}
(dest/'Khojki_prepared_pilot.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(json.dumps({'zip':str(bundle),'sha256':digest,'bytes':bundle.stat().st_size,'notebook':str(dest/'Khojki_prepared_pilot.ipynb')}))
