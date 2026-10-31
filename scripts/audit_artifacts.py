"""Lightweight integrity audit; imports no model runtime."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[1]
checks=[]
for round in [2,3]:
 p=ROOT/f'data/round{round}/test.json';meta=json.loads((p.parent/'test-freeze.json').read_text());rows=json.loads(p.read_text());development=json.loads((p.parent/'train-dev.json').read_text())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 assert not {r['id'] for r in rows}&{r['id'] for r in development}
 assert sum(len(r['text']) for r in rows)==meta['characters']
 checks.append(f'round{round}: frozen hash, count and split disjointness')
for p in (ROOT/'src').rglob('*.py'):ast.parse(p.read_text())
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text())
checks.append('Python syntax')
for name in ['round2-selection','round3-final-selection']:
 report=json.loads((ROOT/f'experiments/{name}.json').read_text());model=ROOT/('models/round2-selected.pt' if name.startswith('round2') else 'models/round3-selected.pt')
 assert hashlib.sha256(model.read_bytes()).hexdigest()==report['checkpoint_sha256'];checks.append(name+': deployment hash')
for tag in ['round2-test','round3-test']:
 e=json.loads((ROOT/f'results/{tag}/evaluation.json').read_text());assert sum(x['automatic']['errors'] for x in e['lines'])==e['summary']['automatic_errors']
 assert sum(x['automatic']['reference_characters'] for x in e['lines'])==e['summary']['characters'];checks.append(tag+': aggregate edit counts')
final=ROOT/'models/round4-final.pt'
assert (ROOT/'models/printed-khojki.pt').read_bytes()==final.read_bytes();checks.append('default model equals round4 frozen deployment')
assert json.loads((ROOT/'models/model-card.json').read_text())['sha256']==hashlib.sha256(final.read_bytes()).hexdigest()
r=json.loads((ROOT/'experiments/round4-followup-selection.json').read_text());assert hashlib.sha256(final.read_bytes()).hexdigest()==r['checkpoint_sha256'];assert r['serialized_prediction_match'] and r['test_used'] is False
p=ROOT/'data/round4/test.json';meta=json.loads((p.parent/'test-freeze.json').read_text());fresh=json.loads(p.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['annotation_sha256'];assert sum(len(r['text']) for r in fresh)==120
train_dev=json.loads((ROOT/'data/round4/train-dev-corrected.json').read_text());assert not {r['id'] for r in fresh}&{r['id'] for r in train_dev}
for name,digest in meta['image_sha256'].items():assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest
original=json.loads((ROOT/'data/round2/train-dev.json').read_text());assert [r for r in original if r['split']=='dev']==[r for r in train_dev if r['split']=='dev'];assert [a['id'] for a,b in zip(original,train_dev) if a['text']!=b['text']]==['p17r-10']
checks.append('round4 frozen images/references, train-test separation, training-only correction')
for tag in ['round4-final-r2','round4-final-r3','round4-final-r4','round4-incumbent-r4']:
 e=json.loads((ROOT/f'results/{tag}/evaluation.json').read_text());assert sum(x['automatic']['errors'] for x in e['lines'])==e['summary']['automatic_errors'];assert sum(x['automatic']['reference_characters'] for x in e['lines'])==e['summary']['characters']
checks.append('round4 evaluation aggregates')
verify=json.loads((ROOT/'results/round4-verification.json').read_text());assert verify['cpu_gpu_page_text_equal'];assert verify['checkpoint_sha256']==hashlib.sha256(final.read_bytes()).hexdigest();checks.append('CPU/GPU parity and deployment hash')
shutdown=json.loads((ROOT/'cloud/round4-shutdown.json').read_text());assert shutdown['remaining_units']>=75 and shutdown['active_sessions']==0 and shutdown['units_per_hour']==0;checks.append('credit reserve and runtime shutdown')

print(json.dumps({'passed':checks},indent=2))
