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
assert (ROOT/'models/printed-khojki.pt').read_bytes()==(ROOT/'models/round3-selected.pt').read_bytes();checks.append('default model equals frozen deployment model')
print(json.dumps({'passed':checks},indent=2))
