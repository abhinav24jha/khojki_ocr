"""Final frozen model evaluation; no further selection from these test outputs."""
from pathlib import Path
import json,subprocess,sys,hashlib
from PIL import Image
import torch
from khojki_ocr.recognize import Recognizer
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
selection=json.loads((ROOT/'experiments/round4-followup-selection.json').read_text());rec=Recognizer(ROOT/'models/round4-final.pt','cuda')
chosen=next(r for r in selection['experiments'] if r['name']==selection['selected']['name']);dev=[r for r in json.loads((ROOT/'data/round2/train-dev.json').read_text()) if r['split']=='dev']
assert [rec(Image.open(ROOT/'data'/r['image'])) for r in dev]==[r['prediction'] for r in chosen['lines']]
selection['serialized_prediction_match']=True;(ROOT/'experiments/round4-followup-selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
freeze=json.loads((ROOT/'data/round4/test-freeze.json').read_text());assert hashlib.sha256((ROOT/'data/round4/test.json').read_bytes()).hexdigest()==freeze['annotation_sha256']
for path,digest in freeze['image_sha256'].items():assert hashlib.sha256((ROOT/'data'/path).read_bytes()).hexdigest()==digest
for name,checkpoint,rounds in [('final','models/round4-final.pt',[2,3,4]),('incumbent','models/round3-selected.pt',[4])]:
 for old in rounds:
  args=[sys.executable,str(ROOT/'scripts/evaluate_pages.py'),'--checkpoint',str(ROOT/checkpoint),'--annotations',str(ROOT/f'data/round{old}/test.json'),'--split','test','--device','cuda','--output',str(ROOT/f'results/round4-{name}-r{old}')]
  subprocess.run(args,check=True,timeout=180)
# Explicitly confirm unchanged default preprocessing semantics on development.
print('Serialized model matches selection; fresh page21 scored only after selection was saved.',flush=True)
subprocess.run([sys.executable,'-m','pytest',str(ROOT/'tests'),'-q'],check=True,timeout=120)
