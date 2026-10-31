"""Prepare an isolated training-reference correction, keeping validation frozen."""
from pathlib import Path
import json,hashlib,datetime
from write_annotations import encode
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/round4'
rows=json.loads((ROOT/'data/round2/train-dev.json').read_text())
for row in rows:
 if row['id']=='p17r-10':
  original=row['text'];row['annotation_tokens']=row['annotation_tokens'].replace('ta ba.i :','ta ba :');row['text']=encode(row['annotation_tokens']);row['reference_revision']={'original':original,'evidence':'data/round4/proposed-reference-revisions.json','status':'Non-expert source-audited training-only correction'}
(OUT/'train-dev-corrected.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('Training-only correction; validation unchanged:',sum(r['split']=='dev' for r in rows))
