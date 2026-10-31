"""Fixed-decoder development comparison; never reads test references."""
from pathlib import Path
import argparse,json,hashlib,datetime,time
import torch
from PIL import Image
from khojki_ocr.recognize import Recognizer
from khojki_ocr.metrics import score
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--phase',choices=['baseline','pilot','final'],default='final');a=p.parse_args();torch.set_num_threads(2)
rows=[r for r in json.loads((ROOT/'data/round3/train-dev.json').read_text()) if r['split']=='dev'];n=sum(len(r['text']) for r in rows)
cases=[('incumbent',[ROOT/'models/round2-selected.pt'])]
if a.phase!='baseline':
 cases += [(name,[ROOT/f'experiments/{name}/best.pt']) for name in ['round3-packed','round3-control']]
 cases += [('round3-ensemble',[ROOT/f'experiments/{name}/best.pt' for name in ['round3-packed','round3-control']])]
records=[]
for name,paths in cases:
 rec=Recognizer(paths,a.device,beam_width=10,constrained=True);start=time.perf_counter();lines=[]
 for r in rows:
  pred=rec(Image.open(ROOT/'data'/r['image']));lines.append({'id':r['id'],'reference':r['text'],'prediction':pred,**score(r['text'],pred)})
 record={'name':name,'paths':[str(x.relative_to(ROOT)) for x in paths],'errors':sum(x['errors'] for x in lines),'characters':n,'seconds':time.perf_counter()-start,'lines':lines};record['cer']=record['errors']/n;records.append(record)
 print(json.dumps({k:v for k,v in record.items() if k!='lines'}),flush=True);del rec
best=min(records,key=lambda x:(x['errors'],2 if x['name'] in ['incumbent','round3-ensemble'] else 1,x['name']!='incumbent'))
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':a.phase,'test_used':False,'selection_rule':'Fixed beam width 10 and structural prefix filter. Minimum development edits; then fewer members; then retain incumbent.','experiments':records,'selected':{k:v for k,v in best.items() if k!='lines'}}
if a.phase=='final':
 saved=[torch.load(ROOT/x,map_location='cpu',weights_only=False) for x in best['paths']]
 if best['name']=='incumbent':compact=saved[0]
 else:
  compact={'vocab':saved[0]['vocab'],'height':48,'layout':'robust','beam_width':10,'constrained':True}
  if len(saved)==1:compact.update(model=saved[0]['model'],packed=saved[0].get('packed',False))
  else:compact['ensemble_states']=[{'model':s['model'],'packed':s.get('packed',False)} for s in saved]
 out=ROOT/'models/round3-selected.pt';torch.save(compact,out)
 rec=Recognizer(out,a.device);assert [rec(Image.open(ROOT/'data'/r['image'])) for r in rows]==[r['prediction'] for r in best['lines']]
 report.update(checkpoint_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),serialized_prediction_match=True,source_checkpoints={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in best['paths']})
(ROOT/f'experiments/round3-{a.phase}-selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
