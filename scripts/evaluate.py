from pathlib import Path
import argparse,json,time,collections
import torch
from PIL import Image
from khojki_ocr.recognize import Recognizer
from khojki_ocr.metrics import score,alignment

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,nargs='+');p.add_argument('--split',default='dev',choices=['train','dev','test','all']);p.add_argument('--preprocess',default='contrast',choices=['raw','contrast','binary']);p.add_argument('--no-columns',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--beam',type=int,default=None);p.add_argument('--output',required=True);p.add_argument('--annotations',default=str(ROOT/'data/annotations.json'));p.add_argument('--layout',choices=['legacy','robust']);p.add_argument('--constrained',action='store_true',default=None);a=p.parse_args()
torch.set_num_threads(4)
rows=json.loads(Path(a.annotations).read_text());rows=[r for r in rows if a.split=='all' or r['split']==a.split]
t=time.perf_counter();rec=Recognizer(a.checkpoint,a.device,a.preprocess,not a.no_columns,a.beam,a.layout,a.constrained);load=time.perf_counter()-t
results=[];confusions=collections.Counter()
for row in rows:
    im=Image.open(ROOT/'data'/row['image']);t=time.perf_counter();pred=rec(im);elapsed=time.perf_counter()-t
    s=score(row['text'],pred);results.append({'id':row['id'],'reference':row['text'],'prediction':pred,'seconds':elapsed,**s})
    for op,r,q,_,_ in alignment(row['text'],pred):
        if op!='equal':confusions[(op,r,q)]+=1
    print(json.dumps(results[-1],ensure_ascii=False),flush=True)
summary={'lines':len(rows),'characters':sum(r['reference_characters'] for r in results),'errors':sum(r['errors'] for r in results),'load_seconds':load,'inference_seconds':sum(r['seconds'] for r in results)}
summary['cer']=summary['errors']/summary['characters']
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'config':vars(a),'summary':summary,'lines':results,'confusions':[{'operation':op,'reference':r,'prediction':q,'count':n} for (op,r,q),n in confusions.most_common()]},ensure_ascii=False,indent=2))
print(json.dumps(summary),flush=True)
