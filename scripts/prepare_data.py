from pathlib import Path
import argparse,json,hashlib,time
from dataclasses import asdict
from pypdf import PdfReader
from khojki_ocr.segment import detect_lines,overlay

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
source=a.source
reader=PdfReader(source)
data=ROOT/'data'
for d in ['pages','lines','overlays']: (data/d).mkdir(exist_ok=True)
manifest=[]
for page in [4,10,17,18,19,20,22,23,24,29,42]:
    im=max(reader.pages[page-1].images,key=lambda x:x.image.width*x.image.height).image.convert('RGB')
    name=f'p{page:02d}'
    im.save(data/'pages'/f'{name}.png')
    t=time.perf_counter(); lines,roi=detect_lines(im,name); elapsed=time.perf_counter()-t
    overlay(im,lines).save(data/'overlays'/f'{name}.png')
    for line in lines: im.crop(line.box).save(data/'lines'/f'{line.id}.png')
    manifest.append({'page':page,'image':f'pages/{name}.png','size':im.size,'roi':roi,'segmentation_seconds':elapsed,'lines':[asdict(x) for x in lines]})
    print(page,len(lines),roi,round(elapsed,4))
(data/'manifest.json').write_text(json.dumps(manifest,indent=2))
(data/'provenance.json').write_text(json.dumps({'source_filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf_pages':len(reader.pages),'usage':'Assessment exercise only; do not publicly redistribute source scans.'},indent=2))
