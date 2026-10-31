"""Create a private exercise package, excluding caches and bulky resumable weights."""
from pathlib import Path
import json,hashlib,zipfile,datetime,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
archives=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((ROOT/'cloud').glob('*.zip'))]
(ROOT/'cloud/archive-index.json').write_text(json.dumps({'note':'Full input/result archives preserved in the working project; excluded from compact submission ZIP. SHA-256 records identify exact resumable experiments.','archives':archives},indent=2))
files=[]
for p in ROOT.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(ROOT);parts=rel.parts
 if any(x in ['.git','__pycache__','.pytest_cache'] or x.endswith('.egg-info') for x in parts):continue
 if parts[0] in ['src','scripts','tests','assets'] and p.suffix in ['.py','.ttf','.txt']:files.append(p)
 elif parts[0]=='data' and (p.suffix=='.json' or (parts[1] in ['pages','lines'] and p.suffix=='.png')):files.append(p)
 elif parts[0]=='results' and p.suffix in ['.json','.txt','.tsv','.csv','.html','.png','.ttf','.log','.md']:files.append(p)
 elif parts[0]=='models' and p.suffix in ['.pt','.json']:files.append(p)
 elif parts[0]=='experiments' and p.suffix in ['.json','.log']:files.append(p)
 elif parts[0]=='cloud' and p.suffix in ['.json','.py','.ipynb']:files.append(p)
 elif len(parts)==1 and p.name in ['README.md','RESULTS.md','ARCHITECTURE.md','MODEL_CARD.md','pyproject.toml','.gitignore','technical-retro.pdf']:files.append(p)
# Every HTML local image/font reference in the package must resolve to a packaged file.
chosen={p.resolve() for p in files}
for p in files:
 if p.suffix!='.html':continue
 for ref in re.findall(r'(?:src="|url\()([^"\)]+)',p.read_text()):
  if ref.startswith(('data:','http:','https:')):continue
  assert (p.parent/ref).resolve() in chosen,(p,ref)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Private take-home exercise package; no public redistribution of scan material.','files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}}
(ROOT/'submission-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'submission-manifest.json')
output=ROOT.parent/'khojki-ocr-submission.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):z.write(p,'khojki-ocr/'+str(p.relative_to(ROOT)))
with zipfile.ZipFile(output) as z:
 assert z.testzip() is None
 for n,v in manifest['files'].items():
  data=z.read('khojki-ocr/'+n);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'],n
receipt={'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'files':len(files),'integrity_verified':True}
(ROOT.parent/'khojki-ocr-submission-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
