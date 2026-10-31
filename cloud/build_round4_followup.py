from pathlib import Path
import json,zipfile,hashlib,ast
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cloud'
files=['scripts/finetune.py','scripts/round4_followup.py','scripts/evaluate_round4_final.py','src/khojki_ocr/recognize.py','data/round4/train-dev-corrected.json','data/round4/test.json','data/round4/test-freeze.json','data/pages/p21.png']+[f'data/lines/p21a-{i:02d}.png' for i in range(1,6)]
bundle=OUT/'khojki-round4-followup.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(ROOT/n,n)
sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
code=f'''# Offline-prepared follow-up: matched low learning rate and source-audited label correction.
bundle=Path('/content/khojki-round4-followup.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='{sha}'
with zipfile.ZipFile(bundle) as z:z.extractall(ROOT)
run([ROOT/'scripts/round4_followup.py'],EXP/'round4-followup.log',2400)
run([ROOT/'scripts/evaluate_round4_final.py'],RES/'round4-final-evaluation.log',900)
'''
export='''EXPORTS_FOLLOWUP=[]
for name in ['r4-low-original','r4-low-corrected','r4-low-white','round4-followup']:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  paths=list((EXP/name).glob('*'))+list(EXP.glob(name+'-*.log')) if name!='round4-followup' else list(RES.rglob('*'))+list(EXP.glob('round4-followup*'))+[ROOT/'models/round4-final.pt']
  for p in paths:
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS_FOLLOWUP.append({'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
print('FOLLOWUP_MANIFEST',json.dumps(EXPORTS_FOLLOWUP))
'''
for name,s in [('round4-followup-cell.py',code),('round4-followup-export-cell.py',export)]:ast.parse(s);(OUT/name).write_text(s)
p=OUT/'Khojki_padding_and_normalization_audit.ipynb';nb=json.loads(p.read_text())
for s in [code,export]:nb['cells'].append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'outputs':[],'execution_count':None})
(OUT/'Khojki_round4_complete.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1));print(bundle.stat().st_size,sha)
