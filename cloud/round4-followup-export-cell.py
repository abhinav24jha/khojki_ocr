EXPORTS_FOLLOWUP=[]
for name in ['r4-low-original','r4-low-corrected','r4-low-white','round4-followup']:
 target=Path('/content')/(name+'-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  paths=list((EXP/name).glob('*'))+list(EXP.glob(name+'-*.log')) if name!='round4-followup' else list(RES.rglob('*'))+list(EXP.glob('round4-followup*'))+[ROOT/'models/round4-final.pt']
  for p in paths:
   if p.is_file():z.write(p,str(p.relative_to(ROOT)))
 display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download '+name+' results</a>'))
 EXPORTS_FOLLOWUP.append({'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
print('FOLLOWUP_MANIFEST',json.dumps(EXPORTS_FOLLOWUP))
