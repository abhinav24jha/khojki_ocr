# Final deployment parity and CPU throughput, run after model selection/evaluation.
out=RES/'round4-cpu-demo'
run(['-m','khojki_ocr',ROOT/'data/pages/p21.png','--checkpoint',ROOT/'models/round4-final.pt','--output',out,'--device','cpu'],RES/'round4-cpu.log',180)
cpu=json.loads((out/'page-001/result.json').read_text());gpu=json.loads((RES/'round4-final-r4/p21/result.json').read_text());assert [r['text'] for r in cpu['lines']]==[r['text'] for r in gpu['lines']]
saved=torch.load(ROOT/'models/round4-final.pt',map_location='cpu',weights_only=False)
members=saved.get('ensemble_states') or [{'model':saved['model'],'packed':saved.get('packed',False)}]
verify={'cpu_gpu_page_text_equal':True,'cpu_page':{k:v for k,v in cpu.items() if k!='lines'},'cpu_run':json.loads((out/'run.json').read_text()),'members':len(members),'parameters':sum(sum(v.numel() for k,v in m['model'].items() if not any(s in k for s in ['running_mean','running_var','num_batches_tracked'])) for m in members),'views':saved.get('views',[[1.,0.]]),'checkpoint_sha256':hashlib.sha256((ROOT/'models/round4-final.pt').read_bytes()).hexdigest()}
(RES/'round4-verification.json').write_text(json.dumps(verify,indent=2));print('VERIFICATION',json.dumps(verify))
target=Path('/content/round4-verification-results.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in [*out.rglob('*'),RES/'round4-verification.json',RES/'round4-cpu.log']:
  if p.is_file():z.write(p,str(p.relative_to(ROOT)))
display(HTML('<a download="'+target.name+'" href="data:application/zip;base64,'+base64.b64encode(target.read_bytes()).decode()+'">Download round4 verification</a>'))
print('VERIFY_MANIFEST',json.dumps([{'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}]))
