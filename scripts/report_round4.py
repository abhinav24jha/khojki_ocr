"""Report new-model effects separately from reference-audit effects; no inference."""
from pathlib import Path
import json,itertools,html,collections,hashlib
from khojki_ocr.metrics import score,alignment
from write_annotations import encode
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'
def load(p):return json.loads((ROOT/p).read_text())
def marked(reference,prediction):
 out=[]
 for op,old,new,_,_ in alignment(reference,prediction):
  if not new:continue
  value=html.escape(new)
  out.append(value if op=='equal' else '<span style="background:#ffe1c4;border-bottom:2px solid #b75b12">'+value+'</span>')
 return ''.join(out)
selection=load('experiments/round4-followup-selection.json');fresh=load('results/round4-final-r4/evaluation.json');old=load('results/round4-incumbent-r4/evaluation.json');references=load('data/round4/test.json')
variants=[]
for texts in itertools.product(*[[r['text']]+r.get('alternate_references',[]) for r in references]):
 vals=[score(t,r['automatic_prediction']) for t,r in zip(texts,fresh['lines'])];n=sum(r['reference_characters'] for r in vals);e=sum(r['errors'] for r in vals);variants.append({'errors':e,'characters':n,'cer':e/n})
revisions=load('data/round4/proposed-reference-revisions.json')['changes'];audit=[]
for model,path in [('round4','results/round4-final-r3/evaluation.json'),('incumbent','results/round3-test/evaluation.json')]:
 evaluation=load(path);rows=[]
 for r in evaluation['lines']:
  text=r['reference'];applied=[]
  for change in revisions:
   if change['id']!=r['id'] or change['scope']!='historical-test-reference-only':continue
   before=encode(change['original_tokens']);after=encode(change['replacement_tokens']);assert text.count(before)==1,(r['id'],before,text)
   text=text.replace(before,after);applied.append(change)
  rows.append({'id':r['id'],'original_reference':r['reference'],'proposed_reference':text,'prediction':r['automatic_prediction'],'original':score(r['reference'],r['automatic_prediction']),'proposed':score(text,r['automatic_prediction']),'changes':applied})
 n=sum(r['proposed']['reference_characters'] for r in rows);e=sum(r['proposed']['errors'] for r in rows)
 audit.append({'model':model,'original_summary':evaluation['summary'],'proposed_errors':e,'proposed_characters':n,'proposed_cer':e/n,'lines':rows})
ops=collections.Counter();space=0
for row in fresh['lines']:
 for op,r,q,_,_ in alignment(row['reference'],row['automatic_prediction']):
  if op!='equal':ops[(op,r,q)]+=1;space+=r==' ' or q==' '
report={'selected':selection['selected'],'fresh':fresh['summary'],'incumbent_fresh':old['summary'],'predeclared_reference_sensitivity':{'minimum':min(variants,key=lambda r:r['cer']),'maximum':max(variants,key=lambda r:r['cer']),'interpretation':'Range across source-declared alternatives, not a confidence interval or adjudicated truth.'},'reference_audit':audit,'audit_interpretation':'Post-hoc non-expert source review; original frozen test labels unchanged. Proposed-label score changes are not model improvements. Report both models against the same reference version.','space_involving_edits':space,'other_edits':fresh['summary']['automatic_errors']-space,'exact_lines':sum(r['automatic']['errors']==0 for r in fresh['lines']),'confusions':[{'operation':op,'reference':r,'prediction':q,'count':count} for (op,r,q),count in ops.most_common()],'checkpoint_sha256':hashlib.sha256((ROOT/'models/round4-final.pt').read_bytes()).hexdigest()}
(RES/'round4-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
style='''<style>@font-face{font-family:Khojki;src:url(../assets/NotoSansKhojki-Regular.ttf)}body{max-width:1050px;margin:40px auto;padding:0 20px;background:#f4f3ee;color:#243540;font:16px/1.55 system-ui}h1,h2{color:#174c52}article{background:white;padding:24px;margin:20px 0;border:1px solid #d7dfde;border-radius:10px}.k{font-family:Khojki;font-size:30px;overflow-wrap:anywhere}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #cfdad8}.muted{color:#52636c}</style>'''
parts=['<!doctype html><meta charset="utf-8"><title>Khojki OCR — round-four evidence</title>'+style,'<h1>Round four: bugs, labels and measured OCR</h1>',f"<p>Selected configuration: <b>{html.escape(report['selected']['name'])}</b>. Development: {report['selected']['errors']}/405 edits. Fresh page 21: <b>{fresh['summary']['automatic_errors']}/120 ({fresh['summary']['automatic_cer']:.2%} CER)</b>; previous model: {old['summary']['automatic_errors']}/120 ({old['summary']['automatic_cer']:.2%}).</p>",'<p class="muted">Five fixed lines, non-expert chart transcriptions, two declared ambiguities. Related printed material; this is not collection-wide or handwriting accuracy. Model selection finished before these predictions were read.</p>',f"<p>Declared alternative-reference sensitivity: {report['predeclared_reference_sensitivity']['minimum']['cer']:.2%}–{report['predeclared_reference_sensitivity']['maximum']['cer']:.2%}. <a href='../data/round4/reference-audit.html'>Inspect the separate label audit</a>.</p>"]
oldby={r['id']:r for r in old['lines']}
for r in fresh['lines']:
 ref=next(x for x in references if x['id']==r['id']);parts.append(f"<article><h2>{r['id']} — {r['automatic']['errors']} edits; previously {oldby[r['id']]['automatic']['errors']}</h2><img src='../data/{ref['image']}' alt='Source line'><p>Frozen reference</p><p class='k'>{html.escape(r['reference'])}</p><p>Final OCR (highlighted substitutions/insertions)</p><p class='k'>{marked(r['reference'],r['automatic_prediction'])}</p><p>Previous OCR</p><p class='k'>{html.escape(oldby[r['id']]['automatic_prediction'])}</p><p class='muted'>{html.escape(ref['uncertainty'])}</p></article>")
parts.append('<h2>Controlled experiments — same 405-character development set</h2><table><tr><th>Experiment</th><th>Edits</th><th>CER</th></tr>')
for filename in ['experiments/round4-pilot-selection.json','experiments/round4-final-selection.json','experiments/round4-followup-selection.json']:
 data=load(filename);parts.append(f'<tr><th colspan="3">{html.escape(Path(filename).stem)}</th></tr>')
 for r in data['experiments']:parts.append(f"<tr><td>{html.escape(r['name'])}</td><td>{r['errors']}</td><td>{r['errors']/405:.2%}</td></tr>")
parts.append('</table><article><h2>A repeated visual ambiguity</h2><p>Independent E and PA plus vowel I have similar modern-font outlines, but are distinct Unicode sequences. Three edit alignments pair an inserted PA with E-to-I substitution; a fourth E location is also confused. E has no positive real training example. These observations were made after evaluation and do not alter frozen references or scores.</p><img src="../data/round4/encoding-ambiguity.png" alt="Independent E versus PA plus I, and HA plus II versus HA TA plus II"><p>This rendering comparison is a diagnostic, not a licence to merge the encodings or relabel the scan from predictions.</p></article><h2>Reference changes are a separate effect</h2><p>The original historical test remains frozen. The following proposed corrections came from matching source images and chart forms; they need expert adjudication.</p><table><tr><th>Model</th><th>Original reference CER</th><th>Proposed revised reference CER</th></tr>')
for row in audit:parts.append(f"<tr><td>{row['model']}</td><td>{row['original_summary']['automatic_errors']}/{row['original_summary']['characters']} ({row['original_summary']['automatic_cer']:.2%})</td><td>{row['proposed_errors']}/{row['proposed_characters']} ({row['proposed_cer']:.2%})</td></tr>")
parts.append('</table><p><a href="round4-report.json">Machine-readable report</a> · <a href="round4-final-r4/p21/review.html">Review the full page output</a></p>')
(RES/'round4-comparison.html').write_text('\n'.join(parts));print(json.dumps({k:v for k,v in report.items() if k not in ['reference_audit','confusions']},indent=2))
