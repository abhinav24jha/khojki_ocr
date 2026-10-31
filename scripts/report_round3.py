"""Build the final comparison and uncertainty analysis from saved predictions only."""
from pathlib import Path
import json,html,csv,itertools,collections,unicodedata
from khojki_ocr.metrics import score,alignment
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
final=json.loads((OUT/'round3-test/evaluation.json').read_text())
baseline=json.loads((OUT/'round3-baseline-test/evaluation.json').read_text())
refs={r['id']:r for r in json.loads((ROOT/'data/round3/test.json').read_text())}
rows=[];min_errors=max_errors=0;space_edits=0;confusions=collections.Counter();variant_counts=[]
for line in final['lines']:
 ref=refs[line['id']];variants={ref['text']}
 # Independent alternatives compose, so sensitivity covers both uncertain glyphs.
 specs={ref['annotation_tokens']}
 for alt in ref.get('uncertain_readings',[]):specs|={s.replace(alt['primary_tokens'],alt['alternative_tokens']) for s in list(specs)}
 # Convert named tokens locally, without importing rendering/model libraries.
 names=['a','aa','i','u','e','ai','o','au','ka','kha','ga','gga','gha','nga','ca','cha','ja','jja',None,'nya','tta','ttha','dda','ddha','nna','ta','tha','da','ddda','dha','na','pa','pha','ba','bba','bha','ma','ya','ra','la','va','sa','ha','lla']
 base={n:chr(0x11200+i) for i,n in enumerate(names) if n};signs={n:chr(0x1122c+i) for i,n in enumerate(['aa','i','ii','u','e','ai','o','au','an','vir','nu','sha'])}
 def encode(spec):
  value=''
  for token in spec.split():
   if token==':':value+='\U0001123a'
   elif token=='SPACE':value+=' '
   elif token.isdigit():value+=''.join(chr(0xae6+int(c)) for c in token)
   else:
    parts=token.split('.');value+=base[parts[0]]+''.join(signs[x] for x in parts[1:])
  return value
 variants|={encode(s) for s in specs}
 measurements=[score(v,line['automatic_prediction']) for v in variants]
 scores=[x['errors'] for x in measurements]
 variant_counts.append({(x['errors'],x['reference_characters']) for x in measurements})
 min_errors+=min(scores);max_errors+=max(scores)
 for op,r,q,_,_ in alignment(line['reference'],line['automatic_prediction']):
  if op!='equal':
   confusions[(op,r,q)]+=1
   if r==' ' or q==' ':space_edits+=1
 rows.append(line|{'image':ref['image'],'annotation_tokens':ref['annotation_tokens'],'uncertain_readings':ref.get('uncertain_readings',[]),'sensitivity_min_errors':min(scores),'sensitivity_max_errors':max(scores)})
n=final['summary']['characters']
totals={(0,0)}
for alternatives in variant_counts:totals={(e+a,n+b) for e,n in totals for a,b in alternatives}
min_cer=min(e/n for e,n in totals);max_cer=max(e/n for e,n in totals)
report={'primary':final['summary'],'baseline_same_fresh_lines':baseline['summary'],'reference_sensitivity':{'characters':n,'min_errors':min_errors,'max_errors':max_errors,'min_cer':min_cer,'max_cer':max_cer,'variant_character_counts':sorted({n for e,n in totals}),'interpretation':'Range over predeclared non-expert alternative glyph readings, not a statistical confidence interval or a corrected primary score.'},'stage_audit':{'missed_selected_lines':final['summary']['missed_lines'],'column_space_edits':space_edits,'other_character_edits':final['summary']['automatic_errors']-space_edits,'invalid_inventory_characters':final['summary']['invalid_inventory_characters'],'utf8_failures':final['summary']['utf8_failures'],'net_crop_error_difference':final['summary']['net_crop_error_difference'],'interpretation':'Operational attribution, not an identifiable causal decomposition: compare checked crops with automatic crops; encoding counts concern mechanical mapping/storage, not correctness of historical glyph identity.'},'confusions':[{'operation':op,'reference':r,'prediction':q,'count':count,'reference_name':unicodedata.name(r,'') if r else '', 'prediction_name':unicodedata.name(q,'') if q else ''} for (op,r,q),count in confusions.most_common()],'lines':rows}
(OUT/'round3-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (OUT/'round3-transcriptions.tsv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['id','reference','automatic_prediction','errors','characters','CER','uncertainty_min_errors','uncertainty_max_errors'])
 for r in rows:w.writerow([r['id'],r['reference'],r['automatic_prediction'],r['automatic']['errors'],r['automatic']['reference_characters'],r['automatic']['cer'],r['sensitivity_min_errors'],r['sensitivity_max_errors']])
parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Khojki OCR: fresh test comparison</title><style>@font-face{font-family:Khojki;src:url(../assets/NotoSansKhojki-Regular.ttf)}body{font:16px/1.55 system-ui;max-width:1080px;margin:32px auto;padding:0 20px;color:#20323d;background:#f7f6f1}article{background:white;border:1px solid #d8dfe3;padding:22px;margin:24px 0}.k{font:30px/1.8 Khojki,serif;overflow-wrap:anywhere}img{max-width:100%}mark{background:#ffe0d8;color:#8a291f}code{font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer}</style><h1>Fresh test: source, reference and OCR</h1>''']
parts.append(f"<p>Primary automatic-page CER: <strong>{100*final['summary']['automatic_cer']:.2f}%</strong> ({final['summary']['automatic_errors']}/{n} edits). Five same-book lines; non-expert chart transcriptions frozen before prediction inspection. This is not a handwriting or collection-wide accuracy estimate.</p>")
parts.append(f"<p>Predeclared reference alternatives give {100*min_cer:.2f}%–{100*max_cer:.2f}% CER. The primary references below remain unchanged. Highlighted OCR characters are substitutions/insertions; deletions appear in the edit details.</p>")
for r in rows:
 pred=[];diff=[]
 for op,a,b,_,_ in alignment(r['reference'],r['automatic_prediction']):
  if op=='equal':pred.append(html.escape(b))
  else:
   if b:pred.append('<mark>'+html.escape(b)+'</mark>')
   diff.append({'operation':op,'reference':a,'prediction':b,'reference_code':f'U+{ord(a):04X}' if a else None,'prediction_code':f'U+{ord(b):04X}' if b else None})
 parts.append(f'''<article><h2>{r['id']} — {r['automatic']['errors']} / {r['automatic']['reference_characters']} edits</h2><img src="../data/{html.escape(r['image'])}" alt="Source scan {r['id']}"><p>Frozen primary reference</p><div class="k">{html.escape(r['reference'])}</div><p>Automatic-page OCR</p><div class="k">{''.join(pred)}</div><details><summary>Edits and reference uncertainty</summary><pre>{html.escape(json.dumps(diff,ensure_ascii=False,indent=2))}</pre><p>{html.escape(json.dumps(r['uncertain_readings'],ensure_ascii=False))}</p><code>{html.escape(r['annotation_tokens'])}</code></details></article>''')
parts.append('</html>');(OUT/'round3-comparison.html').write_text(''.join(parts))
print(json.dumps({k:v for k,v in report.items() if k not in ['lines','confusions']},indent=2))
