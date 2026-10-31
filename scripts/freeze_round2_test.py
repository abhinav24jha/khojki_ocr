"""Image/chart transcriptions, frozen before round-two test predictions are inspected.
Non-expert labels; ambiguous glyph alternatives are recorded, not silently resolved.
"""
from pathlib import Path
import json,hashlib,datetime
from write_annotations import encode
ROOT=Path(__file__).resolve().parents[1]
specs={
'p19r-06':'pa.ii cha.e : ta ba ka.aa : bha aa : ba.i sa ta.aa ra : SPACE 14',
'p19r-07':'ta ba : ja.u ra ta bha : ja.u ga ta : a pa.aa ra : SPACE 15',
'p19r-08':'sa kha ba : sa.aa ma.e : ra ca.e aa : sa.an sa.aa ra : SPACE 16',
'p19r-09':'va.e ddda : ma.ii ha.e : a ga.o ca ra : va.e aa ra.aa : SPACE 17',
'p19r-10':'ta.i sa ka.aa : pa.i : ha pa.i : sa ka la : pa sa.aa ra.aa : SPACE 18',
}
# Alternatives are a sensitivity analysis; primary references remain fixed.
alts={
'p19r-06':[('bha','ma')],
'p19r-07':[('ja.u ra','ga.u ra'),('ta bha','ta ma')],
'p19r-08':[('sa kha ba','sa sa ba'),('ra ca.e','ra cha.e')],
}
protocol=json.loads((ROOT/'data/protocol.json').read_text())
lookup={r['id']:r for r in protocol['lines']}
rows=[]
for id,spec in specs.items():
    row=lookup[id].copy();row.update(split='test',text=encode(spec),annotation_tokens=spec,
      annotation_method='Non-expert AI assistant visual transcription against Unicode chart; not copied from model output',
      uncertain_readings=[{'primary_tokens':x,'alternative_tokens':y,'alternative_text':encode(spec.replace(x,y)), 'reason':'Historical printed glyph shape differs from modern font; expert adjudication needed'} for x,y in alts.get(id,[])])
    rows.append(row)
out=ROOT/'data/round2/test.json'
if out.exists():raise RuntimeError('Frozen references exist; record an explicit revision instead')
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(ROOT/'data/round2/test-freeze.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'lines':len(rows),'characters':sum(len(r['text']) for r in rows),'status':'References frozen before inspecting round2 or prior baseline predictions for these selected lines. Prior whole-page baseline inference included these images but its unscored text was not inspected. Same book/page as development; not an independent-book benchmark.','annotation_limitation':'Non-expert references with explicit alternative readings; primary CER and sensitivity must both be reported.'},indent=2))
print('Frozen',len(rows),'lines',sum(len(r['text']) for r in rows),'code points')
