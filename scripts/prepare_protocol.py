"""Predeclared page splits; region boxes are operator annotations, not OCR labels."""
from pathlib import Path
from dataclasses import asdict
import json,datetime
from PIL import Image,ImageDraw,ImageFont
from khojki_ocr.segment import detect_lines,overlay

ROOT=Path(__file__).resolve().parents[1];data=ROOT/'data'
regions={17:[85,224,654,1084],18:[85,130,710,1110],19:[85,155,690,1125],20:[85,140,700,1115],23:[50,130,703,1100]}
records=[]
for p,region in regions.items():
    im=Image.open(data/'pages'/f'p{p:02d}.png')
    lines,_=detect_lines(im,f'p{p:02d}r',region=region)
    overlay(im,lines).save(data/'overlays'/f'p{p:02d}-region.png')
    for j,line in enumerate(lines):
        crop=im.crop(line.box);crop.save(data/'lines'/f'{line.id}.png')
        split='train' if p==17 else 'dev' if p in [18,23] else 'test'
        if p==20:split='dev' if j<3 else 'test'
        selected=(p==17) or (p==18 and j<3) or (p==19 and j<5) or (p==23 and j<2)
        if p==20:selected=j<8
        records.append({'id':line.id,'page':p,'image':f'lines/{line.id}.png','box':line.box,'region':region,'split':split,'selected_for_transcription':selected})
    # Enlarged, numbered crops for manual transcription from the chart.
    items=[x for x in records if x['page']==p and x['selected_for_transcription']]
    for start in range(0,len(items),5):
        sheet=Image.new('RGB',(1500,5*170),'white');d=ImageDraw.Draw(sheet)
        for j,row in enumerate(items[start:start+5]):
            crop=im.crop(row['box']);crop.thumbnail((1400,120));crop=crop.resize((min(1400,crop.width*2),min(120,crop.height*2)))
            sheet.paste(crop,(50,j*170+25));d.text((20,j*170+5),row['id'],fill='black')
        sheet.save(data/'reference'/f'annotate-p{p:02d}-{start}.png')
    print(p,len(lines))
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'All 15 contents entries on page 17 train; first 3 body lines page18 dev; first5 body lines page19 final test. First2 page23 lines separate body-text stress dev. Selected before recognizer predictions on these lines.','region_policy':'Operator-selected body regions exclude decorative frames and headings. Line locations inside those regions are detected automatically. Report separately from fully automatic page processing.','annotation_policy':'Literal visual transcription by AI assistant against rendered Unicode chart, independently of OCR output; not expert-certified. Space before distant number column. Preserve visible word separators. Unresolved identity must be marked, not silently inferred.','lines':records}
(data/'protocol.json').write_text(json.dumps(protocol,indent=2))
