"""Prepare the next experiment protocol and visual word-bank audit, without ML."""
from pathlib import Path
import json,hashlib,datetime
from PIL import Image,ImageDraw
from khojki_ocr.word_augmentation import training_words
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/round2';OUT.mkdir(exist_ok=True)
rows=json.loads((ROOT/'data/annotations.json').read_text())
for row in rows:
    if row['split']=='test':row['split']='dev';row['former_split']='test'
(OUT/'train-dev.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
bank=[];tiles=[]
for row in rows:
    if row['split']!='train':continue
    words=training_words(Image.open(ROOT/'data'/row['image']),row['text'])
    for i,(im,text) in enumerate(words):
        name=f'{row["id"]}-word-{i:02d}.png';im.save(OUT/name)
        bank.append({'source_id':row['id'],'text':text,'image':name});tiles.append((im,name))
sheet=Image.new('RGB',(1400,((len(tiles)+5)//6)*110),'white');d=ImageDraw.Draw(sheet)
for i,(im,name) in enumerate(tiles):
    x=(i%6)*230;y=(i//6)*110;im=im.convert('RGB');im.thumbnail((220,75));sheet.paste(im,(x,y+20));d.text((x,y),name,fill='black')
sheet.save(OUT/'training-word-bank.png')
(OUT/'word-bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2))
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'training':'Original 15 p17 lines only; printed separator word crops and recombinations use training data exclusively.',
 'development':'Original 6 dev plus original 10 test lines, now explicitly development after inspecting their failures. The previous 8.4% remains a historical baseline only.',
 'new_test_selection':['p19r-06','p19r-07','p19r-08','p19r-09','p19r-10'],
 'new_test_status':'Selected before round2 training. References will be frozen before inspecting recognizer predictions on these lines. These lines share a page/book with development, so not book-disjoint.',
 'word_bank_size':len(bank),'train_dev_sha256':hashlib.sha256((OUT/'train-dev.json').read_bytes()).hexdigest(),
 'experiments':'Small paired runs: robust layout only; 300-step whole-line fine-tune; 300-step real-word recombination fine-tune; expand promising runs after development comparison. Add bold-font synthetic coverage if supported by first pilot.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('Prepared',len(bank),'training word crops;',len(rows),'train/development lines; fresh test excluded.')
