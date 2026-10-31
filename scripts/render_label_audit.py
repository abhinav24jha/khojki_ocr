from pathlib import Path
from PIL import Image,ImageDraw
import sys
sys.path.insert(0,str(Path(__file__).parent))
from write_annotations import encode
from khojki_ocr.render import Renderer
from khojki_ocr.word_augmentation import training_words
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/round4'
rows={}
for f in ['data/annotations.json','data/round2/test.json','data/round3/test.json']:
 for r in json.loads((ROOT/f).read_text()):rows[r['id']]=r
specs=[('p17r-10',4,['ta ba :','ta ba.i :']),('p20r-09',4,['ga.u aa ba :','ga.ii aa na :']),('p18r-14',0,['u pa.e :','u pa.o :','u tha.e :','u tha.o :']),('p18r-14',3,['pa.e nya i :','pa.o i :','pa.o i :','pa.o nya :']),('p19r-06',2,['bha aa :','bba.ii aa :','bha.ii aa :','ma aa :']),('p20r-11',2,['sa ta ga.u ra ka.e :','sa ta ga.u ra ka.u.an :']),('p20r-10',4,['bha ba :','la.ii ta.ii :','la.ii ba.ii :','la.aa ba :']),('p20r-10',3,['ha.aa tha.u.an :','ha.aa pa.u.an :','ha.aa kha.u.an :'])]
ren=Renderer(ROOT/'assets/NotoSansKhojki-Regular.ttf',52)
for id,i,alternatives in specs:
 r=rows[id];words=training_words(Image.open(ROOT/'data'/r['image']),r['text'],preserve_height=True);source=words[i][0];source=source.resize((source.width*3,source.height*3));sheet=Image.new('RGB',(1100,180+len(alternatives)*100),'white');d=ImageDraw.Draw(sheet);sheet.paste(source,(10,20));d.text((600,30),id+f' word {i+1}',fill='black')
 for j,spec in enumerate(alternatives):
  y=180+j*100;sheet.paste(ren.render(encode(spec)),(10,y));d.text((600,y+15),spec,fill='black')
 sheet.save(OUT/f'audit-{id}-w{i+1}.png')
