import itertools
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from khojki_ocr.metrics import score, alignment
from khojki_ocr.unicode import inspect_text
from khojki_ocr.model import decode, VOCAB
from khojki_ocr.decoding import beam_decode
from khojki_ocr.pipeline import process_page

ROOT=Path(__file__).resolve().parents[1]

def test_unicode_cer_counts_scalars_not_utf8_bytes():
    reference='\U00011208\U0001122c\U0001123a૧'
    result=score(reference,'\U00011208\U0001122c\U0001123a૨')
    assert result['reference_characters']==4
    assert result['errors']==1 and result['cer']==.25
    assert score('','x')['cer'] is None
    assert score('abc','ac')['delete']==1
    assert score('ac','abc')['insert']==1

def test_ctc_repeated_letter_requires_blank():
    logits=torch.full((4,1,len(VOCAB)+1),-20.)
    for t,c in enumerate([1,1,0,1]):logits[t,0,c]=0
    assert decode(logits)==[VOCAB[0]*2]

def test_beam_matches_exhaustive_path_sums():
    probs=np.array([[.5,.3,.2],[.4,.35,.25],[.3,.5,.2]])
    totals={}
    for path in itertools.product(range(3),repeat=3):
        collapsed=[];last=0
        for i in path:
            if i and i!=last:collapsed.append(VOCAB[i-1])
            last=i
        text=''.join(collapsed)
        totals[text]=totals.get(text,0)+np.prod([probs[t,c] for t,c in enumerate(path)])
    assert beam_decode(np.log(probs),width=100)==max(totals,key=totals.get)

def test_validator_reports_does_not_repair():
    invalid='\U00011200\U0001122c\U00011230'
    out=inspect_text(invalid)
    assert 'non_atomic_vowel' in {i['code'] for i in out['issues']}
    assert ''.join(c['character'] for c in out['codepoints'])==invalid
    assert inspect_text('\U00011208\U00011232\U0001123a૧')['issues']==[]
    assert not inspect_text('क')['valid_inventory']
    assert inspect_text('\U0001122c')['issues'][0]['code']=='orphan_combining_sign'

def test_frozen_test_is_disjoint_and_unchanged():
    rows=json.loads((ROOT/'data/annotations.json').read_text())
    frozen=json.loads((ROOT/'data/annotations-v2-frozen.json').read_text())
    assert [(r['id'],r['text'],r['split']) for r in rows]==[(r['id'],r['text'],r['split']) for r in frozen]
    groups={s:{r['id'] for r in rows if r['split']==s} for s in ['train','dev','test']}
    assert len(groups['test'])==10
    assert not groups['test'] & (groups['train']|groups['dev'])
    assert all(r['page']!=19 for r in rows if r['split'] in ['train','dev'])

def test_blank_page_writes_valid_empty_artifacts(tmp_path):
    result=process_page(Image.new('RGB',(600,800),'white'),lambda im:'bad',tmp_path)
    assert result['line_count']==0
    assert (tmp_path/'transcription.txt').read_text()==''
    assert json.loads((tmp_path/'result.json').read_text())['lines']==[]
    assert (tmp_path/'review.html').exists()

def test_export_html_escapes_text_and_keeps_utf8(tmp_path):
    from khojki_ocr.pipeline import write_review
    text='<script>\U00011208</script>'
    report={'lines':[{'id':'test','crop':'test.png','text':text,'validation':inspect_text(text)}]}
    write_review(report,tmp_path)
    html=(tmp_path/'review.html').read_text()
    assert '&lt;script&gt;' in html and '<script>\U00011208' not in html

def test_pdf_native_and_rendered_readers(tmp_path):
    from khojki_ocr.pipeline import page_images
    source=tmp_path/'one-page.pdf'
    Image.new('RGB',(200,300),'white').save(source,'PDF',resolution=72)
    assert [(n,im.size) for n,im in page_images(source,[1],native=True)]==[(1,(200,300))]
    rendered=list(page_images(source,[1],native=False))
    assert len(rendered)==1 and rendered[0][1].width>200

def test_robust_layout_keeps_faint_refrain_and_number_gap():
    from khojki_ocr.segment import detect_lines
    from khojki_ocr.recognize import split_columns
    page=Image.open(ROOT/'data/pages/p20.png')
    robust,_=detect_lines(page,mode='robust')
    # This page has sixteen visually audited body lines, including a short refrain.
    assert len(robust)==16
    for id in ['p19r-01','p19r-02']:
        assert len(split_columns(Image.open(ROOT/f'data/lines/{id}.png'),'robust'))==2

def test_word_augmentation_rejects_separator_label_mismatch():
    from khojki_ocr.word_augmentation import training_words,concatenate_words,SEP
    rows=json.loads((ROOT/'data/annotations.json').read_text())
    row=next(r for r in rows if r['id']=='p17r-01')
    image=Image.open(ROOT/'data'/row['image'])
    bank=training_words(image,row['text'])
    assert len(bank)==row['text'].count(SEP)
    assert training_words(image,row['text']+SEP)==[]
    combined,text=concatenate_words([bank[0],bank[0]])
    assert text==bank[0][1]*2 and combined.width>bank[0][0].width

def test_round2_references_frozen_and_not_used_for_training():
    import hashlib
    path=ROOT/'data/round2/test.json'
    meta=json.loads((ROOT/'data/round2/test-freeze.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    test=json.loads(path.read_text());dev=json.loads((ROOT/'data/round2/train-dev.json').read_text())
    assert not {r['id'] for r in test}&{r['id'] for r in dev}
    assert len(test)==5 and all(r['split']=='test' for r in test)

def test_packed_recurrence_ignores_extra_batch_padding():
    from khojki_ocr.model import CRNN
    torch.manual_seed(9)
    model=CRNN(packed=True).eval()
    short=torch.zeros(1,1,48,80);short[:,:,:,:32]=torch.rand(1,1,48,32)
    wider=torch.zeros(1,1,48,128);wider[:,:,:,:80]=short
    with torch.no_grad():
        a=model(short,torch.tensor([8]))[:8]
        b=model(wider,torch.tensor([8]))[:8]
    assert torch.allclose(a,b,atol=1e-6)


def test_optional_beam_filter_is_structural_only():
    from khojki_ocr.unicode import plausible_prefix
    assert plausible_prefix('\U00011208\U0001122c\U00011234')
    assert not plausible_prefix('\U0001122c')
    assert not plausible_prefix('\U00011208\U0001122c\U00011230')
    assert plausible_prefix('\U00011219') and plausible_prefix('\U00011224')


def test_preserved_word_geometry_does_not_shrink_ink_height():
    from khojki_ocr.word_augmentation import concatenate_words
    im=Image.new('L',(60,48),255);ImageDraw.Draw(im).rectangle((10,10,50,37),fill=0)
    combined,text=concatenate_words([(im,'x'),(im,'y')],preserve_height=True)
    assert combined.height==48 and text=='xy'
    y,_=np.where(np.asarray(combined)<128)
    assert (int(y.min()),int(y.max()))==(10,37)

def test_compact_ensemble_loads_and_reproduces_identical_members(tmp_path):
    from khojki_ocr.model import CRNN
    from khojki_ocr.recognize import Recognizer
    torch.manual_seed(11)
    state=CRNN().state_dict()
    single=tmp_path/'single.pt';ensemble=tmp_path/'ensemble.pt'
    torch.save({'model':state,'vocab':VOCAB,'height':48,'layout':'robust'},single)
    torch.save({'ensemble_states':[{'model':state,'packed':False}]*2,'vocab':VOCAB,'height':48,'layout':'robust'},ensemble)
    a=Recognizer(single);b=Recognizer(ensemble)
    image=Image.new('L',(80,48),255)
    assert len(b.models)==2 and a(image)==b(image)
