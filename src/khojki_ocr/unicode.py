"""Conservative validation: report issues; never guess a visual character."""
import unicodedata

LETTERS={chr(c) for c in range(0x11200,0x1122c) if c!=0x11212}|{chr(c) for c in [0x1123f,0x11240]}
VOWELS={chr(c) for c in range(0x1122c,0x11234)}|{'\U00011241'}
MARKS=VOWELS|{chr(c) for c in range(0x11234,0x11238)}|{'\U0001123e'}
ALLOWED={chr(c) for c in range(0x11200,0x11242) if c!=0x11212}|{chr(c) for c in range(0xae6,0xaf0)}|set(' .,()\n\t')

def inspect_text(text):
    issues=[]
    for i,c in enumerate(text):
        if c not in ALLOWED:
            issues.append({'index':i,'code':'unsupported_codepoint','value':f'U+{ord(c):04X}','message':unicodedata.name(c,'UNASSIGNED')})
        if c in MARKS and (i==0 or text[i-1] not in LETTERS|MARKS):
            issues.append({'index':i,'code':'orphan_combining_sign'})
    forbidden={
        '\U00011200\U0001122c':'Use independent AA U+11201, after confirming the source.',
        '\U0001122c\U00011230':'O is U+11232, not AA+E.',
        '\U0001122c\U00011231':'AU is U+11233, not AA+AI.',
        '\U00011230\U0001122c':'O is U+11232, not E+AA.',
        '\U00011231\U0001122c':'AU is U+11233, not AI+AA.',
    }
    for seq,message in forbidden.items():
        start=0
        while (i:=text.find(seq,start))!=-1:
            issues.append({'index':i,'code':'non_atomic_vowel','message':message});start=i+1
    for i in range(len(text)-1):
        if text[i] in VOWELS and text[i+1] in VOWELS:
            issues.append({'index':i,'code':'multiple_vowel_signs','message':'Review against the scan; no automatic correction applied.'})
    return {'valid_inventory':not any(x['code']=='unsupported_codepoint' for x in issues),'issues':issues,
            'codepoints':[{'character':c,'codepoint':f'U+{ord(c):04X}','name':unicodedata.name(c,'UNASSIGNED')} for c in text],
            'utf8_roundtrip':text.encode('utf-8').decode('utf-8')==text,'normalization':'NFC','nfc_changed':unicodedata.normalize('NFC',text)!=text}


def plausible_prefix(text):
    """Optional experimental beam filter; not a visual correctness verifier.

    Only rejects orphan marks and explicitly non-atomic adjacent vowel forms.
    This deliberately contains no dictionary or language frequency prior.
    """
    if not text:return True
    if text[-1] in MARKS and (len(text)==1 or text[-2] not in LETTERS|MARKS):return False
    if len(text)>1:
        if text[-1] in VOWELS and text[-2] in VOWELS:return False
        if text[-2:]=='\U00011200\U0001122c':return False
    return True
