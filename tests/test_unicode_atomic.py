"""Atomic-vowel examples from Unicode 17 Table 15-4; no ML runtime needed."""
import pytest
from khojki_ocr.unicode import inspect_text

@pytest.mark.parametrize('sequence',[
 '\U00011200\U0001122c','\U00011240\U0001122e','\U00011206\U0001122c',
 '\U00011200\U00011231','\U00011200\U00011233','\U00011200\U0001122c\U00011231',
 '\U0001122c\U00011230',
])
def test_table_15_4_non_atomic_forms_are_reported_without_repair(sequence):
 report=inspect_text(sequence)
 assert any(i['code']=='non_atomic_vowel' for i in report['issues'])
 assert ''.join(c['character'] for c in report['codepoints'])==sequence

@pytest.mark.parametrize('text',['\U00011201','\U00011202','\U00011203','\U00011205','\U00011207','\U00011208\U00011232'])
def test_atomic_vowels_and_consonant_plus_o_remain_valid(text):
 assert inspect_text(text)['issues']==[]

def test_similar_independent_e_and_pa_i_are_not_silently_equated():
 for text in ['\U00011204','\U0001121f\U0001122d']:
  report=inspect_text(text)
  assert report['issues']==[]
  assert ''.join(c['character'] for c in report['codepoints'])==text
