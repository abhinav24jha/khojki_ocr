p=ROOT/'scripts/finetune.py'
s=p.read_text().replace('    if a.freeze_rnn:model.rnn.eval()\n','').replace('    for param in model.rnn.parameters():param.requires_grad_(False)','    for param in model.rnn.parameters():param.requires_grad_(False)\n    model.rnn.dropout=0.0')
p.write_text(s)
CASES={'r4-control':[],'r4-word-red':['--word-fraction','.4','--word-geometry','preserve','--legacy-rgb-word-padding'],'r4-word-white':['--word-fraction','.4','--word-geometry','preserve'],'r4-freezebn':['--freeze-bn'],'r4-word-white-freezebn':['--word-fraction','.4','--word-geometry','preserve','--freeze-bn'],'r4-freezernn':['--freeze-rnn']}
def train(name,steps):
 folder=EXP/name;folder.mkdir(exist_ok=True);local=folder/'last.pt';resume=local.exists();checkpoint=local if resume else EXP/'precision-packed/best.pt'
 if resume and torch.load(local,map_location='cpu',weights_only=False)['step']>=steps:print('Already complete',name,steps);return
 run([ROOT/'scripts/finetune.py','--checkpoint',checkpoint,'--annotations',ANN,'--steps',steps,'--lr','0.00005','--seed','503','--layout','robust','--bold-replay','--packed','--device','cuda','--output',name,*CASES[name],*(['--resume'] if resume else [])],EXP/(name+f'-{steps}.log'))
for name in CASES:train(name,600)
run([ROOT/'scripts/select_round4_model.py','--phase','pilot'],EXP/'round4-pilot.log')
report=json.loads((EXP/'round4-pilot-selection.json').read_text());eligible=sorted([r for r in report['experiments'] if r['name'] in CASES and r['errors']<=16],key=lambda r:r['errors'])[:2]
print('Extension rule: at most two pilots with <=16/405 edits; chosen',[r['name'] for r in eligible])
for row in eligible:train(row['name'],3000)
run([ROOT/'scripts/select_round4_model.py','--phase','final'],EXP/'round4-final.log')
# These already-viewed test sets are regression diagnostics, not new blind estimates.
for old in [2,3]:
 run([ROOT/'scripts/evaluate_pages.py','--checkpoint',ROOT/'models/round4-selected.pt','--annotations',ROOT/f'data/round{old}/test.json','--split','test','--device','cuda','--output',RES/f'round4-regression-r{old}'],RES/f'round4-regression-r{old}.log')
