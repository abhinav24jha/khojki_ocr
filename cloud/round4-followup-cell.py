# Offline-prepared follow-up: matched low learning rate and source-audited label correction.
bundle=Path('/content/khojki-round4-followup.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='df4c43f08a52be96bddfb5434e979f1cb3188dbf49005231da5be00047733fca'
with zipfile.ZipFile(bundle) as z:z.extractall(ROOT)
run([ROOT/'scripts/round4_followup.py'],EXP/'round4-followup.log',2400)
run([ROOT/'scripts/evaluate_round4_final.py'],RES/'round4-final-evaluation.log',900)
