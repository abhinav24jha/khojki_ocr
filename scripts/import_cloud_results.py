"""Verify UI-exported cloud archives before extracting experiment results."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
records=json.loads(a.manifest.read_text())
for item in records:
    name=item['name']
    if Path(name).name!=name:raise ValueError('Archive name must be a filename')
    archive=ROOT/'cloud'/name
    assert archive.stat().st_size==item['bytes'],name
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['sha256'],name
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None,name
        for n in z.namelist():
            path=Path(n)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in {'experiments','results','models'}:raise ValueError(f'Unexpected member: {n}')
        z.extractall(ROOT)
    print(name,'verified and extracted')
print('All exported bytes verified. Runtime can now be terminated.')
