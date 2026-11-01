"""Verify UI-exported cloud archives before extracting experiment results."""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {"experiments", "results", "models"}


def import_archive(archive: Path, record: dict, root: Path) -> None:
    """Reject changed bytes and unsafe members before writing any archive content."""
    if archive.stat().st_size != record["bytes"]:
        raise ValueError(f"Archive size mismatch: {archive.name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive.name}")
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {archive.name}")
        seen = set()
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            target = root.joinpath(*path.parts)
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or path.parts[0] not in ALLOWED_ROOTS
                or stat.S_ISLNK(member.external_attr >> 16)
                or not target.resolve().is_relative_to(root)
                or path in seen
            ):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            seen.add(path)
        bundle.extractall(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    for record in records:
        name = record["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Archive name must be a filename")
        import_archive(ROOT / "cloud" / name, record, ROOT)
        print(name, "verified and extracted")
    print("All exported bytes verified. Runtime can now be terminated.")


if __name__ == "__main__":
    main()
