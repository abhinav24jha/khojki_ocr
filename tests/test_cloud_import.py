"""Artifact integrity and extraction boundaries, using tiny synthetic archives."""

import hashlib
import importlib.util
import stat
import zipfile

import pytest


@pytest.fixture
def importer(project_root):
    spec = importlib.util.spec_from_file_location(
        "cloud_import", project_root / "scripts/import_cloud_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.import_archive


def manifest(archive):
    return {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }


def test_valid_archive_verifies_and_extracts_expected_bytes(importer, tmp_path):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("results/metrics.json", '{"cer": 0.25}')
        bundle.writestr("experiments/run/best.pt", b"checkpoint bytes")
    output = tmp_path / "project"
    importer(archive, manifest(archive), output)
    assert (output / "results/metrics.json").read_text() == '{"cer": 0.25}'
    assert (output / "experiments/run/best.pt").read_bytes() == b"checkpoint bytes"


@pytest.mark.parametrize("field,bad_value", [("bytes", -1), ("sha256", "0" * 64)])
def test_changed_archive_rejected_before_any_extraction(
    importer, tmp_path, field, bad_value
):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("results/good.json", "{}")
    record = {**manifest(archive), field: bad_value}
    output = tmp_path / "project"
    with pytest.raises(ValueError, match="mismatch"):
        importer(archive, record, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "/results/absolute.txt",
        "results/../../escape.txt",
        "src/unexpected.py",
        "results\\escape.txt",
    ],
)
def test_unsafe_members_reject_entire_archive_before_writing(
    importer, tmp_path, member
):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("results/otherwise-valid.json", "{}")
        bundle.writestr(member, "bad")
    output = tmp_path / "project"
    with pytest.raises(ValueError, match="Unexpected archive member"):
        importer(archive, manifest(archive), output)
    assert not output.exists()


def test_zip_symlinks_cannot_escape_extraction(importer, tmp_path):
    archive = tmp_path / "export.zip"
    member = zipfile.ZipInfo("results/link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "../../outside")
    with pytest.raises(ValueError, match="Unexpected archive member"):
        importer(archive, manifest(archive), tmp_path / "project")


def test_existing_parent_symlink_cannot_redirect_writes(importer, tmp_path):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("results/escape.txt", "bad")
    output, outside = tmp_path / "project", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unexpected archive member"):
        importer(archive, manifest(archive), output)
    assert not (outside / "escape.txt").exists()


def test_duplicate_members_are_rejected(importer, tmp_path):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("results/same.json", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            bundle.writestr("results/same.json", "second")
    with pytest.raises(ValueError, match="Unexpected archive member"):
        importer(archive, manifest(archive), tmp_path / "project")
