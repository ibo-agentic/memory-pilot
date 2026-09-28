"""Tests for kaggle_session.py's restore_folder -- the part of the
copy_in/copy_out_and_version round trip that's actually testable locally
without a real Kaggle session. copy_in/copy_out_and_version themselves still
inherit that documented "untested outside a real Kaggle session" limitation
(they shell out to the real `kaggle` CLI and touch /kaggle/input); what IS
tested here is the detection/extraction logic a real Kaggle Dataset input
must be handled by, for both forms `kaggle datasets create/version
--dir-mode zip` can leave a folder in on the input side: a plain extracted
directory, or a zip archive (with either of the two zip internal layouts
Kaggle might produce)."""

from __future__ import annotations

import zipfile

from alfworld_pilot.kaggle_session import restore_folder


def test_restore_folder_from_plain_directory(tmp_path):
    src_root = tmp_path / "src"
    (src_root / "logs").mkdir(parents=True)
    (src_root / "logs" / "results.jsonl").write_text('{"a": 1}\n')

    dest = tmp_path / "dest"
    restored = restore_folder(src_root, "logs", dest)

    assert restored is True
    assert (dest / "results.jsonl").read_text() == '{"a": 1}\n'


def test_restore_folder_from_zip_with_contents_at_root(tmp_path):
    src_root = tmp_path / "src"
    src_root.mkdir()
    with zipfile.ZipFile(src_root / "logs.zip", "w") as zf:
        zf.writestr("results.jsonl", '{"a": 1}\n')
        zf.writestr("subdir/state.json", '{"b": 2}\n')

    dest = tmp_path / "dest"
    restored = restore_folder(src_root, "logs", dest)

    assert restored is True
    assert (dest / "results.jsonl").read_text() == '{"a": 1}\n'
    assert (dest / "subdir" / "state.json").read_text() == '{"b": 2}\n'


def test_restore_folder_from_zip_with_nested_folder(tmp_path):
    # The other plausible --dir-mode zip layout: the zip contains one
    # top-level "logs/" folder wrapping the real contents.
    src_root = tmp_path / "src"
    src_root.mkdir()
    with zipfile.ZipFile(src_root / "logs.zip", "w") as zf:
        zf.writestr("logs/results.jsonl", '{"a": 1}\n')

    dest = tmp_path / "dest"
    restored = restore_folder(src_root, "logs", dest)

    assert restored is True
    assert (dest / "results.jsonl").read_text() == '{"a": 1}\n'
    assert not (dest / "logs").exists()  # unwrapped, not double-nested


def test_restore_folder_prefers_plain_directory_over_zip_if_both_exist(tmp_path):
    src_root = tmp_path / "src"
    (src_root / "logs").mkdir(parents=True)
    (src_root / "logs" / "from_dir.txt").write_text("dir")
    with zipfile.ZipFile(src_root / "logs.zip", "w") as zf:
        zf.writestr("from_zip.txt", "zip")

    dest = tmp_path / "dest"
    restore_folder(src_root, "logs", dest)

    assert (dest / "from_dir.txt").exists()
    assert not (dest / "from_zip.txt").exists()


def test_restore_folder_returns_false_when_neither_form_exists(tmp_path):
    src_root = tmp_path / "src"
    src_root.mkdir()

    dest = tmp_path / "dest"
    restored = restore_folder(src_root, "logs", dest)

    assert restored is False
    assert not dest.exists()
