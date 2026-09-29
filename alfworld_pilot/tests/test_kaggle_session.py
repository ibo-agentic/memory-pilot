"""Tests for kaggle_session.py's find_attached_dataset and restore_folder --
the parts of the copy_in/copy_out_and_version round trip that are actually
testable locally without a real Kaggle session. copy_in/copy_out_and_version
themselves still inherit that documented "untested outside a real Kaggle
session" limitation (they shell out to the real `kaggle` CLI); what IS
tested here is the path-discovery and layout-detection logic, including the
two real layouts a real Kaggle session (2026-09-29 handoff test) actually
produced that this project had NOT assumed: the dataset mounted at
/kaggle/input/datasets/<owner>/<name>/ rather than /kaggle/input/<name>/,
and its uploaded logs.zip auto-extracted flat into the dataset root rather
than preserved as logs/ or logs.zip."""

from __future__ import annotations

import zipfile

import pytest

from alfworld_pilot import kaggle_session
from alfworld_pilot.kaggle_session import _zip_folder, copy_in, find_attached_dataset, restore_folder


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


def test_restore_folder_flat_layout_exact_reported_structure(tmp_path):
    # Exactly the structure a real Kaggle session produced: Kaggle
    # auto-extracted logs.zip directly into the dataset root instead of
    # preserving a logs/ subfolder.
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "handoff_test_jobs.json").write_text("[]")
    (src_root / "handoff_test_results.jsonl").write_text('{"job_id": "log_0"}\n')

    dest = tmp_path / "logs_kaggle"
    restored = restore_folder(src_root, "logs", dest, other_names=("cache",))

    assert restored is True
    assert (dest / "handoff_test_jobs.json").read_text() == "[]"
    assert (dest / "handoff_test_results.jsonl").read_text() == '{"job_id": "log_0"}\n'


def test_restore_folder_flat_layout_excludes_other_known_folder_names(tmp_path):
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "handoff_test_jobs.json").write_text("[]")
    (src_root / "cache").mkdir()  # a separate, still-properly-named "cache" folder
    (src_root / "cache" / "some_cached_response.json").write_text("{}")

    dest = tmp_path / "logs_kaggle"
    restore_folder(src_root, "logs", dest, other_names=("cache",))

    assert (dest / "handoff_test_jobs.json").exists()
    assert not (dest / "cache").exists()  # not pulled into the logs destination


def test_find_attached_dataset_original_assumed_layout(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    (input_root / "memory-pilot-handoff-test").mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    found = find_attached_dataset("iboooh/memory-pilot-handoff-test")

    assert found == input_root / "memory-pilot-handoff-test"


def test_find_attached_dataset_real_kaggle_layout(tmp_path, monkeypatch):
    # The actual layout a real Kaggle session mounted an attached dataset at
    # (2026-09-29 handoff test): /kaggle/input/datasets/<owner>/<name>/, not
    # /kaggle/input/<name>/.
    input_root = tmp_path / "input"
    (input_root / "datasets" / "iboooh" / "memory-pilot-handoff-test").mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    found = find_attached_dataset("iboooh/memory-pilot-handoff-test")

    assert found == input_root / "datasets" / "iboooh" / "memory-pilot-handoff-test"


def test_find_attached_dataset_recursive_fallback(tmp_path, monkeypatch):
    # Neither of the two known layouts -- some other nesting entirely --
    # still found via the recursive search.
    input_root = tmp_path / "input"
    (input_root / "something" / "unexpected" / "memory-pilot-handoff-test").mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    found = find_attached_dataset("iboooh/memory-pilot-handoff-test")

    assert found == input_root / "something" / "unexpected" / "memory-pilot-handoff-test"


def test_find_attached_dataset_returns_none_when_not_found(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    input_root.mkdir()
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    found = find_attached_dataset("iboooh/memory-pilot-handoff-test")

    assert found is None


def test_find_attached_dataset_prefers_original_layout_when_both_exist(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    (input_root / "memory-pilot-handoff-test").mkdir(parents=True)
    (input_root / "datasets" / "iboooh" / "memory-pilot-handoff-test").mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    found = find_attached_dataset("iboooh/memory-pilot-handoff-test")

    assert found == input_root / "memory-pilot-handoff-test"


# --- _zip_folder: bakes the folder name in as the archive's own top-level
# directory, which is the actual fix (not --dir-mode) for Kaggle's mount-time
# flattening. ---


def test_zip_folder_prefixes_archive_paths_with_the_folder_name(tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "results.jsonl").write_text('{"a": 1}\n')
    (folder / "sub" / "state.json").parent.mkdir()
    (folder / "sub" / "state.json").write_text('{"b": 2}\n')

    out_path = tmp_path / "logs.zip"
    _zip_folder(folder, "logs", out_path)

    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())
    assert names == {"logs/results.jsonl", "logs/sub/state.json"}


def test_zip_folder_then_restore_folder_round_trip(tmp_path):
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "results.jsonl").write_text('{"a": 1}\n')

    src_root = tmp_path / "dataset_root"
    src_root.mkdir()
    _zip_folder(folder, "logs", src_root / "logs.zip")

    dest = tmp_path / "restored"
    restored = restore_folder(src_root, "logs", dest)

    assert restored is True
    assert (dest / "results.jsonl").read_text() == '{"a": 1}\n'


# --- copy_in: both logs and a non-empty cache present, using the real
# _zip_folder-produced format -- the scenario the flat-layout fallback is
# unsafe for, which is why copy_in must not need it here at all. ---


def test_copy_in_restores_both_logs_and_nonempty_cache(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    dataset_root = input_root / "memory-pilot-handoff-test"
    dataset_root.mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    logs_src = tmp_path / "logs_src"
    logs_src.mkdir()
    (logs_src / "handoff_test_results.jsonl").write_text('{"job_id": "log_0"}\n')
    _zip_folder(logs_src, "logs", dataset_root / "logs.zip")

    cache_src = tmp_path / "cache_src"
    cache_src.mkdir()
    (cache_src / "abcdef123.json").write_text('{"request": {}, "response": {}}\n')
    _zip_folder(cache_src, "cache", dataset_root / "cache.zip")

    logs_dest = tmp_path / "working" / "logs_kaggle"
    cache_dest = tmp_path / "working" / "cache_kaggle"
    copy_in("iboooh/memory-pilot-handoff-test", logs_dest, cache_dest)

    assert (logs_dest / "handoff_test_results.jsonl").read_text() == '{"job_id": "log_0"}\n'
    assert (cache_dest / "abcdef123.json").read_text() == '{"request": {}, "response": {}}\n'
    # Each folder's own content only, never mixed into the other.
    assert not (logs_dest / "abcdef123.json").exists()
    assert not (cache_dest / "handoff_test_results.jsonl").exists()


def test_copy_in_fails_loudly_when_both_folders_are_ambiguously_flat(tmp_path, monkeypatch):
    # Simulates an OLD-format dataset (pushed before the _zip_folder fix)
    # where BOTH logs and cache got flattened into the same dataset root --
    # genuinely unresolvable, must fail rather than guess or mix.
    input_root = tmp_path / "input"
    dataset_root = input_root / "memory-pilot-handoff-test"
    dataset_root.mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    (dataset_root / "handoff_test_results.jsonl").write_text('{"job_id": "log_0"}\n')  # "belongs" to logs
    (dataset_root / "abcdef123.json").write_text('{"request": {}}\n')  # "belongs" to cache

    logs_dest = tmp_path / "working" / "logs_kaggle"
    cache_dest = tmp_path / "working" / "cache_kaggle"

    with pytest.raises(RuntimeError, match="cannot safely fall back"):
        copy_in("iboooh/memory-pilot-handoff-test", logs_dest, cache_dest)


def test_copy_in_flat_fallback_still_works_when_only_one_folder_is_actually_ambiguous(tmp_path, monkeypatch):
    # Mixed old/new dataset: cache has a PROPER form (cache.zip, e.g. pushed
    # by a session already running the _zip_folder fix), so it's not
    # "missing" at all -- only logs is flat (pre-fix leftover). Exactly one
    # folder is genuinely ambiguous, so the flat fallback is safe here: any
    # leftover root file, once cache.zip's own content is excluded, can only
    # belong to logs.
    input_root = tmp_path / "input"
    dataset_root = input_root / "memory-pilot-handoff-test"
    dataset_root.mkdir(parents=True)
    monkeypatch.setattr(kaggle_session, "KAGGLE_INPUT_ROOT", input_root)

    (dataset_root / "handoff_test_jobs.json").write_text("[]")
    (dataset_root / "handoff_test_results.jsonl").write_text('{"job_id": "log_0"}\n')
    cache_src = tmp_path / "cache_src"
    cache_src.mkdir()
    (cache_src / "abcdef123.json").write_text('{"request": {}}\n')
    _zip_folder(cache_src, "cache", dataset_root / "cache.zip")

    logs_dest = tmp_path / "working" / "logs_kaggle"
    cache_dest = tmp_path / "working" / "cache_kaggle"
    copy_in("iboooh/memory-pilot-handoff-test", logs_dest, cache_dest)

    assert (logs_dest / "handoff_test_jobs.json").exists()
    assert (logs_dest / "handoff_test_results.jsonl").exists()
    assert (cache_dest / "abcdef123.json").exists()
    assert not (logs_dest / "abcdef123.json").exists()  # cache's own content never leaked into logs
