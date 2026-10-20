"""Command lines built by the download/unzip helpers."""

import pathlib
import shlex

import pytest

import LION.utils.utils as utils


@pytest.fixture
def commands(monkeypatch):
    calls = []

    def fake_run_cmd(cmd, verbose=True):
        calls.append(shlex.split(cmd))
        if calls[-1][0] in fake_run_cmd.fail:
            raise RuntimeError("failed")
        return ""

    fake_run_cmd.fail = set()
    monkeypatch.setattr(utils, "run_cmd", fake_run_cmd)
    return calls, fake_run_cmd.fail


def test_wget_writes_to_the_requested_file(commands, tmp_path):
    calls, _ = commands
    target = tmp_path / "my data" / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls == [["wget", "https://example.org/file.zip", "-O", str(target)]]


def test_curl_fallback(commands, tmp_path):
    calls, fail = commands
    fail.add("wget")
    target = tmp_path / "file.zip"
    utils.download_file("https://example.org/file.zip", target)
    assert calls[-1] == [
        "curl",
        "-fL",
        "https://example.org/file.zip",
        "-o",
        str(target),
    ]


def test_7z_fallback_output_folder(commands, tmp_path):
    calls, fail = commands
    fail.add("unzip")
    archive = tmp_path / "a b.zip"
    archive.write_bytes(b"")
    out = tmp_path / "out dir"
    utils.unzip_file(archive, out)
    assert calls[0] == ["unzip", str(archive), "-d", str(out)]
    assert calls[1] == ["7z", "x", str(archive), f"-o{out}", "-y"]
